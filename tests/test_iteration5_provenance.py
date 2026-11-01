"""
tests/test_iteration5_provenance.py
Iteration 5 Regression Tests:
1. Split manifest write-once enforcement (refuses overwrite).
2. Model factory builds models from config only.
3. BiLSTM cross-pooling checkpoint loading raises without silent key pruning.
4. Weight initialization standard deviations match theoretical scaling (xavier_normal, he_normal).
5. Shape trace sequence length <= max_len from real DataLoader batch.
"""

import math
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.case_builder import (
    MultimodalCaseItem,
    create_stratified_split,
)
from src.data_loader import (
    ForensicTokenizer,
    TextEvidenceDataset,
    derive_max_len,
)
from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    SyntheticDataVAE,
    init_weights,
)
from src.models.factory import (
    build_model,
    get_default_config,
    save_model_checkpoint,
    load_model_from_checkpoint,
)


def test_manifest_write_once_refuses_overwrite():
    """Validates that create_stratified_split refuses to overwrite an existing manifest."""
    cases = [
        MultimodalCaseItem(
            case_id=f"CASE-{i % 2}-{i:04d}",
            text_content=f"Sample text evidence {i}",
            text_hash=f"thash_{i:04d}",
            image_content=None,
            image_hash=f"ihash_{i:04d}",
            label=i % 2,
            metadata={},
            content_hash=f"hash_{i:04d}",
        )
        for i in range(20)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = os.path.join(tmpdir, "split_manifest.json")

        # First call: manifest does not exist, should succeed
        tr, va, te, stats = create_stratified_split(
            cases, manifest_path=manifest_file, allow_overwrite=False
        )
        assert os.path.exists(manifest_file), "Manifest was not written on first call."

        # Second call: manifest exists, must raise FileExistsError
        with pytest.raises(FileExistsError) as exc_info:
            create_stratified_split(
                cases, manifest_path=manifest_file, allow_overwrite=False
            )
        assert "Write-once violation" in str(exc_info.value)

        # Third call with explicit allow_overwrite=True: should succeed
        tr2, va2, te2, stats2 = create_stratified_split(
            cases, manifest_path=manifest_file, allow_overwrite=True
        )
        assert len(tr2) == len(tr)


def test_factory_builds_all_models_from_config():
    """Validates that src/models/factory.py builds every model from config dict only."""
    # CNN
    cfg_cnn = get_default_config("cnn")
    cnn = build_model(cfg_cnn)
    assert isinstance(cnn, VisualEvidenceCNN)

    # BiLSTM Concat
    cfg_bilstm_concat = get_default_config("bilstm", pooling="concat")
    bilstm_c = build_model(cfg_bilstm_concat)
    assert isinstance(bilstm_c, ConversationBiLSTM)
    assert bilstm_c.pooling == "concat"
    assert bilstm_c.attention is None

    # BiLSTM Attention
    cfg_bilstm_attn = get_default_config("bilstm", pooling="attention")
    bilstm_a = build_model(cfg_bilstm_attn)
    assert isinstance(bilstm_a, ConversationBiLSTM)
    assert bilstm_a.pooling == "attention"
    assert bilstm_a.attention is not None

    # ANN
    cfg_ann = get_default_config("ann")
    ann = build_model(cfg_ann)
    assert isinstance(ann, ANNMultimodalRiskClassifier)

    # VAE
    cfg_vae = get_default_config("vae")
    vae = build_model(cfg_vae)
    assert isinstance(vae, SyntheticDataVAE)


def test_bilstm_cross_pooling_loading_raises():
    """
    Validates that loading a BiLSTM checkpoint into a different pooling mode raises ValueError.
    Ensures zero silent key pruning occurs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save an attention-trained checkpoint
        cfg_attn = get_default_config("bilstm", pooling="attention")
        model_attn = build_model(cfg_attn)
        ckpt_path = os.path.join(tmpdir, "bilstm_attn.pt")
        save_model_checkpoint(model_attn, cfg_attn, ckpt_path)

        # Loading with expected_pooling='concat' must raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_model_from_checkpoint(ckpt_path, expected_pooling="concat")
        assert "FATAL POOLING MISMATCH" in str(exc_info.value)
        assert "no silent key pruning permitted" in str(exc_info.value)

        # Loading with expected_pooling='attention' must succeed
        loaded_model, loaded_cfg = load_model_from_checkpoint(ckpt_path, expected_pooling="attention")
        assert loaded_model.pooling == "attention"
        assert loaded_cfg["pooling"] == "attention"


def test_weight_initialization_std_assertions():
    """
    Validates that layer weight std right after init matches theoretical scaling:
    - xavier_normal ~ sqrt(2 / (fan_in + fan_out))
    - he_normal     ~ sqrt(2 / fan_in)
    """
    layer = nn.Linear(512, 512, bias=False)

    # Xavier Normal
    torch.manual_seed(42)
    init_weights(layer, method="xavier_normal")
    std_xavier = layer.weight.std().item()
    expected_xavier = math.sqrt(2.0 / (512 + 512))  # sqrt(2 / 1024) = sqrt(1 / 512) ≈ 0.04419
    assert pytest.approx(std_xavier, rel=0.10) == expected_xavier

    # He Normal (Kaiming Normal)
    torch.manual_seed(42)
    init_weights(layer, method="he_normal")
    std_he = layer.weight.std().item()
    expected_he = math.sqrt(2.0 / 512)  # sqrt(2 / 512) = sqrt(1 / 256) = 0.0625
    assert pytest.approx(std_he, rel=0.10) == expected_he


def test_shape_trace_real_batch_length_le_max_len():
    """
    Part 1d: Tests that a real batch from TextEvidenceDataset has sequence length <= max_len.
    Verifies that derive_max_len drives the tokenizer limit and trace length <= max_len.
    """
    corpus = [
        "Urgent: wire transfer required immediately to account 123456",
        "Hello team, please find attached the monthly project progress report.",
        "Security alert: unauthorized login detected from suspicious IP address.",
        "Lunch meeting rescheduled to 1pm in conference room B.",
    ]
    derived_len = derive_max_len(corpus, percentile=95.0)
    assert derived_len >= 5

    tokenizer = ForensicTokenizer(max_length=derived_len)
    tokenizer.fit_on_texts(corpus)

    dataset = TextEvidenceDataset(corpus, [1, 0, 1, 0], tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch_tokens, batch_labels = next(iter(loader))
    assert batch_tokens.shape[1] == derived_len, f"Batch seq_len {batch_tokens.shape[1]} != max_len {derived_len}"
    assert batch_tokens.shape[1] <= derived_len

    # Trace model with real batch
    cfg = get_default_config("bilstm", vocab_size=tokenizer.vocab_size, max_length=derived_len)
    model = build_model(cfg)
    traces = model.trace_shapes(batch_tokens)

    # Input shape in trace
    input_shape = traces[0]["shape"]
    assert input_shape[1] <= derived_len
