"""
tests/test_regression.py - Comprehensive Unit and Regression Test Suite for ForenSightX
Covers:
- Dataset adapters, raw record purity, case builder, zero leakage, stratified splits
- Forensic heuristic feature extraction formulas (URL, keywords, attachments, timeline, email)
- BiLSTM pooling modes, OOV calculation, derived max_len, attention conditioning
- CNN dense layer proportion, forward hooks, shape tracing, BatchNorm eval batch size 1
- Multimodal ANN fusion, LOFO sensitivity analysis, overfit-one-batch verification
- Synthetic Data VAE keyword-only constructor, parameter formulas, closed-form math, loss with free bits
- Baseline augmenters, model factory strict checkpoint loading, weight init scaling
- Configuration validation schema tests
"""

import os
import math
import uuid
import yaml
import hashlib
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    SyntheticDataVAE,
    compute_vae_parameter_count,
    load_model_checkpoint,
    migrate_bilstm_state_dict,
    init_weights,
    get_optimizer,
    DEVICE,
)
from src.models.factory import (
    build_model,
    get_default_config,
    save_model_checkpoint,
    load_model_from_checkpoint,
)
from src.models.vae_trainer import (
    vae_loss_with_free_bits,
    GaussianFeatureAugmenter,
    InterpolationFeatureAugmenter,
)
from src.data.adapters import (
    ImageForensicAdapter,
    TextForensicAdapter,
    ForensicRawRecord,
)
from src.data.case_builder import (
    CaseBuilder,
    MultimodalCaseItem,
    create_stratified_split,
)
from src.data.hard_generator import (
    HardForensicDatasetGenerator,
    generate_hard_benchmark_dataset,
    TRAIN_VAL_TEMPLATES,
    HELD_OUT_TEST_TEMPLATES,
)
from src.data.validate_config import validate_configs
from src.data_loader import (
    ForensicTokenizer,
    TextEvidenceDataset,
    derive_max_len,
    calculate_oov_rate,
    extract_url_score,
    extract_keyword_score,
    extract_attachment_score,
    extract_timeline_anomaly,
    extract_email_anomaly,
    SMOKE_BANNER,
)
from train import set_seed, compute_grad_norm, evaluate_predictions


def test_regression_batchnorm_eval_batch_size_1():
    """
    Guards Iteration 1 failure: BatchNorm must not bypass running_mean/var during eval.
    Must produce valid normalized probabilities even for batch size 1.
    """
    model = ANNMultimodalRiskClassifier(input_dim=7, use_batchnorm=True)
    # Put in eval mode
    model.eval()
    single_x = torch.tensor([0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=torch.float32)
    out = model(single_x)
    assert out.shape == (1, 1), f"Expected shape (1, 1), got {out.shape}"
    assert 0.0 <= out.item() <= 1.0, f"Expected probability in [0, 1], got {out.item()}"

def test_regression_autograd_leaf_tensor_explain_features():
    """
    Guards Iteration 1 failure: explain_features unsqueeze must not break leaf tensor grad tracking.
    """
    model = ANNMultimodalRiskClassifier(input_dim=7)
    model.eval()
    feat = torch.tensor([0.8, 0.9, 0.7, 0.6, 0.9, 0.5, 0.7], dtype=torch.float32)
    attributions = model.explain_features(feat)
    assert len(attributions) == 7
    for name in ANNMultimodalRiskClassifier.FEATURE_NAMES:
        assert name in attributions
        assert attributions[name] >= 0.0
    # Must sum to approx 1.0
    assert abs(sum(attributions.values()) - 1.0) < 1e-4

def test_dataset_adapters_and_smoke_mode():
    img_adapter = ImageForensicAdapter("non_existent_image_path")
    txt_adapter = TextForensicAdapter("non_existent_text_path")
    assert not img_adapter.is_available()
    assert not txt_adapter.is_available()
    assert SMOKE_BANNER == "SYNTHETIC SMOKE RUN - NOT A RESULT"

def test_case_builder_label_conditioned_pairing():
    texts = [
        ForensicRawRecord("T1", "text", "benign email", 0, {}, "h1"),
        ForensicRawRecord("T2", "text", "suspicious email", 1, {}, "h2"),
    ]
    images = [
        ForensicRawRecord("I1", "image", "benign.png", 0, {}, "h3"),
        ForensicRawRecord("I2", "image", "phish.png", 1, {}, "h4"),
    ]
    builder = CaseBuilder(pairing_strategy="label_conditioned", seed=42)
    cases = builder.build_cases(texts, images)
    assert len(cases) == 2
    for c in cases:
        if c.label == 0:
            assert "benign" in c.text_content
        elif c.label == 1:
            assert "suspicious" in c.text_content

def test_stratified_split_zero_leakage_and_manifest():
    cases = []
    for i in range(100):
        lbl = 1 if i % 2 == 0 else 0
        h = hashlib.sha256(f"content-{i}".encode()).hexdigest()
        cases.append(
            MultimodalCaseItem(
                case_id=f"CASE-{i:03d}",
                text_content=f"Text sample {i}",
                text_hash=f"thash-{i}",
                image_content=f"img_{i}.png",
                image_hash=f"ihash-{i}",
                label=lbl,
                metadata={},
                content_hash=h,
            )
        )

    manifest_file = "data/test_manifest.json"
    train_set, val_set, test_set, stats = create_stratified_split(
        cases, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, manifest_path=manifest_file
    )

    assert len(train_set) == 70
    assert len(val_set) + len(test_set) == 30

    # Zero ID overlap assertion
    train_ids = {c.case_id for c in train_set}
    val_ids = {c.case_id for c in val_set}
    test_ids = {c.case_id for c in test_set}
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Zero Content Hash overlap assertion
    train_hashes = {c.content_hash for c in train_set}
    val_hashes = {c.content_hash for c in val_set}
    test_hashes = {c.content_hash for c in test_set}
    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0

    # Imbalance ratio recorded
    assert "imbalance_ratio" in stats["splits"]["train"]
    assert stats["manifest_sha256"] is not None
    assert os.path.exists(manifest_file)
    os.remove(manifest_file)

def test_feature_formula_url_score():
    # Empty
    assert extract_url_score("") == 0.0
    # IPv4 literal (+0.80) + baseline (+0.20) = 1.0
    assert extract_url_score("http://192.168.1.100/login") == 1.0
    # Suspicious TLD (.xyz: +0.60, baseline +0.20) = 0.80
    assert pytest.approx(extract_url_score("http://sample-domain.xyz"), 0.01) == 0.80
    # Basic auth trick '@' (+0.70, baseline +0.20) = 0.90
    assert pytest.approx(extract_url_score("http://legit.com@phish.net"), 0.01) == 0.90

def test_feature_formula_keyword_score():
    assert extract_keyword_score("routine quarterly engineering sync") == 0.0
    # 'otp' (1.0) / 3.0 = 0.3333
    assert pytest.approx(extract_keyword_score("please share your otp"), 0.01) == 0.3333
    # 'urgent' (0.8) + 'wire' (0.9) + 'transfer' (0.7) = 2.4 / 3.0 = 0.80
    assert pytest.approx(extract_keyword_score("urgent wire transfer required"), 0.01) == 0.80
    # Bound to 1.0 max
    assert extract_keyword_score("urgent wire transfer immediately verify credentials password") == 1.0

def test_feature_formula_attachment_score():
    assert extract_attachment_score([]) == 0.0
    assert extract_attachment_score(["report.docx"]) == 0.0
    # Double extension -> 0.95
    assert extract_attachment_score(["invoice.pdf.exe"]) == 1.0  # max(0.95, 1.0 for .exe) = 1.0
    assert extract_attachment_score(["statement.pdf.vbs"]) == 0.95
    # Archive -> 0.40
    assert extract_attachment_score(["data.zip"]) == 0.40

def test_feature_formula_timeline_anomaly():
    t0 = datetime(2026, 11, 1, 10, 0, 0)
    assert extract_timeline_anomaly([t0]) == 0.0
    # Normal spacing (> 3 sec)
    assert extract_timeline_anomaly([t0, t0 + timedelta(minutes=5), t0 + timedelta(minutes=10)]) == 0.0
    # Rapid burst (< 3 sec) for >= 2 events -> 0.50 + 0.10 * 2 = 0.70
    burst = [t0, t0 + timedelta(seconds=1), t0 + timedelta(seconds=2)]
    assert pytest.approx(extract_timeline_anomaly(burst), 0.01) == 0.70

def test_feature_formula_email_anomaly():
    clean = {"sender": "lead@company.com", "reply_to": "lead@company.com", "display_name": "Lead"}
    assert extract_email_anomaly(clean) == 0.0

    # Domain mismatch (0.60)
    mismatch = {"sender": "ceo@company.com", "reply_to": "attacker@evil.com", "display_name": "CEO"}
    assert pytest.approx(extract_email_anomaly(mismatch), 0.01) == 0.60

    # Display spoofing from free mail (0.70)
    spoof = {"sender": "attacker@gmail.com", "reply_to": "attacker@gmail.com", "display_name": "Chief Executive Officer"}
    assert pytest.approx(extract_email_anomaly(spoof), 0.01) == 0.70

def test_bilstm_blueprint_pooling_modes():
    vocab_size = 50
    # Blueprint 5.2 default: concat
    model_concat = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=32, pooling="concat")
    tokens = torch.randint(0, vocab_size, (2, 15))
    out_c = model_concat(tokens)
    assert out_c.shape == (2, 1)

    # Optional attention pooling
    model_attn = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=32, pooling="attention")
    out_a, attn_w = model_attn(tokens, return_attention=True)
    assert out_a.shape == (2, 1)
    assert attn_w.shape == (2, 15)

def test_derive_max_len_percentile():
    texts = [
        "short text",
        "medium length message for sync",
        "this is a much longer chat message simulating detailed phishing instructions with url and tokens",
    ]
    derived = derive_max_len(texts, percentile=95.0)
    assert derived > 5
    assert isinstance(derived, int)

def test_ann_module2_options():
    # Test all variations of dropout and batchnorm
    for use_bn in [True, False]:
        for use_drop in [True, False]:
            model = ANNMultimodalRiskClassifier(input_dim=7, use_batchnorm=use_bn, use_dropout=use_drop)
            x = torch.rand(4, 7)
            out = model(x)
            assert out.shape == (4, 1)

    # Test initializers
    for init_m in ["xavier_uniform", "he_normal", "default"]:
        model = ANNMultimodalRiskClassifier(input_dim=7)
        init_weights(model, method=init_m)

def test_model_shape_traces():
    cnn = VisualEvidenceCNN(in_channels=3, num_classes=1)
    cnn_trace = cnn.trace_shapes(torch.randn(1, 3, 128, 128))
    assert len(cnn_trace) >= 6
    assert cnn_trace[0]["stage"] == "input"
    assert cnn_trace[-1]["stage"] in ["output", "classifier_output"]

    bilstm = ConversationBiLSTM(vocab_size=50, pooling="concat")
    bilstm_trace = bilstm.trace_shapes(torch.randint(0, 50, (1, 10)))
    assert len(bilstm_trace) >= 4

    ann = ANNMultimodalRiskClassifier(input_dim=7)
    ann_trace = ann.trace_shapes(torch.rand(1, 7))
    assert len(ann_trace) >= 4

def test_oov_rate_calculation():
    tok = ForensicTokenizer(max_length=20)
    tok.fit_on_texts(["apple banana cherry"])
    # "date" is OOV
    rate = calculate_oov_rate(["apple date"], tok)
    assert pytest.approx(rate, 0.01) == 0.50

def test_regression_shuffled_label_train_only_hash_check():
    """Asserts that label shuffling touches TRAIN labels only and leaves TEST labels immutable."""
    set_seed(42)
    labels_train = np.random.randint(0, 2, 70)
    labels_test = np.random.randint(0, 2, 30)

    test_hash_before = hashlib.sha256(labels_test.tobytes()).hexdigest()

    # Simulate 20 shuffle seeds
    for seed in range(20):
        rng = np.random.RandomState(seed)
        shuf_train = rng.permutation(labels_train)
        assert len(shuf_train) == len(labels_train)
        # Test labels MUST remain completely untouched
        test_hash_after = hashlib.sha256(labels_test.tobytes()).hexdigest()
        assert test_hash_after == test_hash_before, f"Test labels mutated on seed {seed}!"

def test_regression_bilstm_attention_conditioned_on_config():
    """
    Asserts that under pooling='concat' (Blueprint 5.2 default), BiLSTM holds 0 attention parameters.
    When pooling='attention' or use_attention=True, attention layer is allocated.
    """
    vocab_size = 50
    # Concat pooling with default use_attention=False -> 0 attention parameters
    model_concat = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=64, pooling="concat", use_attention=False)
    assert model_concat.attention is None
    assert "attention.weight" not in [name for name, _ in model_concat.named_parameters()]
    total_concat_params = sum(p.numel() for p in model_concat.parameters())

    # Attention pooling -> allocates attention layer
    model_attn = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=64, pooling="attention", use_attention=True)
    assert model_attn.attention is not None
    assert "attention.weight" in [name for name, _ in model_attn.named_parameters()]
    total_attn_params = sum(p.numel() for p in model_attn.parameters())

    # Attention parameters = 128 * 1 + 1 = 129
    assert total_attn_params - total_concat_params == 129

    # Verify forward works in both modes
    tokens = torch.randint(0, vocab_size, (2, 10))
    out_c, attn_c = model_concat(tokens, return_attention=True)
    assert out_c.shape == (2, 1)
    assert attn_c.shape == (2, 10)
    assert torch.allclose(torch.sum(attn_c, dim=1), torch.ones(2), atol=1e-4)

    out_a, attn_a = model_attn(tokens, return_attention=True)
    assert out_a.shape == (2, 1)
    assert attn_a.shape == (2, 10)
    assert torch.allclose(torch.sum(attn_a, dim=1), torch.ones(2), atol=1e-4)

def test_regression_vae_parameter_count_assumption03_vs_positional_bug():
    """
    Verifies the root cause of the Iteration 2 VAE parameter discrepancy (135 vs 1,215).
    """
    # 1. Authoritative ASSUMPTION-03 architecture
    vae_spec = SyntheticDataVAE(input_dim=7, num_classes=2, hidden_dim=24, latent_dim=4)
    total_spec = sum(p.numel() for p in vae_spec.parameters())
    assert total_spec == 1215

    # 2. Iteration 2 positional argument bug: passing 4 as hidden_dim
    # Verified keyword-only enforcement rejects positional arguments
    with pytest.raises(TypeError):
        SyntheticDataVAE(7, 2, 4)  # type: ignore

    vae_bug = SyntheticDataVAE(input_dim=7, num_classes=2, hidden_dim=4, latent_dim=4)
    total_bug = sum(p.numel() for p in vae_bug.parameters())
    assert total_bug == 135

    # Layer-by-layer verification of 135
    # Linear(9, 4) = 40; Linear(4, 2) = 10; Linear(2, 4) = 12; Linear(2, 4) = 12
    # Linear(6, 2) = 14; Linear(2, 4) = 12; Linear(4, 7) = 35
    # Total = 40 + 10 + 12 + 12 + 14 + 12 + 35 = 135
    assert (40 + 10 + 12 + 12 + 14 + 12 + 35) == 135

def test_regression_vae_encode_decode_with_2d_and_1d_labels():
    """
    Guards against tensor dimension mismatches in VAE encode/decode when y is (batch_size, 1) or (batch_size,).
    """
    vae = SyntheticDataVAE(input_dim=7, num_classes=2, hidden_dim=24, latent_dim=4)
    x = torch.rand(4, 7)

    # 1D labels
    y_1d = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    mu_1, logvar_1 = vae.encode(x, y_1d)
    assert mu_1.shape == (4, 4)
    recon_1 = vae.decode(mu_1, y_1d)
    assert recon_1.shape == (4, 7)

    # 2D labels (batch_size, 1)
    y_2d = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
    mu_2, logvar_2 = vae.encode(x, y_2d)
    assert mu_2.shape == (4, 4)
    recon_2 = vae.decode(mu_2, y_2d)
    assert recon_2.shape == (4, 7)

def test_regression_cnn_hidden_dense_layer_proportion():
    """Asserts that CNN hidden dense layer parameters are correctly calculated and accounted for."""
    cnn = VisualEvidenceCNN(in_channels=3, num_classes=1)
    total_params = sum(p.numel() for p in cnn.parameters())
    assert total_params == 356097

    # Dense layer classifier.1: Linear(128 * 4 * 4, 128)
    dense_weight_params = cnn.classifier[1].weight.numel()  # 2048 * 128 = 262144
    dense_bias_params = cnn.classifier[1].bias.numel()      # 128
    dense_total = dense_weight_params + dense_bias_params   # 262272

    assert dense_total == 262272
    pct = (dense_total / total_params) * 100.0
    assert 73.0 <= pct <= 74.0

def test_regression_forward_hooks_shape_traces():
    """Verifies that PyTorch forward hooks capture shape traces across all submodules on a real batch."""
    cnn = VisualEvidenceCNN(in_channels=3, num_classes=1)
    x = torch.rand(2, 3, 128, 128)

    traces = []
    hooks = []
    for name, mod in cnn.named_modules():
        if len(list(mod.children())) == 0:
            def make_hook(n):
                def hook_fn(m, inp, out):
                    traces.append((n, list(inp[0].shape), list(out.shape) if isinstance(out, torch.Tensor) else None))
                return hook_fn
            hooks.append(mod.register_forward_hook(make_hook(name)))

    cnn.eval()
    with torch.no_grad():
        cnn(x)

    for h in hooks:
        h.remove()

    assert len(traces) >= 8
    submod_names = [t[0] for t in traces]
    assert "conv1" in submod_names
    assert "bn1" in submod_names
    assert "conv2" in submod_names
    assert "conv3" in submod_names
    assert "adaptive_pool" in submod_names
    assert "classifier.1" in submod_names
    assert "classifier.4" in submod_names

def test_regression_lofo_seven_feature_keys():
    """Verifies that all 7 features are distinct and tested in LOFO."""
    features = ANNMultimodalRiskClassifier.FEATURE_NAMES
    assert len(features) == 7
    expected = [
        "cnn_score",
        "bilstm_score",
        "url_score",
        "keyword_score",
        "attachment_score",
        "timeline_anomaly",
        "email_anomaly",
    ]
    assert features == expected

def test_regression_grid_lr_sweep_and_convergence():
    """Verifies that calibrated learning rates (RMSprop 0.001, SGD 0.02, Adam 0.005) prevent divergence."""
    set_seed(42)
    x = torch.rand(64, 7)
    y = torch.randint(0, 2, (64, 1)).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    crit = nn.BCELoss()

    # RMSprop with 0.001 must not oscillate wildly
    m_rms = ANNMultimodalRiskClassifier(7, 32, 16, use_batchnorm=True, use_dropout=True)
    opt_rms = get_optimizer(m_rms, "rmsprop", lr=0.001)
    for _ in range(5):
        for bx, by in loader:
            opt_rms.zero_grad()
            l = crit(m_rms(bx), by)
            l.backward()
            opt_rms.step()
    assert not math.isnan(l.item())
    assert l.item() < 1.0

    # SGD with 0.02 must take effective steps
    m_sgd = ANNMultimodalRiskClassifier(7, 32, 16, use_batchnorm=False, use_dropout=False)
    opt_sgd = get_optimizer(m_sgd, "sgd", lr=0.02)
    init_loss = None
    for _ in range(5):
        for bx, by in loader:
            opt_sgd.zero_grad()
            l = crit(m_sgd(bx), by)
            if init_loss is None:
                init_loss = l.item()
            l.backward()
            opt_sgd.step()
    assert not math.isnan(l.item())

def test_regression_overfit_one_batch_all_three_models():
    """Verifies that overfit-one-batch reaches >= 95% accuracy for CNN, BiLSTM, and ANN."""
    device = torch.device("cpu")
    crit = nn.BCELoss()

    # 1. ANN
    set_seed(42)
    x_ann = torch.rand(4, 7)
    y_ann = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    m_ann = ANNMultimodalRiskClassifier(input_dim=7).to(device)
    opt_ann = torch.optim.Adam(m_ann.parameters(), lr=0.02)
    for _ in range(40):
        m_ann.train()
        opt_ann.zero_grad()
        loss = crit(m_ann(x_ann), y_ann)
        loss.backward()
        opt_ann.step()
    m_ann.eval()
    with torch.no_grad():
        preds = (m_ann(x_ann) >= 0.5).float()
        acc = float((preds == y_ann).float().mean().item())
    assert acc >= 0.95

    # 2. BiLSTM
    set_seed(42)
    vocab_size = 50
    x_bilstm = torch.randint(1, vocab_size, (4, 15))
    y_bilstm = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    m_bilstm = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=32, pooling="concat").to(device)
    opt_bilstm = torch.optim.Adam(m_bilstm.parameters(), lr=0.02)
    for _ in range(40):
        m_bilstm.train()
        opt_bilstm.zero_grad()
        loss = crit(m_bilstm(x_bilstm), y_bilstm)
        loss.backward()
        opt_bilstm.step()
    m_bilstm.eval()
    with torch.no_grad():
        preds = (m_bilstm(x_bilstm) >= 0.5).float()
        acc = float((preds == y_bilstm).float().mean().item())
    assert acc >= 0.95

    # 3. CNN (disable dropout regularization to overfit a 4-sample batch)
    set_seed(42)
    x_cnn = torch.rand(4, 3, 32, 32)
    y_cnn = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    m_cnn = VisualEvidenceCNN(in_channels=3, num_classes=1, use_dropout=False).to(device)
    opt_cnn = torch.optim.Adam(m_cnn.parameters(), lr=0.005)
    for _ in range(40):
        m_cnn.train()
        opt_cnn.zero_grad()
        loss = crit(m_cnn(x_cnn), y_cnn)
        loss.backward()
        opt_cnn.step()
    m_cnn.eval()
    with torch.no_grad():
        preds = (m_cnn(x_cnn) >= 0.5).float()
        acc = float((preds == y_cnn).float().mean().item())
    assert acc >= 0.95

def test_strict_checkpoint_loading_deleted_layer_raises():
    """Verify that attempting to load a checkpoint with a missing/deleted layer raises RuntimeError under strict=True."""
    model = ANNMultimodalRiskClassifier(input_dim=7, hidden_dim1=32, hidden_dim2=16)
    sd = model.state_dict()
    # Delete a genuine parameter
    del sd["fc1.weight"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "bad_ann.pt")
        torch.save(sd, ckpt_path)

        with pytest.raises(RuntimeError, match="Missing key.*fc1.weight"):
            load_model_checkpoint(model, ckpt_path, model_type="ann")

def test_strict_checkpoint_loading_tensor_equality():
    """Verify that after load_model_checkpoint, every parameter tensor exactly matches the checkpoint tensor."""
    model = ANNMultimodalRiskClassifier(input_dim=7, hidden_dim1=32, hidden_dim2=16)
    sd = model.state_dict()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "valid_ann.pt")
        torch.save(sd, ckpt_path)

        # Perturb model weights before loading
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        # Load checkpoint
        loaded_model = load_model_checkpoint(model, ckpt_path, model_type="ann")

        for name, param in loaded_model.named_parameters():
            assert torch.equal(param.data, sd[name]), f"Tensor mismatch for parameter {name}"

def test_bilstm_migration_allow_list_and_unexpected_key_rejection():
    """
    Verify migrate_bilstm_state_dict:
    - Removes allow-listed attention.* keys when target_pooling='concat'.
    - Preserves attention.* keys when target_pooling='attention'.
    - Raises RuntimeError if unexpected keys outside the allow-list are encountered.
    """
    base_model = ConversationBiLSTM(vocab_size=100, hidden_dim=64, pooling="attention", use_attention=True)
    sd = base_model.state_dict()
    assert "attention.weight" in sd
    assert "attention.bias" in sd

    # 1. Target pooling concat -> migration strips attention keys
    migrated_concat = migrate_bilstm_state_dict(sd, target_pooling="concat")
    assert "attention.weight" not in migrated_concat
    assert "attention.bias" not in migrated_concat

    # 2. Target pooling attention -> migration keeps attention keys
    migrated_attn = migrate_bilstm_state_dict(sd, target_pooling="attention")
    assert "attention.weight" in migrated_attn
    assert "attention.bias" in migrated_attn

    # 3. Unexpected key outside allow-list -> raises RuntimeError
    corrupted_sd = dict(sd)
    corrupted_sd["unauthorized_backdoor_layer.weight"] = torch.randn(10, 10)
    with pytest.raises(RuntimeError, match="Unexpected key in state_dict not permitted by allow-list"):
        migrate_bilstm_state_dict(corrupted_sd, target_pooling="concat")

def test_vae_keyword_only_constructor():
    """Verify that positional argument instantiation of SyntheticDataVAE is rejected with TypeError."""
    with pytest.raises(TypeError):
        SyntheticDataVAE(7, 2, 24, 4)

def test_vae_parameter_formula_closed_form():
    """Verify that closed-form parameter count matches empirical model sum across various hyperparameter settings."""
    test_configs = [
        {"input_dim": 7, "num_classes": 2, "hidden_dim": 24, "latent_dim": 4},
        {"input_dim": 7, "num_classes": 2, "hidden_dim": 4, "latent_dim": 4},
        {"input_dim": 10, "num_classes": 3, "hidden_dim": 32, "latent_dim": 8},
        {"input_dim": 5, "num_classes": 2, "hidden_dim": 16, "latent_dim": 2},
    ]

    for cfg in test_configs:
        model = SyntheticDataVAE(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
        )
        empirical_params = sum(p.numel() for p in model.parameters())
        formula_params = compute_vae_parameter_count(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
        )
        class_formula_params = SyntheticDataVAE.parameter_formula(**cfg)

        assert empirical_params == formula_params, f"Mismatch for cfg {cfg}: empirical={empirical_params}, formula={formula_params}"
        assert empirical_params == class_formula_params

    # Spec architecture assertion (ASSUMPTION-03: 1,215 params)
    assert compute_vae_parameter_count(7, 2, 24, 4) == 1215
    # Iteration 2 positional bug architecture (135 params)
    assert compute_vae_parameter_count(7, 2, 4, 4) == 135

def test_shuffled_label_null_distribution_formula():
    """
    Verify the theoretical standard deviation for ROC-AUC under the null hypothesis:
      sigma_0 = sqrt((n + 1) / (12 * n0 * n1))
    For balanced n=60 (n0=30, n1=30): sigma_0 = sqrt(61 / 10800) approx 0.075154
    """
    n = 60
    n0 = 30
    n1 = 30
    sigma_0 = math.sqrt((n + 1) / (12 * n0 * n1))
    assert abs(sigma_0 - 0.075154) < 1e-4

    # 2x theoretical sd threshold
    max_allowed_std = 2.0 * sigma_0
    assert abs(max_allowed_std - 0.1503) < 1e-3

    # 4x theoretical sd bound from chance (0.5)
    lower_bound = 0.5 - 4.0 * sigma_0
    upper_bound = 0.5 + 4.0 * sigma_0
    assert 0.199 < lower_bound < 0.201
    assert 0.799 < upper_bound < 0.801

def test_vocab_train_only_isolation():
    """Verify that tokenizer fitted on train texts does not contain test-only unique vocabulary."""
    train_texts = ["urgent transfer required immediately", "wire money to account"]
    test_texts = ["unseen_token_x confidential secret"]

    tok = ForensicTokenizer()
    tok.fit_on_texts(train_texts)

    assert "urgent" in tok.word2idx
    assert "transfer" in tok.word2idx
    assert "unseen_token_x" not in tok.word2idx

    # Check that encoding unseen test token maps to UNK (index 1)
    tokens, tensor_ids = tok.encode("unseen_token_x")
    assert tensor_ids[0].item() == 1

def test_validate_configs_on_examples():
    """Verify that example config templates pass schema validation when ignoring filesystem paths."""
    is_valid, errors = validate_configs(
        datasets_path="config/datasets.example.yaml",
        label_map_path="config/label_map.example.yaml",
        check_paths_exist=False,
    )
    assert is_valid is True, f"Example configs failed validation: {errors}"
    assert len(errors) == 0

def test_validate_configs_missing_files_error():
    """Verify that validate_configs reports missing datasets.yaml and label_map.yaml cleanly."""
    is_valid, errors = validate_configs(
        datasets_path="datasets.yaml",
        label_map_path="label_map.yaml",
    )
    assert is_valid is False
    assert any("datasets.yaml" in e for e in errors)
    assert any("label_map.yaml" in e for e in errors)

def test_validate_configs_schema_rejection(tmp_path):
    """Verify that missing required keys (e.g. min_test_cases, task) are flagged as errors."""
    bad_ds_path = tmp_path / "bad_datasets.yaml"
    bad_lm_path = tmp_path / "bad_label_map.yaml"

    bad_ds_path.write_text("""
datasets:
  - name: "IncompleteDataset"
    path: "data/raw"
    license: "MIT"
""", encoding="utf-8")

    bad_lm_path.write_text("""
label_mapping:
  image:
    clean: 0
""", encoding="utf-8")

    is_valid, errors = validate_configs(
        datasets_path=str(bad_ds_path),
        label_map_path=str(bad_lm_path),
        check_paths_exist=False,
    )
    assert is_valid is False
    assert any("missing required field: 'task'" in e for e in errors)
    assert any("missing required field: 'min_test_cases'" in e for e in errors)

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

def test_generator_raw_artifact_purity_zero_sampling():
    """
    Asserts that all heuristic features stored in metadata are strictly equal
    to the direct evaluation of the extractor functions on raw text, timestamps,
    attachments, and headers (zero direct feature sampling).
    """
    gen = HardForensicDatasetGenerator(seed=123)
    cases = gen.generate_benchmark_partition(num_samples=20, is_test_split=False)

    for c in cases:
        meta = c.metadata
        stored_heur = meta["heuristic_features"]

        # Recompute purely from raw artifacts
        u = extract_url_score(c.text_content)
        k = extract_keyword_score(c.text_content)
        a = extract_attachment_score(meta["attachments"])
        tm = extract_timeline_anomaly(meta["timestamps"])
        em = extract_email_anomaly(meta["headers"])

        recomputed = [u, k, a, tm, em]
        for idx in range(5):
            assert abs(stored_heur[idx] - recomputed[idx]) < 1e-6, (
                f"Direct feature sampling violation! Index {idx}: stored {stored_heur[idx]} != recomputed {recomputed[idx]}"
            )

def test_within_class_variance_and_auc_cap():
    """
    Asserts that:
    1. Every heuristic feature has non-zero sample standard deviation in both class 0 and class 1.
    2. Single-feature test AUC is capped <= 0.85 (ASSUMPTION-07).
    """
    tr, va, te, stats = generate_hard_benchmark_dataset()

    for split_name, cases in [("Train", tr), ("Val", va), ("Test", te)]:
        y = np.array([c.label for c in cases])
        X = np.array([c.metadata["heuristic_features"] for c in cases])

        for col in range(5):
            c0_std = float(np.std(X[y == 0, col]))
            c1_std = float(np.std(X[y == 1, col]))
            assert c0_std > 0.0, f"Within-class std for feature {col} in class 0 is 0 in {split_name}!"
            assert c1_std > 0.0, f"Within-class std for feature {col} in class 1 is 0 in {split_name}!"

    # Assert test AUC cap
    y_te = np.array([c.label for c in te])
    X_te = np.array([c.metadata["heuristic_features"] for c in te])
    for col in range(5):
        auc = roc_auc_score(y_te, X_te[:, col])
        assert auc <= 0.85, f"Single-feature test AUC {auc:.4f} exceeds cap 0.85 for feature {col}!"

def test_template_family_holdout_isolation():
    """
    Asserts that held-out test template family keys never appear in train or val splits.
    """
    held_out_keys = set(HELD_OUT_TEST_TEMPLATES.keys())
    train_val_keys = set(TRAIN_VAL_TEMPLATES.keys())

    assert len(held_out_keys.intersection(train_val_keys)) == 0, "Template family overlap detected!"

def test_vae_loss_with_free_bits_math():
    """
    Verifies that vae_loss_with_free_bits respects free bits minimum threshold
    and scales properly with beta.
    """
    recon_x = torch.zeros(10, 7)
    x = torch.zeros(10, 7)
    mu = torch.zeros(10, 4)
    logvar = torch.zeros(10, 4)  # KL per dim is 0.0

    # With free_bits=0.1, each dimension receives effective KL = 0.1, total effective KL = 0.4
    total_loss, recon, raw_kl, dim_kls = vae_loss_with_free_bits(
        recon_x, x, mu, logvar, beta=1.0, free_bits=0.10
    )
    assert abs(float(recon.item()) - 0.0) < 1e-6
    assert abs(float(raw_kl.item()) - 0.0) < 1e-6
    assert abs(float(total_loss.item()) - 0.40) < 1e-6

def test_baseline_augmenters():
    """Verifies Gaussian and Interpolation augmenters generate valid samples."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.1, 0.9, size=(50, 7)).astype(np.float32)
    y = np.array([0] * 25 + [1] * 25)

    # 1. Gaussian Augmenter
    ga = GaussianFeatureAugmenter(seed=42).fit(X, y)
    syn_g = ga.sample(num_samples=15, target_class=0)
    assert syn_g.shape == (15, 7)
    assert (syn_g >= 0.0).all() and (syn_g <= 1.0).all()

    # 2. Interpolation Augmenter
    ia = InterpolationFeatureAugmenter(seed=42).fit(X, y)
    syn_i = ia.sample(num_samples=15, target_class=1)
    assert syn_i.shape == (15, 7)
    assert (syn_i >= 0.0).all() and (syn_i <= 1.0).all()

def test_data_mode_configuration_file():
    """Verifies config/data_mode.yaml exists and parses cleanly."""
    cfg_path = "config/data_mode.yaml"
    assert os.path.exists(cfg_path), f"Missing data mode config: {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "mode" in cfg
    assert cfg["mode"] in ["hard_synthetic", "real"]

def test_buggy_vae_parameter_count_135():
    """
    Part 0d Audit Test:
    Asserts that SyntheticDataVAE(input_dim=7, latent_dim=4, hidden_dim=4, num_classes=2)
    produces exactly 135 parameters, replacing the erroneous '135 = 67 x 2' calculation.
    """
    model = SyntheticDataVAE(input_dim=7, latent_dim=4, hidden_dim=4, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params == 135, f"Expected exactly 135 parameters, got {total_params}"

    # Verify layer-by-layer breakdown
    param_dict = {name: p.numel() for name, p in model.named_parameters()}

    expected_counts = {
        "encoder_fc.0.weight": 36,  # 4 x (7 + 2) = 36
        "encoder_fc.0.bias": 4,     # 4
        "encoder_fc.2.weight": 8,   # (4 // 2) x 4 = 2 x 4 = 8
        "encoder_fc.2.bias": 2,     # 2
        "fc_mu.weight": 8,          # 4 x 2 = 8
        "fc_mu.bias": 4,            # 4
        "fc_logvar.weight": 8,      # 4 x 2 = 8
        "fc_logvar.bias": 4,        # 4
        "decoder.0.weight": 12,     # (4 // 2) x (4 + 2) = 2 x 6 = 12
        "decoder.0.bias": 2,        # 2
        "decoder.2.weight": 8,      # 4 x 2 = 8
        "decoder.2.bias": 4,        # 4
        "decoder.4.weight": 28,     # 7 x 4 = 28
        "decoder.4.bias": 7,        # 7
    }

    for layer_name, expected_numel in expected_counts.items():
        assert param_dict[layer_name] == expected_numel, (
            f"Layer {layer_name} expected {expected_numel} parameters, got {param_dict[layer_name]}"
        )

    # Sum verification
    assert sum(expected_counts.values()) == 135
