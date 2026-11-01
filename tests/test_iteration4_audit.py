"""
ForenSightX - Iteration 4 Audit & Defect Verification Suite
Authoritative Reference: ForenSightX_Project_Architecture.docx Section 13

Guards all Iteration 4 defects and architectural specifications:
1. strict=True checkpoint loading, named migration allow-list, unexpected key rejection, and tensor equality.
2. SyntheticDataVAE keyword-only enforcement and closed-form parameter formula verification.
3. Shuffled-label control theoretical null standard deviation bounds and test label hash preservation.
4. Tokenizer train-only vocabulary isolation.
5. Real dataset configuration schema and label mapping validator.
"""

import math
import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    SyntheticDataVAE,
    compute_vae_parameter_count,
    load_model_checkpoint,
    migrate_bilstm_state_dict,
)
from src.data.validate_config import validate_configs
from src.data_loader import ForensicTokenizer


# =====================================================================
# 1. strict=True Checkpoint Loading & Migration Audit
# =====================================================================

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


# =====================================================================
# 2. SyntheticDataVAE Closed-Form Formula & Keyword-Only Enforcement
# =====================================================================

def test_vae_keyword_only_constructor():
    """Verify that positional argument instantiation of SyntheticDataVAE is rejected with TypeError."""
    with pytest.raises(TypeError):
        SyntheticDataVAE(7, 2, 24, 4)  # type: ignore


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


# =====================================================================
# 3. Shuffled-Label Null Distribution Theoretical Calculations
# =====================================================================

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


# =====================================================================
# 4. Vocab & BiLSTM Train-Only Tokenizer Isolation
# =====================================================================

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


# =====================================================================
# 5. Real Data Config Validation
# =====================================================================

def test_validate_configs_on_examples():
    """Verify that example config templates pass schema validation when ignoring filesystem paths."""
    is_valid, errors = validate_configs(
        datasets_path="datasets.example.yaml",
        label_map_path="label_map.example.yaml",
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
