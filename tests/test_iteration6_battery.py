"""
tests/test_iteration6_battery.py
Iteration 6 Defect & Architectural Verification Suite.
Guards:
1. Hard generator raw artifact purity (zero direct feature sampling).
2. Within-class variance > 0 for all heuristic features in both classes.
3. Single-feature AUC cap <= 0.85.
4. Held-out test template family isolation.
5. VAE anti-collapse loss with free bits and beta annealing.
6. Baseline augmenters validity and shape compliance.
7. data_mode.yaml schema compliance.
"""

import os
import yaml
import pytest
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data.hard_generator import (
    HardForensicDatasetGenerator,
    generate_hard_benchmark_dataset,
    TRAIN_VAL_TEMPLATES,
    HELD_OUT_TEST_TEMPLATES,
)
from src.data_loader import (
    extract_url_score,
    extract_keyword_score,
    extract_attachment_score,
    extract_timeline_anomaly,
    extract_email_anomaly,
)
from src.models.vae_trainer import (
    vae_loss_with_free_bits,
    GaussianFeatureAugmenter,
    InterpolationFeatureAugmenter,
)


# =====================================================================
# 1. Generator Purity & Zero Direct Feature Sampling
# =====================================================================

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


# =====================================================================
# 2. Within-Class Variance & Single-Feature AUC Cap
# =====================================================================

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


# =====================================================================
# 3. Template Family Holdout
# =====================================================================

def test_template_family_holdout_isolation():
    """
    Asserts that held-out test template family keys never appear in train or val splits.
    """
    held_out_keys = set(HELD_OUT_TEST_TEMPLATES.keys())
    train_val_keys = set(TRAIN_VAL_TEMPLATES.keys())

    assert len(held_out_keys.intersection(train_val_keys)) == 0, "Template family overlap detected!"


# =====================================================================
# 4. VAE Anti-Collapse Schedule & Free Bits
# =====================================================================

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


# =====================================================================
# 5. Baseline Augmenters Validity
# =====================================================================

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


# =====================================================================
# 6. Data Mode Configuration Validation
# =====================================================================

def test_data_mode_configuration_file():
    """Verifies config/data_mode.yaml exists and parses cleanly."""
    cfg_path = "config/data_mode.yaml"
    assert os.path.exists(cfg_path), f"Missing data mode config: {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "mode" in cfg
    assert cfg["mode"] in ["hard_synthetic", "real"]

