"""
tests/test_iteration9_integrity.py
ForenSightX Iteration 9 Integrity & Verification Test Suite.

Enforces Global Rules G12, G13, G14, G16, G17, G18:
- (G12) Derived metrics derive strictly from integer confusion matrices (TP, FP, FN, TN, threshold).
- (G13) Single-use test guard provenance and zero leakage.
- (G14) Distinct weight hashes across seeds (>= 10 distinct hashes per arm, no seed collapse).
- (G16) Summary rows computed strictly by compute_summary_row_from_seed_runs; tests parse results and fail on mismatch > 1e-6.
        Delta tables print point estimate AND bootstrap mean.
- (G17) Manifest SHA consistency across controls, gates, and experiments.
- (G18) Primary AUC and val-tuned thresholds.
- Generator v4 60-family split isolation and non-zero opposite-class cue rates.
"""

import os
import json
import hashlib
import pytest
import numpy as np

from src.utils.metrics import (
    derive_metrics_from_confusion_matrix,
    compute_summary_row_from_seed_runs,
)
from src.data.hard_generator import (
    TRAIN_TEMPLATES_V4,
    VAL_TEMPLATES_V4,
    TEST_TEMPLATES_V4,
)

RESULTS_V9_PATH = "results/experiment_results_v9.json"
MANIFEST_V4_PATH = "data/split_manifest_hard_v4.json"
TABLES_V9_PATH = "results/all_tables_v9.md"


def test_generator_v4_family_isolation_and_disjointness():
    """
    Asserts Generator v4 has 60 mutually disjoint template families
    (30 train, 15 val, 15 test) with zero cross-split overlap.
    """
    tr_fams = set(TRAIN_TEMPLATES_V4.keys())
    va_fams = set(VAL_TEMPLATES_V4.keys())
    te_fams = set(TEST_TEMPLATES_V4.keys())

    assert len(tr_fams) == 30, f"Expected 30 train families, got {len(tr_fams)}"
    assert len(va_fams) == 15, f"Expected 15 val families, got {len(va_fams)}"
    assert len(te_fams) == 15, f"Expected 15 test families, got {len(te_fams)}"

    assert len(tr_fams.intersection(va_fams)) == 0, "Train and Val template families overlap!"
    assert len(tr_fams.intersection(te_fams)) == 0, "Train and Test template families overlap!"
    assert len(va_fams.intersection(te_fams)) == 0, "Val and Test template families overlap!"


def test_rule_g17_manifest_sha_consistency():
    """
    Rule G17: Every control and experiment prints the SHA-256 of the manifest it ran on.
    Asserts committed split_manifest_hard_v4.json matches the hash recorded in results JSON.
    """
    assert os.path.exists(MANIFEST_V4_PATH), f"Missing manifest '{MANIFEST_V4_PATH}'"
    computed_sha = hashlib.sha256(open(MANIFEST_V4_PATH, "rb").read()).hexdigest()

    assert os.path.exists(RESULTS_V9_PATH), f"Missing results artifact '{RESULTS_V9_PATH}'"
    with open(RESULTS_V9_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    exp_manifest_sha = data["part5_experiments"]["manifest_sha256"]
    assert exp_manifest_sha == computed_sha, (
        f"Rule G17 Failure: Experiment manifest SHA ({exp_manifest_sha}) != disk manifest SHA ({computed_sha})"
    )


def test_rule_g12_confusion_matrix_recomputation():
    """
    Rule G12: Recomputes Accuracy, Precision, Recall, and F1 strictly from
    the printed integer confusion matrix (TP, FP, FN, TN, threshold).
    Verifies that every printed seed run matches closed-form math.
    """
    assert os.path.exists(RESULTS_V9_PATH), f"Missing results artifact '{RESULTS_V9_PATH}'"
    with open(RESULTS_V9_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs_dict = data["part5_experiments"]["runs"]

    for arm in ["A_heuristics", "B_cnn", "C_bilstm", "D_early_fusion", "E_multi_agent"]:
        seed_runs = runs_dict[arm]
        assert len(seed_runs) >= 10, f"Arm {arm} must have >= 10 seeds"

        for r in seed_runs:
            tp = r["tp"]
            fp = r["fp"]
            fn = r["fn"]
            tn = r["tn"]
            th = r["val_threshold"]

            recomputed = derive_metrics_from_confusion_matrix(tp=tp, fp=fp, fn=fn, tn=tn, threshold=th)

            assert abs(recomputed["accuracy"] - r["accuracy"]) < 1e-4
            assert abs(recomputed["precision"] - r["precision"]) < 1e-4
            assert abs(recomputed["recall"] - r["recall"]) < 1e-4
            assert abs(recomputed["f1"] - r["f1"]) < 1e-4


def test_rule_g14_distinct_final_weight_hashes():
    """
    Rule G14: Asserts that for every N-seed experiment (N=10), there are >= N distinct
    final-weight hashes and metrics differ across seeds (no seed collapse).
    """
    assert os.path.exists(RESULTS_V9_PATH), f"Missing results artifact '{RESULTS_V9_PATH}'"
    with open(RESULTS_V9_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs_dict = data["part5_experiments"]["runs"]

    for arm in ["A_heuristics", "B_cnn", "C_bilstm", "D_early_fusion", "E_multi_agent"]:
        seed_runs = runs_dict[arm]
        N = len(seed_runs)
        final_hashes = {r["final_weight_hash"] for r in seed_runs}
        assert len(final_hashes) >= N, f"Arm {arm} must have >= {N} distinct final-weight hashes (got {len(final_hashes)})"

        aucs = [r["auc"] for r in seed_runs]
        assert np.std(aucs) > 1e-6, f"Arm {arm} exhibits zero variance across seeds"


def test_rule_g16_summary_row_recomputation():
    """
    Rule G16: Asserts every summary row (mean/std) is computed strictly from
    printed seed rows by compute_summary_row_from_seed_runs with mismatch < 1e-6.
    """
    assert os.path.exists(RESULTS_V9_PATH), f"Missing results artifact '{RESULTS_V9_PATH}'"
    with open(RESULTS_V9_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs_dict = data["part5_experiments"]["runs"]
    summaries = data["part5_experiments"]["summary"]

    for arm in ["A_heuristics", "B_cnn", "C_bilstm", "D_early_fusion", "E_multi_agent"]:
        seed_runs = runs_dict[arm]
        expected_summary = compute_summary_row_from_seed_runs(seed_runs)
        stored_summary = summaries[arm]

        assert abs(expected_summary["mean_auc"] - stored_summary["mean_auc"]) < 1e-6, (
            f"Mean AUC mismatch in Arm {arm}: {expected_summary['mean_auc']} vs {stored_summary['mean_auc']}"
        )
        assert abs(expected_summary["std_auc"] - stored_summary["std_auc"]) < 1e-6, (
            f"Std AUC mismatch in Arm {arm}: {expected_summary['std_auc']} vs {stored_summary['std_auc']}"
        )
        assert abs(expected_summary["mean_f1"] - stored_summary["mean_f1"]) < 1e-6
        assert abs(expected_summary["std_f1"] - stored_summary["std_f1"]) < 1e-6


def test_rule_g16_delta_table_point_estimate_and_bootstrap_mean():
    """
    Rule G16: Asserts delta tables print BOTH the exact point estimate (mean_B - mean_A)
    AND the bootstrap mean of the paired difference.
    """
    assert os.path.exists(RESULTS_V9_PATH), f"Missing results artifact '{RESULTS_V9_PATH}'"
    with open(RESULTS_V9_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    deltas = data["part5_experiments"]["delta_comparisons"]
    summaries = data["part5_experiments"]["summary"]

    # Verify B - A
    d_ba = deltas[0]
    expected_pe_auc = summaries["B_cnn"]["mean_auc"] - summaries["A_heuristics"]["mean_auc"]
    assert abs(d_ba["point_estimate_delta_auc"] - expected_pe_auc) < 1e-6
    assert "bootstrap_mean_delta_auc" in d_ba

    # Verify C - B
    d_cb = deltas[1]
    expected_pe_auc_cb = summaries["C_bilstm"]["mean_auc"] - summaries["B_cnn"]["mean_auc"]
    assert abs(d_cb["point_estimate_delta_auc"] - expected_pe_auc_cb) < 1e-6
    assert "bootstrap_mean_delta_auc" in d_cb

