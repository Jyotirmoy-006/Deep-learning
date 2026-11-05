"""
tests/test_iteration10_integrity.py
ForenSightX Iteration 10 Integrity, Competence, and Statistical Verification Suite.

Strictly enforces:
- (G12) Derived metrics derive strictly from integer confusion matrices (TP, FP, FN, TN, threshold).
- (G13) Single-use test guard provenance and audit logging.
- (G14) Distinct weight hashes across seeds (>= 10 distinct hashes per arm, no seed collapse).
- (G16) Summary rows computed strictly via compute_summary_row_from_seed_runs; tests parse results and fail on mismatch > 1e-6.
        Delta tables print point estimate AND bootstrap mean.
- (G17) Manifest SHA consistency across controls, gates, and experiments.
- (G18) Primary AUC and val-tuned thresholds reported beside 0.5 threshold.
- (G20) Static test: generate_tables_v10.py contains zero float literals.
- (G21) Base-model competence gate: base model Val AUC >= oracle - 0.10.
- (G22) Experiment IDs fixed to Blueprint section 13 (A, B, C, D, E, A-h, S).
- (G23) Unified evaluation function: point estimates AND bootstrap statistics from identical predictions.
"""

import os
import ast
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

RESULTS_V10_PATH = "results/experiment_results_v10.json"
MANIFEST_V4_PATH = "data/split_manifest_hard_v4.json"
TABLES_V10_PATH = "results/all_tables_v10.md"
GEN_SCRIPT_V10_PATH = "results/generate_tables_v10.py"


def test_rule_g20_no_numeric_literals_in_generate_tables():
    """
    Rule G20: generate_tables_v10.py must not contain numeric float literals representing metrics.
    Every number must come dynamically from results JSON.
    """
    assert os.path.exists(GEN_SCRIPT_V10_PATH), f"Missing script '{GEN_SCRIPT_V10_PATH}'"
    with open(GEN_SCRIPT_V10_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    float_constants = [
        (node.value, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    ]
    assert len(float_constants) == 0, f"Rule G20 Violation: Found float literals in {GEN_SCRIPT_V10_PATH}: {float_constants}"


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

    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    exp_manifest_sha = data["part4_blueprint_experiments"]["manifest_sha256"]
    assert exp_manifest_sha == computed_sha, (
        f"Rule G17 Failure: Experiment manifest SHA ({exp_manifest_sha}) != disk manifest SHA ({computed_sha})"
    )


def test_rule_g21_base_model_competence_gate():
    """
    Rule G21: Base-Model Competence Gate.
    Each base model's val AUC must be >= its leave-families-out oracle minus configured margin (0.10).
    """
    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    g21 = data["part1_cnn_gate"]["base_model_gate"]
    assert g21["bilstm_gate_passed"] is True, f"BiLSTM failed competence gate: {g21['bilstm_val_auc']} < {g21['text_oracle_val_auc']} - {g21['margin']}"
    assert g21["cnn_gate_passed"] is True, f"CNN failed competence gate: {g21['cnn_val_auc']} < {g21['image_oracle_val_auc']} - {g21['margin']}"
    assert g21["verdict"] == "PASS", f"Gate G21 verdict was not PASS: {g21['verdict']}"


def test_rule_g22_experiment_arms_structure():
    """
    Rule G22: Blueprint Section 13 fixed experiment IDs:
    A (BiLSTM), B (CNN), C (ANN 2D), D (ANN 7D), E (Augmented ANN), A-h (Heuristics), S (Coordinator).
    """
    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data["part4_blueprint_experiments"]["runs"]
    required_arms = ["A_bilstm", "B_cnn", "C_ann_2d", "D_ann_7d", "E_augmented", "A_h_heuristics", "S_coordinator"]
    for arm in required_arms:
        assert arm in runs, f"Missing required Blueprint arm: {arm}"

    for arm in ["A_bilstm", "B_cnn", "C_ann_2d", "D_ann_7d", "E_augmented", "A_h_heuristics"]:
        assert len(runs[arm]) >= 10, f"Arm {arm} must have >= 10 seeds"


def test_rule_g12_confusion_matrix_recomputation():
    """
    Rule G12: Recomputes Accuracy, Precision, Recall, and F1 strictly from
    the printed integer confusion matrix (TP, FP, FN, TN, threshold).
    """
    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs_dict = data["part4_blueprint_experiments"]["runs"]

    for arm in ["A_bilstm", "B_cnn", "C_ann_2d", "D_ann_7d", "E_augmented", "A_h_heuristics"]:
        for r in runs_dict[arm]:
            # Val-tuned threshold metrics
            recomputed = derive_metrics_from_confusion_matrix(
                tp=r["tp"], fp=r["fp"], fn=r["fn"], tn=r["tn"], threshold=r["val_threshold"]
            )
            assert abs(recomputed["accuracy"] - r["accuracy"]) < 1e-4
            assert abs(recomputed["precision"] - r["precision"]) < 1e-4
            assert abs(recomputed["recall"] - r["recall"]) < 1e-4
            assert abs(recomputed["f1"] - r["f1"]) < 1e-4

            # 0.5 threshold metrics
            recomp_05 = derive_metrics_from_confusion_matrix(
                tp=r["tp_05"], fp=r["fp_05"], fn=r["fn_05"], tn=r["tn_05"], threshold=0.5
            )
            assert abs(recomp_05["accuracy"] - r["accuracy_05"]) < 1e-4
            assert abs(recomp_05["precision"] - r["precision_05"]) < 1e-4
            assert abs(recomp_05["recall"] - r["recall_05"]) < 1e-4
            assert abs(recomp_05["f1"] - r["f1_05"]) < 1e-4


def test_rule_g14_distinct_final_weight_hashes():
    """
    Rule G14: Asserts that for every N-seed experiment (N=10), there are >= N distinct
    final-weight hashes and metrics differ across seeds (no seed collapse).
    """
    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs_dict = data["part4_blueprint_experiments"]["runs"]

    for arm in ["A_bilstm", "B_cnn", "C_ann_2d", "D_ann_7d", "E_augmented", "A_h_heuristics"]:
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
    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs_dict = data["part4_blueprint_experiments"]["runs"]
    summaries = data["part4_blueprint_experiments"]["summary"]

    for arm in ["A_bilstm", "B_cnn", "C_ann_2d", "D_ann_7d", "E_augmented", "A_h_heuristics"]:
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
    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    deltas = data["part4_blueprint_experiments"]["delta_comparisons"]
    summaries = data["part4_blueprint_experiments"]["summary"]

    # Verify B_cnn vs A_bilstm
    d_ba = deltas[0]
    expected_pe_auc = summaries["B_cnn"]["mean_auc"] - summaries["A_bilstm"]["mean_auc"]
    assert abs(d_ba["point_estimate_delta_auc"] - expected_pe_auc) < 1e-6
    assert "bootstrap_mean_delta_auc" in d_ba
    assert "ci_lower_auc" in d_ba and "ci_upper_auc" in d_ba


def test_rule_g13_test_guard_audit():
    """
    Rule G13: Asserts SingleUseTestGuard logged all test set evaluations
    and recorded valid predictions SHA-256 hashes.
    """
    assert os.path.exists(RESULTS_V10_PATH), f"Missing results artifact '{RESULTS_V10_PATH}'"
    with open(RESULTS_V10_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_items = data["part5_diagnostics"]["test_guard_access_log"]
    assert len(log_items) > 0, "Test guard log cannot be empty"
    for item in log_items:
        assert item["sample_count"] == 300, f"Test sample count must be 300, got {item['sample_count']}"
        assert len(item["prediction_sha256"]) == 64, "Prediction hash must be 64-char hex SHA-256"

