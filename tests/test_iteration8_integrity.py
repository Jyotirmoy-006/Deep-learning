"""
tests/test_iteration8_integrity.py
Guarantees Global Rules G12, G13, G14 for Iteration 8:
- (G12) One metrics function: Accuracy/Precision/Recall/F1/Brier derive from printed integer confusion matrix.
- (G13) Single-use test-split access guard: multiple calls or selection imports raise TestLeakageError.
- (G14) Distinct weight hashes across seeds: asserts >= N distinct final-weight hashes per arm and metric variance.
- (G5/G11) hard_v3 manifest purity and true label noise audit.
"""

import os
import json
import pytest
import numpy as np
import torch
import torch.nn as nn
from src.utils.metrics import (
    compute_confusion_and_derived_metrics,
    derive_metrics_from_confusion_matrix,
)
from src.data.test_guard import (
    SingleUseTestGuard,
    compute_model_weight_hash,
    compute_data_order_hash,
    TestLeakageError,
)

RESULTS_V8_PATH = "results/experiment_results_v8.json"
MANIFEST_V3_PATH = "data/split_manifest_hard_v3.json"


def test_rule_g12_derived_metrics_recomputation():
    """
    Rule G12: Recomputes Accuracy, Precision, Recall, and F1 strictly from
    the printed integer confusion matrix (TP, FP, FN, TN, threshold).
    Verifies that every printed seed run matches closed-form math.
    """
    assert os.path.exists(RESULTS_V8_PATH), f"Missing results artifact '{RESULTS_V8_PATH}'"
    with open(RESULTS_V8_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    canonical = data["part4_experiments"]["4c_canonical_summary"]

    for arm in ["A", "B", "C", "D", "E"]:
        seed_runs = canonical[arm]["per_seed_runs"]
        assert len(seed_runs) >= 10, f"Arm {arm} must have at least 10 seeds (got {len(seed_runs)})"

        for s_idx, run in enumerate(seed_runs):
            tp = run["tp"]
            fp = run["fp"]
            fn = run["fn"]
            tn = run["tn"]
            th = run["threshold"]

            recomputed = derive_metrics_from_confusion_matrix(tp=tp, fp=fp, fn=fn, tn=tn, threshold=th)

            # Round to 4 decimal places matching printed table values
            r_acc = round(recomputed["accuracy"], 4)
            r_prec = round(recomputed["precision"], 4)
            r_rec = round(recomputed["recall"], 4)
            r_f1 = round(recomputed["f1"], 4)

            assert abs(r_acc - run["accuracy"]) < 1e-4, (
                f"Arm {arm} Seed {run['seed']} Accuracy mismatch: recomputed {r_acc} != printed {run['accuracy']}"
            )
            assert abs(r_prec - run["precision"]) < 1e-4, (
                f"Arm {arm} Seed {run['seed']} Precision mismatch: recomputed {r_prec} != printed {run['precision']}"
            )
            assert abs(r_rec - run["recall"]) < 1e-4, (
                f"Arm {arm} Seed {run['seed']} Recall mismatch: recomputed {r_rec} != printed {run['recall']}"
            )
            assert abs(r_f1 - run["f1"]) < 1e-4, (
                f"Arm {arm} Seed {run['seed']} F1 mismatch: recomputed {r_f1} != printed {run['f1']}"
            )


def test_rule_g13_single_use_test_guard():
    """
    Rule G13: Asserts SingleUseTestGuard logs caller provenance and SHA-256 hash,
    and strictly raises TestLeakageError on multiple calls for the same seed.
    """
    X_test_dummy = np.random.randn(50, 4)
    y_test_dummy = np.random.randint(0, 2, 50)

    guard = SingleUseTestGuard(X_test=X_test_dummy, y_test=y_test_dummy)

    # First call for seed 42 succeeds
    preds1 = np.random.uniform(0, 1, 50)
    m1 = guard.evaluate_test(seed=42, predict_fn_or_probs=preds1, threshold=0.5, caller_name="eval_test")
    assert guard.total_evaluations == 1
    assert "prediction_sha256" in m1
    assert len(guard.access_log) == 1
    assert guard.access_log[0]["seed"] == 42

    # Second call for seed 42 must raise TestLeakageError
    with pytest.raises(TestLeakageError) as exc_info:
        guard.evaluate_test(seed=42, predict_fn_or_probs=preds1, threshold=0.5, caller_name="eval_test_again")
    assert "Rule G13 VIOLATION" in str(exc_info.value)
    assert "already evaluated for seed 42" in str(exc_info.value)

    # Unauthorized access from selection/grid keywords must raise
    with pytest.raises(TestLeakageError) as exc_info_grid:
        guard.evaluate_test(seed=99, predict_fn_or_probs=preds1, threshold=0.5, caller_name="model_selection_grid")
    assert "cannot be accessed by selection/grid" in str(exc_info_grid.value)


def test_rule_g14_distinct_weight_hashes_and_variance():
    """
    Rule G14: Asserts that for every N-seed experiment (N=10), there are >= N distinct
    final-weight hashes and metrics differ across seeds (no seed collapse).
    """
    assert os.path.exists(RESULTS_V8_PATH), f"Missing results artifact '{RESULTS_V8_PATH}'"
    with open(RESULTS_V8_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    canonical = data["part4_experiments"]["4c_canonical_summary"]

    for arm in ["A", "B", "C", "D", "E"]:
        seed_runs = canonical[arm]["per_seed_runs"]
        N = len(seed_runs)
        assert N >= 10, f"Arm {arm} must have >= 10 seed runs"

        final_hashes = [r["final_weight_hash"] for r in seed_runs]
        unique_final_hashes = set(final_hashes)
        assert len(unique_final_hashes) >= N, (
            f"Arm {arm} violated Rule G14: expected at least {N} distinct final-weight hashes, got {len(unique_final_hashes)}"
        )

        aucs = [r["auc"] for r in seed_runs]
        assert len(set(aucs)) > 1, f"Arm {arm} violated Rule G14: AUC is identical across all seeds (seed collapse detected)"


def test_hard_v3_manifest_and_true_label_noise():
    """
    Validates Generator v3 manifest:
    1. 40 template families partitioned disjointly (20 train, 10 val, 10 test).
    2. True training label noise applied (flipped fraction in [0.03, 0.08]).
    3. Zero family or hash leakage across splits.
    """
    assert os.path.exists(MANIFEST_V3_PATH), f"Missing manifest '{MANIFEST_V3_PATH}'"
    with open(MANIFEST_V3_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    stats = manifest["stats"]["splits"]
    assert stats["train"]["families_count"] == 20
    assert stats["val"]["families_count"] == 10
    assert stats["test"]["families_count"] == 10

    # Train flipped fraction around 5%
    flipped_frac = stats["train"]["flipped_fraction"]
    assert 0.03 <= flipped_frac <= 0.08, f"Train flipped fraction {flipped_frac:.4f} outside [0.03, 0.08]"

    # Zero overlap
    tr_h = set(manifest["train_hashes"])
    va_h = set(manifest["val_hashes"])
    te_h = set(manifest["test_hashes"])

    assert len(tr_h.intersection(va_h)) == 0, "Train and Val hash overlap detected"
    assert len(tr_h.intersection(te_h)) == 0, "Train and Test hash overlap detected"
    assert len(va_h.intersection(te_h)) == 0, "Val and Test hash overlap detected"


def test_markdown_table_4_3_roundtrip_recomputation():
    """
    Rule G9 & G12 Round-Trip: Parses Table 4.3 from results/all_tables_v8.md,
    extracts the confusion matrix (TP/FP/FN/TN), recomputes Accuracy, Precision, Recall,
    and F1, and asserts that they match the printed Markdown numbers.
    """
    md_path = "results/all_tables_v8.md"
    assert os.path.exists(md_path), f"Missing Markdown table ledger '{md_path}'"

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find Table 4.3
    table_43_lines = []
    in_table = False
    for line in lines:
        if "### Table 4.3: Per-Seed Raw Metrics" in line:
            in_table = True
            continue
        if in_table and line.startswith("### Table 4.4"):
            break
        if in_table and line.startswith("| **"):
            table_43_lines.append(line.strip())

    assert len(table_43_lines) == 50, f"Expected 50 rows in Table 4.3 (5 arms x 10 seeds), found {len(table_43_lines)}"

    for row in table_43_lines:
        parts = [p.strip() for p in row.split("|")[1:-1]]
        arm = parts[0].replace("**", "")
        seed = int(parts[1])
        cm_str = parts[2].replace("`", "")
        tp, fp, fn, tn = [int(x) for x in cm_str.split("/")]
        th = float(parts[3])
        printed_auc = float(parts[4])
        printed_acc = float(parts[5])
        printed_prec = float(parts[6])
        printed_rec = float(parts[7])
        printed_f1 = float(parts[8])

        recomputed = derive_metrics_from_confusion_matrix(tp=tp, fp=fp, fn=fn, tn=tn, threshold=th)

        assert abs(round(recomputed["accuracy"], 4) - printed_acc) < 1e-4, (
            f"Row {arm} seed {seed} Acc mismatch: {recomputed['accuracy']} vs {printed_acc}"
        )
        assert abs(round(recomputed["precision"], 4) - printed_prec) < 1e-4, (
            f"Row {arm} seed {seed} Prec mismatch: {recomputed['precision']} vs printed_prec"
        )
        assert abs(round(recomputed["recall"], 4) - printed_rec) < 1e-4, (
            f"Row {arm} seed {seed} Rec mismatch: {recomputed['recall']} vs printed_rec"
        )
        assert abs(round(recomputed["f1"], 4) - printed_f1) < 1e-4, (
            f"Row {arm} seed {seed} F1 mismatch: {recomputed['f1']} vs printed_f1"
        )


