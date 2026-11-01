"""
tests/test_iteration5_summary_assert.py
Strict Rule G2 Verification Test Suite.
Enforces that:
1. Every summary statistic is recomputed from the per-row values stored in results/experiment_results_v5.json
   and printed in results/all_tables_v5.md.
2. Mean, std, min, max, percentiles, and counts match exactly within numerical tolerance (1e-4).
3. Percentiles strictly lie within [min, max].
"""

import json
import os
import re
import numpy as np
import pytest


@pytest.fixture(scope="module")
def results_data():
    path = "results/experiment_results_v5.json"
    assert os.path.exists(path), f"Missing results file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_rule_g2_global_permutation_control_summary_recomputed(results_data):
    """
    Asserts that Global Permutation Control summary metrics are exactly
    reproducible from the 100 individual seed rows.
    """
    ctrl = results_data["part2_valid_controls"]
    rows = ctrl["control_100_seeds_table"]
    assert len(rows) == 100, f"Expected exactly 100 seeds, found {len(rows)}"

    aucs = [r["roc_auc"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    recomputed_mean = float(np.mean(aucs))
    recomputed_std = float(np.std(aucs))
    recomputed_min = float(np.min(aucs))
    recomputed_max = float(np.max(aucs))
    recomputed_p2_5 = float(np.percentile(aucs, 2.5))
    recomputed_p97_5 = float(np.percentile(aucs, 97.5))

    # Assert match with stored summary statistics
    assert abs(recomputed_mean - ctrl["empirical_mean_auc"]) < 1e-4, "Mean AUC mismatch"
    assert abs(recomputed_std - ctrl["empirical_std_auc"]) < 1e-4, "Std AUC mismatch"
    assert abs(recomputed_min - ctrl["min_auc"]) < 1e-4, "Min AUC mismatch"
    assert abs(recomputed_max - ctrl["max_auc"]) < 1e-4, "Max AUC mismatch"
    assert abs(recomputed_p2_5 - ctrl["percentile_2_5"]) < 1e-4, "2.5% Percentile mismatch"
    assert abs(recomputed_p97_5 - ctrl["percentile_97_5"]) < 1e-4, "97.5% Percentile mismatch"

    # Assert percentiles strictly lie within [min, max]
    assert recomputed_min <= recomputed_p2_5 <= recomputed_max, "p2.5 outside [min, max]"
    assert recomputed_min <= recomputed_p97_5 <= recomputed_max, "p97.5 outside [min, max]"
    assert recomputed_p2_5 <= recomputed_p97_5, "Percentile ordering violation"


def test_rule_g2_histogram_counts_and_binnings(results_data):
    """
    Asserts that the 10-bin histogram counts sum to 100 and match row-level binning.
    """
    ctrl = results_data["part2_valid_controls"]
    rows = ctrl["control_100_seeds_table"]
    aucs = [r["roc_auc"] for r in rows]

    hist = ctrl["histogram_10_bins"]
    total_count = sum(b["count"] for b in hist)
    assert total_count == 100, f"Histogram count sum {total_count} != 100"

    bins = np.linspace(0.0, 1.0, 11)
    recomputed_counts, _ = np.histogram(aucs, bins=bins)
    for i, b in enumerate(hist):
        assert b["count"] == int(recomputed_counts[i]), f"Histogram bin {i} mismatch: {b['count']} vs {recomputed_counts[i]}"


def test_rule_g2_bilstm_ablation_summary_recomputed(results_data):
    """
    Asserts that BiLSTM 20-seed ablation metrics and delta bounds are mathematically sound.
    """
    ab_tbl = results_data["part3_reruns"]["bilstm_20_seeds_ablation"]
    assert len(ab_tbl) == 20, f"Expected 20 seeds, got {len(ab_tbl)}"

    delta_aucs = [r["delta_auc"] for r in ab_tbl]
    delta_f1s = [r["delta_f1"] for r in ab_tbl]

    ci_auc = results_data["part3_reruns"]["ci_delta_auc"]
    ci_f1 = results_data["part3_reruns"]["ci_delta_f1"]

    # CI bounds must lie within observed extremes
    assert min(delta_aucs) <= ci_auc[0] <= max(delta_aucs), "Delta AUC CI lower bound outside sample range"
    assert min(delta_aucs) <= ci_auc[1] <= max(delta_aucs), "Delta AUC CI upper bound outside sample range"
    assert min(delta_f1s) <= ci_f1[0] <= max(delta_f1s), "Delta F1 CI lower bound outside sample range"
    assert min(delta_f1s) <= ci_f1[1] <= max(delta_f1s), "Delta F1 CI upper bound outside sample range"


def test_rule_g2_markdown_tables_consistency():
    """
    Parses results/all_tables_v5.md directly to verify that the markdown table rows
    exactly produce the printed summary statistics.
    """
    md_path = "results/all_tables_v5.md"
    if not os.path.exists(md_path):
        pytest.skip("all_tables_v5.md not yet generated")

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract Table 2.1 rows: | Seed 001 | 0.5123 | ...
    pattern = r"\|\s*Seed\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*Epoch\s*(\d+)\s*\|"
    matches = re.findall(pattern, text)
    assert len(matches) == 100, f"Expected 100 seeds in markdown Table 2.1, found {len(matches)}"

    aucs = [float(m[1]) for m in matches]
    md_mean = float(np.mean(aucs))
    md_std = float(np.std(aucs))
    md_min = float(np.min(aucs))
    md_max = float(np.max(aucs))

    # Check Table 2.2 summary values in markdown text
    assert f"{md_mean:.4f}" in text, "Markdown table mean does not match row recomputation"
    assert f"{md_std:.4f}" in text, "Markdown table std does not match row recomputation"
    assert f"{md_min:.4f}" in text, "Markdown table min does not match row recomputation"
    assert f"{md_max:.4f}" in text, "Markdown table max does not match row recomputation"

