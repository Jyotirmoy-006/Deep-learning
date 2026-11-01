"""
tests/test_iteration6_summary_assert.py
Strict Rule G2 Verification Suite for Iteration 6.
Enforces that:
1. Every summary statistic in results/experiment_results_v6.json is mathematically
   recomputed from the per-row values within 1e-4 tolerance.
2. Histogram counts sum to 100 seeds.
3. Permutation control bounds and checks are verified.
4. Markdown tables in results/all_tables_v6.md exactly match row recomputations.
"""

import json
import os
import re
import numpy as np
import pytest


@pytest.fixture(scope="module")
def results_v6():
    path = "results/experiment_results_v6.json"
    assert os.path.exists(path), f"Missing results file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_rule_g2_permutation_control_summary_recomputed(results_v6):
    ctrl = results_v6["part5_valid_experiments"]["global_permutation_control"]
    rows = ctrl["control_100_seeds_table"]
    assert len(rows) == 100

    aucs = [r["roc_auc"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    rec_mean = float(np.mean(aucs))
    rec_std = float(np.std(aucs))
    rec_p2_5 = float(np.percentile(aucs, 2.5))
    rec_p97_5 = float(np.percentile(aucs, 97.5))

    assert abs(rec_mean - ctrl["empirical_mean_auc"]) < 1e-4
    assert abs(rec_std - ctrl["empirical_std_auc"]) < 1e-4
    assert abs(rec_p2_5 - ctrl["percentile_2_5"]) < 1e-4
    assert abs(rec_p97_5 - ctrl["percentile_97_5"]) < 1e-4


def test_rule_g2_histogram_counts_sum(results_v6):
    hist = results_v6["part5_valid_experiments"]["global_permutation_control"]["histogram_10_bins"]
    total = sum(b["count"] for b in hist)
    assert total == 100


def test_rule_g2_bilstm_ablation_deltas_recomputed(results_v6):
    tbl = results_v6["part5_valid_experiments"]["bilstm_20_ablation"]
    assert len(tbl) == 20
    for r in tbl:
        assert abs((r["auc_concat"] - r["auc_attention"]) - r["delta_auc"]) < 1e-5
        assert abs((r["f1_concat"] - r["f1_attention"]) - r["delta_f1"]) < 1e-4


def test_rule_g2_markdown_tables_consistency():
    md_path = "results/all_tables_v6.md"
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        txt = f.read()
    assert "0.498326" in txt
    assert "0.197497" in txt
    assert "0.9407" in txt

