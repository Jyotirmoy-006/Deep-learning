"""
ForenSightX - Iteration 7 Round-Trip Consistency Tests (Rule G9)
Directly parses the printed Markdown tables in EXECUTION_REPORT_v7.md (NOT the JSON)
and recomputes every mean, standard deviation, percentile (via numpy.percentile),
histogram bin count, and ratio from the printed rows.
"""

import re
import numpy as np
import pytest

REPORT_PATH = "EXECUTION_REPORT_v7.md"


def load_report_text() -> str:
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def parse_markdown_table_rows(table_text: str):
    """Parses markdown table into list of row dictionaries."""
    lines = [l.strip() for l in table_text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return []
    header = [c.strip() for c in lines[0].split("|")[1:-1]]
    rows = []
    for l in lines[2:]:
        if not l.startswith("|"):
            continue
        cols = [c.strip() for c in l.split("|")[1:-1]]
        if len(cols) == len(header):
            rows.append(dict(zip(header, cols)))
    return rows


# =====================================================================
# 1. Round-Trip Verification of Null Controls (N1, N2, N3)
# =====================================================================

@pytest.mark.parametrize("ctrl_key", ["N1", "N2", "N3"])
def test_null_control_roundtrip_recomputation(ctrl_key: str):
    """
    Rule G9: Parses the 100 printed per-seed rows for N1, N2, N3 directly from
    EXECUTION_REPORT_v7.md and recomputes mean, std, percentiles, min, max, and
    10-bin histogram counts. Asserts exact match with printed summary table.
    """
    text = load_report_text()

    # Find the section for this control
    sec_pattern = rf"### Table 3\.{ctrl_key}:.*?\n(?P<summary>\|.*?)\n\n#### 10-Bin Histogram for {ctrl_key}.*?\n(?P<hist>\|.*?)\n\n#### Per-Seed Raw Metrics for {ctrl_key}.*?\n(?P<raw>\|.*?)(?=\n\n|\Z)"
    match = re.search(sec_pattern, text, re.DOTALL)
    assert match is not None, f"Could not find section for Table 3.{ctrl_key} in {REPORT_PATH}!"

    raw_table_text = match.group("raw")
    summary_table_text = match.group("summary")
    hist_table_text = match.group("hist")

    raw_rows = parse_markdown_table_rows(raw_table_text)
    assert len(raw_rows) == 100, f"Expected 100 printed rows for {ctrl_key}, found {len(raw_rows)}!"

    # Parse numeric values from printed rows
    aucs = np.array([float(r["ROC-AUC"]) for r in raw_rows])
    accs = np.array([float(r["Accuracy"]) for r in raw_rows])
    z_scores = np.array([float(r["Binomial z-Score"]) for r in raw_rows])
    epochs = np.array([float(r["Selected Epoch"]) for r in raw_rows])

    # Recompute summary statistics directly from printed rows using numpy
    recomputed_mean_auc = float(np.mean(aucs))
    recomputed_std_auc = float(np.std(aucs))
    recomputed_p2_5 = float(np.percentile(aucs, 2.5))
    recomputed_p97_5 = float(np.percentile(aucs, 97.5))
    recomputed_min_auc = float(np.min(aucs))
    recomputed_max_auc = float(np.max(aucs))
    recomputed_mean_acc = float(np.mean(accs))
    recomputed_mean_z = float(np.mean(z_scores))
    recomputed_mean_epoch = float(np.mean(epochs))

    # Parse printed summary values
    summary_rows = parse_markdown_table_rows(summary_table_text)
    summary_dict = {r["Metric"]: r["Value"] for r in summary_rows}

    printed_mean_auc = float(summary_dict["Empirical Mean AUC"])
    printed_std_auc = float(summary_dict["Empirical Std AUC"])
    printed_p2_5 = float(summary_dict["Percentile 2.5%"])
    printed_p97_5 = float(summary_dict["Percentile 97.5%"])
    printed_min_auc = float(summary_dict["Min AUC Observed"])
    printed_max_auc = float(summary_dict["Max AUC Observed"])
    printed_mean_acc = float(summary_dict["Mean Accuracy"])
    printed_mean_z = float(summary_dict["Mean Binomial z-score"])
    printed_mean_epoch = float(summary_dict["Mean Selected Epoch"])

    # Assert match within floating point tolerance
    assert abs(recomputed_mean_auc - printed_mean_auc) < 1e-4, f"{ctrl_key} Mean AUC mismatch!"
    assert abs(recomputed_std_auc - printed_std_auc) < 1e-4, f"{ctrl_key} Std AUC mismatch!"
    assert abs(recomputed_p2_5 - printed_p2_5) < 1e-4, f"{ctrl_key} P2.5 mismatch!"
    assert abs(recomputed_p97_5 - printed_p97_5) < 1e-4, f"{ctrl_key} P97.5 mismatch!"
    assert abs(recomputed_min_auc - printed_min_auc) < 1e-4, f"{ctrl_key} Min AUC mismatch!"
    assert abs(recomputed_max_auc - printed_max_auc) < 1e-4, f"{ctrl_key} Max AUC mismatch!"
    assert abs(recomputed_mean_acc - printed_mean_acc) < 1e-4, f"{ctrl_key} Mean Acc mismatch!"
    assert abs(recomputed_mean_z - printed_mean_z) < 1e-4, f"{ctrl_key} Mean z mismatch!"
    assert abs(recomputed_mean_epoch - printed_mean_epoch) < 1e-2, f"{ctrl_key} Mean epoch mismatch!"

    # Recompute 10-bin histogram counts directly from printed rows
    recomputed_hist, _ = np.histogram(aucs, bins=10, range=(0.35, 0.65))
    hist_rows = parse_markdown_table_rows(hist_table_text)
    assert len(hist_rows) == 10, f"Expected 10 histogram rows for {ctrl_key}!"

    for b_idx, h_row in enumerate(hist_rows):
        printed_count = int(h_row["Count"])
        assert recomputed_hist[b_idx] == printed_count, (
            f"{ctrl_key} Histogram bin {b_idx} count mismatch: recomputed {recomputed_hist[b_idx]} != printed {printed_count}"
        )


# =====================================================================
# 2. Round-Trip Verification of Canonical Experiments A-E
# =====================================================================

def test_canonical_experiments_roundtrip_recomputation():
    """
    Rule G9: Parses the 5 per-seed rows for Experiments A-E in Table 4.3 and
    recomputes mean and std AUC/F1. Asserts exact match with Table 4.2.
    """
    text = load_report_text()

    # Match Table 4.2 (Summary) and Table 4.3 (Per-Seed)
    t43_match = re.search(r"### Table 4\.3: Per-Seed Raw Metrics for Experiments A-E.*?\n(?P<t43>\|.*?)(?=\n\n|\Z)", text, re.DOTALL)
    t42_match = re.search(r"### Table 4\.2: Canonical Experiments A-E Summary Across 5 Seeds.*?\n(?P<t42>\|.*?)(?=\n\n|\Z)", text, re.DOTALL)

    assert t43_match is not None, "Could not find Table 4.3 in report!"
    assert t42_match is not None, "Could not find Table 4.2 in report!"

    t43_rows = parse_markdown_table_rows(t43_match.group("t43"))
    t42_rows = parse_markdown_table_rows(t42_match.group("t42"))

    assert len(t43_rows) == 5, f"Expected 5 seed rows in Table 4.3, found {len(t43_rows)}!"
    assert len(t42_rows) == 5, f"Expected 5 arm rows in Table 4.2, found {len(t42_rows)}!"

    arms = ["A", "B", "C", "D", "E"]
    for idx, arm in enumerate(arms):
        # Extract printed per-seed values from Table 4.3
        arm_aucs = [float(r[f"Exp {arm} AUC"]) for r in t43_rows]
        arm_f1s = [float(r[f"Exp {arm} F1"]) for r in t43_rows]

        recomp_mean_auc = float(np.mean(arm_aucs))
        recomp_std_auc = float(np.std(arm_aucs))
        recomp_mean_f1 = float(np.mean(arm_f1s))
        recomp_std_f1 = float(np.std(arm_f1s))

        # Extract printed summary values from Table 4.2
        t42_row = [r for r in t42_rows if f"Exp {arm}" in r["Experiment Arm"]][0]
        # Format: "**0.986078** (±0.002100)"
        auc_str = t42_row["Mean ROC-AUC (Std)"].replace("**", "")
        m_auc_pr = float(re.search(r"([\d\.]+)\s*\([±\s]*([\d\.]+)\)", auc_str).group(1))
        s_auc_pr = float(re.search(r"([\d\.]+)\s*\([±\s]*([\d\.]+)\)", auc_str).group(2))

        f1_str = t42_row["Mean F1 (Std)"].replace("**", "")
        m_f1_pr = float(re.search(r"([\d\.]+)\s*\([±\s]*([\d\.]+)\)", f1_str).group(1))
        s_f1_pr = float(re.search(r"([\d\.]+)\s*\([±\s]*([\d\.]+)\)", f1_str).group(2))

        assert abs(recomp_mean_auc - m_auc_pr) < 1e-4, f"Exp {arm} Mean AUC mismatch: {recomp_mean_auc} != {m_auc_pr}"
        assert abs(recomp_std_auc - s_auc_pr) < 1e-4, f"Exp {arm} Std AUC mismatch: {recomp_std_auc} != {s_auc_pr}"
        assert abs(recomp_mean_f1 - m_f1_pr) < 1e-4, f"Exp {arm} Mean F1 mismatch: {recomp_mean_f1} != {m_f1_pr}"
        assert abs(recomp_std_f1 - s_f1_pr) < 1e-4, f"Exp {arm} Std F1 mismatch: {recomp_std_f1} != {s_f1_pr}"


# =====================================================================
# 3. Round-Trip Verification of VAE Two-Sided Criteria
# =====================================================================

def test_vae_two_sided_criteria_roundtrip():
    """
    Rule G9: Parses Table 5.1 in EXECUTION_REPORT_v7.md and recomputes the
    standard deviation ratio for each feature and class from the printed stds.
    """
    text = load_report_text()
    t51_match = re.search(r"### Table 5\.1: Two-Sided Criteria for VAE Feature Distribution.*?\n(?P<t51>\|.*?)(?=\n\n|\Z)", text, re.DOTALL)
    assert t51_match is not None, "Could not find Table 5.1 in report!"

    rows = parse_markdown_table_rows(t51_match.group("t51"))
    assert len(rows) == 7, f"Expected 7 feature rows in Table 5.1, found {len(rows)}!"

    for r in rows:
        # Class 0
        r_c0_real = float(r["Class 0 Real Std"])
        r_c0_synth = float(r["Class 0 Synth Std"])
        r_c0_ratio_str = r["Class 0 Std Ratio [0.5, 2.0]"]
        r_c0_ratio_val = float(re.search(r"([\d\.]+)", r_c0_ratio_str).group(1))

        recomputed_c0_ratio = r_c0_synth / r_c0_real
        assert abs(recomputed_c0_ratio - r_c0_ratio_val) < 1e-3, (
            f"Class 0 ratio mismatch for {r['Feature Name']}: recomputed {recomputed_c0_ratio:.4f} != printed {r_c0_ratio_val:.4f}"
        )

        # Class 1
        r_c1_real = float(r["Class 1 Real Std"])
        r_c1_synth = float(r["Class 1 Synth Std"])
        r_c1_ratio_str = r["Class 1 Std Ratio [0.5, 2.0]"]
        r_c1_ratio_val = float(re.search(r"([\d\.]+)", r_c1_ratio_str).group(1))

        recomputed_c1_ratio = r_c1_synth / r_c1_real
        assert abs(recomputed_c1_ratio - r_c1_ratio_val) < 1e-3, (
            f"Class 1 ratio mismatch for {r['Feature Name']}: recomputed {recomputed_c1_ratio:.4f} != printed {r_c1_ratio_val:.4f}"
        )
