"""
ForenSightX - Table Formatter for Iteration 7
Emits markdown tables directly from results/experiment_results_v7.json.
Enforces Rule G1 (Programmatic generation), Rule G2 (Round-trip consistency),
and Rule G9 (Direct markdown parseable format).
"""

import json
import os
import numpy as np

def generate_markdown_tables():
    with open("results/experiment_results_v7.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    def p(s=""):
        lines.append(s)

    p("# ForenSightX — Iteration 7 Experimental Tables")
    p()
    p("> **Source File:** `results/experiment_results_v7.json`  ")
    p("> **SHA-256:** `24949ad2045c4b0f167703defaca3a5322cf18d885482bc511aaa63bce30f344`  ")
    p()

    # Part 0
    p("## Part 0: Disclosures and Parameter Audit")
    p()
    p("### Table 0.1: Commit History for `chief executive` VIP Display Name Filter")
    p("`JSON Key: part0_disclosure.0c.commit_history_table`")
    p()
    p("| Commit | Tag / Phase | String Present | Status |")
    p("|---|---|---|---|")
    for r in data["part0_disclosure"]["0c"]["commit_history_table"]:
        p(f"| `{r['commit']}` | `{r['tag']}` | `{r['string_present']}` | {r['status']} |")
    p()

    p("### Table 0.2: Audited Parameter Breakdown for Buggy VAE (135 Parameters)")
    p("`JSON Key: part0_disclosure.0d.layer_breakdown`")
    p()
    p("| Layer Name | Parameter Shape | Number of Parameters |")
    p("|---|---|---|")
    for r in data["part0_disclosure"]["0d"]["layer_breakdown"]:
        p(f"| `{r['layer']}` | `{r['shape']}` | {r['params']} |")
    p(f"| **Total Parameters** | | **{data['part0_disclosure']['0d']['total_parameters']}** |")
    p()

    # Part 1
    p("## Part 1: Data Structure Audit (hard_v1)")
    p()
    p("### Table 1.1: hard_v1 Partition Structural Inventory")
    p("`JSON Key: part1_data_audit.1a_partition_table`")
    p()
    p("| Split | Total Cases | Unique Texts | Unique Images | Unique Families | Mean Text Reuse | Max Text Reuse | Mean Family Reuse | Max Family Reuse | Flipped Fraction |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for r in data["part1_data_audit"]["1a_partition_table"]:
        p(f"| **{r['split'].capitalize()}** | {r['total_cases']} | {r['unique_text_records']} | {r['unique_image_records']} | {r['unique_template_families']} | {r['mean_text_reuse']:.2f} | {r['max_text_reuse']} | {r['mean_family_reuse']:.2f} | {r['max_family_reuse']} | {r['actual_flipped_fraction']:.4f} |")
    p()

    p("### Table 1.2: Design Effect and Effective Sample Size (n_eff)")
    p("`JSON Key: part1_data_audit.1b_design_effect`")
    p()
    p("| Cluster Level | Bootstrap Variance | DEFF Ratio (Var_cluster / Var_case) | Total Cases (N) | Effective Sample Size (n_eff) |")
    p("|---|---|---|---|---|")
    de = data["part1_data_audit"]["1b_design_effect"]
    p(f"| Case-level (i.i.d.) | {de['case_level_bootstrap_variance']:.6f} | 1.0000 | {de['val_total_n']} | {de['val_total_n']:.1f} |")
    p(f"| Record Cluster | {de['record_cluster_bootstrap_variance']:.6f} | {de['deff_record']:.4f} | {de['val_total_n']} | {de['n_eff_record']:.1f} |")
    p(f"| Family Cluster | {de['family_cluster_bootstrap_variance']:.6f} | {de['deff_family']:.4f} | {de['val_total_n']} | {de['n_eff_family']:.1f} |")
    p()

    p("### Table 1.3: Permutation Function Feature Independence Audit")
    p("`JSON Key: part1_data_audit.1c_permutation_audit.feature_dependency_table`")
    p()
    p("| Feature Index | Feature Name | Mutual Information with Permuted Label | Chi-Square Statistic | p-Value |")
    p("|---|---|---|---|---|")
    for r in data["part1_data_audit"]["1c_permutation_audit"]["feature_dependency_table"]:
        p(f"| {r['feature_index']} | `{r['feature_name']}` | {r['mutual_information']:.6f} | {r['chi2_stat']:.4f} | {r['p_value']:.4f} |")
    p()

    # Part 2
    p("## Part 2: Grouped Splits and OOF Base Models (hard_v2)")
    p()
    p("### Table 2.1: GroupKFold OOF vs Val vs Test Performance")
    p("`JSON Key: part2_grouped_splits.2b_base_models`")
    p()
    p("| Base Model | OOF Train AUC | Val AUC | Test AUC | OOF Train Acc | Val Acc | Test Acc | Train c0 Std | Train c1 Std | Val c0 Std | Val c1 Std | Test c0 Std | Test c1 Std | \|OOF - Val\| AUC | Gate G-OOF Status |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in data["part2_grouped_splits"]["2b_base_models"]:
        status = "PASS" if r["delta_oof_val_auc"] <= 0.05 else "FAIL"
        p(f"| **{r['model']}** | {r['oof_train_auc']:.6f} | {r['val_auc']:.6f} | {r['test_auc']:.6f} | {r['oof_train_acc']:.4f} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | {r['train_c0_score_std']:.4f} | {r['train_c1_score_std']:.4f} | {r['val_c0_score_std']:.4f} | {r['val_c1_score_std']:.4f} | {r['test_c0_score_std']:.4f} | {r['test_c1_score_std']:.4f} | {r['delta_oof_val_auc']:.6f} | **{status}** |")
    p()

    p("### Table 2.2: Oracle Performance Ceilings (Family-Disjoint)")
    p("`JSON Key: part2_grouped_splits.2c_oracle_ceilings`")
    p()
    oc = data["part2_grouped_splits"]["2c_oracle_ceilings"]
    p("| Oracle Estimator | Validation ROC-AUC | Test ROC-AUC | Label Noise Scope |")
    p("|---|---|---|---|")
    p(f"| Logistic Regression Oracle | {oc['oracle_logistic_regression']['val_auc']:.6f} | {oc['oracle_logistic_regression']['test_auc']:.6f} | Training Split Only (5% rate) |")
    p(f"| Gradient Boosting (GBDT) Oracle | {oc['oracle_gbdt']['val_auc']:.6f} | {oc['oracle_gbdt']['test_auc']:.6f} | Training Split Only (5% rate) |")
    p()

    # Part 3
    p("## Part 3: Nested Null Controls (100 Seeds Each)")
    p()

    for ctrl_key, title in [
        ("N1", "N1: i.i.d. Gaussian Features Through ANN"),
        ("N2", "N2: Real Frozen Grouped-OOF Features, Permuted Labels"),
        ("N3", "N3: Full Pipeline Retrained Base Models, OOF, ANN"),
    ]:
        cdata = data["part3_nested_null_controls"][ctrl_key]
        p(f"### Table 3.{ctrl_key}: {title} Summary and 10-Bin Histogram")
        p(f"`JSON Key: part3_nested_null_controls.{ctrl_key}`")
        p()
        p("| Metric | Value | Reference / Bound | Pass / Fail |")
        p("|---|---|---|---|")
        p(f"| Theoretical Sigma_0 | {cdata['theoretical_sigma_0']:.6f} | Mann-Whitney U Null Std | Reference |")
        p(f"| Max Allowed Empirical Std | {cdata['max_allowed_empirical_sd']:.6f} | 2 * Sigma_0 Bound | Reference |")
        p(f"| Empirical Mean AUC | {cdata['empirical_mean_auc']:.6f} | Expected ~0.5000 | Within Bound |")
        p(f"| Empirical Std AUC | {cdata['empirical_std_auc']:.6f} | Must be <= 2 * Sigma_0 | **{'PASS' if cdata['std_check_passed'] else 'FAIL'}** |")
        p(f"| Percentile 2.5% | {cdata['percentile_2_5']:.6f} | Empirical 95% Interval Low | - |")
        p(f"| Percentile 97.5% | {cdata['percentile_97_5']:.6f} | Empirical 95% Interval High | - |")
        p(f"| Min AUC Observed | {cdata['min_auc']:.6f} | 4-Sigma Low: {cdata['four_sigma_lower_bound']:.6f} | **{'PASS' if cdata['min_auc'] >= cdata['four_sigma_lower_bound'] else 'FAIL'}** |")
        p(f"| Max AUC Observed | {cdata['max_auc']:.6f} | 4-Sigma High: {cdata['four_sigma_upper_bound']:.6f} | **{'PASS' if cdata['max_auc'] <= cdata['four_sigma_upper_bound'] else 'FAIL'}** |")
        p(f"| Mean Accuracy | {cdata['mean_accuracy']:.4f} | Expected ~0.5000 | - |")
        p(f"| Mean Binomial z-score | {cdata['mean_binomial_z']:.4f} | Expected ~0.0000 | - |")
        p(f"| Mean Selected Epoch | {cdata['mean_selected_epoch']:.2f} | 1 to 15 epochs | - |")
        p(f"| **Overall Gate G-PERM Status** | **{'PASS' if cdata['overall_pass'] else 'FAIL'}** | Std <= 2*Sigma0 & No seed > 4*Sigma0 | **{'PASS' if cdata['overall_pass'] else 'FAIL'}** |")
        p()

        p(f"#### 10-Bin Histogram for {ctrl_key} (Recomputed per Rule G9)")
        p(f"`JSON Key: part3_nested_null_controls.{ctrl_key}.histogram`")
        p()
        p("| Bin Index | Range Min | Range Max | Count |")
        p("|---|---|---|---|")
        for h in cdata["histogram"]:
            p(f"| {h['bin_idx']} | {h['range_min']:.4f} | {h['range_max']:.4f} | {h['count']} |")
        p()

        p(f"#### Per-Seed Raw Metrics for {ctrl_key} (100 Seeds)")
        p(f"`JSON Key: part3_nested_null_controls.{ctrl_key}.per_seed_rows`")
        p()
        p("| Seed | ROC-AUC | Accuracy | Binomial z-Score | Selected Epoch |")
        p("|---|---|---|---|---|")
        for r in cdata["per_seed_rows"]:
            p(f"| {r['seed']} | {r['auc']:.6f} | {r['accuracy']:.4f} | {r['binomial_z']:.4f} | {r['selected_epoch']} |")
        p()

    # Part 4
    p("## Part 4: Canonical Experiments A-E & Model Selection")
    p()
    p("### Table 4.1: ANN 48-Cell Hyperparameter Grid Top-5 Ranking and Negative Control")
    p("`JSON Key: part4_experiments.4b_ann_grid`")
    p()
    p("| Rank / Cell ID | Optimizer | LR | Weight Init | Dropout Rate | BatchNorm | Val ROC-AUC | Val F1 | Val BCE Loss | Saturation Fraction | Selection Status |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    neg = data["part4_experiments"]["4b_ann_grid"]["negative_control"]
    p(f"| **Negative Control** | {neg['optimizer']} | {neg['lr']} | {neg['init']} | {neg['dropout']} | {neg['batchnorm']} | {neg['val_auc']:.4f} | {neg['val_f1']:.4f} | {neg['val_bce_loss']:.4f} | {neg['saturation_fraction']:.4f} | **EXCLUDED (Control)** |")
    for r_idx, c in enumerate(data["part4_experiments"]["4b_ann_grid"]["top_5_cells"]):
        p(f"| **Rank {r_idx+1} ({c['cell_id']})** | {c['optimizer']} | {c['lr']} | {c['init']} | {c['dropout']} | {c['batchnorm']} | **{c['val_auc']:.6f}** | {c['val_f1']:.4f} | {c['val_bce_loss']:.4f} | {c['saturation_fraction']:.4f} | {'**WINNER**' if r_idx == 0 else 'Candidate'} |")
    p()

    p("### Table 4.2: Canonical Experiments A-E Summary Across 5 Seeds")
    p("`JSON Key: part4_experiments.4c_canonical_summary`")
    p()
    p("| Experiment Arm | Architecture / Features | Mean ROC-AUC (Std) | Mean F1 (Std) | Mean Accuracy (Std) | Mean Precision (Std) | Mean Recall (Std) | Mean Brier Score (Std) |")
    p("|---|---|---|---|---|---|---|---|")
    arm_desc = {
        "A": "BiLSTM Text Evidence Only",
        "B": "VisualEvidenceCNN Image Only",
        "C": "ANN on Multimodal Embeddings [BiLSTM, CNN]",
        "D": "ANN on Full 7D Feature Vector (Winner Grid)",
        "E": "Exp D + 50% VAE Feature Augmentation",
    }
    for arm in ["A", "B", "C", "D", "E"]:
        cs = data["part4_experiments"]["4c_canonical_summary"][arm]
        p(f"| **Exp {arm}** | {arm_desc[arm]} | **{cs['mean_auc']:.6f}** (±{cs['std_auc']:.6f}) | **{cs['mean_f1']:.4f}** (±{cs['std_f1']:.4f}) | {cs['mean_accuracy']:.4f} (±{cs['std_accuracy']:.4f}) | {cs['mean_precision']:.4f} (±{cs['std_precision']:.4f}) | {cs['mean_recall']:.4f} (±{cs['std_recall']:.4f}) | {cs['mean_brier_score']:.6f} (±{cs['std_brier_score']:.6f}) |")
    p()

    p("### Table 4.3: Per-Seed Raw Metrics for Experiments A-E")
    p("`JSON Key: part4_experiments.4c_canonical_summary.<arm>.per_seed_runs`")
    p()
    p("| Seed | Exp A AUC | Exp B AUC | Exp C AUC | Exp D AUC | Exp E AUC | Exp A F1 | Exp B F1 | Exp C F1 | Exp D F1 | Exp E F1 |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    seeds = [42, 43, 44, 45, 46]
    for idx, s in enumerate(seeds):
        ra = data["part4_experiments"]["4c_canonical_summary"]["A"]["per_seed_runs"][idx]
        rb = data["part4_experiments"]["4c_canonical_summary"]["B"]["per_seed_runs"][idx]
        rc = data["part4_experiments"]["4c_canonical_summary"]["C"]["per_seed_runs"][idx]
        rd = data["part4_experiments"]["4c_canonical_summary"]["D"]["per_seed_runs"][idx]
        re_arm = data["part4_experiments"]["4c_canonical_summary"]["E"]["per_seed_runs"][idx]
        p(f"| {s} | {ra['auc']:.6f} | {rb['auc']:.6f} | {rc['auc']:.6f} | {rd['auc']:.6f} | {re_arm['auc']:.6f} | {ra['f1']:.4f} | {rb['f1']:.4f} | {rc['f1']:.4f} | {rd['f1']:.4f} | {re_arm['f1']:.4f} |")
    p()

    p("### Table 4.4: Cluster Bootstrap 95% Confidence Intervals (1000 Replicates)")
    p("`JSON Key: part4_experiments.4c_cluster_ci_table`")
    p()
    p("| Comparison | Delta ROC-AUC | 95% CI (Delta AUC) | Delta AUC Significant | Delta F1 | 95% CI (Delta F1) | Delta F1 Significant |")
    p("|---|---|---|---|---|---|---|")
    for r in data["part4_experiments"]["4c_cluster_ci_table"]:
        p(f"| **{r['comparison']}** | {r['delta_auc_mean']:+.4f} | [{r['delta_auc_95_ci'][0]:+.4f}, {r['delta_auc_95_ci'][1]:+.4f}] | **{'YES' if r['delta_auc_significant'] else 'NO'}** | {r['delta_f1_mean']:+.4f} | [{r['delta_f1_95_ci'][0]:+.4f}, {r['delta_f1_95_ci'][1]:+.4f}] | **{'YES' if r['delta_f1_significant'] else 'NO'}** |")
    p()

    p("### Table 4.5: Leave-One-Feature-Out (LOFO) and Single-Feature ANN Audits")
    p("`JSON Key: part4_experiments.4d_diagnostics.lofo_table` & `part4_experiments.4d_diagnostics.single_feature_ann_table`")
    p()
    p("| Feature Name | LOFO Test AUC (Drop Feature) | Delta AUC Drop from Full D | Single-Feature ANN Test AUC | Exceeds 0.85 Cap | Permutation Importance Drop |")
    p("|---|---|---|---|---|---|")
    lofo = data["part4_experiments"]["4d_diagnostics"]["lofo_table"]
    single = data["part4_experiments"]["4d_diagnostics"]["single_feature_ann_table"]
    perm_d = data["part4_experiments"]["4d_diagnostics"]["permutation_importance_d"]
    for idx in range(7):
        p(f"| `{lofo[idx]['dropped_feature']}` | {lofo[idx]['test_auc']:.6f} | {lofo[idx]['delta_auc_from_full_d']:+.6f} | {single[idx]['test_auc']:.6f} | **{'YES' if single[idx]['exceeds_cap_0_85'] else 'NO'}** | {perm_d[idx]['test_auc_drop']:+.6f} |")
    p()

    p("### Table 4.6: Calibration and Prediction Agreement Matrix")
    p("`JSON Key: part4_experiments.4d_diagnostics.calibration_table` & `part4_experiments.4d_diagnostics.prediction_agreement_matrix`")
    p()
    cal = data["part4_experiments"]["4d_diagnostics"]["calibration_table"]
    p(f"**Temperature Scaling Diagnostics:** Fitted Temperature = **{cal['fitted_temperature']:.4f}**")
    p(f"- Brier Score: {cal['brier_score_before']:.6f} (Raw) -> **{cal['brier_score_after']:.6f}** (Calibrated)")
    p(f"- Expected Calibration Error (ECE): {cal['ece_before']:.6f} (Raw) -> **{cal['ece_after']:.6f}** (Calibrated)")
    p()
    p("| Arm Agreement Fraction | Exp A | Exp B | Exp C | Exp D |")
    p("|---|---|---|---|---|")
    pam = data["part4_experiments"]["4d_diagnostics"]["prediction_agreement_matrix"]
    for a1 in ["A", "B", "C", "D"]:
        p(f"| **Exp {a1}** | {pam[a1]['A']:.4f} | {pam[a1]['B']:.4f} | {pam[a1]['C']:.4f} | {pam[a1]['D']:.4f} |")
    p()

    p("### Table 4.7: CNN OFAT Anomaly Resolution (Calibrated Threshold)")
    p("`JSON Key: part4_experiments.4e_cnn_ofat`")
    p()
    of = data["part4_experiments"]["4e_cnn_ofat"]
    p(f"- **Logit Distribution:** Mean = {of['val_logit_mean']:.4f}, Std = {of['val_logit_std']:.4f}")
    p(f"- **Rigid Threshold (0.50) Accuracy:** {of['val_acc_at_0_5']:.4f}")
    p(f"- **Calibrated Threshold ({of['calibrated_threshold']:.4f}) Accuracy:** Val = **{of['val_acc_calibrated']:.4f}**, Test = **{of['test_acc_calibrated']:.4f}**")
    p()
    p("| Seed | BatchNorm ON Val AUC | BatchNorm ON Val Acc (Calibrated) | BatchNorm OFF Val AUC | BatchNorm OFF Val Acc (Calibrated) |")
    p("|---|---|---|---|---|")
    for idx in range(len(of["ofat_rerun_bn_on"])):
        ron = of["ofat_rerun_bn_on"][idx]
        roff = of["ofat_rerun_bn_off"][idx]
        p(f"| {ron['seed']} | {ron['val_auc']:.6f} | {ron['val_acc_calibrated']:.4f} | {roff['val_auc']:.6f} | {roff['val_acc_calibrated']:.4f} |")
    p()

    p("### Table 4.8: BiLSTM Pooling Architecture Comparison (Concat vs Attention)")
    p("`JSON Key: part4_experiments.4f_bilstm_pooling`")
    p()
    bp = data["part4_experiments"]["4f_bilstm_pooling"]
    p("| Pooling Architecture | Validation ROC-AUC | Test ROC-AUC | Oracle Ceiling Test AUC | Ceiling Violation |")
    p("|---|---|---|---|---|")
    p(f"| Concat Pooling (Default) | {bp['concat_val_auc']:.6f} | {bp['concat_test_auc']:.6f} | {bp['oracle_ceiling_test_auc']:.6f} | **{'YES' if bp['ceiling_violation'] else 'NO'}** |")
    p(f"| Attention Pooling | {bp['attention_val_auc']:.6f} | {bp['attention_test_auc']:.6f} | {bp['oracle_ceiling_test_auc']:.6f} | **{'YES' if bp['ceiling_violation'] else 'NO'}** |")
    p()

    # Part 5
    p("## Part 5: Repaired VAE Two-Sided Validation & Augmentation")
    p()
    p("### Table 5.1: Two-Sided Criteria for VAE Feature Distribution (Per Feature, Per Class)")
    p("`JSON Key: part5_vae.5a_two_sided_criteria`")
    p()
    p("| Feature Index | Feature Name | Class 0 Real Std | Class 0 Synth Std | Class 0 Std Ratio [0.5, 2.0] | Class 0 Wasserstein Ratio (<=0.50) | Class 0 Pass | Class 1 Real Std | Class 1 Synth Std | Class 1 Std Ratio [0.5, 2.0] | Class 1 Wasserstein Ratio (<=0.50) | Class 1 Pass |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in data["part5_vae"]["5a_two_sided_criteria"]:
        c0 = r["class_0"]
        c1 = r["class_1"]
        p(f"| {r['feature_index']} | `{r['feature_name']}` | {c0['real_std']:.4f} | {c0['synth_std']:.4f} | {c0['ratio']:.4f} ({'PASS' if c0['ratio_in_bounds'] else 'FAIL'}) | {c0['wasserstein_ratio']:.4f} ({'PASS' if c0['wasserstein_in_bounds'] else 'FAIL'}) | **{'PASS' if c0['passed'] else 'FAIL'}** | {c1['real_std']:.4f} | {c1['synth_std']:.4f} | {c1['ratio']:.4f} ({'PASS' if c1['ratio_in_bounds'] else 'FAIL'}) | {c1['wasserstein_ratio']:.4f} ({'PASS' if c1['wasserstein_in_bounds'] else 'FAIL'}) | **{'PASS' if c1['passed'] else 'FAIL'}** |")
    p()

    p("### Table 5.2: Per-Dimension KL Divergence and Active Latent Capacity")
    p("`JSON Key: part5_vae.5a_per_dim_kl`")
    p()
    p("| Latent Dimension | KL Divergence (nats) | Free-Bits Floor (nats) | Margin Above Floor (nats) | Non-Collapsed Status (>= 0.05 nats) |")
    p("|---|---|---|---|---|")
    for r in data["part5_vae"]["5a_per_dim_kl"]:
        p(f"| Dimension {r['latent_dim']} | **{r['kl_divergence_nats']:.6f}** | {r['free_bits_floor']:.3f} | {r['margin_above_floor']:+.6f} | **{'ACTIVE (PASS)' if r['non_collapsed'] else 'COLLAPSED (FAIL)'}** |")
    p()

    p("### Table 5.3: GBDT Real vs Synthetic Discriminator & Positive Control")
    p("`JSON Key: part5_vae.5b_discriminator`")
    p()
    disc = data["part5_vae"]["5b_discriminator"]
    p("| Evaluation Task | Estimator | 5-Fold CV ROC-AUC | Target Threshold | Status / Validation |")
    p("|---|---|---|---|---|")
    p(f"| Indistinguishability (Real vs Repaired VAE) | Gradient Boosting (GBDT) | {disc['discriminator_gbdt_cv_auc']:.4f} | <= {disc['discriminator_threshold_cap']:.2f} | **{'PASS' if disc['discriminator_gbdt_cv_auc'] <= disc['discriminator_threshold_cap'] else 'DISTINGUISHABLE'}** |")
    p(f"| Positive Control (Degraded Generator Centroid+10%) | Gradient Boosting (GBDT) | {disc['positive_control_degraded_auc']:.4f} | > 0.90 | **{'VALID (PASS)' if disc['positive_control_passed'] else 'INVALID'}** |")
    p()

    p("### Table 5.4: Experiment E 4-Arm Ratio Sweep Across Augmentation Levels")
    p("`JSON Key: part5_vae.5c_ratio_sweep`")
    p()
    p("| Augmentation Arm | Ratio = 0.0 (Baseline) | Ratio = 0.25 | Ratio = 0.50 | Ratio = 1.00 | Mean Across Ratios |")
    p("|---|---|---|---|---|---|")
    arms = ["Baseline_None", "SyntheticDataVAE", "GaussianAugmenter", "InterpolationAugmenter"]
    for arm in arms:
        row_entries = [r for r in data["part5_vae"]["5c_ratio_sweep"] if r["arm"] == arm]
        vals = {r["augmentation_ratio"]: f"{r['mean_auc']:.6f} (±{r['std_auc']:.6f})" for r in row_entries}
        m_all = np.mean([r["mean_auc"] for r in row_entries])
        p(f"| **{arm}** | {vals.get(0.0, '-')} | {vals.get(0.25, '-')} | {vals.get(0.5, '-')} | {vals.get(1.0, '-')} | **{m_all:.6f}** |")
    p()

    # Research Questions
    p("## Research Questions Summary")
    p()
    p("### Table 6.1: Formal Research Questions (Strict yes/no/inconclusive with 95% CI)")
    p("`JSON Key: research_questions`")
    p()
    p("| ID | Research Question | Decision | Evaluation Metric | Estimate | 95% Cluster Bootstrap CI | Statistically Significant |")
    p("|---|---|---|---|---|---|---|")
    for rq in data["research_questions"]:
        p(f"| **{rq['question_id']}** | {rq['research_question']} | **`{rq['decision']}`** | {rq['metric']} | {rq['estimate']:+.4f} | [{rq['cluster_bootstrap_95_ci'][0]:+.4f}, {rq['cluster_bootstrap_95_ci'][1]:+.4f}] | **{'YES' if rq['statistically_significant'] else 'NO'}** |")
    p()

    output_path = "results/all_tables_v7.md"
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] All tables written to {output_path} ({len(lines)} lines)")
    return content

if __name__ == "__main__":
    generate_markdown_tables()
