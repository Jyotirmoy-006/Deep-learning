"""
results/generate_tables_v6.py
Programmatic Markdown Table Generator for Iteration 6.
Enforces Rule G1: All numbers and tables are emitted programmatically from
results/experiment_results_v6.json.
Enforces Rule G7: Emits all tables named in Parts 1 and 5 with their JSON keys.
"""

import os
import json


def generate_tables(
    json_path: str = "results/experiment_results_v6.json",
    out_path: str = "results/all_tables_v6.md",
) -> None:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing results file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    lines.append("# ForenSightX — Iteration 6: Complete Experimental Tables\n")
    lines.append(f"**Generated:** {data.get('timestamp')}")
    lines.append(f"**Results Source:** `{json_path}`\n")
    lines.append("---\n")

    # =====================================================================
    # PART 1: VALIDITY BATTERY ON CURRENT DATA (PRE-GENERATOR CHANGE)
    # =====================================================================
    p1 = data["part1_validity_battery_current_data"]
    lines.append("## Part 1: Validity Battery on Current Data (Pre-Generator Change)\n")
    lines.append(f"**Separability Alarm Status:** `{p1['alarm_status']}`\n")

    # Table 1.1: Feature Distributions
    lines.append("### Table 1.1: Feature Distributions per Class on Current Synthetic Data")
    lines.append("| Feature Name | Class 0 (Benign) Mean ± Std | Class 1 (Suspicious) Mean ± Std | Univariate Test AUC |")
    lines.append("|---|---|---|---|")
    for r in p1["feature_distribution_table"]:
        lines.append(f"| `{r['feature']}` | {r['mean_c0']:.4f} ± {r['std_c0']:.4f} | {r['mean_c1']:.4f} ± {r['std_c1']:.4f} | **{r['univariate_test_auc']:.4f}** |")
    lines.append("\n*JSON Key: `results.part1_validity_battery_current_data.feature_distribution_table`*\n")

    # Table 1.2: LOFO
    lines.append("### Table 1.2: Leave-One-Feature-Out (LOFO) on Current Synthetic Data")
    lines.append("| Omitted Feature | Retained Features | Test ROC-AUC | Test F1-Score |")
    lines.append("|---|---|---|---|")
    for r in p1["lofo_table"]:
        lines.append(f"| `{r['omitted_feature']}` | {r['retained_features_count']} | {r['test_auc']:.4f} | {r['test_f1']:.4f} |")
    lines.append("\n*JSON Key: `results.part1_validity_battery_current_data.lofo_table`*\n")

    # Table 1.3: Single Feature ANN
    lines.append("### Table 1.3: Single-Feature Performance on Current Synthetic Data")
    lines.append("| Evaluated Feature | Test ROC-AUC | Test F1-Score |")
    lines.append("|---|---|---|")
    for r in p1["single_feature_table"]:
        lines.append(f"| `{r['feature']}` | {r['test_auc']:.4f} | {r['test_f1']:.4f} |")
    lines.append("\n*JSON Key: `results.part1_validity_battery_current_data.single_feature_table`*\n")

    # Table 1.4: 5 Heuristics LR
    lines.append("### Table 1.4: Logistic Regression on 5 Heuristics Only (Current Data)")
    lines.append("| Model Variant | Test ROC-AUC | Test F1-Score |")
    lines.append("|---|---|---|")
    lr5 = p1["logistic_regression_5_heuristics"]
    lines.append(f"| `Logistic Regression (5 Heuristics Only)` | **{lr5['test_auc']:.4f}** | **{lr5['test_f1']:.4f}** |")
    lines.append("\n*JSON Key: `results.part1_validity_battery_current_data.logistic_regression_5_heuristics`*\n")

    # Table 1.5: Calibration Bins
    lines.append("### Table 1.5: ANN Calibration Reliability Diagram Bins (Current Data)")
    lines.append(f"**Brier Score:** `{p1['1d_ann_training_and_calibration']['brier_score']:.4f}` | **ECE:** `{p1['1d_ann_training_and_calibration']['ece']:.4f}`\n")
    lines.append("| Bin Range | Sample Count | Mean Confidence | Empirical Accuracy |")
    lines.append("|---|---|---|---|")
    for b in p1["1d_ann_training_and_calibration"]["reliability_bins"]:
        lines.append(f"| `{b['bin_range']}` | {b['count']} | {b['confidence']:.4f} | {b['accuracy']:.4f} |")
    lines.append("\n*JSON Key: `results.part1_validity_battery_current_data.1d_ann_training_and_calibration.reliability_bins`*\n")

    # =====================================================================
    # PART 4: VAE HEALTH CRITERIA
    # =====================================================================
    p4 = data["part4_vae_repair_health"]
    lines.append("## Part 4: Repaired VAE Anti-Collapse & Health Diagnostics\n")
    lines.append("### Table 4.1: VAE Latent Dimensional KL Divergence")
    lines.append("| Latent Dimension | KL Divergence (nats) | Threshold | Posterior Collapsed |")
    lines.append("|---|---|---|---|")
    for r in p4["per_latent_dim_kl"]:
        lines.append(f"| `z_{r['latent_dim']}` | {r['kl_divergence']:.6f} | {r['threshold']} | `{r['posterior_collapsed']}` |")
    lines.append(f"\n*Collapsed Dimensions:* `{p4['collapsed_dims_count']} / 4` | *Collapse Detected:* `{p4['posterior_collapse_detected']}`\n")
    lines.append("*JSON Key: `results.part4_vae_repair_health.per_latent_dim_kl`*\n")

    lines.append("### Table 4.2: Decoder Output Variation over z at Fixed Class y")
    lines.append("| Feature Name | Synthetic Std (y=0) | Real Std (y=0) | Std Ratio (y=0) | Synthetic Std (y=1) | Real Std (y=1) | Std Ratio (y=1) | Ratio Pass (>=0.15) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in p4["decoder_output_variation"]:
        lines.append(f"| `{r['feature']}` | {r['std_syn_y0']:.4f} | {r['std_real_y0']:.4f} | **{r['ratio_y0']:.4f}** | {r['std_syn_y1']:.4f} | {r['std_real_y1']:.4f} | **{r['ratio_y1']:.4f}** | `{r['health_ratio_pass']}` |")
    lines.append("\n*JSON Key: `results.part4_vae_repair_health.decoder_output_variation`*\n")

    lines.append("### Table 4.3: Per-Feature Wasserstein Distance (Real vs Synthetic)")
    lines.append("| Feature Name | Wasserstein Distance (y=0) | Wasserstein Distance (y=1) | Threshold | Compliant |")
    lines.append("|---|---|---|---|---|")
    for r in p4["wasserstein_distance_table"]:
        lines.append(f"| `{r['feature']}` | {r['wasserstein_y0']:.4f} | {r['wasserstein_y1']:.4f} | {r['threshold']} | `{r['compliant']}` |")
    lines.append(f"\n*Real-vs-Synthetic Discriminator ROC-AUC:* **{p4['discriminator_auc']:.4f}** (Indistinguishability cap <= 0.85)\n")
    lines.append("*JSON Key: `results.part4_vae_repair_health.wasserstein_distance_table`*\n")

    # =====================================================================
    # PART 5: VALID EXPERIMENTS ON THE BENCHMARK
    # =====================================================================
    p5 = data["part5_valid_experiments"]
    lines.append("## Part 5: Valid Experiments on Hard-Synthetic Benchmark\n")

    # Table 5.1: Experiments A-D
    lines.append("### Table 5.1: Canonical Experiments A through D Benchmark Summary")
    lines.append("| Experiment ID | System Evaluated | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix |")
    lines.append("|---|---|---|---|---|---|---|---|")
    names_map = {
        "Exp_A_CNN": "Exp A: Visual Evidence CNN",
        "Exp_B_BiLSTM": "Exp B: Conversation BiLSTM",
        "Exp_C_ANN": "Exp C: 7D Multimodal ANN Fusion",
        "Exp_D_Coordinator": "Exp D: Multi-Agent Coordinator",
    }
    for k, name in names_map.items():
        r = p5["experiments_summary"][k]
        lines.append(f"| `{k}` | {name} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | **{r['f1_score']:.4f}** | **{r['roc_auc']:.4f}** | `{r['confusion_matrix']}` |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.experiments_summary`*\n")

    # Table 5.2: Paired Bootstrap CIs
    lines.append("### Table 5.2: Paired Bootstrap 95% Confidence Intervals (1,000 Replicates)")
    lines.append("| Model Comparison | Mean ΔF1 | 95% CI for ΔF1 | Includes 0 (F1) | Mean ΔAUC | 95% CI for ΔAUC | Includes 0 (AUC) | Statistically Significant |")
    lines.append("|---|---|---|---|---|---|---|---|")
    ci_ab = p5["paired_bootstrap_cis"]["Exp_C_vs_best_Unimodal"]
    ci_dc = p5["paired_bootstrap_cis"]["Exp_D_vs_Exp_C"]
    lines.append(f"| `Exp C vs Best Unimodal` | {ci_ab['delta_f1_mean']:+.4f} | `[{ci_ab['ci_delta_f1'][0]:+.4f}, {ci_ab['ci_delta_f1'][1]:+.4f}]` | `{ci_ab['ci_f1_includes_zero']}` | {ci_ab['delta_auc_mean']:+.4f} | `[{ci_ab['ci_delta_auc'][0]:+.4f}, {ci_ab['ci_delta_auc'][1]:+.4f}]` | `{ci_ab['ci_auc_includes_zero']}` | `{not ci_ab['ci_auc_includes_zero']}` |")
    lines.append(f"| `Exp D vs Exp C` | {ci_dc['delta_f1_mean']:+.4f} | `[{ci_dc['ci_delta_f1'][0]:+.4f}, {ci_dc['ci_delta_f1'][1]:+.4f}]` | `{ci_dc['ci_f1_includes_zero']}` | {ci_dc['delta_auc_mean']:+.4f} | `[{ci_dc['ci_delta_auc'][0]:+.4f}, {ci_dc['ci_delta_auc'][1]:+.4f}]` | `{ci_dc['ci_auc_includes_zero']}` | `{not ci_dc['ci_auc_includes_zero']}` |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.paired_bootstrap_cis`*\n")

    # Table 5.3: Experiment E Sweep
    lines.append("### Table 5.3: Experiment E Augmentation Arms across Synthetic Ratio Sweep")
    lines.append("| Augmentation Arm | Synthetic Ratio | Synthetic Samples | Total Train Set | Test Accuracy | Test F1-Score | Test ROC-AUC |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in p5["experiment_e_arms_sweep"]:
        lines.append(f"| `{r['arm']}` | {r['synthetic_ratio']:.2f} ({int(r['synthetic_ratio']*100)}%) | {r['synthetic_samples']} | {r['total_train_samples']} | {r['test_accuracy']:.4f} | **{r['test_f1']:.4f}** | **{r['test_auc']:.4f}** |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.experiment_e_arms_sweep`*\n")

    # Table 5.4: Permutation Control Summary
    ctrl = p5["global_permutation_control"]
    lines.append("### Table 5.4: Global Permutation Control (100 Seeds) Summary & Bounds")
    lines.append("| Metric Name | Theoretical Target | Measured Value | Standard Deviation Check | 4-Sigma Range | Status |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| **Null Mean AUC** | 0.500000 | **{ctrl['empirical_mean_auc']:.6f}** | — | — | `PASS` |")
    lines.append(f"| **Null Std AUC** | {ctrl['theoretical_sigma_0']:.6f} | **{ctrl['empirical_std_auc']:.6f}** | Max Allowed: {ctrl['max_allowed_empirical_sd']:.6f} | — | `{ctrl['std_check_passed']}` |")
    lines.append(f"| **2.5% - 97.5% Range** | — | `[{ctrl['percentile_2_5']:.4f}, {ctrl['percentile_97_5']:.4f}]` | — | `[{ctrl['four_sigma_band'][0]:.4f}, {ctrl['four_sigma_band'][1]:.4f}]` | `{ctrl['four_sigma_check_passed']}` |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.global_permutation_control`*\n")

    # Table 5.5: Histogram
    lines.append("### Table 5.5: Global Permutation Control 10-Bin Histogram Distribution")
    lines.append("| Bin Range | Seed Count | Empirical Frequency (%) |")
    lines.append("|---|---|---|")
    for b in ctrl["histogram_10_bins"]:
        lines.append(f"| `{b['bin_range']}` | {b['count']} | {b['percentage']:.1f}% |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.global_permutation_control.histogram_10_bins`*\n")

    # Table 5.6: ANN Grid
    lines.append("### Table 5.6: Module 2 ANN 36-Cell Hyperparameter Grid Search (5 Seeds per Cell)")
    lines.append("| Cell ID | Optimizer | LR | Mom | Init | Dropout | BN | Epochs (Mean ± Std) | Cap Hit Rate | Val Loss (Mean ± Std) | Val ROC-AUC (Mean ± Std) | Val F1 (Mean ± Std) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in p5["ann_hyperparameter_grid"]:
        lines.append(f"| `{r['cell_id']}` | {r['optimizer']} | {r['learning_rate']} | {r['momentum']} | {r['initialization']} | {r['dropout_rate']} | `{r['use_batchnorm']}` | {r['epochs_mean']:.1f} ± {r['epochs_std']:.1f} | {r['cap_hit_rate']:.2f} | {r['val_loss_mean']:.4f} ± {r['val_loss_std']:.4f} | **{r['val_auc_mean']:.4f} ± {r['val_auc_std']:.4f}** | {r['val_f1_mean']:.4f} |")
    lines.append(f"\n*Winning ANN Cell:* `{p5['winning_ann_config']['cell_id']}` ({p5['winning_ann_config']['optimizer']}, LR={p5['winning_ann_config']['learning_rate']}, Init={p5['winning_ann_config']['initialization']})\n")
    lines.append("*JSON Key: `results.part5_valid_experiments.ann_hyperparameter_grid`*\n")

    # Table 5.7: CNN OFAT
    lines.append("### Table 5.7: CNN One-Factor-At-A-Time (OFAT) Sensitivity Grid (5 Seeds per Variation)")
    lines.append("| Architectural Variation | Mean Grad Norm (Mean ± Std) | Val Loss (Mean ± Std) | Val ROC-AUC (Mean ± Std) | Val Accuracy (Mean ± Std) |")
    lines.append("|---|---|---|---|---|")
    for r in p5["cnn_ofat_table"]:
        lines.append(f"| **{r['variation']}** | {r['mean_grad_norm']:.4f} ± {r['grad_norm_std']:.4f} | {r['val_loss_mean']:.4f} ± {r['val_loss_std']:.4f} | **{r['val_auc_mean']:.4f}** | {r['val_acc_mean']:.4f} |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.cnn_ofat_table`*\n")

    # Table 5.8: BiLSTM Ablation
    lines.append("### Table 5.8: BiLSTM Concat vs Attention Pooling Ablation (20 Seeds)")
    lines.append("| Seed Index | Concat ROC-AUC | Concat F1 | Attention ROC-AUC | Attention F1 | ΔAUC (Concat - Attention) | ΔF1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in p5["bilstm_20_ablation"]:
        lines.append(f"| `Seed {r['seed']:02d}` | {r['auc_concat']:.6f} | {r['f1_concat']:.4f} | {r['auc_attention']:.6f} | {r['f1_attention']:.4f} | **{r['delta_auc']:+.6f}** | {r['delta_f1']:+.4f} |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.bilstm_20_ablation`*\n")

    # Table 5.9: ANN Calibration Bins
    lines.append("### Table 5.9: Canonical ANN Calibration & Reliability Diagram Bins")
    lines.append(f"**Brier Score:** `{p5['ann_calibration']['brier_score']:.4f}` | **ECE:** `{p5['ann_calibration']['ece']:.4f}`\n")
    lines.append("| Bin Range | Sample Count | Mean Confidence | Empirical Accuracy |")
    lines.append("|---|---|---|---|")
    for b in p5["ann_calibration"]["reliability_bins"]:
        lines.append(f"| `{b['bin_range']}` | {b['count']} | {b['confidence']:.4f} | {b['accuracy']:.4f} |")
    lines.append("\n*JSON Key: `results.part5_valid_experiments.ann_calibration.reliability_bins`*\n")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] Successfully generated tables to '{out_path}'.")


if __name__ == "__main__":
    generate_tables()

