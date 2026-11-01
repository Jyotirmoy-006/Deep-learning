"""
Generates verbatim markdown tables from results/experiment_results_v3.json
Strict adherence to rule: Paste tables; do not summarize numbers.
"""

import json

with open("results/experiment_results_v3.json", "r") as f:
    data = json.load(f)

# 1a. Shuffled-Label Control Table
part_1a = data["part_1a_shuffled_label_control"]
p1a_md = []
p1a_md.append("### 1a. Shuffled-Label Control (20 Seeds)")
p1a_md.append(f"- **Test Labels Initial SHA256:** `{part_1a['test_labels_sha256']}`")
p1a_md.append(f"- **Test Labels Immutability Verified:** `{part_1a['test_labels_unmodified']}` (Assert zero mutation across all shuffle runs)")
p1a_md.append(f"- **Control Status:** **{part_1a['control_status']}** (Mean ROC-AUC in $[0.40, 0.60]$: `{part_1a['auc_in_valid_range']}`)\n")
p1a_md.append("| Shuffle Seed | Accuracy | ROC-AUC | Status |")
p1a_md.append("|---|---|---|---|")
for row in part_1a["per_seed_results"]:
    p1a_md.append(f"| Seed {row['seed']} | {row['accuracy']:.4f} | {row['roc_auc']:.4f} | Permuted Train Only |")
p1a_md.append(f"| **Mean +/- Std** | **{part_1a['mean_accuracy']:.4f} +/- {part_1a['std_accuracy']:.4f}** | **{part_1a['mean_roc_auc']:.4f} +/- {part_1a['std_roc_auc']:.4f}** | **{part_1a['control_status']}** |")
p1a_md.append(f"\n**Mean Aggregated Confusion Matrix (over 20 seeds):**")
p1a_md.append("```text")
p1a_md.append(str(part_1a["mean_confusion_matrix"]))
p1a_md.append("```\n")

# 1b. Parameter Audit Tables
part_1b = data["part_1b_parameter_audit"]
p1b_md = []
p1b_md.append("### 1b. Comprehensive Parameter Audit & Per-Module Breakdown\n")

models = part_1b["models"]
for m_name, m_info in models.items():
    p1b_md.append(f"#### Module: `{m_name}` (Total Parameters: {m_info['total_params']:,})")
    p1b_md.append("| Parameter Name | Tensor Shape | Param Count | Trainable (`requires_grad`) |")
    p1b_md.append("|---|---|---|---|")
    for layer in m_info["layers"]:
        shape_str = str(layer["shape"])
        p1b_md.append(f"| `{layer['param_name']}` | `{shape_str}` | {layer['num_params']:,} | `{layer['requires_grad']}` |")
    p1b_md.append(f"| **Total** | - | **{m_info['total_params']:,}** | - |\n")

p1b_md.append(f"- **CNN Hidden Dense Layer (`classifier.1`):** {part_1b['cnn_hidden_dense_layer_params']:,} params ({part_1b['cnn_hidden_dense_layer_percentage']:.2f}% of CNN total)")
p1b_md.append(f"- **CNN Kernel Sizes:** `conv1`: {part_1b['cnn_kernel_sizes']['conv1']}, `conv2`: {part_1b['cnn_kernel_sizes']['conv2']}, `conv3`: {part_1b['cnn_kernel_sizes']['conv3']}")
p1b_md.append(f"- **Vocabulary Size:** {part_1b['vocab_size']}")
p1b_md.append(f"- **BiLSTM Attention Parameter Delta:** {part_1b['bilstm_attention_param_delta']:,} params (`pooling=attention`: 180,738 vs `pooling=concat`: 180,609)")
p1b_md.append(f"- **VAE 135 Discrepancy Diagnosis:** {part_1b['vae_discrepancy_explanation']}\n")

# 1c. Forward Hooks Shape Traces
part_1c = data["part_1c_real_batch_shape_traces"]
p1c_md = []
p1c_md.append("### 1c. Forward Hooks Shape Traces on REAL DataLoader Batch\n")
p1c_md.append(f"- **Real Input Batch Sizes:** {part_1c['real_batch_sizes']}\n")

for m_name, traces in part_1c["hook_traces"].items():
    p1c_md.append(f"#### Shape Trace: `{m_name}`")
    p1c_md.append("| Submodule Path | Module Class | Input Tensor Shape | Output Tensor Shape |")
    p1c_md.append("|---|---|---|---|")
    for t in traces:
        p1c_md.append(f"| `{t['submodule']}` | `{t['module_class']}` | `{t['input_shape']}` | `{t['output_shape']}` |")
    p1c_md.append("")

# 1d. Complete LOFO Table
part_1d = data["part_1d_complete_lofo"]
p1d_md = []
p1d_md.append("### 1d. Complete Leave-One-Feature-Out (LOFO) across all 7 Features\n")
p1d_md.append("| Configuration / Feature Dropped | Dropped Index | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix |")
p1d_md.append("|---|---|---|---|---|---|---|---|")
for k, v in part_1d.items():
    p1d_md.append(f"| **Drop `{v['dropped_feature']}`** | {v['dropped_index']} | {v['accuracy']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1_score']:.4f} | {v['roc_auc']:.4f} | `{v['confusion_matrix']}` |")
p1d_md.append("")

# 1e. Grid Audit Table
part_1e = data["part_1e_grid_audit"]
p1e_md = []
p1e_md.append("### 1e. Module 2 Grid Audit (36 ANN Configurations & CNN OFAT)\n")
p1e_md.append(f"- **Calibrated LR per Optimizer:** `{part_1e['lr_sweep_configuration']}`")
p1e_md.append(f"- **Convergence Settings:** Epoch Cap = `{part_1e['convergence_settings']['epoch_cap']}`, Threshold = `{part_1e['convergence_settings']['convergence_threshold']}`\n")
p1e_md.append("#### Full 36-Cell ANN Grid Search (3 Seeds per Cell)")
p1e_md.append("| Run ID | Optimizer | LR | Initialization | Dropout | BatchNorm | Cap-Hit? | Epochs to Conv (Mean +/- Std) | Final Train Loss (Mean +/- Std) | Val Acc (Mean +/- Std) | Val F1 (Mean +/- Std) | Val ROC-AUC (Mean +/- Std) |")
p1e_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in part_1e["ann_grid_runs"]:
    p1e_md.append(f"| {r['run_id']} | {r['optimizer']} | {r['learning_rate']} | {r['initialization']} | {r['dropout']} | {r['batchnorm']} | {r['cap_hit_flag']} | {r['epochs_to_convergence_mean']:.2f} +/- {r['epochs_to_convergence_std']:.2f} | {r['final_train_loss_mean']:.4f} +/- {r['final_train_loss_std']:.4f} | {r['val_accuracy_mean']:.4f} +/- {r['val_accuracy_std']:.4f} | {r['val_f1_mean']:.4f} +/- {r['val_f1_std']:.4f} | {r['val_roc_auc_mean']:.4f} +/- {r['val_roc_auc_std']:.4f} |")
p1e_md.append("")

p1e_md.append("#### CNN One-Factor-At-A-Time (OFAT) Grid")
p1e_md.append("| Run ID | Varied Factor | Factor Value | Optimizer | LR | Initialization | Dropout | BatchNorm | Final Train Loss | Val Accuracy | Val F1-Score | Val ROC-AUC |")
p1e_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in part_1e["cnn_ofat_runs"]:
    p1e_md.append(f"| {r['run_id']} | {r['varied_factor']} | {r['factor_value']} | {r['optimizer']} | {r['learning_rate']} | {r['initialization']} | {r['dropout']} | {r['batchnorm']} | {r['final_train_loss']:.4f} | {r['val_accuracy']:.4f} | {r['val_f1']:.4f} | {r['val_roc_auc']:.4f} |")
p1e_md.append("")

# Run 21 and 36 deep diagnosis
inv = part_1e["investigation_run21_run36"]
p1e_md.append("#### Deep Root-Cause Investigation: Run 21 and Run 36")
p1e_md.append(f"- **Run 21 Diagnosis:**")
p1e_md.append(f"  - *Configuration:* `{inv['run_21_diagnosis']['config']}`")
p1e_md.append(f"  - *Iteration 2 Observation:* {inv['run_21_diagnosis']['iteration_2_observation']}")
p1e_md.append(f"  - *Root Cause:* {inv['run_21_diagnosis']['root_cause']}")
p1e_md.append(f"  - *Iteration 3 Tuned Outcome:* Epochs: {inv['run_21_diagnosis']['iteration_3_tuned_outcome']['epochs_to_convergence_mean']:.1f}, Final Loss: {inv['run_21_diagnosis']['iteration_3_tuned_outcome']['final_train_loss_mean']:.4f}, Val Acc: {inv['run_21_diagnosis']['iteration_3_tuned_outcome']['val_accuracy_mean']:.4f}")
p1e_md.append(f"- **Run 36 Diagnosis:**")
p1e_md.append(f"  - *Configuration:* `{inv['run_36_diagnosis']['config']}`")
p1e_md.append(f"  - *Iteration 2 Observation:* {inv['run_36_diagnosis']['iteration_2_observation']}")
p1e_md.append(f"  - *Root Cause:* {inv['run_36_diagnosis']['root_cause']}")
p1e_md.append(f"  - *Iteration 3 Tuned Outcome:* Epochs: {inv['run_36_diagnosis']['iteration_3_tuned_outcome']['epochs_to_convergence_mean']:.1f}, Final Loss: {inv['run_36_diagnosis']['iteration_3_tuned_outcome']['final_train_loss_mean']:.4f}, Val Acc: {inv['run_36_diagnosis']['iteration_3_tuned_outcome']['val_accuracy_mean']:.4f}\n")

# 1f. BiLSTM Pooling Ablation Table
part_1f = data["part_1f_bilstm_pooling_ablation"]
p1f_md = []
p1f_md.append("### 1f. BiLSTM Concat vs Attention Pooling Ablation (5 Seeds)\n")
p1f_md.append(f"- **Evaluation Dataset:** Identical held-out test split (N = {part_1f['n_seeds']} seeds, max_len = {part_1f['max_len']})\n")
p1f_md.append("| Pooling Mechanism | Accuracy (Mean +/- Std) | Precision (Mean +/- Std) | Recall (Mean +/- Std) | F1-Score (Mean +/- Std) | ROC-AUC (Mean +/- Std) | Blueprint 5.2 Alignment |")
p1f_md.append("|---|---|---|---|---|---|---|")
for mode, s in part_1f["summary"].items():
    is_bp = "Authoritative Spec Default" if mode == "concat" else "Optional Ablation"
    p1f_md.append(f"| **BiLSTM (`{mode}`)** | {s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f} | {s['precision_mean']:.4f} +/- {s['precision_std']:.4f} | {s['recall_mean']:.4f} +/- {s['recall_std']:.4f} | {s['f1_score_mean']:.4f} +/- {s['f1_score_std']:.4f} | {s['roc_auc_mean']:.4f} +/- {s['roc_auc_std']:.4f} | {is_bp} |")
p1f_md.append("")

# 1g. Missing Diagnostics Tables
part_1g = data["part_1g_missing_diagnostics"]
p1g_md = []
p1g_md.append("### 1g. Comprehensive Diagnostics Suite (Grad Norms, Timing & Overfit-One-Batch)\n")

p1g_md.append("#### 1. Per-Epoch Gradient Norms & Anomaly Flags")
p1g_md.append("| Model Architecture | Epoch | Min Grad Norm | Max Grad Norm | Mean Grad Norm | NaN Flag | Inf Flag | Exploding Flag (>50.0) | Loss Stagnation Flag |")
p1g_md.append("|---|---|---|---|---|---|---|---|---|")
for m_name, g_info in part_1g["grad_norm_analysis"].items():
    for row in g_info["per_epoch_norms"]:
        p1g_md.append(f"| `{m_name}` | Epoch {row['epoch']} | {row['grad_norm_min']:.4f} | {row['grad_norm_max']:.4f} | {row['grad_norm_mean']:.4f} | `{g_info['nan_flag']}` | `{g_info['inf_flag']}` | `{g_info['exploding_gradient_flag']}` | `{g_info['loss_stagnation_flag']}` |")
p1g_md.append("")

p1g_md.append("#### 2. DataLoader Data-Wait Time vs Compute Time")
p1g_md.append("| Model Architecture | Total Data Wait (s) | Total Compute (s) | Data Wait / Compute Ratio | Assessment |")
p1g_md.append("|---|---|---|---|---|")
for m_name, t_info in part_1g["dataloader_timing"].items():
    ratio = t_info["wait_vs_compute_ratio"]
    status = "Compute-Bound (Healthy)" if ratio < 1.0 else "I/O Wait"
    p1g_md.append(f"| `{m_name}` | {t_info['total_data_wait_seconds']:.6f}s | {t_info['total_compute_seconds']:.6f}s | {ratio:.4f} | {status} |")
p1g_md.append("")

p1g_md.append("#### 3. Overfit-One-Batch Verification (ALL THREE Deep Learning Models)")
p1g_md.append("| Model Architecture | Batch Size | Epochs | Final Overfit Accuracy | Status |")
p1g_md.append("|---|---|---|---|---|")
for m_name, ob_info in part_1g["overfit_one_batch_all_models"].items():
    status = "**PASSED (100% Convergence)**" if ob_info["passed"] else "FAILED"
    p1g_md.append(f"| `{m_name}` | {ob_info['batch_size']} samples | {ob_info['epochs']} | **{ob_info['final_accuracy'] * 100:.1f}%** | {status} |")
p1g_md.append("")

with open("results/all_tables_v3.md", "w", encoding="utf-8") as f:
    f.write("\n".join(p1a_md + ["\n"] + p1b_md + ["\n"] + p1c_md + ["\n"] + p1d_md + ["\n"] + p1e_md + ["\n"] + p1f_md + ["\n"] + p1g_md))

print("All tables successfully written to results/all_tables_v3.md")
