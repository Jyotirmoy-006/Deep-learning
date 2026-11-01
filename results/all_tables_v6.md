# ForenSightX — Iteration 6: Complete Experimental Tables

**Generated:** 2026-11-01 19:54:17
**Results Source:** `results/experiment_results_v6.json`

---

## Part 1: Validity Battery on Current Data (Pre-Generator Change)

**Separability Alarm Status:** `ALARM TRIGGERED: TASK TRIVIALLY SEPARABLE (Univariate AUCs >= 0.95)`

### Table 1.1: Feature Distributions per Class on Current Synthetic Data
| Feature Name | Class 0 (Benign) Mean ± Std | Class 1 (Suspicious) Mean ± Std | Univariate Test AUC |
|---|---|---|---|
| `cnn_score` | 0.1485 ± 0.0462 | 0.8459 ± 0.0487 | **1.0000** |
| `bilstm_score` | 0.1185 ± 0.0448 | 0.8752 ± 0.0515 | **1.0000** |
| `url_score` | 0.0000 ± 0.0000 | 0.2375 ± 0.4136 | **0.6250** |
| `keyword_score` | 0.0000 ± 0.0000 | 0.6984 ± 0.2440 | **0.9844** |
| `attachment_score` | 0.0304 ± 0.1577 | 0.4250 ± 0.4250 | **0.7321** |
| `timeline_anomaly` | 0.0500 ± 0.0000 | 0.5187 ± 0.2480 | **0.8906** |
| `email_anomaly` | 0.0500 ± 0.0000 | 0.1312 ± 0.2150 | **0.5625** |

*JSON Key: `results.part1_validity_battery_current_data.feature_distribution_table`*

### Table 1.2: Leave-One-Feature-Out (LOFO) on Current Synthetic Data
| Omitted Feature | Retained Features | Test ROC-AUC | Test F1-Score |
|---|---|---|---|
| `cnn_score` | 6 | 1.0000 | 1.0000 |
| `bilstm_score` | 6 | 1.0000 | 1.0000 |
| `url_score` | 6 | 1.0000 | 1.0000 |
| `keyword_score` | 6 | 1.0000 | 1.0000 |
| `attachment_score` | 6 | 1.0000 | 1.0000 |
| `timeline_anomaly` | 6 | 1.0000 | 1.0000 |
| `email_anomaly` | 6 | 1.0000 | 1.0000 |

*JSON Key: `results.part1_validity_battery_current_data.lofo_table`*

### Table 1.3: Single-Feature Performance on Current Synthetic Data
| Evaluated Feature | Test ROC-AUC | Test F1-Score |
|---|---|---|
| `cnn_score` | 1.0000 | 1.0000 |
| `bilstm_score` | 1.0000 | 1.0000 |
| `url_score` | 0.6250 | 0.4000 |
| `keyword_score` | 0.9844 | 0.9841 |
| `attachment_score` | 0.7321 | 0.6531 |
| `timeline_anomaly` | 0.8906 | 0.8772 |
| `email_anomaly` | 0.5625 | 0.6957 |

*JSON Key: `results.part1_validity_battery_current_data.single_feature_table`*

### Table 1.4: Logistic Regression on 5 Heuristics Only (Current Data)
| Model Variant | Test ROC-AUC | Test F1-Score |
|---|---|---|
| `Logistic Regression (5 Heuristics Only)` | **1.0000** | **1.0000** |

*JSON Key: `results.part1_validity_battery_current_data.logistic_regression_5_heuristics`*

### Table 1.5: ANN Calibration Reliability Diagram Bins (Current Data)
**Brier Score:** `0.0077` | **ECE:** `0.0681`

| Bin Range | Sample Count | Mean Confidence | Empirical Accuracy |
|---|---|---|---|
| `[0.0, 0.1)` | 27 | 0.0752 | 0.0000 |
| `[0.1, 0.2)` | 1 | 0.1374 | 0.0000 |
| `[0.2, 0.3)` | 0 | 0.2500 | 0.0000 |
| `[0.3, 0.4)` | 0 | 0.3500 | 0.0000 |
| `[0.4, 0.5)` | 0 | 0.4500 | 0.0000 |
| `[0.5, 0.6)` | 0 | 0.5500 | 0.0000 |
| `[0.6, 0.7)` | 0 | 0.6500 | 0.0000 |
| `[0.7, 0.8)` | 3 | 0.7502 | 1.0000 |
| `[0.8, 0.9)` | 3 | 0.8659 | 1.0000 |
| `[0.9, 1.0)` | 26 | 0.9706 | 1.0000 |

*JSON Key: `results.part1_validity_battery_current_data.1d_ann_training_and_calibration.reliability_bins`*

## Part 4: Repaired VAE Anti-Collapse & Health Diagnostics

### Table 4.1: VAE Latent Dimensional KL Divergence
| Latent Dimension | KL Divergence (nats) | Threshold | Posterior Collapsed |
|---|---|---|---|
| `z_0` | 0.056196 | 0.05 | `False` |
| `z_1` | 0.067283 | 0.05 | `False` |
| `z_2` | 0.065292 | 0.05 | `False` |
| `z_3` | 0.048978 | 0.05 | `True` |

*Collapsed Dimensions:* `1 / 4` | *Collapse Detected:* `False`

*JSON Key: `results.part4_vae_repair_health.per_latent_dim_kl`*

### Table 4.2: Decoder Output Variation over z at Fixed Class y
| Feature Name | Synthetic Std (y=0) | Real Std (y=0) | Std Ratio (y=0) | Synthetic Std (y=1) | Real Std (y=1) | Std Ratio (y=1) | Ratio Pass (>=0.15) |
|---|---|---|---|---|---|---|---|
| `cnn_score` | 0.0129 | 0.2093 | **0.0617** | 0.0160 | 0.1982 | **0.0807** | `False` |
| `bilstm_score` | 0.0034 | 0.0017 | **1.9567** | 0.0051 | 0.0001 | **37.1575** | `True` |
| `url_score` | 0.0127 | 0.0936 | **0.1357** | 0.0434 | 0.4347 | **0.0999** | `False` |
| `keyword_score` | 0.0104 | 0.2473 | **0.0421** | 0.0227 | 0.3103 | **0.0733** | `False` |
| `attachment_score` | 0.0099 | 0.0933 | **0.1064** | 0.2671 | 0.4669 | **0.5721** | `False` |
| `timeline_anomaly` | 0.0176 | 0.2615 | **0.0675** | 0.0337 | 0.3599 | **0.0937** | `False` |
| `email_anomaly` | 0.0323 | 0.2979 | **0.1083** | 0.0309 | 0.4257 | **0.0725** | `False` |

*JSON Key: `results.part4_vae_repair_health.decoder_output_variation`*

### Table 4.3: Per-Feature Wasserstein Distance (Real vs Synthetic)
| Feature Name | Wasserstein Distance (y=0) | Wasserstein Distance (y=1) | Threshold | Compliant |
|---|---|---|---|---|
| `cnn_score` | 0.0960 | 0.0773 | 0.2 | `True` |
| `bilstm_score` | 0.0098 | 0.0096 | 0.2 | `True` |
| `url_score` | 0.0782 | 0.3916 | 0.2 | `False` |
| `keyword_score` | 0.1928 | 0.2415 | 0.2 | `False` |
| `attachment_score` | 0.0309 | 0.2364 | 0.2 | `False` |
| `timeline_anomaly` | 0.1875 | 0.3317 | 0.2 | `False` |
| `email_anomaly` | 0.2424 | 0.3708 | 0.2 | `False` |

*Real-vs-Synthetic Discriminator ROC-AUC:* **0.4654** (Indistinguishability cap <= 0.85)

*JSON Key: `results.part4_vae_repair_health.wasserstein_distance_table`*

## Part 5: Valid Experiments on Hard-Synthetic Benchmark

### Table 5.1: Canonical Experiments A through D Benchmark Summary
| Experiment ID | System Evaluated | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|
| `Exp_A_CNN` | Exp A: Visual Evidence CNN | 0.9433 | 0.9571 | 0.9241 | **0.9404** | **0.9317** | `[[149, 6], [11, 134]]` |
| `Exp_B_BiLSTM` | Exp B: Conversation BiLSTM | 0.8667 | 0.8261 | 0.9172 | **0.8693** | **0.8447** | `[[127, 28], [12, 133]]` |
| `Exp_C_ANN` | Exp C: 7D Multimodal ANN Fusion | 0.8667 | 0.8261 | 0.9172 | **0.8693** | **0.9407** | `[[127, 28], [12, 133]]` |
| `Exp_D_Coordinator` | Exp D: Multi-Agent Coordinator | 0.8667 | 0.8261 | 0.9172 | **0.8693** | **0.9228** | `[[127, 28], [12, 133]]` |

*JSON Key: `results.part5_valid_experiments.experiments_summary`*

### Table 5.2: Paired Bootstrap 95% Confidence Intervals (1,000 Replicates)
| Model Comparison | Mean ΔF1 | 95% CI for ΔF1 | Includes 0 (F1) | Mean ΔAUC | 95% CI for ΔAUC | Includes 0 (AUC) | Statistically Significant |
|---|---|---|---|---|---|---|---|
| `Exp C vs Best Unimodal` | -0.0707 | `[-0.1158, -0.0262]` | `False` | +0.0085 | `[-0.0253, +0.0450]` | `True` | `False` |
| `Exp D vs Exp C` | +0.0000 | `[+0.0000, +0.0000]` | `True` | -0.0179 | `[-0.0338, -0.0060]` | `False` | `True` |

*JSON Key: `results.part5_valid_experiments.paired_bootstrap_cis`*

### Table 5.3: Experiment E Augmentation Arms across Synthetic Ratio Sweep
| Augmentation Arm | Synthetic Ratio | Synthetic Samples | Total Train Set | Test Accuracy | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| `None` | 0.00 (0%) | 0 | 700 | 0.8667 | **0.8693** | **0.9487** |
| `VAE` | 0.25 (25%) | 175 | 875 | 0.8667 | **0.8693** | **0.9610** |
| `Gaussian` | 0.25 (25%) | 175 | 875 | 0.8667 | **0.8693** | **0.9580** |
| `Interpolation` | 0.25 (25%) | 175 | 875 | 0.8667 | **0.8693** | **0.9472** |
| `VAE` | 0.50 (50%) | 350 | 1050 | 0.8667 | **0.8693** | **0.9537** |
| `Gaussian` | 0.50 (50%) | 350 | 1050 | 0.8667 | **0.8693** | **0.9396** |
| `Interpolation` | 0.50 (50%) | 350 | 1050 | 0.8667 | **0.8693** | **0.9426** |
| `VAE` | 1.00 (100%) | 700 | 1400 | 0.8667 | **0.8693** | **0.8749** |
| `Gaussian` | 1.00 (100%) | 700 | 1400 | 0.8667 | **0.8693** | **0.8827** |
| `Interpolation` | 1.00 (100%) | 700 | 1400 | 0.8667 | **0.8693** | **0.8305** |

*JSON Key: `results.part5_valid_experiments.experiment_e_arms_sweep`*

### Table 5.4: Global Permutation Control (100 Seeds) Summary & Bounds
| Metric Name | Theoretical Target | Measured Value | Standard Deviation Check | 4-Sigma Range | Status |
|---|---|---|---|---|---|
| **Null Mean AUC** | 0.500000 | **0.498326** | — | — | `PASS` |
| **Null Std AUC** | 0.033407 | **0.197497** | Max Allowed: 0.066815 | — | `False` |
| **2.5% - 97.5% Range** | — | `[0.1599, 0.8835]` | — | `[0.3664, 0.6336]` | `False` |

*JSON Key: `results.part5_valid_experiments.global_permutation_control`*

### Table 5.5: Global Permutation Control 10-Bin Histogram Distribution
| Bin Range | Seed Count | Empirical Frequency (%) |
|---|---|---|
| `[0.0, 0.1)` | 1 | 1.0% |
| `[0.1, 0.2)` | 5 | 5.0% |
| `[0.2, 0.3)` | 8 | 8.0% |
| `[0.3, 0.4)` | 21 | 21.0% |
| `[0.4, 0.5)` | 19 | 19.0% |
| `[0.5, 0.6)` | 13 | 13.0% |
| `[0.6, 0.7)` | 15 | 15.0% |
| `[0.7, 0.8)` | 11 | 11.0% |
| `[0.8, 0.9)` | 5 | 5.0% |
| `[0.9, 1.0)` | 2 | 2.0% |

*JSON Key: `results.part5_valid_experiments.global_permutation_control.histogram_10_bins`*

### Table 5.6: Module 2 ANN 36-Cell Hyperparameter Grid Search (5 Seeds per Cell)
| Cell ID | Optimizer | LR | Mom | Init | Dropout | BN | Epochs (Mean ± Std) | Cap Hit Rate | Val Loss (Mean ± Std) | Val ROC-AUC (Mean ± Std) | Val F1 (Mean ± Std) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `C01` | Adam | 0.001 | 0.0 | he_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0359 ± 0.0055 | **1.0000 ± 0.0000** | 1.0000 |
| `C02` | Adam | 0.001 | 0.0 | xavier_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0284 ± 0.0053 | **1.0000 ± 0.0000** | 1.0000 |
| `C03` | Adam | 0.001 | 0.0 | default | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0317 ± 0.0059 | **1.0000 ± 0.0000** | 1.0000 |
| `C04` | Adam | 0.01 | 0.0 | he_normal | 0.2 | `True` | 13.4 ± 0.5 | 0.00 | 0.0001 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C05` | Adam | 0.01 | 0.0 | xavier_normal | 0.2 | `True` | 13.4 ± 1.0 | 0.20 | 0.0001 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C06` | Adam | 0.01 | 0.0 | default | 0.2 | `True` | 14.0 ± 0.9 | 0.40 | 0.0002 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C07` | Adam | 0.5 | 0.0 | he_normal | 0.2 | `True` | 4.4 ± 0.5 | 0.00 | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C08` | Adam | 0.5 | 0.0 | xavier_normal | 0.2 | `True` | 4.4 ± 0.5 | 0.00 | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C09` | Adam | 0.5 | 0.0 | default | 0.2 | `True` | 4.2 ± 0.4 | 0.00 | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C10` | RMSprop | 0.001 | 0.0 | he_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0107 ± 0.0018 | **1.0000 ± 0.0000** | 1.0000 |
| `C11` | RMSprop | 0.001 | 0.0 | xavier_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0087 ± 0.0026 | **1.0000 ± 0.0000** | 1.0000 |
| `C12` | RMSprop | 0.001 | 0.0 | default | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0110 ± 0.0022 | **1.0000 ± 0.0000** | 1.0000 |
| `C13` | RMSprop | 0.01 | 0.0 | he_normal | 0.2 | `True` | 9.0 ± 1.1 | 0.00 | 0.0001 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C14` | RMSprop | 0.01 | 0.0 | xavier_normal | 0.2 | `True` | 9.0 ± 0.6 | 0.00 | 0.0001 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C15` | RMSprop | 0.01 | 0.0 | default | 0.2 | `True` | 10.8 ± 0.7 | 0.00 | 0.0001 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C16` | RMSprop | 0.5 | 0.0 | he_normal | 0.2 | `True` | 5.8 ± 1.7 | 0.00 | 0.0164 ± 0.0328 | **1.0000 ± 0.0000** | 0.9980 |
| `C17` | RMSprop | 0.5 | 0.0 | xavier_normal | 0.2 | `True` | 4.8 ± 1.0 | 0.00 | 0.1000 ± 0.2000 | **0.9980 ± 0.0040** | 0.9970 |
| `C18` | RMSprop | 0.5 | 0.0 | default | 0.2 | `True` | 5.2 ± 1.5 | 0.00 | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C19` | SGD_momentum_0.0 | 0.001 | 0.0 | he_normal | 0.2 | `True` | 12.4 ± 4.2 | 0.40 | 0.5919 ± 0.0556 | **0.8071 ± 0.0784** | 0.7183 |
| `C20` | SGD_momentum_0.0 | 0.001 | 0.0 | xavier_normal | 0.2 | `True` | 12.6 ± 4.3 | 0.60 | 0.5200 ± 0.1225 | **0.8548 ± 0.2104** | 0.8478 |
| `C21` | SGD_momentum_0.0 | 0.001 | 0.0 | default | 0.2 | `True` | 12.8 ± 4.4 | 0.80 | 0.5732 ± 0.1178 | **0.7843 ± 0.3923** | 0.8059 |
| `C22` | SGD_momentum_0.0 | 0.01 | 0.0 | he_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.1629 ± 0.0129 | **0.9998 ± 0.0003** | 0.9898 |
| `C23` | SGD_momentum_0.0 | 0.01 | 0.0 | xavier_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.1171 ± 0.0076 | **1.0000 ± 0.0000** | 0.9980 |
| `C24` | SGD_momentum_0.0 | 0.01 | 0.0 | default | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.1375 ± 0.0177 | **1.0000 ± 0.0000** | 1.0000 |
| `C25` | SGD_momentum_0.0 | 0.5 | 0.0 | he_normal | 0.2 | `True` | 13.4 ± 0.5 | 0.00 | 0.0002 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C26` | SGD_momentum_0.0 | 0.5 | 0.0 | xavier_normal | 0.2 | `True` | 14.2 ± 0.7 | 0.40 | 0.0002 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C27` | SGD_momentum_0.0 | 0.5 | 0.0 | default | 0.2 | `True` | 14.2 ± 0.7 | 0.40 | 0.0002 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C28` | SGD_momentum_0.9 | 0.001 | 0.9 | he_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.1670 ± 0.0138 | **0.9998 ± 0.0003** | 0.9898 |
| `C29` | SGD_momentum_0.9 | 0.001 | 0.9 | xavier_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.1208 ± 0.0078 | **1.0000 ± 0.0000** | 0.9970 |
| `C30` | SGD_momentum_0.9 | 0.001 | 0.9 | default | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.1438 ± 0.0199 | **1.0000 ± 0.0000** | 1.0000 |
| `C31` | SGD_momentum_0.9 | 0.01 | 0.9 | he_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0041 ± 0.0011 | **1.0000 ± 0.0000** | 1.0000 |
| `C32` | SGD_momentum_0.9 | 0.01 | 0.9 | xavier_normal | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0027 ± 0.0004 | **1.0000 ± 0.0000** | 1.0000 |
| `C33` | SGD_momentum_0.9 | 0.01 | 0.9 | default | 0.2 | `True` | 15.0 ± 0.0 | 1.00 | 0.0028 ± 0.0002 | **1.0000 ± 0.0000** | 1.0000 |
| `C34` | SGD_momentum_0.9 | 0.5 | 0.9 | he_normal | 0.2 | `True` | 4.2 ± 0.4 | 0.00 | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C35` | SGD_momentum_0.9 | 0.5 | 0.9 | xavier_normal | 0.2 | `True` | 4.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |
| `C36` | SGD_momentum_0.9 | 0.5 | 0.9 | default | 0.2 | `True` | 4.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 |

*Winning ANN Cell:* `C18` (RMSprop, LR=0.5, Init=default)

*JSON Key: `results.part5_valid_experiments.ann_hyperparameter_grid`*

### Table 5.7: CNN One-Factor-At-A-Time (OFAT) Sensitivity Grid (5 Seeds per Variation)
| Architectural Variation | Mean Grad Norm (Mean ± Std) | Val Loss (Mean ± Std) | Val ROC-AUC (Mean ± Std) | Val Accuracy (Mean ± Std) |
|---|---|---|---|---|
| **Baseline (Adam, LR=0.001, Dropout=0.3, BN=True)** | 1.2265 ± 0.0809 | 1.1482 ± 0.3747 | **0.9366** | 0.5000 |
| **OFAT 1: Dropout OFF (p=0.0)** | 1.1111 ± 0.0926 | 1.1850 ± 0.3935 | **0.9386** | 0.5000 |
| **OFAT 2: BatchNorm OFF** | 1.3224 ± 0.5049 | 0.3302 ± 0.0642 | **0.9501** | 0.9400 |
| **OFAT 3: LR Half (0.0005)** | 1.0275 ± 0.0483 | 1.3382 ± 0.3212 | **0.9377** | 0.5000 |
| **OFAT 4: LR Double (0.002)** | 1.6973 ± 0.1501 | 0.7236 ± 0.0665 | **0.9352** | 0.5000 |
| **OFAT 5: Optimizer RMSprop** | 3.6638 ± 0.5137 | 0.9389 ± 0.2564 | **0.9304** | 0.5000 |
| **OFAT 6: Optimizer SGD (mom=0.9)** | 1.2484 ± 0.0472 | 0.9418 ± 0.1657 | **0.9369** | 0.5000 |

*JSON Key: `results.part5_valid_experiments.cnn_ofat_table`*

### Table 5.8: BiLSTM Concat vs Attention Pooling Ablation (20 Seeds)
| Seed Index | Concat ROC-AUC | Concat F1 | Attention ROC-AUC | Attention F1 | ΔAUC (Concat - Attention) | ΔF1 |
|---|---|---|---|---|---|---|
| `Seed 01` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 02` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 03` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 04` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 05` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 06` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 07` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 08` | 1.000000 | 0.9901 | 1.000000 | 1.0000 | **-0.000000** | -0.0099 |
| `Seed 09` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 10` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 11` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 12` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 13` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 14` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 15` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 16` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 17` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 18` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 19` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |
| `Seed 20` | 1.000000 | 1.0000 | 1.000000 | 1.0000 | **+0.000000** | +0.0000 |

*JSON Key: `results.part5_valid_experiments.bilstm_20_ablation`*

### Table 5.9: Canonical ANN Calibration & Reliability Diagram Bins
**Brier Score:** `0.1330` | **ECE:** `0.1326`

| Bin Range | Sample Count | Mean Confidence | Empirical Accuracy |
|---|---|---|---|
| `[0.0, 0.1)` | 139 | 0.0007 | 0.0863 |
| `[0.1, 0.2)` | 0 | 0.1500 | 0.0000 |
| `[0.2, 0.3)` | 0 | 0.2500 | 0.0000 |
| `[0.3, 0.4)` | 0 | 0.3500 | 0.0000 |
| `[0.4, 0.5)` | 0 | 0.4500 | 0.0000 |
| `[0.5, 0.6)` | 0 | 0.5500 | 0.0000 |
| `[0.6, 0.7)` | 0 | 0.6500 | 0.0000 |
| `[0.7, 0.8)` | 0 | 0.7500 | 0.0000 |
| `[0.8, 0.9)` | 0 | 0.8500 | 0.0000 |
| `[0.9, 1.0)` | 161 | 0.9992 | 0.8261 |

*JSON Key: `results.part5_valid_experiments.ann_calibration.reliability_bins`*
