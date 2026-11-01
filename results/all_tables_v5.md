# ForenSightX — Iteration 5 Programmatic Companion Tables

*Rule G1 Compliance: Every table value is emitted programmatically from results/experiment_results_v5.json with exact key citations.*

## Table 0.1: Split Manifest Provenance (HEAD vs Working Tree)

| Property | Previous Commit (HEAD) | Working Tree (Iter 4/5) | Invariant Status |
|---|---|---|---|
| **Train Set Cases** | 140 | 280 | Scaled 2x |
| **Val Set Cases** | 30 | 60 | Scaled 2x |
| **Test Set Cases** | 30 | 60 | Scaled 2x |
| **Train IDs SHA-256** | `4d8719594546d0dd...` | `e45c887ab713a83a...` | Changed |
| **Test IDs SHA-256** | `7f8710b95bb39662...` | `58a150b066a5de89...` | Changed |
| **Test Set ID Overlap** | — | 0 cases | ZERO OVERLAP (Incomparable to Iter 1-3) |
| **Write-Once Protection** | Not Enforced | Enforced | ENFORCED (FileExistsError raised if manifest_path already exists) |

*JSON Path: `results/experiment_results_v5.json -> part0_provenance.manifest_comparison`*

## Table 0.2: Rebuilt Iteration 3 Failure Log (From Saved Raw Task Logs)

| Launch / Step | Log File Citation | Verbatim Error Message | Real Cause & Status |
|---|---|---|---|
| **Launch 1 (step 507)** | `c09614bb-97d9-40c2-a5f2-97ab549b5327/.system_generated/steps/507/output.txt` | `ImportError: cannot import name 'build_multimodal_cases' from 'src.data.case_builder'` | Attempted to import build_multimodal_cases from case_builder where only CaseBuilder class was defined. |
| **Launch 2 (step 529)** | `c09614bb-97d9-40c2-a5f2-97ab549b5327/.system_generated/steps/529/output.txt` | `ImportError: cannot import name 'CNNFeatureDataset' from 'train'` | Import of CNNFeatureDataset from train module failed because dataset was named ImageEvidenceDataset in data_loader.<br>*RETRACTED: 'results/ did not exist' was false; directory existed.* |
| **Launch 3 (step 557)** | `c09614bb-97d9-40c2-a5f2-97ab549b5327/.system_generated/steps/557/output.txt` | `TypeError: ForensicRawRecord.__init__() missing 2 required positional arguments: 'metadata' and 'content_hash'` | ForensicRawRecord dataclass instantiation in investigation script omitted metadata and content_hash fields.<br>*RETRACTED: 'KeyError: confusion_matrix' was false.* |
| **Launch 4 (step 563)** | `c09614bb-97d9-40c2-a5f2-97ab549b5327/.system_generated/steps/563/output.txt` | `TypeError: tuple indices must be integers or slices, not str` | create_stratified_split returned a 4-element tuple (train, val, test, stats), but caller indexed splits['train'] as a dictionary.<br>*RETRACTED: 'TypeError: unhashable type: numpy.ndarray' was false.* |
| **Launch 5 (task-569 / step 588)** | `c09614bb-97d9-40c2-a5f2-97ab549b5327/.system_generated/tasks/task-569.log` | `RuntimeError: Tensors must have same number of dimensions: got 2 and 3` | SyntheticDataVAE.encode attempted torch.cat([x, y_onehot]) where x was 2D (batch, 7) and y_onehot was unsqueezed into 3D (batch, 1, 2).<br>*RETRACTED: 'PowerShell inline SyntaxError' was false.* |

*JSON Path: `results/experiment_results_v5.json -> part0_provenance.rebuilt_failure_log`*

## Table 1.1: Canonical Model Factory Parameter Counts

| Architecture Name | Total Parameters | Trainable Parameters | Non-Trainable Parameters | Specification Reference |
|---|---|---|---|---|
| **VisualEvidenceCNN** | 356,097 | 356,097 | 0 | Blueprint 5.1 |
| **ConversationBiLSTM** | 190,593 | 190,593 | 0 | Blueprint 5.2 |
| **ANNMultimodalRiskClassifier** | 2,817 | 2,817 | 0 | Blueprint 5.3 |
| **SyntheticDataVAE** | 1,871 | 1,871 | 0 | Blueprint 16 |

*JSON Path: `results/experiment_results_v5.json -> part1_model_identity.factory_parameter_counts`*

## Table 1.2: CNN Checkpoint State-Dict Keys & Tensor Shapes

| State-Dict Key | Weight Tensor Shape | Parameter Count | Layer Function |
|---|---|---|---|
| `conv1.weight` | `[32, 3, 3, 3]` | 864 | Conv Feature Extractor |
| `conv1.bias` | `[32]` | 32 | Conv Feature Extractor |
| `bn1.weight` | `[32]` | 32 | Batch Normalization |
| `bn1.bias` | `[32]` | 32 | Batch Normalization |
| `bn1.running_mean` | `[32]` | 32 | Batch Normalization |
| `bn1.running_var` | `[32]` | 32 | Batch Normalization |
| `bn1.num_batches_tracked` | `[]` | 1 | Batch Normalization |
| `conv2.weight` | `[64, 32, 3, 3]` | 18,432 | Conv Feature Extractor |
| `conv2.bias` | `[64]` | 64 | Conv Feature Extractor |
| `bn2.weight` | `[64]` | 64 | Batch Normalization |
| `bn2.bias` | `[64]` | 64 | Batch Normalization |
| `bn2.running_mean` | `[64]` | 64 | Batch Normalization |
| `bn2.running_var` | `[64]` | 64 | Batch Normalization |
| `bn2.num_batches_tracked` | `[]` | 1 | Batch Normalization |
| `conv3.weight` | `[128, 64, 3, 3]` | 73,728 | Conv Feature Extractor |
| `conv3.bias` | `[128]` | 128 | Conv Feature Extractor |
| `bn3.weight` | `[128]` | 128 | Batch Normalization |
| `bn3.bias` | `[128]` | 128 | Batch Normalization |
| `bn3.running_mean` | `[128]` | 128 | Batch Normalization |
| `bn3.running_var` | `[128]` | 128 | Batch Normalization |
| `bn3.num_batches_tracked` | `[]` | 1 | Batch Normalization |
| `classifier.1.weight` | `[128, 2048]` | 262,144 | Classification Head |
| `classifier.1.bias` | `[128]` | 128 | Classification Head |
| `classifier.4.weight` | `[1, 128]` | 128 | Classification Head |
| `classifier.4.bias` | `[1]` | 1 | Classification Head |
| **TOTAL** | — | **356,548** | Input: [3, 128, 128] |

*JSON Path: `results/experiment_results_v5.json -> part1_model_identity.cnn_keys_shapes`*

## Table 1.3: Real-Batch Layer Shape Traces (Batch Size = 16)

*Derived Max Length: 19 (95th percentile). Actual Batch Seq Len: 19. Assertion Passed: True*

### CNN Real-Batch Layer Shape Trace
| Layer Stage | Output Tensor Dimensions | Notes |
|---|---|---|
| `input` | `[16, 3, 128, 128]` | Verified via Factory Hook |
| `conv1` | `[16, 32, 128, 128]` | Verified via Factory Hook |
| `pool1` | `[16, 32, 64, 64]` | Verified via Factory Hook |
| `conv2` | `[16, 64, 64, 64]` | Verified via Factory Hook |
| `pool2` | `[16, 64, 32, 32]` | Verified via Factory Hook |
| `conv3` | `[16, 128, 32, 32]` | Verified via Factory Hook |
| `adaptive_pool` | `[16, 128, 4, 4]` | Verified via Factory Hook |
| `flatten` | `[16, 2048]` | Verified via Factory Hook |
| `output` | `[16, 1]` | Verified via Factory Hook |

*JSON Path: `results/experiment_results_v5.json -> part1_model_identity.cnn_trace`*

### BiLSTM Real-Batch Layer Shape Trace
| Layer Stage | Output Tensor Dimensions | Notes |
|---|---|---|
| `input` | `[16, 19]` | Pooling: concat |
| `embedding` | `[16, 19, 64]` | Pooling: concat |
| `lstm_out` | `[16, 19, 128]` | Pooling: concat |
| `lstm_h_n` | `[4, 16, 64]` | Pooling: concat |
| `context_concat` | `[16, 128]` | Pooling: concat |
| `output` | `[16, 1]` | Pooling: concat |

*JSON Path: `results/experiment_results_v5.json -> part1_model_identity.bilstm_trace`*

## Table 2.1: Global Permutation Control (100 Seeds — Labels Permuted BEFORE Splitting)

| Seed Index | Test ROC-AUC | Test Accuracy | Selected Epoch | Status |
|---|---|---|---|---|
| Seed 001 | 0.4915 | 0.4667 | Epoch 1 | `Within Null Band` |
| Seed 002 | 0.5950 | 0.5500 | Epoch 1 | `Within Null Band` |
| Seed 003 | 0.4839 | 0.5000 | Epoch 2 | `Within Null Band` |
| Seed 004 | 0.6594 | 0.4667 | Epoch 1 | `Within Null Band` |
| Seed 005 | 0.4494 | 0.5000 | Epoch 7 | `Within Null Band` |
| Seed 006 | 0.4588 | 0.4667 | Epoch 8 | `Within Null Band` |
| Seed 007 | 0.5633 | 0.5667 | Epoch 6 | `Within Null Band` |
| Seed 008 | 0.5072 | 0.5000 | Epoch 1 | `Within Null Band` |
| Seed 009 | 0.6257 | 0.5833 | Epoch 7 | `Within Null Band` |
| Seed 010 | 0.5251 | 0.5667 | Epoch 1 | `Within Null Band` |
| Seed 011 | 0.4947 | 0.3833 | Epoch 1 | `Within Null Band` |
| Seed 012 | 0.5619 | 0.4833 | Epoch 4 | `Within Null Band` |
| Seed 013 | 0.5724 | 0.4500 | Epoch 4 | `Within Null Band` |
| Seed 014 | 0.4485 | 0.4667 | Epoch 7 | `Within Null Band` |
| Seed 015 | 0.5617 | 0.5667 | Epoch 10 | `Within Null Band` |
| Seed 016 | 0.5424 | 0.4833 | Epoch 5 | `Within Null Band` |
| Seed 017 | 0.5289 | 0.4833 | Epoch 10 | `Within Null Band` |
| Seed 018 | 0.5712 | 0.5000 | Epoch 2 | `Within Null Band` |
| Seed 019 | 0.6532 | 0.5667 | Epoch 1 | `Within Null Band` |
| Seed 020 | 0.4576 | 0.4667 | Epoch 4 | `Within Null Band` |
| Seed 021 | 0.4771 | 0.4667 | Epoch 4 | `Within Null Band` |
| Seed 022 | 0.3176 | 0.3500 | Epoch 1 | `Within Null Band` |
| Seed 023 | 0.5527 | 0.5500 | Epoch 2 | `Within Null Band` |
| Seed 024 | 0.4832 | 0.5333 | Epoch 1 | `Within Null Band` |
| Seed 025 | 0.4436 | 0.4167 | Epoch 3 | `Within Null Band` |
| Seed 026 | 0.5406 | 0.5833 | Epoch 6 | `Within Null Band` |
| Seed 027 | 0.5028 | 0.5333 | Epoch 1 | `Within Null Band` |
| Seed 028 | 0.5206 | 0.5000 | Epoch 1 | `Within Null Band` |
| Seed 029 | 0.5536 | 0.5167 | Epoch 8 | `Within Null Band` |
| Seed 030 | 0.5748 | 0.6000 | Epoch 1 | `Within Null Band` |
| Seed 031 | 0.5857 | 0.5000 | Epoch 1 | `Within Null Band` |
| Seed 032 | 0.5993 | 0.6167 | Epoch 6 | `Within Null Band` |
| Seed 033 | 0.5854 | 0.5167 | Epoch 6 | `Within Null Band` |
| Seed 034 | 0.4609 | 0.4833 | Epoch 9 | `Within Null Band` |
| Seed 035 | 0.4638 | 0.4833 | Epoch 9 | `Within Null Band` |
| Seed 036 | 0.6753 | 0.5833 | Epoch 3 | `Within Null Band` |
| Seed 037 | 0.5236 | 0.4500 | Epoch 1 | `Within Null Band` |
| Seed 038 | 0.4478 | 0.5167 | Epoch 1 | `Within Null Band` |
| Seed 039 | 0.5123 | 0.4833 | Epoch 6 | `Within Null Band` |
| Seed 040 | 0.4508 | 0.4333 | Epoch 1 | `Within Null Band` |
| Seed 041 | 0.5943 | 0.5833 | Epoch 1 | `Within Null Band` |
| Seed 042 | 0.4315 | 0.4833 | Epoch 1 | `Within Null Band` |
| Seed 043 | 0.6218 | 0.5667 | Epoch 5 | `Within Null Band` |
| Seed 044 | 0.5000 | 0.5000 | Epoch 1 | `Within Null Band` |
| Seed 045 | 0.4853 | 0.4500 | Epoch 10 | `Within Null Band` |
| Seed 046 | 0.5525 | 0.5667 | Epoch 2 | `Within Null Band` |
| Seed 047 | 0.5398 | 0.5167 | Epoch 1 | `Within Null Band` |
| Seed 048 | 0.4484 | 0.4833 | Epoch 4 | `Within Null Band` |
| Seed 049 | 0.5221 | 0.5167 | Epoch 1 | `Within Null Band` |
| Seed 050 | 0.5143 | 0.5500 | Epoch 1 | `Within Null Band` |
| Seed 051 | 0.5829 | 0.5667 | Epoch 6 | `Within Null Band` |
| Seed 052 | 0.5514 | 0.5000 | Epoch 2 | `Within Null Band` |
| Seed 053 | 0.4439 | 0.5000 | Epoch 10 | `Within Null Band` |
| Seed 054 | 0.6033 | 0.6167 | Epoch 8 | `Within Null Band` |
| Seed 055 | 0.4886 | 0.5167 | Epoch 1 | `Within Null Band` |
| Seed 056 | 0.6107 | 0.5167 | Epoch 1 | `Within Null Band` |
| Seed 057 | 0.5857 | 0.5500 | Epoch 3 | `Within Null Band` |
| Seed 058 | 0.4916 | 0.5167 | Epoch 2 | `Within Null Band` |
| Seed 059 | 0.4740 | 0.4167 | Epoch 2 | `Within Null Band` |
| Seed 060 | 0.5079 | 0.4667 | Epoch 1 | `Within Null Band` |
| Seed 061 | 0.6367 | 0.5667 | Epoch 1 | `Within Null Band` |
| Seed 062 | 0.4806 | 0.5500 | Epoch 1 | `Within Null Band` |
| Seed 063 | 0.5275 | 0.5000 | Epoch 2 | `Within Null Band` |
| Seed 064 | 0.5351 | 0.4500 | Epoch 8 | `Within Null Band` |
| Seed 065 | 0.4860 | 0.4167 | Epoch 1 | `Within Null Band` |
| Seed 066 | 0.5394 | 0.4833 | Epoch 10 | `Within Null Band` |
| Seed 067 | 0.5034 | 0.4167 | Epoch 5 | `Within Null Band` |
| Seed 068 | 0.5385 | 0.5500 | Epoch 6 | `Within Null Band` |
| Seed 069 | 0.5551 | 0.4833 | Epoch 2 | `Within Null Band` |
| Seed 070 | 0.3432 | 0.4167 | Epoch 9 | `Within Null Band` |
| Seed 071 | 0.5478 | 0.5333 | Epoch 1 | `Within Null Band` |
| Seed 072 | 0.6089 | 0.5167 | Epoch 1 | `Within Null Band` |
| Seed 073 | 0.4208 | 0.5333 | Epoch 1 | `Within Null Band` |
| Seed 074 | 0.5302 | 0.4167 | Epoch 2 | `Within Null Band` |
| Seed 075 | 0.4276 | 0.4333 | Epoch 3 | `Within Null Band` |
| Seed 076 | 0.4783 | 0.5333 | Epoch 1 | `Within Null Band` |
| Seed 077 | 0.6021 | 0.5667 | Epoch 5 | `Within Null Band` |
| Seed 078 | 0.5571 | 0.6167 | Epoch 1 | `Within Null Band` |
| Seed 079 | 0.4604 | 0.4667 | Epoch 5 | `Within Null Band` |
| Seed 080 | 0.3634 | 0.4167 | Epoch 3 | `Within Null Band` |
| Seed 081 | 0.4422 | 0.4667 | Epoch 4 | `Within Null Band` |
| Seed 082 | 0.4877 | 0.5000 | Epoch 9 | `Within Null Band` |
| Seed 083 | 0.4791 | 0.4333 | Epoch 1 | `Within Null Band` |
| Seed 084 | 0.4916 | 0.4167 | Epoch 1 | `Within Null Band` |
| Seed 085 | 0.4833 | 0.4667 | Epoch 1 | `Within Null Band` |
| Seed 086 | 0.3230 | 0.3833 | Epoch 5 | `Within Null Band` |
| Seed 087 | 0.5961 | 0.5000 | Epoch 1 | `Within Null Band` |
| Seed 088 | 0.5413 | 0.4833 | Epoch 5 | `Within Null Band` |
| Seed 089 | 0.5798 | 0.5833 | Epoch 6 | `Within Null Band` |
| Seed 090 | 0.4410 | 0.4667 | Epoch 4 | `Within Null Band` |
| Seed 091 | 0.4451 | 0.4167 | Epoch 1 | `Within Null Band` |
| Seed 092 | 0.4703 | 0.5500 | Epoch 1 | `Within Null Band` |
| Seed 093 | 0.5606 | 0.5333 | Epoch 2 | `Within Null Band` |
| Seed 094 | 0.4023 | 0.4333 | Epoch 3 | `Within Null Band` |
| Seed 095 | 0.4739 | 0.5000 | Epoch 4 | `Within Null Band` |
| Seed 096 | 0.5283 | 0.5000 | Epoch 1 | `Within Null Band` |
| Seed 097 | 0.5461 | 0.5000 | Epoch 1 | `Within Null Band` |
| Seed 098 | 0.3837 | 0.6000 | Epoch 1 | `Within Null Band` |
| Seed 099 | 0.3977 | 0.4000 | Epoch 3 | `Within Null Band` |
| Seed 100 | 0.4522 | 0.4167 | Epoch 2 | `Within Null Band` |

*JSON Path: `results/experiment_results_v5.json -> part2_valid_controls.control_100_seeds_table`*

## Table 2.2: Global Permutation Control Summary Statistics

| Metric | Empirical Value | Theoretical Reference | Compliance Verdict |
|---|---|---|---|
| **Mean ROC-AUC** | 0.5124 | 0.5000 (Chance) | Near Chance (Delta = 0.0124) |
| **Standard Deviation** | 0.0710 | 0.0752 (Max: 0.1503) | **PASS (<= 2x sigma_0)** |
| **2.5th Percentile** | 0.3528 | 0.1994 | Lower Tail Bound |
| **97.5th Percentile** | 0.6454 | 0.8006 | Upper Tail Bound |
| **Minimum AUC** | 0.3176 | >= 0.1994 | PASS (Within 4-sigma) |
| **Maximum AUC** | 0.6753 | <= 0.8006 | PASS (Within 4-sigma) |

*JSON Path: `results/experiment_results_v5.json -> part2_valid_controls` (empirical_mean_auc, empirical_std_auc, etc.)*

## Table 2.3: Global Permutation Control 10-Bin Histogram Distribution

| AUC Bin Range | Seed Count | Percentage (%) | Distribution Representation |
|---|---|---|---|
| [0.0, 0.1) | 0 | 0.0% | `` |
| [0.1, 0.2) | 0 | 0.0% | `` |
| [0.2, 0.3) | 0 | 0.0% | `` |
| [0.3, 0.4) | 6 | 6.0% | `███` |
| [0.4, 0.5) | 38 | 38.0% | `███████████████████` |
| [0.5, 0.6) | 46 | 46.0% | `███████████████████████` |
| [0.6, 0.7) | 10 | 10.0% | `█████` |
| [0.7, 0.8) | 0 | 0.0% | `` |
| [0.8, 0.9) | 0 | 0.0% | `` |
| [0.9, 1.0) | 0 | 0.0% | `` |

*JSON Path: `results/experiment_results_v5.json -> part2_valid_controls.histogram_10_bins`*

## Table 2.4: Label-Proxy Sensitivity Probe (Train-Labels-Only Shuffle)

*Overall Model (7 Features): Fraction AUC > 0.8: 0.19 | Fraction AUC < 0.2: 0.15*

| Feature Evaluated | Frac AUC > 0.8 | Frac AUC < 0.2 | Mean Test AUC | Std Test AUC | Collinearity Status |
|---|---|---|---|---|---|
| **cnn_score** | 0.53 | 0.37 | 0.5800 | 0.4675 | Strong Label Proxy |
| **bilstm_score** | 0.53 | 0.37 | 0.5800 | 0.4675 | Strong Label Proxy |
| **url_score** | 0.00 | 0.00 | 0.4920 | 0.1998 | Weak / Independent |
| **keyword_score** | 0.59 | 0.41 | 0.5900 | 0.4918 | Strong Label Proxy |
| **attachment_score** | 0.00 | 0.00 | 0.5050 | 0.0942 | Weak / Independent |
| **timeline_anomaly** | 0.00 | 0.00 | 0.5567 | 0.2776 | Weak / Independent |
| **email_anomaly** | 0.00 | 0.00 | 0.5000 | 0.0667 | Weak / Independent |

*JSON Path: `results/experiment_results_v5.json -> part2_valid_controls.per_feature_probe`*

## Table 3.1: BiLSTM Concat vs Attention 20-Seed Ablation

| Seed | Concat Ep1 Loss | Concat Final Loss | Concat AUC | Concat F1 | Attn Ep1 Loss | Attn Final Loss | Attn AUC | Attn F1 | Delta AUC | Delta F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Seed 0042 | 0.3012 | 0.0004 | 1.0000 | 1.0000 | 0.4281 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0101 | 0.3096 | 0.0003 | 1.0000 | 1.0000 | 0.3816 | 0.0001 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 2024 | 0.3446 | 0.0001 | 1.0000 | 1.0000 | 0.3261 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0007 | 0.3104 | 0.0000 | 1.0000 | 1.0000 | 0.3705 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0013 | 0.3152 | 0.0011 | 1.0000 | 1.0000 | 0.3051 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0019 | 0.3136 | 0.0000 | 1.0000 | 1.0000 | 0.3890 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0023 | 0.3546 | 0.0000 | 1.0000 | 1.0000 | 0.3592 | 0.0002 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0029 | 0.3020 | 0.0000 | 1.0000 | 1.0000 | 0.2985 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0031 | 0.4221 | 0.0002 | 1.0000 | 1.0000 | 0.3528 | 0.0107 | 1.0000 | 0.9831 | +0.0000 | +0.0169 |
| Seed 0037 | 0.3677 | 0.0007 | 1.0000 | 1.0000 | 0.3056 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0041 | 0.3255 | 0.0011 | 1.0000 | 1.0000 | 0.3250 | 0.0003 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0043 | 0.3313 | 0.0000 | 1.0000 | 1.0000 | 0.3228 | 0.0006 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0047 | 0.3549 | 0.0000 | 1.0000 | 1.0000 | 0.3721 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0053 | 0.2997 | 0.0000 | 1.0000 | 1.0000 | 0.3051 | 0.0335 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0059 | 0.3068 | 0.0004 | 1.0000 | 1.0000 | 0.3108 | 0.1031 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0061 | 0.3611 | 0.0315 | 1.0000 | 1.0000 | 0.3410 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0067 | 0.3064 | 0.0001 | 1.0000 | 1.0000 | 0.3719 | 0.1468 | 1.0000 | 0.9677 | +0.0000 | +0.0323 |
| Seed 0071 | 0.3224 | 0.0000 | 1.0000 | 1.0000 | 0.3750 | 0.0004 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| Seed 0073 | 0.3235 | 0.3689 | 1.0000 | 0.9231 | 0.3650 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | -0.0769 |
| Seed 0079 | 0.3355 | 0.0000 | 1.0000 | 1.0000 | 0.3760 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |

*JSON Path: `results/experiment_results_v5.json -> part3_reruns.bilstm_20_seeds_ablation`*

### Table 3.2: Paired Bootstrap 95% Confidence Intervals for BiLSTM Pooling Delta
| Metric Comparison | Delta Mean | 95% Bootstrap CI [2.5%, 97.5%] | Statistically Significant Difference? |
|---|---|---|---|
| **Delta ROC-AUC (Concat - Attn)** | +0.0000 | [+0.0000, +0.0000] | No (Includes 0.0) |
| **Delta F1-Score (Concat - Attn)** | -0.0014 | [-0.0115, +0.0057] | No (Includes 0.0) |

*JSON Path: `results/experiment_results_v5.json -> part3_reruns` (ci_delta_auc, ci_delta_f1)*

## Table 3.3: Weight Initialization Scaling Invariance Verification

| Init Method | FC1 Empirical Std | FC1 Theoretical Std | FC2 Empirical Std | FC2 Theoretical Std | Scaling Law Compliance |
|---|---|---|---|---|---|
| **xavier_normal** | 0.2414 | 0.2265 | 0.2096 | 0.2041 | `True` |
| **he_normal** | 0.5477 | 0.5345 | 0.2554 | 0.2500 | `True` |

*JSON Path: `results/experiment_results_v5.json -> part3_reruns.weight_init_std_logs`*

## Table 3.4: Full 36-Cell ANN Multimodal Grid Search (5 Seeds per Cell)

| Cell | Optimizer | LR | Mom | Init | Dropout | BN | Cap | Crit | Epochs (Mean ± Std) | Cap-Hit | Val Loss (Mean ± Std) | Val AUC (Mean ± Std) | Val F1 (Mean ± Std) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | Adam | 0.001 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.0521 ± 0.0217 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C02 | Adam | 0.001 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 14.8 ± 0.4 | 0.80 | 0.0467 ± 0.0210 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C03 | Adam | 0.001 | 0.0 | default | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.0641 ± 0.0082 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C04 | Adam | 0.01 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 12.6 ± 1.4 | 0.20 | 0.0002 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C05 | Adam | 0.01 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 12.2 ± 1.7 | 0.20 | 0.0002 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C06 | Adam | 0.01 | 0.0 | default | 0.2 | True | 15 | pat=3 | 13.2 ± 0.8 | 0.00 | 0.0003 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C07 | Adam | 0.5 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 1.2 ± 0.4 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C08 | Adam | 0.5 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 1.2 ± 0.4 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C09 | Adam | 0.5 | 0.0 | default | 0.2 | True | 15 | pat=3 | 1.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C10 | RMSprop | 0.001 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 14.8 ± 0.4 | 0.80 | 0.0195 ± 0.0114 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C11 | RMSprop | 0.001 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 14.8 ± 0.4 | 0.80 | 0.0177 ± 0.0111 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C12 | RMSprop | 0.001 | 0.0 | default | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.0234 ± 0.0037 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C13 | RMSprop | 0.01 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 7.0 ± 1.4 | 0.00 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C14 | RMSprop | 0.01 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 7.2 ± 1.7 | 0.00 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C15 | RMSprop | 0.01 | 0.0 | default | 0.2 | True | 15 | pat=3 | 7.8 ± 0.8 | 0.00 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C16 | RMSprop | 0.5 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 1.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C17 | RMSprop | 0.5 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 1.6 ± 1.2 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C18 | RMSprop | 0.5 | 0.0 | default | 0.2 | True | 15 | pat=3 | 1.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C19 | SGD_momentum_0.0 | 0.01 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 14.8 ± 0.4 | 0.80 | 0.1305 ± 0.0472 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C20 | SGD_momentum_0.0 | 0.01 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.1077 ± 0.0405 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C21 | SGD_momentum_0.0 | 0.01 | 0.0 | default | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.1427 ± 0.0118 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C22 | SGD_momentum_0.0 | 0.1 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.0037 ± 0.0009 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C23 | SGD_momentum_0.0 | 0.1 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 14.8 ± 0.4 | 0.80 | 0.0032 ± 0.0007 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C24 | SGD_momentum_0.0 | 0.1 | 0.0 | default | 0.2 | True | 15 | pat=3 | 14.8 ± 0.4 | 0.80 | 0.0038 ± 0.0004 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C25 | SGD_momentum_0.0 | 1.0 | 0.0 | he_normal | 0.2 | True | 15 | pat=3 | 7.4 ± 1.0 | 0.00 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C26 | SGD_momentum_0.0 | 1.0 | 0.0 | xavier_normal | 0.2 | True | 15 | pat=3 | 8.4 ± 0.8 | 0.00 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C27 | SGD_momentum_0.0 | 1.0 | 0.0 | default | 0.2 | True | 15 | pat=3 | 7.8 ± 1.7 | 0.00 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C28 | SGD_momentum_0.9 | 0.005 | 0.9 | he_normal | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.0136 ± 0.0041 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C29 | SGD_momentum_0.9 | 0.005 | 0.9 | xavier_normal | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.0111 ± 0.0026 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C30 | SGD_momentum_0.9 | 0.005 | 0.9 | default | 0.2 | True | 15 | pat=3 | 15.0 ± 0.0 | 1.00 | 0.0124 ± 0.0011 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C31 | SGD_momentum_0.9 | 0.02 | 0.9 | he_normal | 0.2 | True | 15 | pat=3 | 14.4 ± 0.8 | 0.60 | 0.0009 ± 0.0002 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C32 | SGD_momentum_0.9 | 0.02 | 0.9 | xavier_normal | 0.2 | True | 15 | pat=3 | 14.4 ± 0.8 | 0.60 | 0.0008 ± 0.0002 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C33 | SGD_momentum_0.9 | 0.02 | 0.9 | default | 0.2 | True | 15 | pat=3 | 14.2 ± 0.8 | 0.40 | 0.0009 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C34 | SGD_momentum_0.9 | 0.5 | 0.9 | he_normal | 0.2 | True | 15 | pat=3 | 1.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C35 | SGD_momentum_0.9 | 0.5 | 0.9 | xavier_normal | 0.2 | True | 15 | pat=3 | 1.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| C36 | SGD_momentum_0.9 | 0.5 | 0.9 | default | 0.2 | True | 15 | pat=3 | 1.0 ± 0.0 | 0.00 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

*JSON Path: `results/experiment_results_v5.json -> part3_reruns.ann_grid_table`*

### Winning ANN Configuration Final Test Metrics (Evaluated ONCE)
**Selected Winning Configuration:** Cell C16 (RMSprop, LR=0.5, Init=he_normal, Lowest Val Loss=0.0000)

| Test Metric | Measured Value |
|---|---|
| **Test ROC-AUC** | 1.0000 |
| **Test Accuracy** | 1.0000 |
| **Test F1-Score** | 1.0000 |
| **Test Precision** | 1.0000 |
| **Test Recall** | 1.0000 |

*JSON Path: `results/experiment_results_v5.json -> part3_reruns.final_test_metrics_winning_config`*

## Table 3.5: CNN OFAT Parameter Sensitivity (N=5 Seeds)

| Variation Description | Mean Grad Norm (Mean ± Std) | Val Loss (Mean ± Std) | Val ROC-AUC (Mean ± Std) | Val F1 (Mean ± Std) | Val Accuracy (Mean ± Std) |
|---|---|---|---|---|---|
| **Baseline (Adam, LR=0.001, Dropout=0.3, BN=True)** | 0.1527 ± 0.0195 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |
| **OFAT 1: Dropout OFF (p=0.0)** | 0.1503 ± 0.0380 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |
| **OFAT 2: BatchNorm OFF** | 0.9376 ± 0.4270 | 0.0014 ± 0.0028 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |
| **OFAT 3: LR Half (0.0005)** | 0.2468 ± 0.0272 | 0.0001 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |
| **OFAT 4: LR Double (0.002)** | 0.1236 ± 0.0208 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |
| **OFAT 5: Optimizer RMSprop** | 0.6618 ± 0.2669 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |
| **OFAT 6: Optimizer SGD (mom=0.9)** | 0.5540 ± 0.0278 | 0.0009 ± 0.0004 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |

*JSON Path: `results/experiment_results_v5.json -> part3_reruns.cnn_ofat_table`*

## Table 4.1: VAE Latent Space KL Divergence & Posterior Collapse

*Total KL Divergence: 0.000216 across 4 latent dimensions. Posterior Collapse Flagged: True*

| Latent Dimension | Measured KL Divergence | Epsilon Threshold | Posterior Collapse Flagged? | Interpretation |
|---|---|---|---|---|
| Latent Dim 0 | 0.000045 | 0.05 | `True` | Dimension Inactive / Collapsed |
| Latent Dim 1 | 0.000033 | 0.05 | `True` | Dimension Inactive / Collapsed |
| Latent Dim 2 | 0.000055 | 0.05 | `True` | Dimension Inactive / Collapsed |
| Latent Dim 3 | 0.000084 | 0.05 | `True` | Dimension Inactive / Collapsed |

*JSON Path: `results/experiment_results_v5.json -> part4_vae_health.per_latent_dim_kl`*

## Table 4.2: Decoder Output Variation over z ~ N(0, I) at Fixed Class

| Feature | Std across z (y=0 Fixed) | Std across z (y=1 Fixed) | Mean Output Variation | Decoder Ignores Latent Vector z? |
|---|---|---|---|---|
| **cnn_score** | 0.0058 | 0.0033 | 0.0045 | `True` |
| **bilstm_score** | 0.0058 | 0.0039 | 0.0048 | `True` |
| **url_score** | 0.0027 | 0.0185 | 0.0106 | `True` |
| **keyword_score** | 0.0031 | 0.0157 | 0.0094 | `True` |
| **attachment_score** | 0.0057 | 0.0128 | 0.0092 | `True` |
| **timeline_anomaly** | 0.0072 | 0.0092 | 0.0082 | `True` |
| **email_anomaly** | 0.0066 | 0.0092 | 0.0079 | `True` |

*JSON Path: `results/experiment_results_v5.json -> part4_vae_health.decoder_output_variation`*

## Table 4.3: Per-Feature Per-Class Wasserstein Distance (Real vs Synthetic)

| Feature Name | Wasserstein Dist (y=0) | Compliant (<=0.15)? | Wasserstein Dist (y=1) | Compliant (<=0.15)? | Quality Assessment |
|---|---|---|---|---|---|
| **cnn_score** | 0.0172 | `True` | 0.0226 | `True` | Compliant |
| **bilstm_score** | 0.0168 | `True` | 0.0193 | `True` | Compliant |
| **url_score** | 0.0145 | `True` | 0.4104 | `False` | Distribution Shift Detected |
| **keyword_score** | 0.0134 | `True` | 0.2099 | `False` | Distribution Shift Detected |
| **attachment_score** | 0.1110 | `True` | 0.3763 | `False` | Distribution Shift Detected |
| **timeline_anomaly** | 0.0051 | `True` | 0.2784 | `False` | Distribution Shift Detected |
| **email_anomaly** | 0.0051 | `True` | 0.1269 | `True` | Compliant |

*JSON Path: `results/experiment_results_v5.json -> part4_vae_health.wasserstein_distance_table`*

## Table 5.1: Part 5 Data Gate Check

| Inspection Target | File Status | Operational Decision | Directive Action |
|---|---|---|---|
| `config/data_mode.yaml` | Exists = `False` | `STOP_AFTER_PART_4` | Specification directive: 'read config/data_mode.yaml; if missing, stop after Part 4 and say so.' |

*JSON Path: `results/experiment_results_v5.json -> part5_data_gate`*
