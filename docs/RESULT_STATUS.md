# ForenSightX — Result Status Ledger (Iterations 1 through 5)

This document establishes the official validity ledger for every table across `EXECUTION_REPORT_v1.md` through `EXECUTION_REPORT_v5.md`, adhering to Global Rules G1, G2, G5, and G6.

### Status Definitions:
- **VALID**: Backed by a verified results JSON key, recomputable from raw rows, evaluated under clean non-leaking conditions with matching model configurations.
- **INVALID**: Flawed due to identified architectural discrepancies (e.g. BiLSTM pooling mismatch, VAE positional argument bug, post-OOF label shuffling, or data leakage).
- **UNVERIFIED**: Historical table lacking programmatic results JSON backing, raw row records, or reproducible seeds.

---

## 1. Iteration 1 Baseline Report (`EXECUTION_REPORT.md`)

| Table Name | Description | Status | Reason & Evidence |
|---|---|---|---|
| **Table 1: Baseline Architecture Parameters** | Parameter totals for CNN, BiLSTM, ANN, VAE | **INVALID** | VAE instantiated with 135 parameters due to positional argument bug (`SyntheticDataVAE(7, 2, 4)`); BiLSTM parameters included unmanaged attention head. |
| **Table 2: Baseline Experiments A through E** | Test AUC, F1, Precision, Recall across 5 experiments | **UNVERIFIED** | No JSON results file persisted; single-seed run; uncalibrated heuristic scores on synthetic data. |
| **Table 3: Multi-Agent Case Lifecycle** | Case coordinator execution trace | **UNVERIFIED** | No JSON persistence; illustrative synthetic execution. |

---

## 2. Iteration 2 Audit & Leakage-Proofing Report (`EXECUTION_REPORT_v2.md`)

| Table Name | Description | Status | Reason & Evidence |
|---|---|---|---|
| **Table 1.1: Dataset Split Distribution** | 280 Train, 60 Val, 60 Test stratified split | **VALID** | Split manifest `data/split_manifest.json` committed in `a985ee7` (`manifest_sha256: b4f83fce...`). Zero sample overlap verified. |
| **Table 2.1: Out-Of-Fold Base Model Metrics** | 5-Fold OOF CV scores for CNN and BiLSTM | **UNVERIFIED** | OOF scores were computed in memory and not saved to a write-once results JSON. |
| **Table 2.2: Baseline ANN Multimodal Risk Metrics** | ANN evaluation on test split (5 seeds) | **UNVERIFIED** | No JSON results key persisted; BiLSTM checkpoint was trained with `pooling="attention"` while ANN inference assumed default concat. |
| **Table 3.1: Circularity Controls** | Shuffled labels, 5-heuristic LR baseline, LOFO | **INVALID** | Shuffled-label control was applied after OOF extraction; label shuffling touched train labels post-extraction. |
| **Table 4.1: Checkpoint Evaluation Summary** | CNN, BiLSTM, and ANN test performance | **INVALID** | BiLSTM checkpoint contained attention weights incompatible with blueprint concat pooling. |
| **Table 4.2: Investigation Agent Pipeline Trace** | Case coordinator results on test cases | **UNVERIFIED** | Output printed to stdout without write-once JSON backing. |

---

## 3. Iteration 3 Anomaly Investigations Report (`EXECUTION_REPORT_v3.md`)

| Table Name | Description | Status | Reason & Evidence |
|---|---|---|---|
| **Table 1a: Shuffled-Label Control (20 Seeds)** | Mean AUC 0.5793 +/- 0.3002 | **INVALID** | Failed theoretical null threshold ($\sigma_0 \approx 0.075$, empirical $0.3002 > 2\sigma_0$); shuffle applied after OOF base features were extracted using true labels. |
| **Table 1b: Neural Parameter Breakdown** | Parameter counts per module | **INVALID** | VAE parameter count reported as 135 due to positional argument bug; BiLSTM held dormant attention parameters under `pooling="concat"`. |
| **Table 1c: Layer-by-Layer Shape Traces** | Tensor shapes across modules | **INVALID** | Hook shape trace passed dummy input `(1, 3, 32, 32)` rather than canonical `128x128` image resolution. |
| **Table 1d: Leave-One-Feature-Out (LOFO) 7D** | Drop in AUC per omitted feature | **UNVERIFIED** | Stale feature array; missing programmatic results JSON backing. |
| **Table 1e: Optimization & Initialization Grid (36 Cells)** | Multi-seed optimizer/LR/initialization sweep | **INVALID** | Evaluated on validation set with early-stopping leakage; BiLSTM pooling discrepancy present. |
| **Table 1f: BiLSTM Concat vs Attention Pooling Ablation** | 20 seeds comparison | **INVALID** | Evaluated on 30-case test set from 200-case manifest; attention weights migrated by pruning. |
| **Table 1g: Empirical Diagnostics** | Grad norms, overfit-one-batch, loss stagnation | **UNVERIFIED** | Missing write-once JSON results backing. |

---

## 4. Iteration 4 Audit & Verification Report (`EXECUTION_REPORT_v4.md`)

| Table Name | Description | Status | JSON Backing Key | Reason & Evidence |
|---|---|---|---|---|
| **Table 1.1: strict=True Checkpoint Loading** | Checkpoint key count and tensor equality | **VALID** | `part1_checkpoint_audit.checkpoint_audit` | Verifies `strict=True` loading and parameter tensor equality. BiLSTM keys migrated via allow-list. |
| **Table 2.1: Theoretical Null Thresholds** | Null distribution formulas ($N=60, \sigma_0=0.075154$) | **VALID** | `part2_shuffled_label_control.theoretical_null_sd` | Exact mathematical derivations for balanced Mann-Whitney U test statistic. |
| **Table 2.2: Five Diagnostic Conditions (A-E)** | Evaluation of Conditions A, B, C, D, E | **INVALID** | `part2_shuffled_label_control.five_diagnostic_conditions` | Condition B shuffled train/val labels before OOF, but underlying BiLSTM checkpoint still possessed attention pooling mismatch. |
| **Table 2.3: Shuffled-Label 100 Seeds Table** | Per-seed AUC, accuracy, selected epoch | **INVALID** | Contradicted by JSON | Markdown table printed stale values from an earlier run (mean=0.7175) contradicting `experiment_results_v4.json` (mean=0.5938). |
| **Table 2.4: Shuffled-Label Control Summary** | Empirical mean, std, percentiles | **INVALID** | `part2_shuffled_label_control.final_control_summary` | Empirical std ($0.2836$) violated $2\sigma_0$ threshold ($0.1503$); 4-sigma check failed. |
| **Table 3.1: VAE Parameter Closed-Form Formula** | Closed-form formula vs instantiated (1,215) | **VALID** | `part3_vae_evaluation.parameter_formula_verification` | Proves closed-form formula matches 1,215 parameters under keyword-only constructor. |
| **Table 3.2: VAE "Aligned" Feature Table** | Feature distribution comparison | **INVALID** | `part3_vae_evaluation.feature_distributions` | Labeled "Aligned" without quantitative justification; total KL was $0.000216$, indicating total posterior collapse across all 4 latent dimensions. |
| **Table 4.1: Tokenizer Vocabulary Isolation** | Vocab size (251) fit on train split only | **VALID** | `part4_vocab_and_bilstm.vocab_audit` | Asserted zero vocabulary leakage from val or test splits. |
| **Table 4.2: BiLSTM Pooling Ablation (20 Seeds)** | Concat vs Attention AUC, F1, Loss | **INVALID** | `part4_vocab_and_bilstm.bilstm_ablation_20_seeds` | Evaluated on BiLSTM checkpoint requiring key migration rather than canonical retraining. |
| **Table 5.1: ANN 36-Cell Hyperparameter Grid** | Optimizer, LR, init, Dropout, BN | **INVALID** | `part5_grid_audit_diagnostics.ann_grid_table` | Stale OOF scores derived from attention-migrated base model. |
| **Table 5.2: CNN One-Factor-At-A-Time (OFAT)** | Variations on CNN architecture | **INVALID** | `part5_grid_audit_diagnostics.cnn_ofat_table` | Evaluated on train images rather than isolated validation set. |
| **Table 5.3: Neural Parameter Audit** | CNN (356,097), BiLSTM (190,593), ANN (801), VAE (1,215) | **VALID** | `part5_grid_audit_diagnostics.parameter_audit` | Canonical parameter counts matching exact layer shapes. |
| **Table 6.1: Real Data Config Validation** | Schema validation results for datasets.yaml | **VALID** | `part6_config_validation` | Validator halted execution due to missing configuration files. |

---

## 5. Iteration 5 Provenance & Controls Report (`EXECUTION_REPORT_v5.md`)

| Table Name | Description | Status | JSON Backing Key | Reason & Evidence |
|---|---|---|---|---|
| **Table 0.1: Split Manifest Provenance** | 200 vs 400 cases comparison | **VALID** | `part0_provenance.manifest_comparison` | Exact SHA-256 hashes for train/test IDs and write-once enforcement. |
| **Table 0.2: Rebuilt Iteration 3 Failure Log** | Rebuilt log from raw task outputs | **VALID** | `part0_provenance.rebuilt_failure_log` | Cites exact step log paths and explains root causes. |
| **Table 1.1: Canonical Factory Parameter Counts** | Total, trainable, non-trainable params | **VALID** | `part1_model_identity.factory_parameter_counts` | Built through canonical factory `src/models/factory.py`. |
| **Table 1.2: CNN Checkpoint Keys & Shapes** | Layer-by-layer weight shapes (356,097 params) | **VALID** | `part1_model_identity.cnn_keys_shapes` | Matches `models_saved/cnn_model.pt` with resolution $128 \times 128$. |
| **Table 1.3: Real-Batch Shape Traces** | Forward hook shapes on real batch ($N=16, L=19$) | **VALID** | `part1_model_identity.shape_traces` | Verified batch length $\le 19$ (train 95th percentile). |
| **Table 2.1: Global Permutation Control (100 Seeds)** | Per-seed AUC, accuracy, selected epoch | **VALID** | `part2_valid_controls.control_100_seeds_table` | Clean train-only permutation control; recomputed in `test_iteration5_summary_assert.py`. |
| **Table 2.2: Permutation Control Summary** | Mean (0.5124), Std (0.0768), 2.5% (0.3542), 97.5% (0.6558) | **VALID** | `part2_valid_controls.empirical_mean_auc` | Passes $2\sigma_0$ bound ($0.0768 \le 0.1503$) and 4-sigma range. |
| **Table 2.3: 10-Bin Histogram Distribution** | Bin counts and percentages over $[0, 1]$ | **VALID** | `part2_valid_controls.histogram_10_bins` | Sums to 100 seeds; verified against raw seed rows. |
| **Table 2.4: Label-Proxy Sensitivity Probe** | Sensitivity of 5 heuristics and 2 model scores | **VALID** | `part2_valid_controls.per_feature_probe` | Demonstrates heuristic sensitivity to label shuffling. |
| **Table 3.1: BiLSTM Concat vs Attention Ablation** | 20 seeds ablation on canonical retrained BiLSTM | **VALID** | `part3_reruns.bilstm_20_seeds_ablation` | Canonical retrained weights; CI bounds checked. |
| **Table 3.2: Weight Initialization Std Audit** | Layer weight std vs theoretical bounds | **VALID** | `part3_reruns.weight_init_std_logs` | Confirms He and Xavier formulas. |
| **Table 3.3: ANN 36-Cell Grid Search** | Hyperparameter sweep over clean OOF features | **VALID** | `part3_reruns.ann_grid_table` | Evaluated strictly on validation split. |
| **Table 3.4: Winning ANN Test Evaluation** | Final test metrics (AUC 0.9978, F1 0.9655) | **VALID** | `part3_reruns.final_test_metrics_winning_config` | Evaluated exactly once on test split. |
| **Table 3.5: CNN One-Factor-At-A-Time (OFAT)** | Isolated CNN architectural variants | **VALID** | `part3_reruns.cnn_ofat_table` | Evaluated on validation split. |
| **Table 4.1: VAE Per-Latent-Dim KL Divergence** | KL per dimension ($\approx 10^{-5}$) | **VALID** | `part4_vae_health.per_latent_dim_kl` | Quantitatively diagnoses posterior collapse ($D_{KL} < 0.05$ across all 4 dims). |
| **Table 4.2: VAE Decoder Output Variation** | Std of decoder output over $z$ at fixed $y$ | **VALID** | `part4_vae_health.decoder_output_variation` | Quantitatively confirms decoder ignores latent $z$ ($std < 0.01$). |
| **Table 4.3: VAE Synthetic vs Real Wasserstein** | Per-feature Wasserstein distance | **VALID** | `part4_vae_health.wasserstein_distance_table` | Replaces "Aligned" label with quantitative transport metrics. |
| **Table 5.1: Real Data Execution Gate** | Configuration file check | **VALID** | `part5_data_gate` | Execution stopped at Part 4 due to missing real dataset configs. |



---

## 6. Iteration 6 Benchmark Disclosures & Valid Experiments (`EXECUTION_REPORT_v6.md`)

| Table Name | Description | Status | JSON Backing Key | Reason & Evidence |
|---|---|---|---|---|
| **Table 1.1: Feature Distributions per Class on Current Synthetic Data** | 7-feature distribution on old data | **VALID** | `part1_validity_battery_current_data.feature_distribution_table` | Correctly demonstrates trivial separability of old synthetic data. |
| **Table 1.2: Leave-One-Feature-Out (LOFO) on Current Data** | 7 rows LOFO | **VALID** | `part1_validity_battery_current_data.lofo_table` | Demonstrates all 1.0000 AUCs on trivially separable data. |
| **Table 1.3: Single-Feature Performance on Current Data** | Univariate AUCs | **VALID** | `part1_validity_battery_current_data.single_feature_table` | Confirms multiple features alone achieve AUC $\ge 0.95$. |
| **Table 1.4: Logistic Regression on 5 Heuristics Only** | 5-heuristic baseline on old data | **VALID** | `part1_validity_battery_current_data.logistic_regression_5_heuristics` | Demonstrates 1.0000 AUC on heuristics alone. |
| **Table 1.5: ANN Calibration Reliability Bins (Current Data)** | Brier 0.0077, ECE 0.0681 | **VALID** | `part1_validity_battery_current_data.1d_ann_training_and_calibration.reliability_bins` | Accurate calibration measurement on legacy data. |
| **Table 4.1: VAE Latent Dimensional KL Divergence** | Per-dim KL on hard benchmark | **INVALID** | `part4_vae_repair_health.per_latent_dim_kl` | Evaluated on non-disjoint train/val template families; pending Part 3 nested null validation. |
| **Table 4.2: Decoder Output Variation over z** | Decoder std vs real std | **INVALID** | `part4_vae_repair_health.decoder_output_variation` | Non-disjoint template grouping; pending Part 3 nested null validation. |
| **Table 4.3: Per-Feature Wasserstein Distance** | Real vs synthetic transport distances | **INVALID** | `part4_vae_repair_health.wasserstein_distance_table` | Evaluated on hard_v1 manifest which shares train/val families; pending Part 3 nested null validation. |
| **Table 5.1: Canonical Experiments A-D Summary** | CNN, BiLSTM, ANN, Coordinator metrics | **INVALID** | `part5_valid_experiments.experiments_summary` | Evaluated on hard_v1 split with train/val family sharing; pending Part 3 nested null validation. |
| **Table 5.2: Paired Bootstrap 95% CIs** | Exp C vs Unimodal, Exp D vs C | **INVALID** | `part5_valid_experiments.paired_bootstrap_cis` | Standard bootstrap rather than clustered by family; pending Part 3 nested null validation. |
| **Table 5.3: Exp E Augmentation Arms Sweep** | None, VAE, Gaussian, Interpolation | **INVALID** | `part5_valid_experiments.experiment_e_arms_sweep` | Hard_v1 data split artifact; pending Part 3 nested null validation. |
| **Table 5.4: Global Permutation Control Summary** | 100 seeds bounds check | **INVALID** | `part5_valid_experiments.global_permutation_control` | FAILED null control ($sd = 0.1975 > 2\sigma_0$); permuted train labels only against unpermuted test labels. |
| **Table 5.5: Global Permutation 10-Bin Histogram** | 100 seeds histogram | **INVALID** | `part5_valid_experiments.global_permutation_control.histogram_10_bins` | Reflects bimodal artifact of train-only label permutation. |
| **Table 5.6: ANN 36-Cell Hyperparameter Grid** | Hyperparameter sweep on hard data | **INVALID** | `part5_valid_experiments.ann_hyperparameter_grid` | Evaluated on non-disjoint validation template families. |
| **Table 5.7: CNN OFAT Parameter Sensitivity Grid** | CNN OFAT variations | **INVALID** | `part5_valid_experiments.cnn_ofat_table` | Uncalibrated BatchNorm threshold artifact (accuracy 0.5000 at AUC 0.9366). |
| **Table 5.8: BiLSTM Concat vs Attention Ablation** | 20 seeds ablation | **INVALID** | `part5_valid_experiments.bilstm_20_ablation` | Evaluated on hard_v1 split; pending Part 3 nested null validation. |
| **Table 5.9: Canonical ANN Calibration Bins** | Brier 0.1330, ECE 0.1326 | **INVALID** | `part5_valid_experiments.ann_calibration.reliability_bins` | Hard_v1 manifest evaluation; pending Part 3 nested null validation. |

---

## 7. Audit of Historical 135-Parameter VAE Formulation

The parameter count of 135 reported in Iterations 1, 2, and 3 was previously explained in `EXECUTION_REPORT_v6.md` using the hand-waving approximation "$67 \times 2 \approx 135$". 

This statement is officially retracted. The exact architectural mechanism producing 135 parameters has been empirically verified and is permanently guarded by unit test `tests/test_iteration7_audit.py::test_buggy_vae_parameter_count_135`:

When `SyntheticDataVAE` is instantiated as `SyntheticDataVAE(input_dim=7, latent_dim=4, hidden_dim=4, num_classes=2)`, the resulting layer dimensions and parameter counts are:

| Layer Name | Weight Tensor Shape | Bias Tensor Shape | Total Layer Parameters | Mathematical Formula |
|---|---|---|---|---|
| `encoder_fc.0` | `[4, 9]` | `[4]` | **40** | $4 \times (7 + 2) + 4 = 40$ |
| `encoder_fc.2` | `[2, 4]` | `[2]` | **10** | $(4 // 2) \times 4 + 2 = 10$ |
| `fc_mu` | `[4, 2]` | `[4]` | **12** | $4 \times 2 + 4 = 12$ |
| `fc_logvar` | `[4, 2]` | `[4]` | **12** | $4 \times 2 + 4 = 12$ |
| `decoder.0` | `[2, 6]` | `[2]` | **14** | $(4 // 2) \times (4 + 2) + 2 = 14$ |
| `decoder.2` | `[4, 2]` | `[4]` | **12** | $4 \times 2 + 4 = 12$ |
| `decoder.4` | `[7, 4]` | `[7]` | **35** | $7 \times 4 + 7 = 35$ |
| **Total Model Parameters** | — | — | **135** | $\sum = 40 + 10 + 12 + 12 + 14 + 12 + 35 = 135$ |

---

## 8. Iteration 7 Grouped Splits, Null Controls & Redone Experiments (`EXECUTION_REPORT_v7.md`)

**Results Artifact:** `results/experiment_results_v7.json` (SHA-256: `24949ad2045c4b0f167703defaca3a5322cf18d885482bc511aaa63bce30f344`)  
**Split Manifest:** `data/split_manifest_hard_v2.json` (SHA-256: `b22fb35755469044a262739f8254b60de4fd1c0116af7cf744a4995330698f6a`)  

| Table Name | Description | Status | JSON Backing Key | Reason & Evidence |
|---|---|---|---|---|
| **Table 0.1: Commit History for `chief executive`** | `src/data_loader.py` git log history | **VALID** | `part0_disclosure.0c.commit_history_table` | Verified across commits `06f391f` through `c71fd06`. |
| **Table 0.2: Audited Parameter Breakdown for Buggy VAE** | 14-layer exact parameter breakdown | **VALID** | `part0_disclosure.0d.layer_breakdown` | Guarded by `tests/test_iteration7_audit.py::test_buggy_vae_parameter_count_135`. |
| **Table 1.1: hard_v1 Partition Structural Inventory** | Partition cases, unique records, families | **VALID** | `part1_data_audit.1a_partition_table` | Triggered Gate G-DATA due to 9 shared template families between train and val. |
| **Table 1.2: Design Effect and Effective Sample Size** | DEFF and $n_{\text{eff}}$ by record and family | **VALID** | `part1_data_audit.1b_design_effect` | Cluster bootstrap vs case-level bootstrap ($DEFF = 1.3993$). |
| **Table 1.3: Permutation Function Feature Independence Audit** | MI and $\chi^2$ dependencies | **VALID** | `part1_data_audit.1c_permutation_audit.feature_dependency_table` | Confirms mutual information $\approx 0$ and non-significant $\chi^2$. |
| **Table 2.1: GroupKFold OOF Base Model Performance** | BiLSTM and CNN OOF vs Val vs Test | **VALID** | `part2_grouped_splits.2b_base_models` | Gate G-OOF PASSED ($|\text{OOF} - \text{Val}| \le 0.05$). |
| **Table 2.2: Oracle Performance Ceilings** | LR and GBDT ceilings on family-disjoint splits | **VALID** | `part2_grouped_splits.2c_oracle_ceilings` | Test ceilings: LR $0.9935$, GBDT $0.9932$. |
| **Table 3.N1: N1 i.i.d. Gaussian Null Control** | 100 seeds summary, histogram, raw rows | **VALID** | `part3_nested_null_controls.N1` | Gate G-PERM PASSED: Std $0.035185 \le 2\sigma_0$ ($0.066791$). |
| **Table 3.N2: N2 Frozen Grouped-OOF Null Control** | 100 seeds summary, histogram, raw rows | **VALID** | `part3_nested_null_controls.N2` | Gate G-PERM PASSED: Std $0.031918 \le 2\sigma_0$ ($0.066791$). |
| **Table 3.N3: N3 Full Pipeline Null Control** | 100 seeds summary, histogram, raw rows | **VALID** | `part3_nested_null_controls.N3` | Gate G-PERM PASSED: Std $0.031705 \le 2\sigma_0$ ($0.066791$). |
| **Table 4.1: ANN 48-Cell Hyperparameter Grid** | Grid search with negative control | **VALID** | `part4_experiments.4b_ann_grid` | Winner: CELL_45 (SGD_m09, Init=default, Drop=0.2, BN=True). |
| **Table 4.2: Canonical Experiments A-E Summary** | 5 seeds mean $\pm$ std across A-E | **VALID** | `part4_experiments.4c_canonical_summary` | Evaluated on family-disjoint test partition. |
| **Table 4.3: Per-Seed Raw Metrics for Experiments A-E** | 5 seeds raw metrics | **VALID** | `part4_experiments.4c_canonical_summary.<arm>.per_seed_runs` | Recomputed in `tests/test_iteration7_roundtrip.py`. |
| **Table 4.4: Cluster Bootstrap 95% CIs** | Resample families (1000 reps) | **VALID** | `part4_experiments.4c_cluster_ci_table` | Rigorous cluster bootstrap accounting for family intra-cluster correlation. |
| **Table 4.5: LOFO and Single-Feature Audits** | 7 rows LOFO and univariate ANNs | **VALID** | `part4_experiments.4d_diagnostics` | Single-feature AUCs strictly capped $\le 0.85$. |
| **Table 4.6: Calibration and Prediction Agreement** | Temperature scaling and pairwise agreement | **VALID** | `part4_experiments.4d_diagnostics` | Brier $0.0163 \to 0.0142$, ECE $0.0632 \to 0.0526$. |
| **Table 4.7: CNN OFAT Anomaly Resolution** | Calibrated threshold vs rigid 0.5 | **VALID** | `part4_experiments.4e_cnn_ofat` | Resolves iteration 6 0.5000 accuracy artifact ($0.5000 \to 0.8800$). |
| **Table 4.8: BiLSTM Concat vs Attention Comparison** | Pooling comparison on val and test | **VALID** | `part4_experiments.4f_bilstm_pooling` | Zero ceiling violations observed. |
| **Table 5.1: Two-Sided Criteria for VAE Distribution** | Std ratios and Wasserstein per class | **VALID** | `part5_vae.5a_two_sided_criteria` | All 7 features satisfy $[0.5, 2.0]$ ratio and $\le 0.50$ Wasserstein ratio. |
| **Table 5.2: Per-Dimension KL Divergence** | Active latent capacity | **VALID** | `part5_vae.5a_per_dim_kl` | All 4 dimensions active ($\ge 0.05$ nats). |
| **Table 5.3: GBDT Real vs Synthetic Discriminator** | 5-fold CV AUC and positive control | **VALID** | `part5_vae.5b_discriminator` | Positive control AUC $1.0000 > 0.90$ verifies discriminator power. |
| **Table 5.4: Experiment E 4-Arm Ratio Sweep** | Sweep across ratios $\{0.0, 0.25, 0.50, 1.00\}$ | **VALID** | `part5_vae.5c_ratio_sweep` | Demonstrates performance stability across augmentation ratios. |
| **Table 6.1: Formal Research Questions (RQ1, RQ2)** | Strict yes/no/inconclusive with 95% CI | **VALID** | `research_questions` | RQ1 (Fusion): `inconclusive` (95% CI $[-0.0002, 0.0041]$); RQ2 (Augmentation): `inconclusive` (95% CI $[0.0000, 0.0100]$). |

---

## 9. Iteration 8 Integrity First, Benchmark v3 & Valid Experiments (`EXECUTION_REPORT_v8.md`)

**Results Artifact:** `results/experiment_results_v8.json` (SHA-256: `786a05cf071944395a1a243c28305202a431b835a6d6a64eb7931bdb96e69d8d`)  
**Split Manifest:** `data/split_manifest_hard_v3.json` (SHA-256: `a2a569c9354b040dc942096bab2b67bfec3cbee127b4661f9dd6a317031f31b8`)  

| Table Name | Description | Status | JSON Backing Key | Reason & Evidence |
|---|---|---|---|---|
| **Table 0.1: Git Diff & Provenance of v5 JSON** | Verification of commit 0bfb4f1 | **VALID** | `part0_disclosure.0a_v5_json_git_diff` | Confirms file was created in single commit and never edited; reconciles exploratory log numbers. |
| **Table 0.2: Failure Log for Iterations 5, 6, 7** | Comprehensive failure ledger | **VALID** | `part0_disclosure.0b_failure_log_iterations_5_6_7` | Lists all failed launches with task log paths, verbatim errors, causes, and target changed. |
| **Table 0.3: Pre-Iteration 3 Verification** | Git worktree verification of commit a985ee7 | **VALID** | `part0_disclosure.0c_git_worktree_pre_iter3` | Confirmed 28/28 passed at commit a985ee7; uncommitted edit explanation officially retracted. |
| **Table 0.4: Pytest Suite Collected Counts** | Test counts across 12 test files | **VALID** | `part0_disclosure.0d_pytest_collected_counts` | Confirms 77 tests collected and 100% passing across the suite. |
| **Table 1.1: Iteration 7 Seed Invariance Root Cause** | Static array and DataLoader seed diagnosis | **VALID** | `part1_integrity_audit.1a_seed_audit` | Explains how static arrays in v7 caused seed collapse; guarded by Rule G14 weight hashes. |
| **Table 1.2: Test-Split Access Audit** | CELL_45 Val AUC vs Exp D Test AUC | **VALID** | `part1_integrity_audit.1b_test_access_audit` | Confirms model selection strictly used validation split; test evaluation strictly guarded by Rule G13 SingleUseTestGuard. |
| **Table 1.3: G12 Confusion Matrix & Sigma_0** | Actual vs balanced test partition null bounds | **VALID** | `part1_integrity_audit.1c_confusion_matrix_and_sigma0` | Derives sigma_0 = 0.033396 for n0=147, n1=153 test partition. |
| **Table 1.4: Ground-Truth Label Noise Audit** | hard_v1/v2 vs hard_v3 noise comparison | **VALID** | `part1_integrity_audit.1d_noise_audit` | Shows hard_v1 and hard_v2 had 0% noise relative to artifacts; hard_v3 enforces 5.86% true label noise. |
| **Table 2.N1: N1 i.i.d. Gaussian Null Control** | 100 seeds Gaussian null on hard_v2 | **VALID** | `part2_controls.N1` | Mean 0.501905, Std 0.035048 <= 2*sigma_0 (0.066791). Passed. |
| **Table 2.N2: N2 Real Frozen Grouped-OOF Null** | 100 seeds frozen OOF features with permuted labels | **VALID** | `part2_controls.N2` | Mean 0.499119, Std 0.034509 <= 2*sigma_0 (0.066791). Passed. |
| **Table 2.N3: N3 Full Pipeline Null Control** | 100 seeds retrained pipeline on permuted labels | **VALID** | `part2_controls.N3` | Mean 0.502213, Std 0.034185 <= 2*sigma_0 (0.066791). Passed. |
| **Table 2.N4: N4 Diagnostic Permutation** | Record vs family level permutation | **VALID** | `part2_controls.N4_diagnostic` | Design effect ratio = 1.4339 (n_eff = 207 families). |
| **Table 2.N5: N5 Positive Control Detection Rates** | Injected cue detection at 0.60 and 0.70 AUC | **VALID** | `part2_controls.N5_positive_control` | 85.0% detection at 0.60, 100.0% detection at 0.70. Passed. |
| **Table 2.N6: N6 Leak Canaries Sensitivity** | Moved family and duplicated text canaries | **VALID** | `part2_controls.N6_leak_canary` | Both canary 1 and canary 2 successfully flagged. Passed Gate G-CTRL. |
| **Table 3.1: 40-Family Ontological Taxonomy** | 20 train, 10 val, 10 test family partition | **VALID** | `part3_generator_v3.template_families_count` | 1200 total cases with zero cross-split family or hash overlap. |
| **Table 3.2: 3-Round Oracle Tuning Log** | Validation split oracle calibration | **VALID** | `part3_generator_v3.tuning_rounds_log` | Fused oracle AUC = 0.9330, margin over best unimodal = +0.0731 >= +0.05. |
| **Table 3.3: Final Oracle Ceilings & Headroom** | Text (0.8599), Img (0.8215), Heur (0.8142), Fused (0.9330) | **VALID** | `part3_generator_v3.final_oracle_ceilings` | Headroom = 0.0670, Half Headroom cap = 0.0335. |
| **Table 3.4: MDE Simulation vs Headroom Cap** | Statistical power vs Delta-AUC | **VALID** | `part3_generator_v3.mde_simulation` | MDE = 0.0150 <= 0.0335 cap. Gate G-MDE PASSED. |
| **Table 4.1: ANN 48-Cell Hyperparameter Grid** | Grid search strictly on validation split | **VALID** | `part4_experiments.4b_ann_grid` | Winner: CELL_20 (RMSprop, Init=xavier, Drop=0.2, BN=True) Val AUC = 0.974200. |
| **Table 4.2: Canonical Experiments A-E Summary** | 10 seeds mean +/- std across Arms A-E | **VALID** | `part4_experiments.4c_canonical_summary` | Evaluated on hard_v3 test split with distinct weight hashes. |
| **Table 4.3: Per-Seed Raw Metrics for Exp A-E** | 10 seeds integer confusion matrices and hashes | **VALID** | `part4_experiments.4c_canonical_summary.<arm>.per_seed_runs` | Rule G12 and G14 enforced; recomputed in tests/test_iteration8_integrity.py. |
| **Table 4.4: Paired Cluster Bootstrap 95% CIs** | Resampling 10 whole families (1000 reps) | **VALID** | `part4_experiments.4c_cluster_ci_table` | Exp C vs Uni: dAUC = +0.094829 [0.062003, 0.136159] (Significant); Exp E vs D: dAUC = +0.000315 [-0.005457, 0.007316] (Inconclusive). |
| **Table 4.5: Prediction Agreement Matrix** | Pairwise agreement across Arms A-D | **VALID** | `part4_experiments.4d_diagnostics.agreement_matrix` | Unimodal agreement 0.5767; Multimodal to Full 7D agreement 0.8900. |
| **Table 4.6: Retrained LOFO 7D with Hashes** | Leave-one-feature-out with weight hashes | **VALID** | `part4_experiments.4d_diagnostics.lofo_table` | CNN and BiLSTM show largest drops when omitted. |
| **Table 4.7: Single-Feature ANN Univariate** | 7 individual feature ANNs | **VALID** | `part4_experiments.4d_diagnostics.single_feature_ann_table` | Text, URL, Heuristics <= 0.85; CNN at 0.8556. |
| **Table 4.8: Linear and Tree Baselines** | Logistic Regression (0.8872) and GBDT (0.9533) | **VALID** | `part4_experiments.4d_diagnostics.baselines_table` | Benchmark comparisons on full 7D test features. |
| **Table 4.9: Temperature Scaling Calibration** | Raw vs calibrated Brier and ECE | **VALID** | `part4_experiments.4d_diagnostics.calibration` | Brier 0.0927 -> 0.0912, ECE 0.0720 -> 0.0588. |
| **Table 4.10: CNN OFAT Parameter Sensitivity** | BatchNorm running stats vs batch stats | **VALID** | `part4_experiments.4e_cnn_ofat` | Running stats AUC = 0.8438, Batch stats AUC = 0.8496. |
| **Table 4.11: BiLSTM Concat vs Attention** | Pooling comparison across 10 seeds | **VALID** | `part4_experiments.4f_bilstm_pooling` | Concat: 0.8148 +/- 0.0055, Attention: 0.8082 +/- 0.0191. Zero ceiling violations. |
| **Table 5.1: Two-Sided Criteria for VAE** | Std ratios and Wasserstein per class | **VALID** | `part5_vae.5a_two_sided_criteria` | Quantitative transport metrics on hard_v3. |
| **Table 5.2: Per-Dimension KL Divergence** | Latent capacity across 4 dims | **VALID** | `part5_vae.5a_per_dim_kl` | All 4 dimensions active (non-collapsed, margin above 0.05 nats). |
| **Table 5.3: GBDT Real vs Synthetic Discriminator** | 5-fold CV AUC and positive control | **VALID** | `part5_vae.5b_discriminator` | Positive control AUC = 1.0000 > 0.90 confirms discriminator validity. |
| **Table 5.4: Exp E 4-Arm Ratio Sweep** | Sweep across {0.0, 0.25, 0.50, 1.00} x 10 seeds | **VALID** | `part5_vae.5c_ratio_sweep` | Demonstrates performance stability across VAE, Gaussian, and Interpolation augmenters. |
| **Table 6.1: Formal Research Questions Ledger** | Final empirical answers for RQ1 and RQ2 | **VALID** | `research_questions` | RQ1 (Fusion): YES (dAUC = +0.094829 [0.062003, 0.136159], MDE=0.0150); RQ2 (Augmentation): INCONCLUSIVE (dAUC = +0.000315 [-0.005457, 0.007316], MDE=0.0150). |


