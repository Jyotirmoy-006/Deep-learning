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

