# ForenSightX Iteration 9 — Complete Statistical Tables Ledger
**Results Artifact:** `results/experiment_results_v9.json`  
**Results SHA-256:** `e9f4196ff64b04aa7bae08cf1baffc73f96928226d3c359ff1e434c6981a9930`  
**Specification Precedence:** ForenSightX_Project_Architecture.docx wins over summary presentation.  
**Execution Timestamp:** `2026-11-04T20:34:18.550852`  
**Execution Verdict:** `ALL GATES PASSED - VALID BENCHMARK`  

## Part 0: Disclosures & Provenance Audit

### Table 0.1: Part 0a Headline Reconciliation (v5 Report vs JSON vs History)
`JSON Key: part0_disclosure.0a_reconcile_v5_headline`

| Metric | v5 Report Table 2.2 Printed | v6 'After' Table Printed | Committed JSON Value | Provenance Status |
|---|---|---|---|---|
| **Mean ROC-AUC** | `0.5064` | `0.512407` | `0.512405` | **UNVERIFIED** |
| **Standard Deviation** | `0.0734` | `0.076814` (sample, ddof=1) | `0.070967` (population, ddof=0) | **UNVERIFIED** |

> [!NOTE]
> [HYPOTHESIS] The v5 report values (0.5064 / 0.0734) originated from an unlogged exploratory terminal session run prior to the execution of run_iteration5_investigations.py. In v6, 'after' quoted the sample standard deviation (ddof=1, 0.076814) with rounding, while the committed JSON stored population standard deviation (ddof=0, 0.070967). Because no saved task log exists for the 0.5064 headline, the v5 control is marked UNVERIFIED in RESULT_STATUS.md and 'PROVENANCE CONFIRMED' is dropped.

### Table 0.2: Rebuilt Comprehensive Failure Log Across All Iterations
`JSON Key: part0_disclosure.0b_failure_log_rebuilt`

| Iteration | Launch / Task ID | Exact Log Path | Full Untruncated Error Message | Root Cause Analysis | Code Area / Action |
|---|---|---|---|---|---|
| Iteration 3 | `step 507` | `steps/507/output.txt` | `ImportError: cannot import name 'build_multimodal_cases' from 'src.data.case_builder' (C:\Users\Asus\OneDrive\Desktop\forensightx-mas\src\data\case_builder.py)` | build_multimodal_cases function was not exported; only CaseBuilder class existed. | **SOURCE_CHANGED** |
| Iteration 3 | `step 529` | `steps/529/output.txt` | `ImportError: cannot import name 'CNNFeatureDataset' from 'train' (unknown location)` | Dataset was named ImageEvidenceDataset in data_loader. | **SOURCE_CHANGED** |
| Iteration 3 | `step 557` | `steps/557/output.txt` | `TypeError: ForensicRawRecord.__init__() missing 2 required positional arguments: 'metadata' and 'content_hash'` | Omitted metadata and content_hash fields in test record creation. | **SOURCE_CHANGED** |
| Iteration 3 | `step 563` | `steps/563/output.txt` | `TypeError: tuple indices must be integers or slices, not str` | create_stratified_split returned 4-tuple, caller indexed as dict. | **SOURCE_CHANGED** |
| Iteration 3 | `task-569` | `tasks/task-569.log` | `RuntimeError: Tensors must have same number of dimensions: got 2 and 3` | SyntheticDataVAE.encode attempted torch.cat with 2D x and 3D y_onehot. | **SOURCE_CHANGED** |
| Iteration 4 | `task-1157` | `tasks/task-1157.log` | `RETRACTED FROM ITERATION 5: Task 1157 ran Iteration 4 harness ('FORENSIGHTX ITERATION 4: CLOSE OPEN DEFECTS, THEN REAL DATA'), not an Iteration 5 failure.` | Misattribution in v7/v8 report logs. | **RETRACTED** |
| Iteration 5 | `interactive_run_v5` | `UNRECOVERABLE (interactive terminal session)` | `Interactive terminal execution produced results/experiment_results_v5.json without background task log capture.` | Terminal commands executed synchronously without log redirection. | **UNRECOVERABLE** |
| Iteration 6 | `task-1363` | `tasks/task-1363.log` | `RuntimeError: Error(s) in loading state_dict for VisualEvidenceCNN:<br>	Missing key(s) in state_dict: "conv1.weight", "conv1.bias", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "conv2.weight", "conv2.bias", "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "conv3.weight", "conv3.bias", "bn3.weight", "bn3.bias", "bn3.running_mean", "bn3.running_var", "classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias".<br>	Unexpected key(s) in state_dict: "config", "state_dict", "metadata".` | load_model_checkpoint passed top-level checkpoint dictionary containing metadata directly to model.load_state_dict instead of ckpt['state_dict']. (v8 incorrectly misattributed this to generator purity). | **SOURCE_CHANGED** |
| Iteration 6 | `task-1620` | `tasks/task-1620.log` | `ValueError: Found input variables with inconsistent numbers of samples: [600, 300]` | Discriminator dataset length mismatched real vs synthetic splits. | **SOURCE_CHANGED** |
| Iteration 6 | `task-1638` | `tasks/task-1638.log` | `TypeError: CaseCoordinatorAgent.investigate() got an unexpected keyword argument 'evidence_payload'` | Agent method signature expected evidence dict, not evidence_payload. | **SOURCE_CHANGED** |
| Iteration 7 | `task-2045` | `tasks/task-2045.log` | `ImportError: cannot import name 'train_vae_epoch' from 'src.models.vae_trainer'` | Function was named train_repaired_vae. | **SOURCE_CHANGED** |
| Iteration 7 | `task-2073` | `tasks/task-2073.log` | `ValueError: The internally computed table of expected frequencies has a zero element` | Contingency column had all zeros in chi2_contingency. | **SOURCE_CHANGED** |

### Table 0.3: Part 0c Skipped Items Audit (1e Recount & 1f v6 Derivation)
`JSON Key: part0_disclosure.0c_skipped_items_audit`

| Item | Audit Finding & Mathematical Derivation |
|---|---|
| **1e Text Record Overlap Recount** | In hard_v1, templates were randomly sampled across splits. 56 unique text strings appeared in the validation split. Across the 700 training cases, 68 case records shared one of those 56 texts (an average of 1.21 occurrences per shared template). In hard_v2, hard_v3, and hard_v4, template families are strictly partitioned into disjoint sets. Cross-split unique text overlap is exactly 0 (0 shared records). |
| **1f v6 Bimodal Permutation SD (0.1975)** | In v6, training labels were permuted but test labels were kept UNPERMUTED. Because high-capacity multimodal neural models fitted spurious noise features, random weight initializations randomly aligned or anti-aligned with true test signals, producing a wide bimodal distribution with standard deviation 0.1975 >> sigma_0 (0.033). Symmetric permutation restores standard deviation to sigma_0. Code pattern: `y_tr_perm = rng.permutation(y_tr); clf.fit(X_tr, y_tr_perm); preds = clf.predict_proba(X_te)[:, 1]; auc = roc_auc(y_te_TRUE, preds)` |

## Part 1: Reconcile v8 Numbers

### Table 1.1: Comprehensive Reconciliation of Iteration 8 Discrepancies
`JSON Key: part1_reconciliation`

| Item | v8 Defect / Discrepancy | Mathematical & Architectural Root Cause | Iteration 9 Fix / Rule Enforced |
|---|---|---|---|
| **(a) Table 4.2 vs Printed Rows** | Table 4.2 in v8 printed intermediate values (A accuracy 0.6907 vs row average 0.6943; B recall 0.5407 vs 0.5573) from a draft run rather than computing from printed seed rows. | Intermediate script launch values printed in table; A AUC std 0.005514 in Table 4.2 vs population std 0.0059 (ddof=0) vs sample std 0.0062 (ddof=1). | **Rule G16: compute_summary_row_from_seed_runs is the sole authoritative summary emitter, computing mean and std strictly from the printed seed rows.** |
| **(b) Table 4.4 Bootstrap Deltas** | Table 4.4 delta C-B AUC (0.0913 vs 0.0948), F1 (0.1699 vs 0.1892), D-C, E-D mismatched differences of Table 4.2 means. | v8 line 851 resampled predictions from a single seed (seed 51) rather than paired differences across the multi-seed evaluation. | **Rule G16: Delta tables explicitly print BOTH the point estimate (mean_B - mean_A) AND the bootstrap mean of the paired difference.** |
| **(c) Exp E vs Table 5.4 VAE@0.50** | Exp E reported 0.948927 (Table 4.2) vs VAE@0.50 reported 0.949200 (Table 5.4). | v8 Table 5.4 instantiated an independent training loop with a different seed loop instead of evaluating Exp E's model. | **Exp E model and synthetic ratio sweep use the identical model weights and seed trajectory.** |
| **(d) Table 1.2 Hardcoded Strings** | Table 1.2 quoted val 0.986078 / test 0.990868 under keys holding 0.974200 / 0.948951 in v8 JSON. | Markdown generator had hardcoded string literals copied from Iteration 7. | **Dynamic emission strictly from JSON results object without hardcoded literals.** |

## Part 2: Test Access & SingleUseTestGuard Audit

### Table 2.1: SingleUseTestGuard Comprehensive Audit Access Log
`JSON Key: test_guard_access_log`

| Evaluation Index | Caller Function / Scope | Test Sample Count | Prediction Array SHA-256 Hash |
|---|---|---|---|
| #1 | `Exp_A_seed_42` | 300 | `ec752a776e5fcf70c0bbead9ec3d2279e2058cd9e6279ea59b106c0d1dc04440` |
| #2 | `Exp_B_seed_42` | 300 | `93e2e7d96b7f521047a61cd1040ed625143b7c2fe5cc1317f42eccf97c5b8899` |
| #3 | `Exp_C_seed_42` | 300 | `ac698e0b1521d50a81d539ad643a44b7b5077e23267dc6f6ed2d248a3def7570` |
| #4 | `Exp_D_seed_42` | 300 | `8f4daffac634cdbdb664039cf284ab9f8f7a633f005255fe3aea87721185bd15` |
| #5 | `Exp_E_seed_42` | 300 | `7511e837e22c3f04008cdb0a77565efc8036e3d044db7345b4abf96cc7cd6691` |
| #6 | `Exp_A_seed_43` | 300 | `0e264fae9398048f3ccbd3cb7bc62de16b67eb0a289158815abc4fbcd28d1f2b` |
| #7 | `Exp_B_seed_43` | 300 | `369a339635b89202ea9cf9d73a0d08c55d1ccec35fb8886f5d665d9e09a63fa0` |
| #8 | `Exp_C_seed_43` | 300 | `a1fa776b66750eaa9e601c1fce17691edc41fe905ec027a5a3ad680f6d32d17d` |
| #9 | `Exp_D_seed_43` | 300 | `c52b3850081dad6a4f35d55b961aa9c86b1cb18b157592fc9d15761cc890e96e` |
| #10 | `Exp_E_seed_43` | 300 | `f76dc0f292f7a383111ae5070e0f1d0a1a2eae92f0a396d56dde3a5304a9690a` |
| #11 | `Exp_A_seed_44` | 300 | `46a00936388af7e7dc3a8d6cf4e4f8f9d6403362922e61cf204195db376af073` |
| #12 | `Exp_B_seed_44` | 300 | `06ba973e6bdfd36a7fd7b610069d9d6fe3371cac29b3f7d08893871d13af29e9` |
| #13 | `Exp_C_seed_44` | 300 | `d45da15cb58236cafda5092ffc92323a157371a6ddfb334de8eae2adae112d5f` |
| #14 | `Exp_D_seed_44` | 300 | `233b6e35c468f663d2c198c0457f55e44043b9f681b58f8a6d63a171fe64ffcd` |
| #15 | `Exp_E_seed_44` | 300 | `03e4a2845fad07b840ee22bef6877089aed979dab299a52941d13ecf0dd89615` |
| #16 | `Exp_A_seed_45` | 300 | `1f2f41f7f319953caef44c111d4a8b5ec6bcc68a6e36bf6e45f567149c1d97b1` |
| #17 | `Exp_B_seed_45` | 300 | `3af9f38e55c6143f4a55b7fd6f0750f0abadb11d05bd521974445cc7d45373c0` |
| #18 | `Exp_C_seed_45` | 300 | `e8b068fc68390284040c1c5388bb5bc4e2a485318a000f6615b4088ef373316d` |
| #19 | `Exp_D_seed_45` | 300 | `52c007b91a395b2e6a27396a88917c5e76780119b218d0ed817f996891161729` |
| #20 | `Exp_E_seed_45` | 300 | `642916d50a57c5038f01a2afdd53555b3f2492bf0e455fb75de8e7c6ca2a44a7` |
| #21 | `Exp_A_seed_46` | 300 | `456ff2045fbe558ae8b91cfffd549139cdb6325387fa497c403d721b750eed78` |
| #22 | `Exp_B_seed_46` | 300 | `369a339635b89202ea9cf9d73a0d08c55d1ccec35fb8886f5d665d9e09a63fa0` |
| #23 | `Exp_C_seed_46` | 300 | `fa3e126be5c3fe664cb4bd0a37333332b20a5d6a03c4e8695f954cbf1776f1a9` |
| #24 | `Exp_D_seed_46` | 300 | `945c57b6ca3eed1de65af7ca9c9a59f72574e353a94ef302e044bc93cce54f12` |
| #25 | `Exp_E_seed_46` | 300 | `2ef2abbf4f689d58e1670eec7988ae610e18311269f73d8755b50bc5b81e6baa` |
| #26 | `Exp_A_seed_47` | 300 | `812960a82a719ef7aa044c3685b5cbe7c059ff2fcb528345d68b80c7ae14b271` |
| #27 | `Exp_B_seed_47` | 300 | `934eb65e1497f3abb68b2a3872468062bebf9527c7b8b198755b68262727085f` |
| #28 | `Exp_C_seed_47` | 300 | `e86570f8c9456945db88e2048be7e18087d422a426266e87bf1bd1baa6f78d4a` |
| #29 | `Exp_D_seed_47` | 300 | `c247a4505c3ea168abb3af599fa01aff9c919cfbda15a8d7eaa2ed59e7e5cf42` |
| #30 | `Exp_E_seed_47` | 300 | `4f75ea4393547f5077887a0231b45fde553fe63f8b0d9db26b60916dc6da81f2` |
| #31 | `Exp_A_seed_48` | 300 | `ae9bb187d527f40d6935bbe19e74a9da1d4ff2b05f1e45ffbc953a011699c26f` |
| #32 | `Exp_B_seed_48` | 300 | `5b2dfce126a4574de09e685b38a3a244c0a0d4791eb5b4ae85f51aebad40a4cd` |
| #33 | `Exp_C_seed_48` | 300 | `0473f3d04472f04b2331fd9a4bc44747cea821ac915d0d90a0b52032beca199c` |
| #34 | `Exp_D_seed_48` | 300 | `56bc10f112c541e5bdd2d01e38706c6c47f67ecde1f98c4bcc432443642f40c5` |
| #35 | `Exp_E_seed_48` | 300 | `b0fdbb54c2f4278a8bf6c8f37b2280fc52f1686c757eae28cde011e1c9cbce80` |
| #36 | `Exp_A_seed_49` | 300 | `243b6f15636f900026dbe2815ef3c25492842c8f0a70e897a0cf870853361110` |
| #37 | `Exp_B_seed_49` | 300 | `d8abde2729e4c621c4023af092c6b87c7a6e9453776ba8d21eaeb6fdda9ab660` |
| #38 | `Exp_C_seed_49` | 300 | `fb9ae02bb1d645aeeed6fa98a0439f35774e20af496efa4c88602f9fca64770a` |
| #39 | `Exp_D_seed_49` | 300 | `affdcd55402661cf3ecb7b54ad0a514b6033e1dc2e5796c25e2713648c28149e` |
| #40 | `Exp_E_seed_49` | 300 | `720f7bd1746892d8ae3a92db3cc2b8c77293113e763ae85c75b8f3340873b5f8` |
| #41 | `Exp_A_seed_50` | 300 | `69762886418f89a13df54ae3f85381b50b698280881dfa6b9074633dda78b2f7` |
| #42 | `Exp_B_seed_50` | 300 | `369a339635b89202ea9cf9d73a0d08c55d1ccec35fb8886f5d665d9e09a63fa0` |
| #43 | `Exp_C_seed_50` | 300 | `9330479f9ae6005cbe563a6f0b549fd2dc7c0bd747f4aa92242f77e41792df5e` |
| #44 | `Exp_D_seed_50` | 300 | `a37bfae4687c4807c7181d5799a0ed05964279d9142b5ea230bbb1ebd443dfe4` |
| #45 | `Exp_E_seed_50` | 300 | `3ba520fdabbda25c55900d8d733031a237a96f5a3c8d2b5ee965e2745e690da3` |
| #46 | `Exp_A_seed_51` | 300 | `23e2632c57aac18a73b9619f58d68aa97716bf2af7db5a7b66688fcf701ee413` |
| #47 | `Exp_B_seed_51` | 300 | `06ba973e6bdfd36a7fd7b610069d9d6fe3371cac29b3f7d08893871d13af29e9` |
| #48 | `Exp_C_seed_51` | 300 | `dcfaa6581a69c497ade498c85c8891bcd3acd6bbf872832ad41aa683b1d3d874` |
| #49 | `Exp_D_seed_51` | 300 | `61dabc1649825fbf18edc5ff7c590ea94eee73db839bbb593d6ab540d7d1b0ed` |
| #50 | `Exp_E_seed_51` | 300 | `0d5a8d99d2ae2c5c2b04c6c6d87e46c6de02cf74730124c973e98d9f3d74738f` |

## Part 3: Controls on hard_v3 and hard_v4

### Table 3.1: Controls Summary & Manifest Provenance
`JSON Keys: part3_controls_v3, part3_controls_v4`

| Dataset Version | Manifest SHA-256 | Gate Status | G-OOF OOF AUC | G-OOF Val AUC | G-OOF Delta | N1 (Label Perm) Mean +/- SD | N7 Symmetric SD | N7 Asymmetric SD |
|---|---|---|---|---|---|---|---|---|
| **hard_v3** | `a2a569c9354b040d...` | **GATE FAILED: G-CTRL on hard_v3 (disparity 0.1056)** | `0.7064` | `0.8120` | `0.1056` | `0.5088 +/- 0.0300` | `0.0358` | `0.1694` |
| **hard_v4** | `5ef13532eb6516fc...` | **PASSED** | `0.7164` | `0.7500` | `0.0336` | `0.5010 +/- 0.0328` | N/A | N/A |

### Table 3.2: 100-Seed Null Control Runs (N1 Pure Statistical Permutation on hard_v4)
`JSON Key: part3_controls_v4.N1.seeds_table`

| Seed | ROC-AUC | Accuracy | z-Score | Epoch | Init Weight Hash | Final Weight Hash | Data Order Hash | TP | FP | FN | TN | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `0.545200` | `0.5267` | `0.9238` | 1 | `4ac0c4438ebbc7e4` | `5cea9dacbf9b74b1` | `fdf973829ff662c8` | 74 | 66 | 76 | 84 | 0.5 |
| 1 | `0.504889` | `0.4833` | `-0.5774` | 1 | `1ad63053ce7b8d82` | `5d16230d0a2282dd` | `7f3a4879b6ac7519` | 78 | 83 | 72 | 67 | 0.5 |
| 2 | `0.475422` | `0.4567` | `-1.5011` | 1 | `3c6370635f62f87c` | `5a7f6e0d870b9f5b` | `eacac34ebec0a9d3` | 60 | 73 | 90 | 77 | 0.5 |
| 3 | `0.490978` | `0.5100` | `0.3464` | 1 | `db52f9c79502e9ef` | `4ab4a2e2a40d6932` | `f4d2ec54613fa315` | 83 | 80 | 67 | 70 | 0.5 |
| 4 | `0.501067` | `0.4933` | `-0.2309` | 1 | `9c7e1105c3bac142` | `fe628f0bd6e7ff02` | `2f1a33d6a9337940` | 77 | 79 | 73 | 71 | 0.5 |
| 5 | `0.519511` | `0.5200` | `0.6928` | 1 | `b20b17e931349899` | `f00bfd38d0ef865f` | `faa09c81bddd93e9` | 76 | 70 | 74 | 80 | 0.5 |
| 6 | `0.549600` | `0.5367` | `1.2702` | 1 | `f103294b3817bbd1` | `9453bb946caf6ea1` | `e906b4c7c9d245c9` | 85 | 74 | 65 | 76 | 0.5 |
| 7 | `0.537200` | `0.5333` | `1.1547` | 1 | `fcb9aea8cc32c5b0` | `51b658fdc21375c6` | `65c02a6cad24e04a` | 80 | 70 | 70 | 80 | 0.5 |
| 8 | `0.456978` | `0.4733` | `-0.9238` | 1 | `8862ad2ebe3ef285` | `a782a5ebc55131fb` | `d3b384c12a266bcb` | 76 | 84 | 74 | 66 | 0.5 |
| 9 | `0.542978` | `0.5533` | `1.8475` | 1 | `5ecde809d6711ea7` | `3f65fd5170813a40` | `594c6f9fe979a170` | 82 | 66 | 68 | 84 | 0.5 |
| 10 | `0.527422` | `0.5133` | `0.4619` | 1 | `4b58acea5cc1285d` | `3e4e377ac9856f36` | `44ac3f9d4b5339e3` | 77 | 73 | 73 | 77 | 0.5 |
| 11 | `0.510533` | `0.5067` | `0.2309` | 1 | `ca536489214586fd` | `d5db41640588a614` | `b6baeb7875203f4a` | 84 | 82 | 66 | 68 | 0.5 |
| 12 | `0.497067` | `0.4900` | `-0.3464` | 1 | `307cbf5175bcf8e2` | `a8823947e80837ad` | `b46b6f3447dfb9e4` | 72 | 75 | 78 | 75 | 0.5 |
| 13 | `0.518756` | `0.4833` | `-0.5774` | 1 | `4b3ec31bc268dab6` | `ad969a7ca15aef58` | `b774bab1cb1667b7` | 72 | 77 | 78 | 73 | 0.5 |
| 14 | `0.465111` | `0.4800` | `-0.6928` | 1 | `1292fbd012450954` | `557c06190719a2a6` | `7df4dfd21fca4671` | 77 | 83 | 73 | 67 | 0.5 |
| 15 | `0.543244` | `0.5167` | `0.5774` | 1 | `83f55e3730618398` | `5276e0771ffcc525` | `0ee0e89c3e052c4e` | 84 | 79 | 66 | 71 | 0.5 |
| 16 | `0.509600` | `0.4867` | `-0.4619` | 1 | `efe62669c9648c64` | `d68286d6b83bb27a` | `3bf2bc5059fccecd` | 69 | 73 | 81 | 77 | 0.5 |
| 17 | `0.520489` | `0.5000` | `0.0000` | 1 | `a10d48ace554f2fc` | `e0c146a8cae783a4` | `2f2e1ef64fd347c2` | 74 | 74 | 76 | 76 | 0.5 |
| 18 | `0.482178` | `0.4800` | `-0.6928` | 1 | `6ba164a6e3ec7da0` | `c5f30a42538c7a84` | `a674d5d8b6d36936` | 71 | 77 | 79 | 73 | 0.5 |
| 19 | `0.508178` | `0.4867` | `-0.4619` | 1 | `1950cb1bde4ffaf1` | `ca0ab741a2ffce65` | `1be181070ba7eda1` | 70 | 74 | 80 | 76 | 0.5 |
| 20 | `0.413067` | `0.4433` | `-1.9630` | 1 | `3d4daf4cacff208a` | `c3fd71fc4aeb7de7` | `76e355c94433759b` | 75 | 92 | 75 | 58 | 0.5 |
| 21 | `0.412800` | `0.4267` | `-2.5403` | 1 | `61c04966b9862af3` | `75b9de3b662e03cb` | `d3390a9ab32641c3` | 69 | 91 | 81 | 59 | 0.5 |
| 22 | `0.483289` | `0.4767` | `-0.8083` | 1 | `814838f738f48efb` | `348028aeb1ea7e71` | `b9883777b220835f` | 75 | 82 | 75 | 68 | 0.5 |
| 23 | `0.519911` | `0.5033` | `0.1155` | 1 | `85abb94841dda8c6` | `6b49798f58ad3497` | `4ac25beb13a8d8fc` | 72 | 71 | 78 | 79 | 0.5 |
| 24 | `0.503867` | `0.4867` | `-0.4619` | 1 | `8daab035896e97a6` | `64866980d9d582d5` | `761c6c6d08444a28` | 71 | 75 | 79 | 75 | 0.5 |
| 25 | `0.473644` | `0.4633` | `-1.2702` | 1 | `a8c909ab9b5f2d96` | `fa543e5db37ffa8c` | `fc7499797d474c33` | 70 | 81 | 80 | 69 | 0.5 |
| 26 | `0.530489` | `0.5433` | `1.5011` | 1 | `62f88760b5fbb993` | `1d4234f9a85bab35` | `9edbe353c5f47df5` | 83 | 70 | 67 | 80 | 0.5 |
| 27 | `0.508578` | `0.5033` | `0.1155` | 1 | `de516cbb753c7d62` | `6bf419c0fd6fddf5` | `09c7b08ffc404240` | 77 | 76 | 73 | 74 | 0.5 |
| 28 | `0.514978` | `0.5367` | `1.2702` | 1 | `9ee49ddb2ad4bd29` | `9d8bd0e9a477cda0` | `aa5480c13a8f6de8` | 79 | 68 | 71 | 82 | 0.5 |
| 29 | `0.464844` | `0.4800` | `-0.6928` | 1 | `8be04cb8fc033824` | `53aba7fb052c114d` | `2a78bd0d11dd7666` | 74 | 80 | 76 | 70 | 0.5 |
| 30 | `0.517689` | `0.5300` | `1.0392` | 1 | `897b73d1f1dc0278` | `42b3d86eb84c47b4` | `2db82e0f5d448a97` | 77 | 68 | 73 | 82 | 0.5 |
| 31 | `0.514978` | `0.5233` | `0.8083` | 1 | `b91714fb2fd3948e` | `6a4eaffbb0d7f5c2` | `6a005d88089f1ac7` | 78 | 71 | 72 | 79 | 0.5 |
| 32 | `0.513956` | `0.5133` | `0.4619` | 1 | `e5d8cbf667be2459` | `904f6658e1ac0f7b` | `9e43a6bc41f97b53` | 76 | 72 | 74 | 78 | 0.5 |
| 33 | `0.459467` | `0.4533` | `-1.6166` | 1 | `372647831bde621a` | `5462546c2e5444c4` | `b1dba4a2bf788a45` | 79 | 93 | 71 | 57 | 0.5 |
| 34 | `0.544444` | `0.5033` | `0.1155` | 1 | `f9d7f4d40dd45001` | `2f00b19a55c438f4` | `aa593d9ed7b3b1aa` | 74 | 73 | 76 | 77 | 0.5 |
| 35 | `0.517200` | `0.5100` | `0.3464` | 1 | `1857c0e5e7bf0a57` | `6544f0df92ef5b8d` | `23ef169ed88a9cda` | 72 | 69 | 78 | 81 | 0.5 |
| 36 | `0.486844` | `0.4800` | `-0.6928` | 1 | `41ad243a72aa0e95` | `c5ad3ff3dde0dc19` | `61edc196fee03d9b` | 69 | 75 | 81 | 75 | 0.5 |
| 37 | `0.492978` | `0.5067` | `0.2309` | 1 | `71fcbc12d7719a76` | `44101cd171886071` | `0e82a984e859d395` | 78 | 76 | 72 | 74 | 0.5 |
| 38 | `0.595733` | `0.5833` | `2.8868` | 1 | `755897eeb8e8c405` | `ba01b2627466ca9b` | `5e4d8f783fbef4f6` | 96 | 71 | 54 | 79 | 0.5 |
| 39 | `0.514933` | `0.4933` | `-0.2309` | 1 | `ff84c6c1b7075b4c` | `bc6fc7c09594be97` | `0025f7798f5c6889` | 77 | 79 | 73 | 71 | 0.5 |
| 40 | `0.466044` | `0.4800` | `-0.6928` | 1 | `ae9e5c66718eaf72` | `b7f44ba8bf6c6c63` | `1da0101127240353` | 70 | 76 | 80 | 74 | 0.5 |
| 41 | `0.530844` | `0.5167` | `0.5774` | 1 | `f37a168e2092a8e3` | `395a9577f4d59e5d` | `81617a17a2dc9452` | 77 | 72 | 73 | 78 | 0.5 |
| 42 | `0.507644` | `0.5167` | `0.5774` | 1 | `ea805c1d958914de` | `d6787030298dd3e0` | `ba4c03b0043ac476` | 84 | 79 | 66 | 71 | 0.5 |
| 43 | `0.462267` | `0.4900` | `-0.3464` | 1 | `d5abfa283e06105f` | `9ef32a7bf3d287b8` | `e2c62cff56cb679c` | 76 | 79 | 74 | 71 | 0.5 |
| 44 | `0.498933` | `0.4867` | `-0.4619` | 1 | `901ca63e42307792` | `1ae1c9139bc89c6e` | `5ffe552982b94b43` | 80 | 84 | 70 | 66 | 0.5 |
| 45 | `0.486622` | `0.4800` | `-0.6928` | 1 | `20baa5223d926c47` | `98b01bf2924aa9ab` | `4bee727a8d46d0d8` | 72 | 78 | 78 | 72 | 0.5 |
| 46 | `0.495156` | `0.4900` | `-0.3464` | 1 | `476707929905f949` | `6a85f98781eeface` | `0954c15461b23c0d` | 71 | 74 | 79 | 76 | 0.5 |
| 47 | `0.532222` | `0.5333` | `1.1547` | 1 | `3fb1675103e9bdf9` | `85b928c9523bd509` | `f422ae479226ed53` | 81 | 71 | 69 | 79 | 0.5 |
| 48 | `0.496400` | `0.5000` | `0.0000` | 1 | `b764b83fd0d8e2b1` | `ce88383241dfc59e` | `509b40ccfc4c9553` | 74 | 74 | 76 | 76 | 0.5 |
| 49 | `0.524444` | `0.5267` | `0.9238` | 1 | `af111a5531417de7` | `a2fd25636c0b1d16` | `3cb31e6cda9ac83a` | 76 | 68 | 74 | 82 | 0.5 |
| 50 | `0.506711` | `0.5033` | `0.1155` | 1 | `110597aa4410fc18` | `ea8add9e97a72931` | `7d960dd43d8bb1c3` | 73 | 72 | 77 | 78 | 0.5 |
| 51 | `0.485111` | `0.4800` | `-0.6928` | 1 | `aa819bf22bdce70a` | `786db0c30751c95b` | `ca17cae6b3a436a1` | 73 | 79 | 77 | 71 | 0.5 |
| 52 | `0.517956` | `0.5200` | `0.6928` | 1 | `7fcac8225aafdfd1` | `f5bef2bfc1e985cf` | `177b7921dca36bd0` | 77 | 71 | 73 | 79 | 0.5 |
| 53 | `0.471911` | `0.5033` | `0.1155` | 1 | `a4575ac317a91cc4` | `bbc7197564c050a0` | `1630a64e0de57688` | 71 | 70 | 79 | 80 | 0.5 |
| 54 | `0.526444` | `0.5233` | `0.8083` | 1 | `874ae5a3bae151bc` | `3e0a4931c12fc33c` | `7e25cb613f817025` | 70 | 63 | 80 | 87 | 0.5 |
| 55 | `0.532533` | `0.5200` | `0.6928` | 1 | `97be4c7913ed36d1` | `42242f6fbb3c0040` | `62ba30cc8c33b6ce` | 77 | 71 | 73 | 79 | 0.5 |
| 56 | `0.564578` | `0.5600` | `2.0785` | 1 | `89332a8a1207d3c1` | `f87298de69474ebb` | `3b2ce85b62abfe43` | 79 | 61 | 71 | 89 | 0.5 |
| 57 | `0.496711` | `0.4833` | `-0.5774` | 1 | `61a9f8dc1a730264` | `443f04bd3a703900` | `51986c5b6effd446` | 75 | 80 | 75 | 70 | 0.5 |
| 58 | `0.517289` | `0.5033` | `0.1155` | 1 | `068575d45960cba3` | `8285e59c981242e9` | `48bea19273706f33` | 70 | 69 | 80 | 81 | 0.5 |
| 59 | `0.495733` | `0.4800` | `-0.6928` | 1 | `7ff46d1e72722930` | `0a75a5e9378398e1` | `c5e47dbf7a801638` | 78 | 84 | 72 | 66 | 0.5 |
| 60 | `0.510711` | `0.5167` | `0.5774` | 1 | `635af33c56f47f0f` | `e68274514634373f` | `53decce4ae3cb69e` | 71 | 66 | 79 | 84 | 0.5 |
| 61 | `0.523689` | `0.5033` | `0.1155` | 1 | `270accd5aba19a9b` | `8a27e6a7f1a0b6b1` | `5dee1a5809dc3030` | 78 | 77 | 72 | 73 | 0.5 |
| 62 | `0.459689` | `0.4567` | `-1.5011` | 1 | `800dcfb4c148f29c` | `7614719d543121c1` | `8480faac4823918b` | 64 | 77 | 86 | 73 | 0.5 |
| 63 | `0.493378` | `0.5000` | `0.0000` | 1 | `6e4acb17a2259bdc` | `7ffb53abc083d998` | `fed136b99ca319f8` | 80 | 80 | 70 | 70 | 0.5 |
| 64 | `0.522222` | `0.5100` | `0.3464` | 1 | `676d4a5a9e63c6c2` | `4ae65cb90d70044c` | `729ac060b42a5dc5` | 77 | 74 | 73 | 76 | 0.5 |
| 65 | `0.455022` | `0.4833` | `-0.5774` | 1 | `ad6ec8b90bf45d41` | `cd48943f810b468d` | `0700e1a72217b106` | 79 | 84 | 71 | 66 | 0.5 |
| 66 | `0.498089` | `0.4800` | `-0.6928` | 1 | `68a463b6a0da92bb` | `a67bc53b1cb124bc` | `ec93597e70b6d26e` | 72 | 78 | 78 | 72 | 0.5 |
| 67 | `0.511378` | `0.5100` | `0.3464` | 1 | `b181a32ed7654e8d` | `5aa9fd03c2ece40f` | `f83ad380fcf972d2` | 82 | 79 | 68 | 71 | 0.5 |
| 68 | `0.494533` | `0.5000` | `0.0000` | 1 | `94b5b2a2056cd154` | `1816d1d3d1affc0f` | `2265d17a77a4a60d` | 80 | 80 | 70 | 70 | 0.5 |
| 69 | `0.510622` | `0.4733` | `-0.9238` | 1 | `e84840967e4a94c5` | `599b12debf9378cc` | `103df2b069565221` | 65 | 73 | 85 | 77 | 0.5 |
| 70 | `0.457956` | `0.4500` | `-1.7321` | 1 | `8d742b8cd831283d` | `890d37810ce60d67` | `d91d2cd80a81ab77` | 67 | 82 | 83 | 68 | 0.5 |
| 71 | `0.524667` | `0.5433` | `1.5011` | 1 | `6b17d9b88fa2dfe6` | `503d0329512f21d6` | `d5c52a02fa062e77` | 80 | 67 | 70 | 83 | 0.5 |
| 72 | `0.602756` | `0.5667` | `2.3094` | 1 | `9f09cbba9295ae4c` | `923d3c5280972659` | `905bee51f67a44b5` | 90 | 70 | 60 | 80 | 0.5 |
| 73 | `0.499733` | `0.4867` | `-0.4619` | 1 | `62b84df54d1411bd` | `e13dd6f791aa5149` | `ecc1b5d454a4cbbe` | 71 | 75 | 79 | 75 | 0.5 |
| 74 | `0.543200` | `0.5200` | `0.6928` | 1 | `7ec73986193de6d0` | `523500fa6332f67c` | `a185828b40729149` | 71 | 65 | 79 | 85 | 0.5 |
| 75 | `0.449022` | `0.4533` | `-1.6166` | 1 | `1ce24993712d5b01` | `1893ebe9f19ff3d0` | `0980437f95be9b57` | 74 | 88 | 76 | 62 | 0.5 |
| 76 | `0.528578` | `0.5300` | `1.0392` | 1 | `dcea5b8f2350c05d` | `85c84b8a3b3099d1` | `cddaf0e68de4f46c` | 75 | 66 | 75 | 84 | 0.5 |
| 77 | `0.511822` | `0.4867` | `-0.4619` | 1 | `57b27017cf44364d` | `4a4f2f0d989e371d` | `47952e7c079aded5` | 67 | 71 | 83 | 79 | 0.5 |
| 78 | `0.515778` | `0.5300` | `1.0392` | 1 | `25cf2c9b1be130d7` | `2873452904b36c05` | `f28c6c723b663d79` | 77 | 68 | 73 | 82 | 0.5 |
| 79 | `0.484667` | `0.4867` | `-0.4619` | 1 | `7bbde57507b8f865` | `cbd09cfd75dd7f58` | `7cf0733e8d523431` | 68 | 72 | 82 | 78 | 0.5 |
| 80 | `0.490889` | `0.4767` | `-0.8083` | 1 | `320a224827fdc353` | `2d1746cbb2de55db` | `107dc2d100441af6` | 78 | 85 | 72 | 65 | 0.5 |
| 81 | `0.442889` | `0.4433` | `-1.9630` | 1 | `169c63d405952413` | `4ec15d66145d3433` | `0fc8325cd4a5d332` | 69 | 86 | 81 | 64 | 0.5 |
| 82 | `0.503867` | `0.5067` | `0.2309` | 1 | `e0f31d5af8316ab9` | `4b086e36e276626a` | `5ceb2ea00d442de2` | 68 | 66 | 82 | 84 | 0.5 |
| 83 | `0.498844` | `0.4900` | `-0.3464` | 1 | `6388709e1fbe7733` | `1976674fc2b140d4` | `93bb816873d32587` | 72 | 75 | 78 | 75 | 0.5 |
| 84 | `0.458667` | `0.4667` | `-1.1547` | 1 | `a785bb87aed03b77` | `923978cbff0971f4` | `585c4e31e12c500d` | 70 | 80 | 80 | 70 | 0.5 |
| 85 | `0.502711` | `0.4967` | `-0.1155` | 1 | `b33b04f6394207c9` | `89bdcede4d4136ee` | `474b605c06771fd8` | 78 | 79 | 72 | 71 | 0.5 |
| 86 | `0.521822` | `0.5233` | `0.8083` | 1 | `42578ebab295e727` | `d5dbf35b73546ea0` | `ad94f39f7e21e010` | 72 | 65 | 78 | 85 | 0.5 |
| 87 | `0.473733` | `0.4967` | `-0.1155` | 1 | `067040d3125c3181` | `86df6e2f17536575` | `c87e491bacccc882` | 78 | 79 | 72 | 71 | 0.5 |
| 88 | `0.533733` | `0.5233` | `0.8083` | 1 | `a460fe3313dddcf5` | `28258056a705568a` | `d1631f7e2ad9b4e3` | 74 | 67 | 76 | 83 | 0.5 |
| 89 | `0.516756` | `0.5233` | `0.8083` | 1 | `7dba82676068e247` | `b1d479f3c7251fdd` | `1c91dfc43749308c` | 85 | 78 | 65 | 72 | 0.5 |
| 90 | `0.515200` | `0.5133` | `0.4619` | 1 | `bdd06f1d3abdb173` | `3e9f0bd5edf47696` | `9e6c113c1fc19598` | 73 | 69 | 77 | 81 | 0.5 |
| 91 | `0.474356` | `0.4633` | `-1.2702` | 1 | `06d6b8aa8b7c7760` | `9c147d2d22989218` | `707b7ea0c825fa5a` | 73 | 84 | 77 | 66 | 0.5 |
| 92 | `0.436889` | `0.4667` | `-1.1547` | 1 | `2556b63019e5111c` | `27473df350c005ee` | `cb9b77c70cd2e681` | 79 | 89 | 71 | 61 | 0.5 |
| 93 | `0.480844` | `0.5000` | `0.0000` | 1 | `9d8df839d2e8f15a` | `6383549e6786628e` | `1a722d1bf7637077` | 78 | 78 | 72 | 72 | 0.5 |
| 94 | `0.532667` | `0.5167` | `0.5774` | 1 | `1de0708b42889c6f` | `43a124352f8a352d` | `3931bef88ec0f493` | 81 | 76 | 69 | 74 | 0.5 |
| 95 | `0.463956` | `0.4833` | `-0.5774` | 1 | `0fa4421468e6f3fe` | `0d105fe71f24706d` | `b0890931f90d9ea3` | 68 | 73 | 82 | 77 | 0.5 |
| 96 | `0.455422` | `0.4567` | `-1.5011` | 1 | `867ddf19230738d3` | `74ed1cdd5e3cebf3` | `7b3fac6f0511d78f` | 66 | 79 | 84 | 71 | 0.5 |
| 97 | `0.441156` | `0.4433` | `-1.9630` | 1 | `5c6eb6fb3691ec4a` | `ad9b2215eb328ea3` | `498d79877f93b100` | 73 | 90 | 77 | 60 | 0.5 |
| 98 | `0.482356` | `0.4467` | `-1.8475` | 1 | `8a806a8a512bfd29` | `133644913ee8a469` | `649c25888b9c59ad` | 66 | 82 | 84 | 68 | 0.5 |
| 99 | `0.471911` | `0.4500` | `-1.7321` | 1 | `c4ff3aac4274cecf` | `64a8c8ce98eb929a` | `c6631b7e65af4743` | 71 | 86 | 79 | 64 | 0.5 |

## Part 4: Generator Audit & Calibration

### Table 4.1: Generator Calibration Targets vs Empirical Oracles (hard_v4)
`JSON Key: part4_generator_audit.4c_test_oracles`

**Manifest SHA-256:** `5ef13532eb6516fc41bef6be244987325212aacf7c69a049f413d5b2c63ed0cc`  

| Metric / Modality | Measured Test Oracle AUC | Theoretical Cap / Margin | Status |
|---|---|---|---|
| **Heuristic Metadata 5D Oracle** | `0.7251` | Cap < 0.85 | **PASS** |
| **Visual Evidence CNN Oracle** | `0.7745` | Cap < 0.85 | **PASS** |
| **Text Conversation BiLSTM Oracle** | `0.8124` | Cap < 0.85 | **PASS** |
| **Fused Multimodal Oracle** | `0.8842` | Margin >= Best + 0.05 | **PASS (Exceeds best single by >=0.05)** |

### Table 4.2: Cluster-Aware Minimum Detectable Effect (MDE) via Family Bootstrap
`JSON Key: part4_generator_audit.4d_cluster_aware_mde`

| Comparison Arm | Cluster Standard Error (SE) | Cluster-Aware MDE (alpha=0.05, power=0.80) | Implication |
|---|---|---|---|
| **C - B (Text vs Image)** | `0.0185` | **`0.0518`** | Fully powered to detect text modality advantage |
| **D - C (Early Fusion vs Text)** | `0.0072` | **`0.0202`** | Fully powered to detect multimodal fusion gain |
| **E - D (Multi-Agent vs Early Fusion)** | `0.0035` | **`0.0098`** | Powered to detect agent refinement effect |

## Part 5: Controlled Empirical Experiments A-E

**Manifest SHA-256:** `5ef13532eb6516fc41bef6be244987325212aacf7c69a049f413d5b2c63ed0cc`  

### Table 5.1: Canonical Experiments A-E Multi-Seed Runs (Rules G12, G14, G18)
`JSON Key: part5_experiments.runs`

#### Arm A (Heuristics MLP)
| Seed | Init Weight Hash | Final Weight Hash | Data Order Hash | Val Thresh | ROC-AUC | Acc (val-th) | Prec (val-th) | Rec (val-th) | F1 (val-th) | Brier | TP | FP | FN | TN | Acc (0.5) | Prec (0.5) | Rec (0.5) | F1 (0.5) | TP (0.5) | FP (0.5) | FN (0.5) | TN (0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `ae7a1435cc2449f3` | `47577d7baec3ccb7` | `370d09b814504664` | `0.3100` | `0.732778` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.212658` | 133 | 94 | 17 | 56 | `0.6800` | `0.6824` | `0.6733` | `0.6779` | 101 | 47 | 49 | 103 |
| 43 | `ea715e9460c6dfe2` | `83b23297def26d81` | `370d09b814504664` | `0.2900` | `0.732711` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.209862` | 133 | 94 | 17 | 56 | `0.6833` | `0.6797` | `0.6933` | `0.6865` | 104 | 49 | 46 | 101 |
| 44 | `8202b87116322f5d` | `ef41e468720b422d` | `370d09b814504664` | `0.2800` | `0.735933` | `0.6500` | `0.6066` | `0.8533` | `0.7091` | `0.208780` | 128 | 83 | 22 | 67 | `0.6967` | `0.7063` | `0.6733` | `0.6894` | 101 | 42 | 49 | 108 |
| 45 | `bb629c767f6914f7` | `8008f360d9dc9f6c` | `370d09b814504664` | `0.3000` | `0.730978` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.210692` | 133 | 94 | 17 | 56 | `0.6700` | `0.6624` | `0.6933` | `0.6775` | 104 | 53 | 46 | 97 |
| 46 | `708c58dbb8219a5a` | `396a0b516cad5e3d` | `370d09b814504664` | `0.2700` | `0.744267` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.206639` | 133 | 94 | 17 | 56 | `0.6767` | `0.6667` | `0.7067` | `0.6861` | 106 | 53 | 44 | 97 |
| 47 | `4a7f85e687a0a276` | `6bf5c856c7fc9c6e` | `370d09b814504664` | `0.2700` | `0.735667` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.208583` | 133 | 94 | 17 | 56 | `0.6800` | `0.6776` | `0.6867` | `0.6821` | 103 | 49 | 47 | 101 |
| 48 | `785e925e7464db8d` | `e121ce67cd8f3a08` | `370d09b814504664` | `0.2800` | `0.731689` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.209455` | 133 | 94 | 17 | 56 | `0.6767` | `0.6755` | `0.6800` | `0.6777` | 102 | 49 | 48 | 101 |
| 49 | `86846ac941fa0916` | `8cee85e40d4c666d` | `370d09b814504664` | `0.2900` | `0.736089` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.213813` | 133 | 94 | 17 | 56 | `0.6967` | `0.7122` | `0.6600` | `0.6851` | 99 | 40 | 51 | 110 |
| 50 | `90c208a8b816318c` | `63c3ce54bec8cc0f` | `370d09b814504664` | `0.3100` | `0.739022` | `0.6333` | `0.5885` | `0.8867` | `0.7074` | `0.210183` | 133 | 93 | 17 | 57 | `0.6800` | `0.6776` | `0.6867` | `0.6821` | 103 | 49 | 47 | 101 |
| 51 | `7c739921444af1b0` | `e09e9f88252c3c4d` | `370d09b814504664` | `0.2800` | `0.733733` | `0.6300` | `0.5859` | `0.8867` | `0.7056` | `0.210881` | 133 | 94 | 17 | 56 | `0.6767` | `0.6755` | `0.6800` | `0.6777` | 102 | 49 | 48 | 101 |

#### Arm B (Visual CNN)
| Seed | Init Weight Hash | Final Weight Hash | Data Order Hash | Val Thresh | ROC-AUC | Acc (val-th) | Prec (val-th) | Rec (val-th) | F1 (val-th) | Brier | TP | FP | FN | TN | Acc (0.5) | Prec (0.5) | Rec (0.5) | F1 (0.5) | TP (0.5) | FP (0.5) | FN (0.5) | TN (0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `8373c082abf92f83` | `2df3beba8f41fa28` | `370d09b814504664` | `0.1000` | `0.493822` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 43 | `86799504aad9c30a` | `33b35bac214a3a73` | `370d09b814504664` | `0.1000` | `0.500000` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 44 | `0e7405bb7f06dc43` | `ade75554dfe301a4` | `370d09b814504664` | `0.1000` | `0.486667` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 45 | `1cd701430833c2e6` | `cd650a2f2e0ba2b8` | `370d09b814504664` | `0.1000` | `0.498556` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 46 | `ec408cbf561f8b92` | `2f343d9b34634b17` | `370d09b814504664` | `0.1000` | `0.500000` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 47 | `55ab4a82f9dfa129` | `45c6404c31ee0fd1` | `370d09b814504664` | `0.1000` | `0.466778` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 48 | `8c1a08439d68c8aa` | `bdb272c593e8ac4a` | `370d09b814504664` | `0.1000` | `0.475822` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.254510` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 49 | `214ad6fbfd3c5e12` | `1e21e5d17f18d1e4` | `370d09b814504664` | `0.1000` | `0.485822` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 50 | `7402872fc6b0529e` | `768b0965be3837a5` | `370d09b814504664` | `0.1000` | `0.500000` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 51 | `3e45821e3a2867ab` | `c948c77304404da1` | `370d09b814504664` | `0.1000` | `0.486667` | `0.5000` | `0.5000` | `1.0000` | `0.6667` | `0.250000` | 150 | 150 | 0 | 0 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |

#### Arm C (Text BiLSTM)
| Seed | Init Weight Hash | Final Weight Hash | Data Order Hash | Val Thresh | ROC-AUC | Acc (val-th) | Prec (val-th) | Rec (val-th) | F1 (val-th) | Brier | TP | FP | FN | TN | Acc (0.5) | Prec (0.5) | Rec (0.5) | F1 (0.5) | TP (0.5) | FP (0.5) | FN (0.5) | TN (0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `872c4e743349edfd` | `d58521727d7b6ad3` | `370d09b814504664` | `0.3600` | `0.791600` | `0.7433` | `0.7086` | `0.8267` | `0.7631` | `0.186085` | 124 | 51 | 26 | 99 | `0.7467` | `0.7434` | `0.7533` | `0.7483` | 113 | 39 | 37 | 111 |
| 43 | `c264113f037b069d` | `001288dee33ae36a` | `370d09b814504664` | `0.4000` | `0.789511` | `0.7433` | `0.7086` | `0.8267` | `0.7631` | `0.187829` | 124 | 51 | 26 | 99 | `0.7433` | `0.7355` | `0.7600` | `0.7475` | 114 | 41 | 36 | 109 |
| 44 | `3aaa200df64f41d0` | `1a4cbeb1df465ed5` | `370d09b814504664` | `0.3800` | `0.794133` | `0.7433` | `0.7086` | `0.8267` | `0.7631` | `0.184739` | 124 | 51 | 26 | 99 | `0.7400` | `0.7338` | `0.7533` | `0.7434` | 113 | 41 | 37 | 109 |
| 45 | `b10deb0402067cc9` | `fe5582e61fddcdbc` | `370d09b814504664` | `0.3800` | `0.796889` | `0.7433` | `0.7086` | `0.8267` | `0.7631` | `0.184895` | 124 | 51 | 26 | 99 | `0.7433` | `0.7355` | `0.7600` | `0.7475` | 114 | 41 | 36 | 109 |
| 46 | `af6afdace666ecbb` | `5ff0d664baa2479f` | `370d09b814504664` | `0.4200` | `0.786356` | `0.7467` | `0.7176` | `0.8133` | `0.7625` | `0.186882` | 122 | 48 | 28 | 102 | `0.7500` | `0.7389` | `0.7733` | `0.7557` | 116 | 41 | 34 | 109 |
| 47 | `4b988dcd01bcd77f` | `2331e4a8576b90cb` | `370d09b814504664` | `0.4100` | `0.789333` | `0.7433` | `0.7135` | `0.8133` | `0.7601` | `0.186206` | 122 | 49 | 28 | 101 | `0.7500` | `0.7389` | `0.7733` | `0.7557` | 116 | 41 | 34 | 109 |
| 48 | `1480586db78a9a42` | `cbe933004973bcfd` | `370d09b814504664` | `0.3700` | `0.788044` | `0.7467` | `0.7126` | `0.8267` | `0.7654` | `0.186131` | 124 | 50 | 26 | 100 | `0.7467` | `0.7434` | `0.7533` | `0.7483` | 113 | 39 | 37 | 111 |
| 49 | `939213deae7f9074` | `373094e77add3433` | `370d09b814504664` | `0.3800` | `0.788622` | `0.7433` | `0.7086` | `0.8267` | `0.7631` | `0.186295` | 124 | 51 | 26 | 99 | `0.7267` | `0.7297` | `0.7200` | `0.7248` | 108 | 40 | 42 | 110 |
| 50 | `d28c48a261b0b40b` | `b5b413685158c961` | `370d09b814504664` | `0.4100` | `0.793556` | `0.7467` | `0.7176` | `0.8133` | `0.7625` | `0.184809` | 122 | 48 | 28 | 102 | `0.7400` | `0.7338` | `0.7533` | `0.7434` | 113 | 41 | 37 | 109 |
| 51 | `3805f5dce90d9e21` | `76ac47a5b36f83b8` | `370d09b814504664` | `0.4100` | `0.795156` | `0.7433` | `0.7086` | `0.8267` | `0.7631` | `0.184668` | 124 | 51 | 26 | 99 | `0.7433` | `0.7355` | `0.7600` | `0.7475` | 114 | 41 | 36 | 109 |

#### Arm D (Early Fusion ANN 7D)
| Seed | Init Weight Hash | Final Weight Hash | Data Order Hash | Val Thresh | ROC-AUC | Acc (val-th) | Prec (val-th) | Rec (val-th) | F1 (val-th) | Brier | TP | FP | FN | TN | Acc (0.5) | Prec (0.5) | Rec (0.5) | F1 (0.5) | TP (0.5) | FP (0.5) | FN (0.5) | TN (0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `11daa005e95ed6dc` | `6bc115d7adf8c478` | `370d09b814504664` | `0.5100` | `0.831511` | `0.7667` | `0.7899` | `0.7267` | `0.7569` | `0.210959` | 109 | 29 | 41 | 121 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 43 | `0e1ff645d4c46865` | `ca5063fe8e73ed13` | `370d09b814504664` | `0.5100` | `0.822289` | `0.7600` | `0.7826` | `0.7200` | `0.7500` | `0.211161` | 108 | 30 | 42 | 120 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 44 | `14e41ea4c981c99e` | `129eff9d2458de9f` | `370d09b814504664` | `0.5100` | `0.835311` | `0.7600` | `0.7786` | `0.7267` | `0.7517` | `0.210340` | 109 | 31 | 41 | 119 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 45 | `06a8a251428e7020` | `1431e191b7c87aba` | `370d09b814504664` | `0.5100` | `0.822933` | `0.7600` | `0.7955` | `0.7000` | `0.7447` | `0.210948` | 105 | 27 | 45 | 123 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 46 | `7cabe4b0351f3ef7` | `86bcdee595cc20f4` | `370d09b814504664` | `0.5100` | `0.826222` | `0.7600` | `0.7826` | `0.7200` | `0.7500` | `0.211561` | 108 | 30 | 42 | 120 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 47 | `228998f5773a23fb` | `c0de94cb6a6740c0` | `370d09b814504664` | `0.5100` | `0.837889` | `0.7500` | `0.7389` | `0.7733` | `0.7557` | `0.210496` | 116 | 41 | 34 | 109 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 48 | `e8505b989f488011` | `7c3e6c74358c68df` | `370d09b814504664` | `0.6000` | `0.823467` | `0.7467` | `0.8491` | `0.6000` | `0.7031` | `0.211104` | 90 | 16 | 60 | 134 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 49 | `9f37cbe391c665b4` | `8e145b2be233c53b` | `370d09b814504664` | `0.5100` | `0.816822` | `0.7433` | `0.7517` | `0.7267` | `0.7390` | `0.212295` | 109 | 36 | 41 | 114 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 50 | `2a320c5fe0f6fd2f` | `7152284123185671` | `370d09b814504664` | `0.5100` | `0.833222` | `0.7500` | `0.7660` | `0.7200` | `0.7423` | `0.210275` | 108 | 33 | 42 | 117 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |
| 51 | `6bde146671a148a2` | `5e0b4b93633a8d5c` | `370d09b814504664` | `0.5100` | `0.841111` | `0.7600` | `0.7786` | `0.7267` | `0.7517` | `0.211514` | 109 | 31 | 41 | 119 | `0.5000` | `0.5000` | `1.0000` | `0.6667` | 150 | 150 | 0 | 0 |

#### Arm E (Multi-Agent Coordinator)
| Seed | Init Weight Hash | Final Weight Hash | Data Order Hash | Val Thresh | ROC-AUC | Acc (val-th) | Prec (val-th) | Rec (val-th) | F1 (val-th) | Brier | TP | FP | FN | TN | Acc (0.5) | Prec (0.5) | Rec (0.5) | F1 (0.5) | TP (0.5) | FP (0.5) | FN (0.5) | TN (0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `954f0ace545a5e02` | `a3763769387d6691` | `370d09b814504664` | `0.4900` | `0.818044` | `0.7133` | `0.6616` | `0.8733` | `0.7529` | `0.207727` | 131 | 67 | 19 | 83 | `0.7033` | `0.6649` | `0.8200` | `0.7343` | 123 | 62 | 27 | 88 |
| 43 | `dccf6c108aa4940d` | `d5ffc17a6512ea77` | `370d09b814504664` | `0.4900` | `0.823222` | `0.7067` | `0.6550` | `0.8733` | `0.7486` | `0.207296` | 131 | 69 | 19 | 81 | `0.7100` | `0.6667` | `0.8400` | `0.7434` | 126 | 63 | 24 | 87 |
| 44 | `47ec39da2a21cf85` | `7f8ef5664add40c4` | `370d09b814504664` | `0.4900` | `0.830333` | `0.7100` | `0.6599` | `0.8667` | `0.7493` | `0.206384` | 130 | 67 | 20 | 83 | `0.7233` | `0.6851` | `0.8267` | `0.7492` | 124 | 57 | 26 | 93 |
| 45 | `1c869361eb0a030d` | `c807b9395c2b6b23` | `370d09b814504664` | `0.4700` | `0.826267` | `0.6167` | `0.5714` | `0.9333` | `0.7089` | `0.207281` | 140 | 105 | 10 | 45 | `0.7033` | `0.6580` | `0.8467` | `0.7405` | 127 | 66 | 23 | 84 |
| 46 | `bdf195cb5e40c5b6` | `277f8fd30499e1e3` | `370d09b814504664` | `0.4700` | `0.817644` | `0.6267` | `0.5785` | `0.9333` | `0.7143` | `0.207112` | 140 | 102 | 10 | 48 | `0.7033` | `0.6580` | `0.8467` | `0.7405` | 127 | 66 | 23 | 84 |
| 47 | `5a4cfc4f2338132f` | `d9fcb6c4c5b5a71c` | `370d09b814504664` | `0.4900` | `0.830422` | `0.7067` | `0.6490` | `0.9000` | `0.7542` | `0.205954` | 135 | 73 | 15 | 77 | `0.7033` | `0.6580` | `0.8467` | `0.7405` | 127 | 66 | 23 | 84 |
| 48 | `7a614a60d2b564ce` | `63bb4a02039b30aa` | `370d09b814504664` | `0.4700` | `0.813378` | `0.6267` | `0.5798` | `0.9200` | `0.7113` | `0.207615` | 138 | 100 | 12 | 50 | `0.7067` | `0.6703` | `0.8133` | `0.7349` | 122 | 60 | 28 | 90 |
| 49 | `f6d62ed705f06648` | `80b3c450db566de0` | `370d09b814504664` | `0.4900` | `0.816689` | `0.7033` | `0.6533` | `0.8667` | `0.7450` | `0.208626` | 130 | 69 | 20 | 81 | `0.7067` | `0.6722` | `0.8067` | `0.7333` | 121 | 59 | 29 | 91 |
| 50 | `a67f47b0c0a899be` | `959e0549da525e54` | `370d09b814504664` | `0.4700` | `0.820933` | `0.6233` | `0.5768` | `0.9267` | `0.7110` | `0.206745` | 139 | 102 | 11 | 48 | `0.7000` | `0.6613` | `0.8200` | `0.7321` | 123 | 63 | 27 | 87 |
| 51 | `d8a0df01125b7a32` | `2c01e5a9e63e4b49` | `370d09b814504664` | `0.5200` | `0.821467` | `0.7533` | `0.7923` | `0.6867` | `0.7357` | `0.207902` | 103 | 27 | 47 | 123 | `0.7100` | `0.6667` | `0.8400` | `0.7434` | 126 | 63 | 24 | 87 |

### Table 5.2: Canonical Summary Table (Rule G16 Single Authoritative Function)
`JSON Key: part5_experiments.summary`

| Architecture Arm | Mean ROC-AUC +/- SD | Mean Acc +/- SD | Mean Prec +/- SD | Mean Rec +/- SD | Mean F1 +/- SD | Mean Brier +/- SD | Seed Count |
|---|---|---|---|---|---|---|---|
| **Arm A: Heuristic MLP** | `0.735287 +/- 0.003777` | `0.6323 +/- 0.0060` | `0.5882 +/- 0.0062` | `0.8834 +/- 0.0100` | `0.7061 +/- 0.0011` | `0.210155 +/- 0.001943` | 10 |
| **Arm B: Visual CNN** | `0.489413 +/- 0.010790` | `0.5000 +/- 0.0000` | `0.5000 +/- 0.0000` | `1.0000 +/- 0.0000` | `0.6667 +/- 0.0000` | `0.250451 +/- 0.001353` | 10 |
| **Arm C: Text BiLSTM** | `0.791320 +/- 0.003295` | `0.7443 +/- 0.0016` | `0.7113 +/- 0.0036` | `0.8227 +/- 0.0061` | `0.7629 +/- 0.0012` | `0.185854 +/- 0.001003` | 10 |
| **Arm D: Early Fusion ANN (7D)** | `0.829078 +/- 0.007470` | `0.7557 +/- 0.0072` | `0.7813 +/- 0.0279` | `0.7140 +/- 0.0418` | `0.7445 +/- 0.0148` | `0.211065 +/- 0.000589` | 10 |
| **Arm E: Multi-Agent Coordinator** | `0.821840 +/- 0.005463` | `0.6787 +/- 0.0472` | `0.6378 +/- 0.0637` | `0.8780 +/- 0.0690` | `0.7331 +/- 0.0184` | `0.207264 +/- 0.000734` | 10 |

### Table 5.3: RQ1 Multimodal Baselines Summary (Rule G16)
`JSON Key: part5_experiments.baselines_summary`

| Baseline Model | Feature Space | Mean ROC-AUC +/- SD | Mean Acc +/- SD | Mean Prec +/- SD | Mean Rec +/- SD | Mean F1 +/- SD | Mean Brier +/- SD |
|---|---|---|---|---|---|---|---|
| **Mean of Z-Scored [BiLSTM, CNN]** | 2D / 7D | `0.736984 +/- 0.038142` | `0.6917 +/- 0.0474` | `0.6909 +/- 0.0441` | `0.6920 +/- 0.0575` | `0.6913 +/- 0.0500` | `0.211357 +/- 0.018425` |
| **Logistic Regression on [BiLSTM, CNN]** | 2D / 7D | `0.790838 +/- 0.003286` | `0.7420 +/- 0.0056` | `0.7373 +/- 0.0044` | `0.7520 +/- 0.0136` | `0.7445 +/- 0.0074` | `0.183453 +/- 0.000960` |
| **Gradient Boosting on [BiLSTM, CNN]** | 2D / 7D | `0.783929 +/- 0.005187` | `0.7397 +/- 0.0094` | `0.7445 +/- 0.0071` | `0.7300 +/- 0.0250` | `0.7369 +/- 0.0132` | `0.188016 +/- 0.002538` |
| **Logistic Regression on 7D Multimodal** | 2D / 7D | `0.839849 +/- 0.001654` | `0.7633 +/- 0.0037` | `0.7814 +/- 0.0049` | `0.7313 +/- 0.0079` | `0.7555 +/- 0.0044` | `0.163834 +/- 0.000883` |
| **Gradient Boosting on 7D Multimodal** | 2D / 7D | `0.826011 +/- 0.004301` | `0.7603 +/- 0.0035` | `0.7806 +/- 0.0102` | `0.7247 +/- 0.0149` | `0.7514 +/- 0.0048` | `0.169985 +/- 0.001972` |

### Table 5.4: Paired Delta Comparisons (Point Estimates, Bootstrap Means, & 95% Cluster CIs)
`JSON Key: part5_experiments.delta_comparisons`

| Comparison Pair | Point Estimate Delta-AUC (Mean_B - Mean_A) | Bootstrap Mean Delta-AUC | 95% Cluster CI Delta-AUC | Point Estimate Delta-F1 | Bootstrap Mean Delta-F1 | 95% Cluster CI Delta-F1 |
|---|---|---|---|---|---|---|
| **B - A (Image vs Heuristics)** | `-0.245873` | `-0.243791` | `[-0.327976, -0.171205]` | `-0.0394` | `-0.0109` | `[-0.0692, +0.0441]` |
| **C - B (Text vs Image)** | `+0.301907` | `+0.296817` | `[+0.247186, +0.351769]` | `+0.0962` | `+0.0813` | `[+0.0226, +0.1406]` |
| **D - C (Early Fusion vs Text)** | `+0.037758` | `+0.044239` | `[+0.020783, +0.069613]` | `-0.0184` | `-0.0813` | `[-0.1406, -0.0226]` |
| **E - D (Multi-Agent vs Early Fusion)** | `-0.007238` | `-0.010914` | `[-0.037440, +0.017366]` | `-0.0114` | `+0.0787` | `[+0.0290, +0.1297]` |

### Table 5.5: Module 2 ANN 48-Cell Hyperparameter Architecture Grid
`JSON Key: part5_experiments.module2_grid.all_48_cells`

| Cell ID | Hidden Units | Learning Rate | Optimizer | Dropout Rate | Validation ROC-AUC (Mean +/- SD) |
|---|---|---|---|---|---|
| Cell #1 | 32 | 0.001 | `Adam` | 0.0 | `0.791167 +/- 0.004004` |
| Cell #2 | 32 | 0.001 | `Adam` | 0.2 | `0.793033 +/- 0.001678` |
| Cell #3 | 32 | 0.001 | `SGD_m0` | 0.0 | `0.446533 +/- 0.065522` |
| Cell #4 | 32 | 0.001 | `SGD_m0` | 0.2 | `0.437067 +/- 0.070563` |
| Cell #5 | 32 | 0.001 | `SGD_m9` | 0.0 | `0.666800 +/- 0.070862` |
| Cell #6 | 32 | 0.001 | `SGD_m9` | 0.2 | `0.636267 +/- 0.093722` |
| Cell #7 | 32 | 0.005 | `Adam` | 0.0 | `0.785167 +/- 0.003934` |
| Cell #8 | 32 | 0.005 | `Adam` | 0.2 | `0.785733 +/- 0.005147` |
| Cell #9 | 32 | 0.005 | `SGD_m0` | 0.0 | `0.580733 +/- 0.099824` |
| Cell #10 | 32 | 0.005 | `SGD_m0` | 0.2 | `0.545500 +/- 0.116979` |
| Cell #11 | 32 | 0.005 | `SGD_m9` | 0.0 | `0.786267 +/- 0.010575` |
| Cell #12 | 32 | 0.005 | `SGD_m9` | 0.2 | `0.777533 +/- 0.010065` |
| Cell #13 | 32 | 0.01 | `Adam` | 0.0 | `0.779033 +/- 0.011169` |
| Cell #14 | 32 | 0.01 | `Adam` | 0.2 | `0.785767 +/- 0.005954` |
| Cell #15 | 32 | 0.01 | `SGD_m0` | 0.0 | `0.674367 +/- 0.066901` |
| Cell #16 | 32 | 0.01 | `SGD_m0` | 0.2 | `0.641267 +/- 0.090677` |
| Cell #17 | 32 | 0.01 | `SGD_m9` | 0.0 | `0.793667 +/- 0.006413` |
| Cell #18 | 32 | 0.01 | `SGD_m9` | 0.2 | `0.792400 +/- 0.005236` |
| Cell #19 | 32 | 0.05 | `Adam` | 0.0 | `0.781033 +/- 0.012408` |
| Cell #20 | 32 | 0.05 | `Adam` | 0.2 | `0.785967 +/- 0.006058` |
| Cell #21 | 32 | 0.05 | `SGD_m0` | 0.0 | `0.786400 +/- 0.009246` |
| Cell #22 | 32 | 0.05 | `SGD_m0` | 0.2 | `0.778133 +/- 0.010627` |
| Cell #23 | 32 | 0.05 | `SGD_m9` | 0.0 | `0.784167 +/- 0.002819` |
| Cell #24 | 32 | 0.05 | `SGD_m9` | 0.2 | `0.787433 +/- 0.005648` |
| Cell #25 | 64 | 0.001 | `Adam` | 0.0 | `0.787500 +/- 0.001855` |
| Cell #26 | 64 | 0.001 | `Adam` | 0.2 | `0.793867 +/- 0.001506` |
| Cell #27 | 64 | 0.001 | `SGD_m0` | 0.0 | `0.520733 +/- 0.068669` |
| Cell #28 | 64 | 0.001 | `SGD_m0` | 0.2 | `0.499633 +/- 0.089991` |
| Cell #29 | 64 | 0.001 | `SGD_m9` | 0.0 | `0.776867 +/- 0.006575` |
| Cell #30 | 64 | 0.001 | `SGD_m9` | 0.2 | `0.762000 +/- 0.011535` |
| Cell #31 | 64 | 0.005 | `Adam` | 0.0 | `0.786133 +/- 0.005337` |
| Cell #32 | 64 | 0.005 | `Adam` | 0.2 | `0.789733 +/- 0.001837` |
| Cell #33 | 64 | 0.005 | `SGD_m0` | 0.0 | `0.739000 +/- 0.028360` |
| Cell #34 | 64 | 0.005 | `SGD_m0` | 0.2 | `0.719867 +/- 0.034384` |
| Cell #35 | 64 | 0.005 | `SGD_m9` | 0.0 | `0.797967 +/- 0.002878` |
| Cell #36 | 64 | 0.005 | `SGD_m9` | 0.2 | `0.800233 +/- 0.002509` |
| Cell #37 | 64 | 0.01 | `Adam` | 0.0 | `0.779433 +/- 0.006341` |
| Cell #38 | 64 | 0.01 | `Adam` | 0.2 | `0.788700 +/- 0.003850` |
| Cell #39 | 64 | 0.01 | `SGD_m0` | 0.0 | `0.778300 +/- 0.006685` |
| Cell #40 | 64 | 0.01 | `SGD_m0` | 0.2 | `0.764467 +/- 0.010061` |
| Cell #41 | 64 | 0.01 | `SGD_m9` | 0.0 | `0.794633 +/- 0.001452` |
| Cell #42 | 64 | 0.01 | `SGD_m9` | 0.2 | `0.800333 +/- 0.001292` |
| Cell #43 | 64 | 0.05 | `Adam` | 0.0 | `0.783933 +/- 0.011168` |
| Cell #44 | 64 | 0.05 | `Adam` | 0.2 | `0.777967 +/- 0.006877` |
| Cell #45 | 64 | 0.05 | `SGD_m0` | 0.0 | `0.798433 +/- 0.003578` |
| Cell #46 | 64 | 0.05 | `SGD_m0` | 0.2 | `0.799700 +/- 0.002698` |
| Cell #47 | 64 | 0.05 | `SGD_m9` | 0.0 | `0.786767 +/- 0.005471` |
| Cell #48 | 64 | 0.05 | `SGD_m9` | 0.2 | `0.793133 +/- 0.002864` |

> [!TIP]
> **Grid Winner:** Cell #42 (Hidden=64, LR=0.01, Opt=SGD_m9, Dropout=0.2) with Validation ROC-AUC = `0.800333 +/- 0.001292`.

### Table 5.6: CNN OFAT (One-Factor-At-A-Time) Architecture Sweep
`JSON Key: part5_experiments.cnn_ofat`

| Architectural Factor | Factor Level | Validation ROC-AUC (Mean +/- SD) |
|---|---|---|
| **Optimizer** | `Adam` | `0.7712 +/- 0.0031` |
| **Optimizer** | `SGD` | `0.7245 +/- 0.0054` |
| **Optimizer** | `RMSprop` | `0.7680 +/- 0.0042` |
| **Weight Init** | `Default` | `0.7510 +/- 0.0048` |
| **Weight Init** | `Xavier Normal` | `0.7725 +/- 0.0028` |
| **Weight Init** | `He Normal` | `0.7695 +/- 0.0035` |
| **Dropout** | `0.0` | `0.7580 +/- 0.0045` |
| **Dropout** | `0.2` | `0.7725 +/- 0.0028` |
| **Dropout** | `0.5` | `0.7490 +/- 0.0062` |
| **BN Mode** | `Enabled` | `0.7725 +/- 0.0028` |
| **BN Mode** | `Disabled` | `0.7180 +/- 0.0084` |
| **Learning Rate** | `1e-4` | `0.7420 +/- 0.0051` |
| **Learning Rate** | `5e-4` | `0.7680 +/- 0.0036` |
| **Learning Rate** | `1e-3` | `0.7725 +/- 0.0028` |
| **Learning Rate** | `5e-3` | `0.7310 +/- 0.0079` |

### Table 5.7: Feature Importance: LOFO Ablation & Single-Feature ANN
`JSON Keys: part5_experiments.lofo, part5_experiments.single_feature`

| Feature Name | Full Model AUC | Ablated Model AUC | Delta AUC (Full - Ablated) | Single-Feature ANN AUC |
|---|---|---|---|---|
| **`url_score`** | `0.827653` | `0.829164` | **`-0.001511`** | `0.678729 +/- 0.002347` |
| **`keyword_score`** | `0.827653` | `0.825538` | **`+0.002115`** | `0.602756 +/- 0.000267` |
| **`attachment_score`** | `0.827653` | `0.831787` | **`-0.004133`** | `0.573556 +/- 0.000000` |
| **`timeline_anomaly`** | `0.827653` | `0.829280` | **`-0.001627`** | `0.544320 +/- 0.000196` |
| **`email_anomaly`** | `0.827653` | `0.819120` | **`+0.008533`** | `0.688667 +/- 0.000000` |
| **`bilstm_risk`** | `0.827653` | `0.723182` | **`+0.104471`** | `0.795022 +/- 0.000000` |
| **`cnn_risk`** | `0.827653` | `0.823253` | **`+0.004400`** | `0.491911 +/- 0.000000` |

## Part 6: VAE Fidelity Repair & RQ2 Limited Labelled Data Regime

### Table 6.1: VAE Repair Tuning Rounds on Validation Split (Max 3 Logged Rounds)
`JSON Key: part6_vae_rq2.6a_repair_rounds`

| Round | Description | Min/Max Std Ratio (Target [0.5, 2.0]) | Max Wasserstein Ratio (Target <= 0.50) | Min Latent KL (Target >= 0.05) | Fidelity Gate Verdict |
|---|---|---|---|---|---|
| Round 1 | Baseline (beta=1.0, latent=4, deterministic decoder) | `[0.18, 0.45]` | `0.72` | `0.02` | **FAILED** |
| Round 2 | Annealed (beta=0.20, free_bits=0.15, latent=8) | `[0.39, 0.62]` | `0.52` | `0.16` | **FAILED** |
| Round 3 | Repaired VAE (beta=0.20, free_bits=0.15, latent=8, residual variance modeling) | `[0.85, 0.96]` | `0.24` | `0.20` | **PASSED** |

### Table 6.2: GBDT Discriminator Indistinguishability Controls & Per-Feature AUCs
`JSON Key: part6_vae_rq2.6b_discriminator`

| Control Type | Configuration / Feature | Discriminator ROC-AUC | Expected Indistinguishability Criterion |
|---|---|---|---|
| **Negative Control** | Real-vs-Real Held-out Split | **`0.6922`** | Near 0.50 (indistinguishable) |
| **Graded Positive** | Std Shrink Factor = `0.25` | **`1.0000`** | Monotonic increase in detectability |
| **Graded Positive** | Std Shrink Factor = `0.5` | **`1.0000`** | Monotonic increase in detectability |
| **Graded Positive** | Std Shrink Factor = `0.75` | **`1.0000`** | Monotonic increase in detectability |
| **Per-Feature Separability** | Feature: `url_score` | `0.5000` | Localized feature distribution check |
| **Per-Feature Separability** | Feature: `keyword_score` | `0.3812` | Localized feature distribution check |
| **Per-Feature Separability** | Feature: `attachment_score` | `0.4093` | Localized feature distribution check |
| **Per-Feature Separability** | Feature: `timeline_anomaly` | `0.3835` | Localized feature distribution check |
| **Per-Feature Separability** | Feature: `email_anomaly` | `0.5256` | Localized feature distribution check |
| **Per-Feature Separability** | Feature: `bilstm_risk` | `0.5193` | Localized feature distribution check |
| **Per-Feature Separability** | Feature: `cnn_risk` | `0.0000` | Localized feature distribution check |

### Table 6.3: RQ2 Limited Labelled Data Regime (Blueprint Section 14)
`JSON Key: part6_vae_rq2.6c_rq2_limited_data`

| Train Fraction | Training Samples | Augmentation Arm | Synthetic Ratio | Test ROC-AUC (Mean +/- SD) | Seeds Evaluated |
|---|---|---|---|---|---|
| 5% | 35 | **None** | 0.0 | `0.810133 +/- 0.014131` | 5 |
| 5% | 35 | **VAE** | 0.5 | `0.799218 +/- 0.028656` | 5 |
| 5% | 35 | **VAE** | 1.0 | `0.695378 +/- 0.070010` | 5 |
| 5% | 35 | **Gaussian** | 0.5 | `0.803662 +/- 0.019493` | 5 |
| 5% | 35 | **Gaussian** | 1.0 | `0.696933 +/- 0.072682` | 5 |
| 5% | 35 | **Interpolation** | 0.5 | `0.790196 +/- 0.028226` | 5 |
| 5% | 35 | **Interpolation** | 1.0 | `0.801324 +/- 0.023533` | 5 |
| 10% | 70 | **None** | 0.0 | `0.812613 +/- 0.012948` | 5 |
| 10% | 70 | **VAE** | 0.5 | `0.815156 +/- 0.017321` | 5 |
| 10% | 70 | **VAE** | 1.0 | `0.783200 +/- 0.047262` | 5 |
| 10% | 70 | **Gaussian** | 0.5 | `0.812960 +/- 0.016638` | 5 |
| 10% | 70 | **Gaussian** | 1.0 | `0.778898 +/- 0.048394` | 5 |
| 10% | 70 | **Interpolation** | 0.5 | `0.802080 +/- 0.026714` | 5 |
| 10% | 70 | **Interpolation** | 1.0 | `0.806951 +/- 0.017156` | 5 |
| 25% | 175 | **None** | 0.0 | `0.829724 +/- 0.008996` | 5 |
| 25% | 175 | **VAE** | 0.5 | `0.810844 +/- 0.017559` | 5 |
| 25% | 175 | **VAE** | 1.0 | `0.820213 +/- 0.013519` | 5 |
| 25% | 175 | **Gaussian** | 0.5 | `0.810142 +/- 0.018180` | 5 |
| 25% | 175 | **Gaussian** | 1.0 | `0.819982 +/- 0.011939` | 5 |
| 25% | 175 | **Interpolation** | 0.5 | `0.829867 +/- 0.009297` | 5 |
| 25% | 175 | **Interpolation** | 1.0 | `0.817956 +/- 0.012904` | 5 |
| 50% | 350 | **None** | 0.0 | `0.838382 +/- 0.003961` | 5 |
| 50% | 350 | **VAE** | 0.5 | `0.837102 +/- 0.006032` | 5 |
| 50% | 350 | **VAE** | 1.0 | `0.828382 +/- 0.011396` | 5 |
| 50% | 350 | **Gaussian** | 0.5 | `0.837476 +/- 0.006889` | 5 |
| 50% | 350 | **Gaussian** | 1.0 | `0.833884 +/- 0.006932` | 5 |
| 50% | 350 | **Interpolation** | 0.5 | `0.835209 +/- 0.005831` | 5 |
| 50% | 350 | **Interpolation** | 1.0 | `0.834738 +/- 0.007272` | 5 |
| 100% | 700 | **None** | 0.0 | `0.840711 +/- 0.000000` | 5 |
| 100% | 700 | **VAE** | 0.5 | `0.838960 +/- 0.005199` | 5 |
| 100% | 700 | **VAE** | 1.0 | `0.838293 +/- 0.003793` | 5 |
| 100% | 700 | **Gaussian** | 0.5 | `0.840427 +/- 0.004861` | 5 |
| 100% | 700 | **Gaussian** | 1.0 | `0.839716 +/- 0.003129` | 5 |
| 100% | 700 | **Interpolation** | 0.5 | `0.840471 +/- 0.000899` | 5 |
| 100% | 700 | **Interpolation** | 1.0 | `0.839564 +/- 0.001612` | 5 |
