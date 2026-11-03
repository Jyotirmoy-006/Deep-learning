# ForenSightX — Iteration 7: Gate on the Failed Null, Fix Grouped Splits, Redo Experiments

> **MANDATORY BANNER:**  
> **SYNTHETIC BENCHMARK - NOT REAL-WORLD EVIDENCE**  
> All experimental results in this report are derived from synthetic benchmark data generated under controlled experimental protocols. They demonstrate algorithmic properties, convergence behavior, and statistical validation limits, but do not represent empirical real-world cybersecurity performance.

**Authoritative Spec:** `ForenSightX_Project_Architecture.docx` (Section 13)  
**Secondary Spec:** `Green_and_Gold_Modern_Cybersecurity_Presentation.pdf`  
*Spec Precedence: The .docx specification is authoritative and strictly governs architecture. All numerical metrics in both documents are illustrative.*  
**Execution Environment:** Windows, Python 3.11.9, PyTorch 2.14.0+cpu, Deterministic Execution  
**Current Baseline Git Commit:** `c71fd06`  
**Git Tags:** `iter1-snapshot` (`06f391f`), `iter4` (`5d08356`), `iter5` (`0bfb4f1`), `iter6` (`c71fd06`)  
**Scope Freeze:** FROZEN (No new agents, endpoints, or UI dashboards. Scope modifications strictly limited to grouped split/OOF code, ANN selection, experiment definitions, and VAE evaluation).  
**Results Artifact SHA-256:** `24949ad2045c4b0f167703defaca3a5322cf18d885482bc511aaa63bce30f344` (`results/experiment_results_v7.json`)  
**Grouped Hard Split Manifest SHA-256:** `b22fb35755469044a262739f8254b60de4fd1c0116af7cf744a4995330698f6a` (`data/split_manifest_hard_v2.json`)  

---

## Executive Summary & Global Rule Adherence

Iteration 7 establishes complete statistical control over the benchmark pipeline, fixes template family leakage via grouped splits, verifies out-of-fold generalization, enforces three rigorous validation gates, and reports canonical model evaluations with cluster bootstrap confidence intervals:

1. **Rule G1 (Programmatic Tables):** Every numeric table in this report is emitted programmatically from `results/experiment_results_v7.json`. The exact JSON path and key are printed beneath each table header.
2. **Rule G2 (Recomputed Summary Statistics):** Every summary statistic (mean, standard deviation, percentiles, minimum, maximum, and histogram counts) is verified against the underlying per-row values.
3. **Rule G3 (Explicit Device Placement):** All PyTorch tensors and modules are placed via explicit `.to(DEVICE)` calls.
4. **Rule G4 (Assumption Marking):** All unstated architectural defaults are explicitly marked with `[ASSUMPTION-xx]`.
5. **Rule G5 (Write-Once Results JSON):** `results/experiment_results_v7.json` is write-once. Its SHA-256 (`24949ad2045c4b0f167703defaca3a5322cf18d885482bc511aaa63bce30f344`) is permanently recorded.
6. **Rule G6 & G10 (Hypothesis Labeling):** Any causal claim is explicitly tagged `[HYPOTHESIS]` unless citing an experimental run ID.
7. **Rule G7 (Full Table Inclusion):** Every table named in Parts 1 through 5 is included in full Markdown with no truncated rows or summarizing.
8. **Rule G8 (Validation Gates Enforcement):**
   - **Gate G-DATA:** Evaluated in Part 1. `hard_v1` partition shared 9 template families between train and val; the gate triggered partition regeneration in Part 2.
   - **Gate G-OOF:** Evaluated in Part 2. GroupKFold OOF base models achieved $|\text{OOF AUC} - \text{Val AUC}| \le 0.05$ (BiLSTM: $0.000341$, CNN: $0.038804$). **PASSED**.
   - **Gate G-PERM:** Evaluated in Part 3. All three nested null controls (N1 i.i.d. Gaussian, N2 frozen OOF with permuted labels, N3 full retrained base model pipeline) satisfied $\text{std} \le 2\sigma_0$ ($0.066791$) with zero seeds outside $4\sigma_0$ ($[0.3664, 0.6336]$). **PASSED**.
9. **Rule G9 (Round-Trip Test Enforcement):** Unit tests in `tests/test_iteration7_roundtrip.py` directly parse the Markdown tables in this document, recomputing all means, standard deviations, percentiles (`numpy.percentile`), histogram counts, and confidence intervals from the text rows.
10. **Rule G11 (Immutability):** No committed results or manifests were modified. New results and manifest files are versioned with unique write-once paths.

---

## Validation Gates Status Matrix

| Gate Identifier | Target Evaluation | Specification / Criterion | Measured Value / Finding | Gate Status |
|---|---|---|---|---|
| **GATE G-DATA** | Part 1 Data Structure Leakage | Zero shared template families or records between Train and Val | `hard_v1` shared 9 template families and 68 text records | **TRIGGERED REGENERATION** (Regenerated in Part 2) |
| **GATE G-OOF** | Part 2 Base Model Generalization | $|\text{OOF AUC} - \text{Val AUC}| \le 0.05$ (config tolerance) | BiLSTM: $0.000341$, CNN: $0.038804$ | **PASSED** |
| **GATE G-PERM** | Part 3 Nested Null Controls | $\text{Empirical Std} \le 2\sigma_0$ ($0.066791$) & Zero seeds $> 4\sigma_0$ | N1: Std $0.035185$, N2: Std $0.031918$, N3: Std $0.031705$ | **PASSED** |

> [!NOTE]
> `[HYPOTHESIS]` The failure of the unconstrained permutation control in Iteration 4 and Iteration 5 was caused by label leakage across shared template families between train and validation partitions. When template families are partitioned into mutually disjoint sets (6 train, 3 validation, 3 test) and base models are evaluated using GroupKFold OOF, empirical null dispersion strictly aligns with the theoretical Mann-Whitney U standard error bound $\sigma_0 = \sqrt{(n_0 + n_1 + 1)/(12 n_0 n_1)} \approx 0.033396$.

---

## Part 0: Disclosures and Parameter Audit

### Table 0.1: Commit History for `chief executive` VIP Display Name Filter
`JSON Key: part0_disclosure.0c.commit_history_table`

| Commit | Tag / Phase | String Present | Status |
|---|---|---|---|
| `06f391f` | `iter1-snapshot` | `False` | Base initial state |
| `a985ee7` | `iteration-2` | `True` | Added 'chief executive' to VIP display name check |
| `5824a72` | `iteration-3` | `True` | Preserved |
| `5d08356` | `iter4` | `True` | Preserved |
| `0bfb4f1` | `iter5` | `True` | Preserved |
| `c71fd06` | `iter6` | `True` | Preserved |

### Table 0.2: Audited Parameter Breakdown for Buggy VAE (135 Parameters)
`JSON Key: part0_disclosure.0d.layer_breakdown`

| Layer Name | Parameter Shape | Number of Parameters |
|---|---|---|
| `encoder.0.weight` | `[4, 9]` | 36 |
| `encoder.0.bias` | `[4]` | 4 |
| `encoder.2.weight` | `[2, 4]` | 8 |
| `encoder.2.bias` | `[2]` | 2 |
| `fc_mu.weight` | `[4, 2]` | 8 |
| `fc_mu.bias` | `[4]` | 4 |
| `fc_logvar.weight` | `[4, 2]` | 8 |
| `fc_logvar.bias` | `[4]` | 4 |
| `decoder.0.weight` | `[2, 6]` | 12 |
| `decoder.0.bias` | `[2]` | 2 |
| `decoder.2.weight` | `[4, 2]` | 8 |
| `decoder.2.bias` | `[4]` | 4 |
| `decoder.4.weight` | `[7, 4]` | 28 |
| `decoder.4.bias` | `[7]` | 7 |
| **Total Parameters** | | **135** |

## Part 1: Data Structure Audit (hard_v1)

### Table 1.1: hard_v1 Partition Structural Inventory
`JSON Key: part1_data_audit.1a_partition_table`

| Split | Total Cases | Unique Texts | Unique Images | Unique Families | Mean Text Reuse | Max Text Reuse | Mean Family Reuse | Max Family Reuse | Flipped Fraction |
|---|---|---|---|---|---|---|---|---|---|
| **Train** | 700 | 262 | 700 | 9 | 2.67 | 14 | 77.78 | 98 | 0.0643 |
| **Val** | 200 | 109 | 200 | 9 | 1.83 | 6 | 22.22 | 28 | 0.0600 |
| **Test** | 300 | 65 | 300 | 3 | 4.62 | 18 | 100.00 | 115 | 0.0567 |

### Table 1.2: Design Effect and Effective Sample Size (n_eff)
`JSON Key: part1_data_audit.1b_design_effect`

| Cluster Level | Bootstrap Variance | DEFF Ratio (Var_cluster / Var_case) | Total Cases (N) | Effective Sample Size (n_eff) |
|---|---|---|---|---|
| Case-level (i.i.d.) | 0.000307 | 1.0000 | 200 | 200.0 |
| Record Cluster | 0.000300 | 0.9773 | 200 | 204.6 |
| Family Cluster | 0.000430 | 1.3993 | 200 | 142.9 |

### Table 1.3: Permutation Function Feature Independence Audit
`JSON Key: part1_data_audit.1c_permutation_audit.feature_dependency_table`

| Feature Index | Feature Name | Mutual Information with Permuted Label | Chi-Square Statistic | p-Value |
|---|---|---|---|---|
| 0 | `url_score` | 0.001750 | 1.8439 | 0.3977 |
| 1 | `keyword_score` | 0.001869 | 2.4462 | 0.6543 |
| 2 | `attachment_score` | 0.002780 | 0.7791 | 0.6774 |
| 3 | `timeline_anomaly` | 0.000285 | 0.2046 | 0.6510 |
| 4 | `email_anomaly` | 0.003022 | 4.2122 | 0.2394 |

## Part 2: Grouped Splits and OOF Base Models (hard_v2)

### Table 2.1: GroupKFold OOF vs Val vs Test Performance
`JSON Key: part2_grouped_splits.2b_base_models`

| Base Model | OOF Train AUC | Val AUC | Test AUC | OOF Train Acc | Val Acc | Test Acc | Train c0 Std | Train c1 Std | Val c0 Std | Val c1 Std | Test c0 Std | Test c1 Std | \|OOF - Val\| AUC | Gate G-OOF Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ConversationBiLSTM** | 0.598617 | 0.598958 | 0.710884 | 0.5886 | 0.5650 | 0.5367 | 0.4825 | 0.4838 | 0.2624 | 0.0000 | 0.1316 | 0.0000 | 0.000341 | **PASS** |
| **VisualEvidenceCNN** | 0.964485 | 0.925681 | 0.960740 | 0.9557 | 0.9400 | 0.9600 | 0.1511 | 0.1662 | 0.2172 | 0.2091 | 0.2035 | 0.1431 | 0.038804 | **PASS** |

### Table 2.2: Oracle Performance Ceilings (Family-Disjoint)
`JSON Key: part2_grouped_splits.2c_oracle_ceilings`

| Oracle Estimator | Validation ROC-AUC | Test ROC-AUC | Label Noise Scope |
|---|---|---|---|
| Logistic Regression Oracle | 0.985377 | 0.993464 | Training Split Only (5% rate) |
| Gradient Boosting (GBDT) Oracle | 0.992087 | 0.993242 | Training Split Only (5% rate) |

## Part 3: Nested Null Controls (100 Seeds Each)

### Table 3.N1: N1: i.i.d. Gaussian Features Through ANN Summary and 10-Bin Histogram
`JSON Key: part3_nested_null_controls.N1`

| Metric | Value | Reference / Bound | Pass / Fail |
|---|---|---|---|
| Theoretical Sigma_0 | 0.033396 | Mann-Whitney U Null Std | Reference |
| Max Allowed Empirical Std | 0.066791 | 2 * Sigma_0 Bound | Reference |
| Empirical Mean AUC | 0.496980 | Expected ~0.5000 | Within Bound |
| Empirical Std AUC | 0.035185 | Must be <= 2 * Sigma_0 | **PASS** |
| Percentile 2.5% | 0.435688 | Empirical 95% Interval Low | - |
| Percentile 97.5% | 0.560866 | Empirical 95% Interval High | - |
| Min AUC Observed | 0.407281 | 4-Sigma Low: 0.366418 | **PASS** |
| Max AUC Observed | 0.569783 | 4-Sigma High: 0.633582 | **PASS** |
| Mean Accuracy | 0.4979 | Expected ~0.5000 | - |
| Mean Binomial z-score | -0.0739 | Expected ~0.0000 | - |
| Mean Selected Epoch | 4.83 | 1 to 15 epochs | - |
| **Overall Gate G-PERM Status** | **PASS** | Std <= 2*Sigma0 & No seed > 4*Sigma0 | **PASS** |

#### 10-Bin Histogram for N1 (Recomputed per Rule G9)
`JSON Key: part3_nested_null_controls.N1.histogram`

| Bin Index | Range Min | Range Max | Count |
|---|---|---|---|
| 0 | 0.3500 | 0.3800 | 0 |
| 1 | 0.3800 | 0.4100 | 1 |
| 2 | 0.4100 | 0.4400 | 2 |
| 3 | 0.4400 | 0.4700 | 21 |
| 4 | 0.4700 | 0.5000 | 28 |
| 5 | 0.5000 | 0.5300 | 28 |
| 6 | 0.5300 | 0.5600 | 16 |
| 7 | 0.5600 | 0.5900 | 4 |
| 8 | 0.5900 | 0.6200 | 0 |
| 9 | 0.6200 | 0.6500 | 0 |

#### Per-Seed Raw Metrics for N1 (100 Seeds)
`JSON Key: part3_nested_null_controls.N1.per_seed_rows`

| Seed | ROC-AUC | Accuracy | Binomial z-Score | Selected Epoch |
|---|---|---|---|---|
| 0 | 0.535248 | 0.5200 | 0.6928 | 1 |
| 1 | 0.489298 | 0.4833 | -0.5774 | 5 |
| 2 | 0.455257 | 0.4533 | -1.6166 | 2 |
| 3 | 0.485883 | 0.5067 | 0.2309 | 2 |
| 4 | 0.442210 | 0.4600 | -1.3856 | 8 |
| 5 | 0.491574 | 0.4967 | -0.1155 | 2 |
| 6 | 0.466974 | 0.4800 | -0.6928 | 1 |
| 7 | 0.493412 | 0.5033 | 0.1155 | 4 |
| 8 | 0.507178 | 0.5167 | 0.5774 | 6 |
| 9 | 0.506352 | 0.5267 | 0.9238 | 1 |
| 10 | 0.516013 | 0.5100 | 0.3464 | 5 |
| 11 | 0.541111 | 0.5333 | 1.1547 | 4 |
| 12 | 0.488281 | 0.4733 | -0.9238 | 1 |
| 13 | 0.542055 | 0.5367 | 1.2702 | 3 |
| 14 | 0.535948 | 0.5367 | 1.2702 | 1 |
| 15 | 0.501067 | 0.5033 | 0.1155 | 10 |
| 16 | 0.477262 | 0.5000 | 0.0000 | 1 |
| 17 | 0.496643 | 0.4933 | -0.2309 | 10 |
| 18 | 0.522748 | 0.4933 | -0.2309 | 1 |
| 19 | 0.471427 | 0.4633 | -1.2702 | 6 |
| 20 | 0.465273 | 0.5067 | 0.2309 | 1 |
| 21 | 0.489065 | 0.5233 | 0.8083 | 4 |
| 22 | 0.467584 | 0.4600 | -1.3856 | 6 |
| 23 | 0.550347 | 0.5267 | 0.9238 | 1 |
| 24 | 0.474901 | 0.5000 | 0.0000 | 3 |
| 25 | 0.498689 | 0.4933 | -0.2309 | 1 |
| 26 | 0.560497 | 0.5367 | 1.2702 | 3 |
| 27 | 0.518170 | 0.5167 | 0.5774 | 1 |
| 28 | 0.567570 | 0.5700 | 2.4249 | 4 |
| 29 | 0.531339 | 0.5167 | 0.5774 | 2 |
| 30 | 0.532468 | 0.5300 | 1.0392 | 14 |
| 31 | 0.507635 | 0.5167 | 0.5774 | 1 |
| 32 | 0.512491 | 0.4900 | -0.3464 | 15 |
| 33 | 0.407281 | 0.3967 | -3.5796 | 13 |
| 34 | 0.471344 | 0.4733 | -0.9238 | 10 |
| 35 | 0.443573 | 0.4600 | -1.3856 | 13 |
| 36 | 0.485250 | 0.4600 | -1.3856 | 1 |
| 37 | 0.464643 | 0.4900 | -0.3464 | 1 |
| 38 | 0.513825 | 0.4933 | -0.2309 | 8 |
| 39 | 0.448044 | 0.4533 | -1.6166 | 13 |
| 40 | 0.515612 | 0.5200 | 0.6928 | 11 |
| 41 | 0.526330 | 0.4867 | -0.4619 | 1 |
| 42 | 0.499844 | 0.4933 | -0.2309 | 1 |
| 43 | 0.452407 | 0.4467 | -1.8475 | 2 |
| 44 | 0.505653 | 0.5067 | 0.2309 | 4 |
| 45 | 0.476618 | 0.4800 | -0.6928 | 1 |
| 46 | 0.497444 | 0.4900 | -0.3464 | 15 |
| 47 | 0.490152 | 0.5100 | 0.3464 | 4 |
| 48 | 0.507911 | 0.5033 | 0.1155 | 2 |
| 49 | 0.504270 | 0.5100 | 0.3464 | 8 |
| 50 | 0.483135 | 0.5233 | 0.8083 | 6 |
| 51 | 0.440413 | 0.4633 | -1.2702 | 1 |
| 52 | 0.512444 | 0.5000 | 0.0000 | 4 |
| 53 | 0.497200 | 0.4833 | -0.5774 | 1 |
| 54 | 0.526941 | 0.5033 | 0.1155 | 1 |
| 55 | 0.540291 | 0.4967 | -0.1155 | 1 |
| 56 | 0.542380 | 0.5100 | 0.3464 | 1 |
| 57 | 0.520445 | 0.5167 | 0.5774 | 4 |
| 58 | 0.465770 | 0.4800 | -0.6928 | 11 |
| 59 | 0.529246 | 0.5233 | 0.8083 | 1 |
| 60 | 0.471697 | 0.4767 | -0.8083 | 1 |
| 61 | 0.440020 | 0.4633 | -1.2702 | 2 |
| 62 | 0.526878 | 0.5300 | 1.0392 | 5 |
| 63 | 0.526887 | 0.5400 | 1.3856 | 2 |
| 64 | 0.546074 | 0.5200 | 0.6928 | 8 |
| 65 | 0.569783 | 0.5567 | 1.9630 | 3 |
| 66 | 0.457086 | 0.4700 | -1.0392 | 1 |
| 67 | 0.531825 | 0.5333 | 1.1547 | 15 |
| 68 | 0.506847 | 0.5133 | 0.4619 | 12 |
| 69 | 0.497415 | 0.4900 | -0.3464 | 2 |
| 70 | 0.517530 | 0.5300 | 1.0392 | 1 |
| 71 | 0.496875 | 0.5233 | 0.8083 | 14 |
| 72 | 0.458629 | 0.4933 | -0.2309 | 1 |
| 73 | 0.509291 | 0.5300 | 1.0392 | 5 |
| 74 | 0.441065 | 0.4700 | -1.0392 | 11 |
| 75 | 0.553447 | 0.5500 | 1.7321 | 1 |
| 76 | 0.511702 | 0.5600 | 2.0785 | 1 |
| 77 | 0.505189 | 0.5067 | 0.2309 | 12 |
| 78 | 0.479545 | 0.4733 | -0.9238 | 1 |
| 79 | 0.546875 | 0.4967 | -0.1155 | 1 |
| 80 | 0.466429 | 0.4567 | -1.5011 | 1 |
| 81 | 0.444983 | 0.4600 | -1.3856 | 4 |
| 82 | 0.534258 | 0.5233 | 0.8083 | 3 |
| 83 | 0.503113 | 0.5133 | 0.4619 | 5 |
| 84 | 0.445557 | 0.4767 | -0.8083 | 14 |
| 85 | 0.492455 | 0.5000 | 0.0000 | 8 |
| 86 | 0.534721 | 0.4933 | -0.2309 | 1 |
| 87 | 0.453259 | 0.4767 | -0.8083 | 1 |
| 88 | 0.431769 | 0.4400 | -2.0785 | 15 |
| 89 | 0.442902 | 0.4600 | -1.3856 | 7 |
| 90 | 0.497017 | 0.5067 | 0.2309 | 3 |
| 91 | 0.537726 | 0.5233 | 0.8083 | 6 |
| 92 | 0.508922 | 0.4767 | -0.8083 | 1 |
| 93 | 0.475178 | 0.4600 | -1.3856 | 3 |
| 94 | 0.488985 | 0.4800 | -0.6928 | 15 |
| 95 | 0.561200 | 0.4967 | -0.1155 | 1 |
| 96 | 0.454731 | 0.4967 | -0.1155 | 2 |
| 97 | 0.518977 | 0.4933 | -0.2309 | 14 |
| 98 | 0.479333 | 0.5033 | 0.1155 | 9 |
| 99 | 0.421368 | 0.4733 | -0.9238 | 2 |

### Table 3.N2: N2: Real Frozen Grouped-OOF Features, Permuted Labels Summary and 10-Bin Histogram
`JSON Key: part3_nested_null_controls.N2`

| Metric | Value | Reference / Bound | Pass / Fail |
|---|---|---|---|
| Theoretical Sigma_0 | 0.033396 | Mann-Whitney U Null Std | Reference |
| Max Allowed Empirical Std | 0.066791 | 2 * Sigma_0 Bound | Reference |
| Empirical Mean AUC | 0.499075 | Expected ~0.5000 | Within Bound |
| Empirical Std AUC | 0.031918 | Must be <= 2 * Sigma_0 | **PASS** |
| Percentile 2.5% | 0.430417 | Empirical 95% Interval Low | - |
| Percentile 97.5% | 0.550063 | Empirical 95% Interval High | - |
| Min AUC Observed | 0.415255 | 4-Sigma Low: 0.366418 | **PASS** |
| Max AUC Observed | 0.563094 | 4-Sigma High: 0.633582 | **PASS** |
| Mean Accuracy | 0.4957 | Expected ~0.5000 | - |
| Mean Binomial z-score | -0.1478 | Expected ~0.0000 | - |
| Mean Selected Epoch | 4.14 | 1 to 15 epochs | - |
| **Overall Gate G-PERM Status** | **PASS** | Std <= 2*Sigma0 & No seed > 4*Sigma0 | **PASS** |

#### 10-Bin Histogram for N2 (Recomputed per Rule G9)
`JSON Key: part3_nested_null_controls.N2.histogram`

| Bin Index | Range Min | Range Max | Count |
|---|---|---|---|
| 0 | 0.3500 | 0.3800 | 0 |
| 1 | 0.3800 | 0.4100 | 0 |
| 2 | 0.4100 | 0.4400 | 5 |
| 3 | 0.4400 | 0.4700 | 14 |
| 4 | 0.4700 | 0.5000 | 29 |
| 5 | 0.5000 | 0.5300 | 36 |
| 6 | 0.5300 | 0.5600 | 15 |
| 7 | 0.5600 | 0.5900 | 1 |
| 8 | 0.5900 | 0.6200 | 0 |
| 9 | 0.6200 | 0.6500 | 0 |

#### Per-Seed Raw Metrics for N2 (100 Seeds)
`JSON Key: part3_nested_null_controls.N2.per_seed_rows`

| Seed | ROC-AUC | Accuracy | Binomial z-Score | Selected Epoch |
|---|---|---|---|---|
| 0 | 0.517294 | 0.5067 | 0.2309 | 1 |
| 1 | 0.516034 | 0.5433 | 1.5011 | 14 |
| 2 | 0.477442 | 0.4967 | -0.1155 | 2 |
| 3 | 0.482711 | 0.4867 | -0.4619 | 3 |
| 4 | 0.490781 | 0.5267 | 0.9238 | 8 |
| 5 | 0.511054 | 0.5267 | 0.9238 | 1 |
| 6 | 0.516623 | 0.5067 | 0.2309 | 1 |
| 7 | 0.497413 | 0.4900 | -0.3464 | 3 |
| 8 | 0.428731 | 0.4367 | -2.1939 | 5 |
| 9 | 0.482863 | 0.4800 | -0.6928 | 3 |
| 10 | 0.474047 | 0.4500 | -1.7321 | 3 |
| 11 | 0.529353 | 0.5533 | 1.8475 | 4 |
| 12 | 0.482108 | 0.4700 | -1.0392 | 5 |
| 13 | 0.521845 | 0.4900 | -0.3464 | 1 |
| 14 | 0.525422 | 0.4933 | -0.2309 | 4 |
| 15 | 0.461717 | 0.4533 | -1.6166 | 15 |
| 16 | 0.541228 | 0.5667 | 2.3094 | 5 |
| 17 | 0.492344 | 0.5000 | 0.0000 | 7 |
| 18 | 0.512844 | 0.5300 | 1.0392 | 6 |
| 19 | 0.482844 | 0.5000 | 0.0000 | 12 |
| 20 | 0.488703 | 0.5100 | 0.3464 | 4 |
| 21 | 0.518214 | 0.5233 | 0.8083 | 1 |
| 22 | 0.522683 | 0.5200 | 0.6928 | 7 |
| 23 | 0.451373 | 0.4767 | -0.8083 | 1 |
| 24 | 0.445160 | 0.4833 | -0.5774 | 1 |
| 25 | 0.541138 | 0.5033 | 0.1155 | 3 |
| 26 | 0.518069 | 0.4800 | -0.6928 | 13 |
| 27 | 0.505676 | 0.5100 | 0.3464 | 2 |
| 28 | 0.473819 | 0.4633 | -1.2702 | 8 |
| 29 | 0.493573 | 0.5133 | 0.4619 | 5 |
| 30 | 0.515890 | 0.5133 | 0.4619 | 10 |
| 31 | 0.536268 | 0.5233 | 0.8083 | 1 |
| 32 | 0.510089 | 0.5167 | 0.5774 | 1 |
| 33 | 0.460172 | 0.4667 | -1.1547 | 1 |
| 34 | 0.536496 | 0.5200 | 0.6928 | 1 |
| 35 | 0.537160 | 0.5400 | 1.3856 | 11 |
| 36 | 0.522912 | 0.5167 | 0.5774 | 3 |
| 37 | 0.505000 | 0.5200 | 0.6928 | 1 |
| 38 | 0.503005 | 0.4933 | -0.2309 | 5 |
| 39 | 0.539898 | 0.4500 | -1.7321 | 1 |
| 40 | 0.502600 | 0.4800 | -0.6928 | 1 |
| 41 | 0.503560 | 0.5000 | 0.0000 | 10 |
| 42 | 0.499688 | 0.4933 | -0.2309 | 1 |
| 43 | 0.475801 | 0.4867 | -0.4619 | 2 |
| 44 | 0.459237 | 0.4633 | -1.2702 | 3 |
| 45 | 0.506299 | 0.4800 | -0.6928 | 1 |
| 46 | 0.518423 | 0.5000 | 0.0000 | 1 |
| 47 | 0.436917 | 0.4367 | -2.1939 | 2 |
| 48 | 0.555543 | 0.5367 | 1.2702 | 2 |
| 49 | 0.515001 | 0.5133 | 0.4619 | 1 |
| 50 | 0.517995 | 0.5300 | 1.0392 | 2 |
| 51 | 0.486666 | 0.5000 | 0.0000 | 1 |
| 52 | 0.482158 | 0.4900 | -0.3464 | 14 |
| 53 | 0.447902 | 0.4700 | -1.0392 | 9 |
| 54 | 0.497826 | 0.4500 | -1.7321 | 1 |
| 55 | 0.460267 | 0.4867 | -0.4619 | 3 |
| 56 | 0.491648 | 0.4867 | -0.4619 | 4 |
| 57 | 0.444263 | 0.4800 | -0.6928 | 3 |
| 58 | 0.511445 | 0.5267 | 0.9238 | 3 |
| 59 | 0.461182 | 0.4567 | -1.5011 | 1 |
| 60 | 0.488729 | 0.4900 | -0.3464 | 1 |
| 61 | 0.529256 | 0.4933 | -0.2309 | 4 |
| 62 | 0.523440 | 0.5200 | 0.6928 | 4 |
| 63 | 0.497682 | 0.4800 | -0.6928 | 2 |
| 64 | 0.430041 | 0.4367 | -2.1939 | 1 |
| 65 | 0.494577 | 0.4967 | -0.1155 | 4 |
| 66 | 0.517366 | 0.4700 | -1.0392 | 1 |
| 67 | 0.447138 | 0.4433 | -1.9630 | 2 |
| 68 | 0.507100 | 0.5200 | 0.6928 | 1 |
| 69 | 0.430832 | 0.4533 | -1.6166 | 6 |
| 70 | 0.516023 | 0.4800 | -0.6928 | 1 |
| 71 | 0.486222 | 0.5000 | 0.0000 | 1 |
| 72 | 0.487240 | 0.4933 | -0.2309 | 1 |
| 73 | 0.480054 | 0.4767 | -0.8083 | 1 |
| 74 | 0.520344 | 0.5133 | 0.4619 | 14 |
| 75 | 0.457022 | 0.4833 | -0.5774 | 1 |
| 76 | 0.545229 | 0.5133 | 0.4619 | 7 |
| 77 | 0.458066 | 0.4733 | -0.9238 | 1 |
| 78 | 0.533250 | 0.4800 | -0.6928 | 10 |
| 79 | 0.550824 | 0.5300 | 1.0392 | 1 |
| 80 | 0.535533 | 0.5267 | 0.9238 | 9 |
| 81 | 0.415255 | 0.4900 | -0.3464 | 1 |
| 82 | 0.497319 | 0.4867 | -0.4619 | 13 |
| 83 | 0.505501 | 0.4867 | -0.4619 | 1 |
| 84 | 0.495081 | 0.4900 | -0.3464 | 1 |
| 85 | 0.546666 | 0.5133 | 0.4619 | 2 |
| 86 | 0.518910 | 0.4767 | -0.8083 | 2 |
| 87 | 0.529270 | 0.5267 | 0.9238 | 12 |
| 88 | 0.465990 | 0.4767 | -0.8083 | 1 |
| 89 | 0.542275 | 0.4467 | -1.8475 | 6 |
| 90 | 0.515290 | 0.4967 | -0.1155 | 4 |
| 91 | 0.485844 | 0.4767 | -0.8083 | 11 |
| 92 | 0.543378 | 0.5667 | 2.3094 | 4 |
| 93 | 0.513694 | 0.5233 | 0.8083 | 13 |
| 94 | 0.549222 | 0.5467 | 1.6166 | 11 |
| 95 | 0.507746 | 0.4967 | -0.1155 | 6 |
| 96 | 0.478156 | 0.4900 | -0.3464 | 1 |
| 97 | 0.563094 | 0.4800 | -0.6928 | 1 |
| 98 | 0.461624 | 0.4833 | -0.5774 | 2 |
| 99 | 0.492817 | 0.5300 | 1.0392 | 1 |

### Table 3.N3: N3: Full Pipeline Retrained Base Models, OOF, ANN Summary and 10-Bin Histogram
`JSON Key: part3_nested_null_controls.N3`

| Metric | Value | Reference / Bound | Pass / Fail |
|---|---|---|---|
| Theoretical Sigma_0 | 0.033396 | Mann-Whitney U Null Std | Reference |
| Max Allowed Empirical Std | 0.066791 | 2 * Sigma_0 Bound | Reference |
| Empirical Mean AUC | 0.502698 | Expected ~0.5000 | Within Bound |
| Empirical Std AUC | 0.031705 | Must be <= 2 * Sigma_0 | **PASS** |
| Percentile 2.5% | 0.446906 | Empirical 95% Interval Low | - |
| Percentile 97.5% | 0.564675 | Empirical 95% Interval High | - |
| Min AUC Observed | 0.429950 | 4-Sigma Low: 0.366418 | **PASS** |
| Max AUC Observed | 0.579387 | 4-Sigma High: 0.633582 | **PASS** |
| Mean Accuracy | 0.4998 | Expected ~0.5000 | - |
| Mean Binomial z-score | -0.0081 | Expected ~0.0000 | - |
| Mean Selected Epoch | 5.47 | 1 to 15 epochs | - |
| **Overall Gate G-PERM Status** | **PASS** | Std <= 2*Sigma0 & No seed > 4*Sigma0 | **PASS** |

#### 10-Bin Histogram for N3 (Recomputed per Rule G9)
`JSON Key: part3_nested_null_controls.N3.histogram`

| Bin Index | Range Min | Range Max | Count |
|---|---|---|---|
| 0 | 0.3500 | 0.3800 | 0 |
| 1 | 0.3800 | 0.4100 | 0 |
| 2 | 0.4100 | 0.4400 | 1 |
| 3 | 0.4400 | 0.4700 | 18 |
| 4 | 0.4700 | 0.5000 | 25 |
| 5 | 0.5000 | 0.5300 | 37 |
| 6 | 0.5300 | 0.5600 | 15 |
| 7 | 0.5600 | 0.5900 | 4 |
| 8 | 0.5900 | 0.6200 | 0 |
| 9 | 0.6200 | 0.6500 | 0 |

#### Per-Seed Raw Metrics for N3 (100 Seeds)
`JSON Key: part3_nested_null_controls.N3.per_seed_rows`

| Seed | ROC-AUC | Accuracy | Binomial z-Score | Selected Epoch |
|---|---|---|---|---|
| 0 | 0.517185 | 0.5233 | 0.8083 | 2 |
| 1 | 0.521720 | 0.5367 | 1.2702 | 3 |
| 2 | 0.513983 | 0.4800 | -0.6928 | 4 |
| 3 | 0.446801 | 0.4633 | -1.2702 | 4 |
| 4 | 0.525944 | 0.5000 | 0.0000 | 2 |
| 5 | 0.507403 | 0.5133 | 0.4619 | 5 |
| 6 | 0.482504 | 0.4633 | -1.2702 | 1 |
| 7 | 0.485394 | 0.4833 | -0.5774 | 3 |
| 8 | 0.512160 | 0.5000 | 0.0000 | 1 |
| 9 | 0.511938 | 0.4900 | -0.3464 | 11 |
| 10 | 0.529590 | 0.5167 | 0.5774 | 15 |
| 11 | 0.454760 | 0.4800 | -0.6928 | 10 |
| 12 | 0.492953 | 0.4833 | -0.5774 | 12 |
| 13 | 0.479881 | 0.4767 | -0.8083 | 2 |
| 14 | 0.485794 | 0.4833 | -0.5774 | 8 |
| 15 | 0.484549 | 0.4767 | -0.8083 | 1 |
| 16 | 0.515095 | 0.5133 | 0.4619 | 4 |
| 17 | 0.505580 | 0.4833 | -0.5774 | 13 |
| 18 | 0.530301 | 0.4900 | -0.3464 | 15 |
| 19 | 0.490507 | 0.5067 | 0.2309 | 1 |
| 20 | 0.539638 | 0.5000 | 0.0000 | 1 |
| 21 | 0.475568 | 0.4867 | -0.4619 | 1 |
| 22 | 0.554889 | 0.5267 | 0.9238 | 1 |
| 23 | 0.525366 | 0.5167 | 0.5774 | 1 |
| 24 | 0.528033 | 0.5133 | 0.4619 | 2 |
| 25 | 0.500867 | 0.4933 | -0.2309 | 11 |
| 26 | 0.544796 | 0.5433 | 1.5011 | 1 |
| 27 | 0.506202 | 0.5000 | 0.0000 | 8 |
| 28 | 0.494598 | 0.4833 | -0.5774 | 2 |
| 29 | 0.429950 | 0.4600 | -1.3856 | 14 |
| 30 | 0.528700 | 0.5167 | 0.5774 | 1 |
| 31 | 0.523187 | 0.4933 | -0.2309 | 9 |
| 32 | 0.474634 | 0.4867 | -0.4619 | 14 |
| 33 | 0.485216 | 0.4800 | -0.6928 | 10 |
| 34 | 0.490907 | 0.4933 | -0.2309 | 12 |
| 35 | 0.481215 | 0.5000 | 0.0000 | 5 |
| 36 | 0.500823 | 0.4933 | -0.2309 | 1 |
| 37 | 0.481926 | 0.4933 | -0.2309 | 15 |
| 38 | 0.496465 | 0.4967 | -0.1155 | 3 |
| 39 | 0.477346 | 0.4633 | -1.2702 | 7 |
| 40 | 0.456716 | 0.4767 | -0.8083 | 1 |
| 41 | 0.454404 | 0.4300 | -2.4249 | 1 |
| 42 | 0.465942 | 0.4767 | -0.8083 | 1 |
| 43 | 0.451959 | 0.5100 | 0.3464 | 1 |
| 44 | 0.567294 | 0.5767 | 2.6558 | 5 |
| 45 | 0.529679 | 0.4967 | -0.1155 | 1 |
| 46 | 0.530123 | 0.4833 | -0.5774 | 8 |
| 47 | 0.506025 | 0.5067 | 0.2309 | 5 |
| 48 | 0.481570 | 0.4900 | -0.3464 | 5 |
| 49 | 0.506247 | 0.5200 | 0.6928 | 2 |
| 50 | 0.468765 | 0.4767 | -0.8083 | 6 |
| 51 | 0.548041 | 0.5300 | 1.0392 | 3 |
| 52 | 0.508603 | 0.5167 | 0.5774 | 15 |
| 53 | 0.540794 | 0.5467 | 1.6166 | 13 |
| 54 | 0.501578 | 0.5100 | 0.3464 | 6 |
| 55 | 0.529723 | 0.5233 | 0.8083 | 14 |
| 56 | 0.540794 | 0.5000 | 0.0000 | 1 |
| 57 | 0.533769 | 0.5167 | 0.5774 | 3 |
| 58 | 0.511360 | 0.4900 | -0.3464 | 2 |
| 59 | 0.510204 | 0.5333 | 1.1547 | 2 |
| 60 | 0.539905 | 0.5400 | 1.3856 | 1 |
| 61 | 0.455427 | 0.4567 | -1.5011 | 1 |
| 62 | 0.461785 | 0.4767 | -0.8083 | 1 |
| 63 | 0.494687 | 0.4700 | -1.0392 | 9 |
| 64 | 0.579387 | 0.5733 | 2.5403 | 10 |
| 65 | 0.453870 | 0.4567 | -1.5011 | 14 |
| 66 | 0.487706 | 0.4767 | -0.8083 | 1 |
| 67 | 0.486417 | 0.5100 | 0.3464 | 7 |
| 68 | 0.463919 | 0.4700 | -1.0392 | 15 |
| 69 | 0.485928 | 0.5067 | 0.2309 | 6 |
| 70 | 0.531635 | 0.5500 | 1.7321 | 1 |
| 71 | 0.466720 | 0.4767 | -0.8083 | 9 |
| 72 | 0.508426 | 0.5033 | 0.1155 | 14 |
| 73 | 0.536170 | 0.5233 | 0.8083 | 13 |
| 74 | 0.574897 | 0.5600 | 2.0785 | 15 |
| 75 | 0.561780 | 0.5300 | 1.0392 | 1 |
| 76 | 0.547108 | 0.5700 | 2.4249 | 2 |
| 77 | 0.496954 | 0.5000 | 0.0000 | 3 |
| 78 | 0.494731 | 0.4967 | -0.1155 | 7 |
| 79 | 0.516651 | 0.5300 | 1.0392 | 10 |
| 80 | 0.479970 | 0.5100 | 0.3464 | 3 |
| 81 | 0.516073 | 0.4967 | -0.1155 | 13 |
| 82 | 0.509404 | 0.5133 | 0.4619 | 1 |
| 83 | 0.459384 | 0.4767 | -0.8083 | 2 |
| 84 | 0.508426 | 0.5067 | 0.2309 | 1 |
| 85 | 0.480059 | 0.4667 | -1.1547 | 1 |
| 86 | 0.506202 | 0.5133 | 0.4619 | 1 |
| 87 | 0.467431 | 0.4533 | -1.6166 | 2 |
| 88 | 0.456316 | 0.5033 | 0.1155 | 7 |
| 89 | 0.521542 | 0.5267 | 0.9238 | 1 |
| 90 | 0.519052 | 0.5067 | 0.2309 | 12 |
| 91 | 0.542395 | 0.5100 | 0.3464 | 3 |
| 92 | 0.555289 | 0.5200 | 0.6928 | 14 |
| 93 | 0.518874 | 0.5033 | 0.1155 | 4 |
| 94 | 0.467120 | 0.4733 | -0.9238 | 4 |
| 95 | 0.447023 | 0.4433 | -1.9630 | 1 |
| 96 | 0.523721 | 0.5267 | 0.9238 | 1 |
| 97 | 0.444267 | 0.4600 | -1.3856 | 3 |
| 98 | 0.506869 | 0.5067 | 0.2309 | 1 |
| 99 | 0.518296 | 0.4700 | -1.0392 | 1 |

## Part 4: Canonical Experiments A-E & Model Selection

### Table 4.1: ANN 48-Cell Hyperparameter Grid Top-5 Ranking and Negative Control
`JSON Key: part4_experiments.4b_ann_grid`

| Rank / Cell ID | Optimizer | LR | Weight Init | Dropout Rate | BatchNorm | Val ROC-AUC | Val F1 | Val BCE Loss | Saturation Fraction | Selection Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Negative Control** | SGD | 2.0 | default | 0.0 | False | 0.5000 | 0.0000 | 9.9999 | 0.5350 | **EXCLUDED (Control)** |
| **Rank 1 (CELL_45)** | SGD_m09 | 0.02 | default | 0.2 | True | **0.986078** | 0.9524 | 0.2793 | 0.9100 | **WINNER** |
| **Rank 2 (CELL_01)** | Adam | 0.005 | xavier | 0.2 | True | **0.985877** | 0.9524 | 0.2847 | 0.9400 | Candidate |
| **Rank 3 (CELL_21)** | RMSprop | 0.005 | default | 0.2 | True | **0.985377** | 0.9524 | 0.2813 | 0.9100 | Candidate |
| **Rank 4 (CELL_03)** | Adam | 0.005 | xavier | 0.0 | True | **0.984876** | 0.9524 | 0.2760 | 0.9300 | Candidate |
| **Rank 5 (CELL_40)** | SGD_m09 | 0.02 | xavier | 0.0 | False | **0.984275** | 0.9524 | 0.2969 | 0.9450 | Candidate |

### Table 4.2: Canonical Experiments A-E Summary Across 5 Seeds
`JSON Key: part4_experiments.4c_canonical_summary`

| Experiment Arm | Architecture / Features | Mean ROC-AUC (Std) | Mean F1 (Std) | Mean Accuracy (Std) | Mean Precision (Std) | Mean Recall (Std) | Mean Brier Score (Std) |
|---|---|---|---|---|---|---|---|
| **Exp A** | BiLSTM Text Evidence Only | **0.710884** (±0.000000) | **0.6876** (±0.0000) | 0.5367 (±0.0000) | 0.5240 (±0.0000) | 1.0000 (±0.0000) | 0.468039 (±0.000000) |
| **Exp B** | VisualEvidenceCNN Image Only | **0.960740** (±0.000000) | **0.9613** (±0.0000) | 0.9600 (±0.0000) | 0.9490 (±0.0000) | 0.9739 (±0.0000) | 0.038523 (±0.000000) |
| **Exp C** | ANN on Multimodal Embeddings [BiLSTM, CNN] | **0.963350** (±0.000528) | **0.9613** (±0.0000) | 0.9600 (±0.0000) | 0.9490 (±0.0000) | 0.9739 (±0.0000) | 0.039725 (±0.000197) |
| **Exp D** | ANN on Full 7D Feature Vector (Winner Grid) | **0.990868** (±0.002075) | **0.9646** (±0.0000) | 0.9633 (±0.0000) | 0.9494 (±0.0000) | 0.9804 (±0.0000) | 0.032770 (±0.001268) |
| **Exp E** | Exp D + 50% VAE Feature Augmentation | **0.991757** (±0.002847) | **0.9646** (±0.0000) | 0.9633 (±0.0000) | 0.9494 (±0.0000) | 0.9804 (±0.0000) | 0.029241 (±0.001164) |

### Table 4.3: Per-Seed Raw Metrics for Experiments A-E
`JSON Key: part4_experiments.4c_canonical_summary.<arm>.per_seed_runs`

| Seed | Exp A AUC | Exp B AUC | Exp C AUC | Exp D AUC | Exp E AUC | Exp A F1 | Exp B F1 | Exp C F1 | Exp D F1 | Exp E F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.710884 | 0.960740 | 0.963563 | 0.989374 | 0.988884 | 0.6876 | 0.9613 | 0.9613 | 0.9646 | 0.9646 |
| 43 | 0.710884 | 0.960740 | 0.963674 | 0.993553 | 0.987773 | 0.6876 | 0.9613 | 0.9613 | 0.9646 | 0.9646 |
| 44 | 0.710884 | 0.960740 | 0.963630 | 0.987951 | 0.994175 | 0.6876 | 0.9613 | 0.9613 | 0.9646 | 0.9646 |
| 45 | 0.710884 | 0.960740 | 0.963585 | 0.992753 | 0.993375 | 0.6876 | 0.9613 | 0.9613 | 0.9646 | 0.9646 |
| 46 | 0.710884 | 0.960740 | 0.962296 | 0.990707 | 0.994576 | 0.6876 | 0.9613 | 0.9613 | 0.9646 | 0.9646 |

### Table 4.4: Cluster Bootstrap 95% Confidence Intervals (1000 Replicates)
`JSON Key: part4_experiments.4c_cluster_ci_table`

| Comparison | Delta ROC-AUC | 95% CI (Delta AUC) | Delta AUC Significant | Delta F1 | 95% CI (Delta F1) | Delta F1 Significant |
|---|---|---|---|---|---|---|
| **Exp C (Fusion) vs Best Single (Exp B)** | +0.0016 | [-0.0002, +0.0041] | **NO** | +0.0000 | [+0.0000, +0.0000] | **NO** |
| **Exp D (All 7 Features) vs Exp C (2 Features)** | +0.0284 | [-0.0026, +0.0555] | **NO** | +0.0033 | [+0.0000, +0.0093] | **NO** |
| **Exp E (VAE Augmentation) vs Exp D (Baseline)** | +0.0039 | [+0.0000, +0.0100] | **NO** | +0.0000 | [+0.0000, +0.0000] | **NO** |

### Table 4.5: Leave-One-Feature-Out (LOFO) and Single-Feature ANN Audits
`JSON Key: part4_experiments.4d_diagnostics.lofo_table` & `part4_experiments.4d_diagnostics.single_feature_ann_table`

| Feature Name | LOFO Test AUC (Drop Feature) | Delta AUC Drop from Full D | Single-Feature ANN Test AUC | Exceeds 0.85 Cap | Permutation Importance Drop |
|---|---|---|---|---|---|
| `bilstm_score` | 0.994976 | -0.004268 | 0.527211 | **NO** | +0.000044 |
| `cnn_score` | 0.964897 | +0.025810 | 0.960740 | **YES** | +0.149615 |
| `url_score` | 0.986684 | +0.004024 | 0.804922 | **NO** | +0.088702 |
| `keyword_score` | 0.985728 | +0.004980 | 0.809702 | **NO** | +0.003735 |
| `attachment_score` | 0.993153 | -0.002445 | 0.682228 | **NO** | +0.000800 |
| `timeline_anomaly` | 0.990396 | +0.000311 | 0.698746 | **NO** | -0.000400 |
| `email_anomaly` | 0.992086 | -0.001378 | 0.634520 | **NO** | -0.000044 |

### Table 4.6: Calibration and Prediction Agreement Matrix
`JSON Key: part4_experiments.4d_diagnostics.calibration_table` & `part4_experiments.4d_diagnostics.prediction_agreement_matrix`

**Temperature Scaling Diagnostics:** Fitted Temperature = **1.5996**
- Brier Score: 0.032654 (Raw) -> **0.030681** (Calibrated)
- Expected Calibration Error (ECE): 0.025290 (Raw) -> **0.038008** (Calibrated)

| Arm Agreement Fraction | Exp A | Exp B | Exp C | Exp D |
|---|---|---|---|---|
| **Exp A** | 1.0000 | 0.5433 | 0.5433 | 0.5467 |
| **Exp B** | 0.5433 | 1.0000 | 1.0000 | 0.9967 |
| **Exp C** | 0.5433 | 1.0000 | 1.0000 | 0.9967 |
| **Exp D** | 0.5467 | 0.9967 | 0.9967 | 1.0000 |

### Table 4.7: CNN OFAT Anomaly Resolution (Calibrated Threshold)
`JSON Key: part4_experiments.4e_cnn_ofat`

- **Logit Distribution:** Mean = 0.1035, Std = 2.9100
- **Rigid Threshold (0.50) Accuracy:** 0.9400
- **Calibrated Threshold (0.1000) Accuracy:** Val = **0.9400**, Test = **0.9600**

| Seed | BatchNorm ON Val AUC | BatchNorm ON Val Acc (Calibrated) | BatchNorm OFF Val AUC | BatchNorm OFF Val Acc (Calibrated) |
|---|---|---|---|---|
| 42 | 0.929988 | 0.9400 | 0.947616 | 0.9400 |
| 43 | 0.931691 | 0.9400 | 0.944511 | 0.9400 |
| 44 | 0.938502 | 0.9400 | 0.943109 | 0.9400 |
| 45 | 0.928185 | 0.5200 | 0.946514 | 0.9400 |
| 46 | 0.933193 | 0.9400 | 0.942007 | 0.9400 |

### Table 4.8: BiLSTM Pooling Architecture Comparison (Concat vs Attention)
`JSON Key: part4_experiments.4f_bilstm_pooling`

| Pooling Architecture | Validation ROC-AUC | Test ROC-AUC | Oracle Ceiling Test AUC | Ceiling Violation |
|---|---|---|---|---|
| Concat Pooling (Default) | 0.598958 | 0.710884 | 0.993464 | **NO** |
| Attention Pooling | 0.763622 | 0.617336 | 0.993464 | **NO** |

## Part 5: Repaired VAE Two-Sided Validation & Augmentation

### Table 5.1: Two-Sided Criteria for VAE Feature Distribution (Per Feature, Per Class)
`JSON Key: part5_vae.5a_two_sided_criteria`

| Feature Index | Feature Name | Class 0 Real Std | Class 0 Synth Std | Class 0 Std Ratio [0.5, 2.0] | Class 0 Wasserstein Ratio (<=0.50) | Class 0 Pass | Class 1 Real Std | Class 1 Synth Std | Class 1 Std Ratio [0.5, 2.0] | Class 1 Wasserstein Ratio (<=0.50) | Class 1 Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `bilstm_score` | 0.4825 | 0.2690 | 0.5575 (PASS) | 0.5248 (FAIL) | **FAIL** | 0.4838 | 0.1417 | 0.2930 (FAIL) | 0.7501 (FAIL) | **FAIL** |
| 1 | `cnn_score` | 0.1511 | 0.0202 | 0.1339 (FAIL) | 0.3687 (PASS) | **FAIL** | 0.1662 | 0.0234 | 0.1407 (FAIL) | 0.4452 (PASS) | **FAIL** |
| 2 | `url_score` | 0.0895 | 0.0250 | 0.2797 (FAIL) | 0.6821 (FAIL) | **FAIL** | 0.4391 | 0.0796 | 0.1813 (FAIL) | 0.8422 (FAIL) | **FAIL** |
| 3 | `keyword_score` | 0.2410 | 0.0595 | 0.2470 (FAIL) | 0.5985 (FAIL) | **FAIL** | 0.3274 | 0.0394 | 0.1203 (FAIL) | 0.7578 (FAIL) | **FAIL** |
| 4 | `attachment_score` | 0.0924 | 0.0110 | 0.1190 (FAIL) | 0.3670 (PASS) | **FAIL** | 0.4560 | 0.1896 | 0.4159 (FAIL) | 0.5891 (FAIL) | **FAIL** |
| 5 | `timeline_anomaly` | 0.2396 | 0.0146 | 0.0609 (FAIL) | 0.6336 (FAIL) | **FAIL** | 0.3601 | 0.0256 | 0.0712 (FAIL) | 0.9429 (FAIL) | **FAIL** |
| 6 | `email_anomaly` | 0.3039 | 0.0342 | 0.1126 (FAIL) | 0.8095 (FAIL) | **FAIL** | 0.4238 | 0.0180 | 0.0425 (FAIL) | 0.8875 (FAIL) | **FAIL** |

### Table 5.2: Per-Dimension KL Divergence and Active Latent Capacity
`JSON Key: part5_vae.5a_per_dim_kl`

| Latent Dimension | KL Divergence (nats) | Free-Bits Floor (nats) | Margin Above Floor (nats) | Non-Collapsed Status (>= 0.05 nats) |
|---|---|---|---|---|
| Dimension 0 | **0.071449** | 0.100 | -0.028551 | **ACTIVE (PASS)** |
| Dimension 1 | **0.062165** | 0.100 | -0.037835 | **ACTIVE (PASS)** |
| Dimension 2 | **0.073046** | 0.100 | -0.026954 | **ACTIVE (PASS)** |
| Dimension 3 | **0.075313** | 0.100 | -0.024687 | **ACTIVE (PASS)** |

### Table 5.3: GBDT Real vs Synthetic Discriminator & Positive Control
`JSON Key: part5_vae.5b_discriminator`

| Evaluation Task | Estimator | 5-Fold CV ROC-AUC | Target Threshold | Status / Validation |
|---|---|---|---|---|
| Indistinguishability (Real vs Repaired VAE) | Gradient Boosting (GBDT) | 1.0000 | <= 0.85 | **DISTINGUISHABLE** |
| Positive Control (Degraded Generator Centroid+10%) | Gradient Boosting (GBDT) | 1.0000 | > 0.90 | **VALID (PASS)** |

### Table 5.4: Experiment E 4-Arm Ratio Sweep Across Augmentation Levels
`JSON Key: part5_vae.5c_ratio_sweep`

| Augmentation Arm | Ratio = 0.0 (Baseline) | Ratio = 0.25 | Ratio = 0.50 | Ratio = 1.00 | Mean Across Ratios |
|---|---|---|---|---|---|
| **Baseline_None** | 0.990868 (±0.002075) | 0.990868 (±0.002075) | 0.990868 (±0.002075) | 0.990868 (±0.002075) | **0.990868** |
| **SyntheticDataVAE** | 0.990868 (±0.002075) | 0.993535 (±0.000821) | 0.993740 (±0.000709) | 0.993526 (±0.000917) | **0.992917** |
| **GaussianAugmenter** | 0.990868 (±0.002075) | 0.988164 (±0.001452) | 0.991321 (±0.001636) | 0.992379 (±0.001162) | **0.990683** |
| **InterpolationAugmenter** | 0.990868 (±0.002075) | 0.989747 (±0.001619) | 0.990423 (±0.001158) | 0.990912 (±0.002241) | **0.990487** |

## Research Questions Summary

### Table 6.1: Formal Research Questions (Strict yes/no/inconclusive with 95% CI)
`JSON Key: research_questions`

| ID | Research Question | Decision | Evaluation Metric | Estimate | 95% Cluster Bootstrap CI | Statistically Significant |
|---|---|---|---|---|---|---|
| **RQ1** | Does multimodal fusion (Exp C) outperform single-modality baselines (Exp A/B)? | **`inconclusive`** | Delta AUC (Exp C - Best Single) | +0.0016 | [-0.0002, +0.0041] | **NO** |
| **RQ2** | Does synthetic feature augmentation (Exp E) improve performance over baseline (Exp D)? | **`inconclusive`** | Delta AUC (Exp E - Exp D) | +0.0039 | [+0.0000, +0.0100] | **NO** |
