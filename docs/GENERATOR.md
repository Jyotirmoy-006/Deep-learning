# ForenSightX Hard-Synthetic Benchmark Generator v3 Specification
**Document ID:** `docs/GENERATOR.md`  
**Iteration:** 8 (Integrity First & Generalizable Benchmark)  
**Manifest:** `data/split_manifest_hard_v3.json`  

---

## 1. Executive Summary & Design Rationale

Historical iterations (v1 through v7) revealed critical methodological defects in synthetic benchmark datasets:
1. **Zero Observed Label Noise (v1–v7):** While generator code attempted label perturbation, it assigned `item.label = actual_label` where `actual_label` was also used to render text and images. This meant the observed dataset labels matched generated forensic evidence with 0% noise, contradicting the claimed 5% label noise specification.
2. **Template Family Overlap & Small Cluster Counts:** Iterations 1–6 suffered from template sharing across splits (e.g. 9 shared families and 68 case template pairings between train and val in `hard_v1`). Although Iteration 7 established family-disjoint splits in `hard_v2`, it utilized only 12 total families (6 train, 3 val, 3 test), creating high cluster correlation and low effective sample size ($n_{\text{eff}}$).
3. **Lack of Generalizable Shared Intent Cues:** In `hard_v2`, models trained on disjoint families struggled to learn generalizable multimodal concepts because vocabulary was tightly tied to family nuisance rather than cross-family forensic intent.
4. **Trivially Separable Heuristics:** Uncalibrated heuristic extractors in legacy data produced univariate AUCs near 1.0000, masking any potential benefit of neural multimodal fusion.

**Generator v3 (`hard_v3`)** resolves every defect through rigorous statistical calibration, an expanded 40-family ontology, shared cross-family forensic intent cues, true ground-truth label noise, and write-once cryptographic provenance.

---

## 2. 40-Family Ontological Taxonomy

Generator v3 scales the template ontology to 40 distinct operational families partitioned strictly across splits:

| Split | Family Count | Case Count | Template Family Identifiers |
|---|---|---|---|
| **Train** | **20** | 700 | `invoice_payment`, `account_suspension`, `urgent_wire`, `meeting_reschedule`, `project_status`, `executive_travel`, `vendor_onboarding`, `payroll_discrepancy`, `tax_compliance`, `legal_subpoena`, `system_alert`, `domain_renewal`, `audit_request`, `quarterly_budget`, `facility_maintenance`, `office_relocation`, `contract_amendment`, `procurement_order`, `customer_complaint`, `press_release` |
| **Validation** | **10** | 200 | `it_support_ticket`, `hr_benefit_update`, `cloud_access_request`, `software_license`, `data_backup_notice`, `training_seminar`, `telecom_bill`, `travel_expense_report`, `security_incident_report`, `marketing_campaign` |
| **Test** | **10** | 300 | `shipping_delivery_notice`, `vpn_credential_update`, `conference_invitation`, `patent_filing_notice`, `regulatory_disclosure`, `board_meeting_agenda`, `supply_chain_dispatch`, `charity_donation_match`, `health_safety_inspection`, `infrastructure_upgrade` |
| **Total** | **40** | **1,200** | Strict disjoint partitioning: zero family or text record overlap across splits. |

---

## 3. Class Cues vs Family-Specific Nuisance

To ensure models learn generalizable forensic representations that transfer to completely unseen test families, evidence generation decomposes text, visual, and metadata artifacts into:

### 3.1 Text Evidence
1. **Shared Forensic Intent Cues:**
   - *Suspicious ($y^*=1$):* Urgent authorization demands, credential re-verification warnings, account termination threats, confidential executive directives.
   - *Benign ($y^*=0$):* Routine informational updates, standard operating summaries, collaborative project milestones, archival receipts.
2. **Family-Specific Nuisance:** Domain-specific vocabulary (e.g. patent docket references, telecom roaming data, shipping bill of ladings) that varies independently of class identity.
3. **Ambiguity Calibration:** Controlled mixture of neutral phrases and subtle wording ensuring text leave-families-out oracle AUC falls within $[0.75, 0.90]$.

### 3.2 Visual Evidence (128x128 RGB Images)
1. **Shared Visual Cues:** Subtle warning banners, red-tinted security callout boxes, and badge highlights ($y^*=1$) versus neutral blue/gray corporate letterheads and divider bars ($y^*=0$).
2. **Nuisance Variance:** Variable bounding box coordinates, font sizes, line weights, and background shading conditioned on template family.
3. **Controlled Sensor Noise:** Additive Gaussian pixel noise $\mathcal{N}(0, \sigma^2)$ ensuring visual leave-families-out oracle AUC falls within $[0.75, 0.90]$.

### 3.3 Heuristic Metadata Evidence
1. **Extractors:** Evaluated strictly through canonical feature extraction functions:
   - `extract_url_score`: Evaluates IP address hostnames, suspicious TLDs, and URL structure.
   - `extract_keyword_score`: Evaluates lexical matches against forensic threat terms.
   - `extract_attachment_score`: Identifies executable, script, and macro-enabled extensions.
   - `extract_timeline_anomaly`: Measures inter-message timestamp burstiness and off-hours transmission.
   - `extract_email_anomaly`: Analyzes SPF/DKIM validation status and display-name spoofing.
2. **Realistic Imperfection:** Benign emails contain realistic mailing-list SPF glitches, external vendor links, and compressed archives. Suspicious lures feature stealthy compromised accounts with passing SPF/DKIM and clean links. Calibrated so heuristic leave-families-out oracle AUC falls within $[0.75, 0.90]$.

---

## 4. True Label Noise Injection

Under Generator v3, ground-truth label noise is mathematically enforced:
1. For every case $i$, artifacts are generated from the true uncorrupted state $y_i^* \in \{0, 1\}$.
2. In the training split, the observed classification label $y_i^{\text{obs}}$ provided to training algorithms is perturbed with probability $\eta = 0.05$:
   $$y_i^{\text{obs}} = \begin{cases} 1 - y_i^* & \text{with probability } \eta \\ y_i^* & \text{with probability } 1 - \eta \end{cases}$$
3. Case items store `item.label = y_obs` alongside metadata fields `metadata["true_label"] = y_true` and `metadata["is_label_flipped"] = bool(y_obs != y_true)`.
4. Unit tests assert that empirical flipped fractions $\frac{1}{N} \sum \mathbb{I}(y_i^{\text{obs}} \ne y_i^*)$ match the configured rate within binomial sampling tolerances.

---

## 5. Statistical Performance Ceilings & Minimum Detectable Effect (MDE)

### 5.1 Oracle Ceilings & Headroom
Oracle models are trained on the training partition and evaluated strictly on the family-disjoint validation partition:
- **Text Oracle Ceiling:** Logistic regression on TF-IDF text features $\in [0.75, 0.90]$.
- **Image Oracle Ceiling:** Logistic regression on color-spatial image features $\in [0.75, 0.90]$.
- **Heuristic Oracle Ceiling:** Logistic regression on the 5 extracted heuristic scores $\in [0.75, 0.90]$.
- **Fused Multimodal Oracle:** Joint logistic model combining text, visual, and heuristic representations:
  $$\text{AUC}_{\text{fused}} \ge \max(\text{AUC}_{\text{text}}, \text{AUC}_{\text{image}}, \text{AUC}_{\text{heur}}) + 0.05$$
- **Benchmark Headroom:**
  $$\text{Headroom} = 1.0000 - \text{AUC}_{\text{fused}}$$

### 5.2 Minimum Detectable Effect (MDE) Simulation
To verify that the benchmark has adequate statistical power to evaluate research questions RQ1 and RQ2, MDE is computed via cluster simulation:
1. At sample size $N_{\text{test}} = 300$ across $K_{\text{test}} = 10$ families, scores are simulated with empirical intra-cluster correlation.
2. 1,000 paired cluster bootstrap replications determine the smallest effect size $\Delta \text{AUC}$ achieving $80\%$ power ($\beta = 0.20$) at two-sided significance $\alpha = 0.05$.
3. **Execution Gate G-MDE:**
   $$\text{MDE} \le \frac{\text{Headroom}}{2}$$
   If $\text{MDE} > \frac{\text{Headroom}}{2}$, the benchmark is insufficiently powered and execution halts.
