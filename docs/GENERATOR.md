# Hard-Synthetic Forensic Dataset Generator (`docs/GENERATOR.md`)

> [!WARNING]
> **SYNTHETIC BENCHMARK - NOT REAL-WORLD EVIDENCE**  
> This dataset generator produces synthetic forensic cases designed strictly to audit model robustness, multimodal fusion benefits, and statistical controls. All synthetic records, probabilities, and labels are benchmark artifacts and must never be interpreted as real-world evidence or empirical case ground truth.

---

## 1. Architectural Philosophy & Integrity Guarantees

In Iterations 1 through 5, forensic features were either partially or wholly sampled from synthetic Gaussian distributions (e.g. `np.random.normal(0.82, 0.10)`), or created from small keyword templates that caused near-perfect linear separability ($\text{AUC} \approx 1.0$) and posterior collapse in the VAE.

The **Hard-Synthetic Benchmark Generator** (`src/data/hard_generator.py`) enforces strict architectural separation:

```mermaid
flowchart TD
    subgraph Raw Artifact Generation
        TG[Text Generator] -->|Raw Message Text| RT[Evidence Text]
        IG[Image Generator] -->|Synthetic Screenshot/Doc| RI[Evidence Image]
        MG[Metadata Generator] -->|Headers, Events, Attachments| RM[Metadata Dict]
    end

    subgraph Feature Extraction Pipeline (Pure Extractors)
        RT -->|extract_url_score| F1[url_score]
        RT -->|extract_keyword_score| F2[keyword_score]
        RM -->|extract_attachment_score| F3[attachment_score]
        RM -->|extract_timeline_anomaly| F4[timeline_anomaly]
        RM -->|extract_email_anomaly| F5[email_anomaly]
    end

    subgraph Neural Evidence Models
        RI -->|VisualEvidenceCNN| F6[cnn_score]
        RT -->|ConversationBiLSTM| F7[bilstm_score]
    end

    subgraph 7D Fusion Vector
        F1 & F2 & F3 & F4 & F5 & F6 & F7 --> FUSE[7D Multimodal Feature Vector]
    end
```

### Invariants Enforced:
1. **Zero Direct Feature Sampling**: No feature vector column is drawn from a random number generator or hardcoded distribution. All features are extracted strictly by executing the certified extractor functions from `src/data_loader.py` on the raw evidence strings, images, and metadata dicts.
2. **Within-Class Variance Guarantee**: In both benign ($y=0$) and suspicious ($y=1$) classes, every heuristic feature has a non-zero sample variance ($\sigma > 0$). Benign samples occasionally contain suspicious cues (e.g., promotional urgency, legitimate zip attachments), while suspicious samples occasionally use benign phrasing.
3. **Single-Feature Ceiling ($\text{AUC} \le 0.85$)**: No single feature is allowed to achieve $\text{AUC} > 0.85$ on the benchmark (ASSUMPTION-07). Multimodal fusion is required to achieve high classification accuracy.
4. **Single-Modality Ceiling**: Text-only and image-only unimodal ceilings (measured by Logistic Regression and GBDT) remain strictly below the multimodal fused ceiling.
5. **Template Family Holdout**: The test split contains held-out template families (`shipping_delivery_notice`, `vpn_credential_update`, `conference_invitation`) that never appear in the training or validation splits.
6. **Record-Level Pre-Split**: Raw records (texts and images) are partitioned into train, validation, and test splits *prior* to case pairing, ensuring zero cross-split contamination.

---

## 2. Benchmark Parameters & Assumptions

| Parameter Name | Value | Scope / Location | Spec Reference / Tag |
|---|---|---|---|
| `total_samples` | 1,200 | Dataset size | ASSUMPTION-10 |
| `train_samples` | 700 (58.33%) | Training split size | ASSUMPTION-10 |
| `val_samples` | 200 (16.67%) | Validation split size | ASSUMPTION-10 |
| `test_samples` | 300 (25.00%) | Test split size ($\ge 300$) | ASSUMPTION-06 |
| `single_feature_auc_cap` | 0.85 | Feature ceiling | ASSUMPTION-07 |
| `label_noise_rate` | 0.05 (5.0%) | Label perturbation | ASSUMPTION-08 |
| `modality_conflict_rate` | 0.05 (5.0%) | Cross-modal cue conflict | ASSUMPTION-09 |
| `manifest_path` | `data/split_manifest_hard_v1.json` | Manifest path | Write-Once Enforced |

---

## 3. Template Families & Modality Artifacts

### 3.1. Training & Validation Template Families
- `invoice_payment`: Benign vendor invoices vs fraudulent wire diversion requests.
- `account_suspension`: Legitimate IT maintenance warnings vs credential phishing lures.
- `urgent_wire`: Routine expense approvals vs CEO wire impersonation.
- `meeting_reschedule`: Calendar updates with harmless zoom links vs malicious redirect links.
- `project_status`: Engineering sprint summaries vs exfiltration alert memos.

### 3.2. Test Split Held-Out Template Families
- `shipping_delivery_notice`: Package delivery tracking emails with parcel receipt images.
- `vpn_credential_update`: Enterprise remote-access policy memos with IT portal screenshots.
- `conference_invitation`: Academic/industry summit registration invitations with itinerary graphics.

### 3.3. Raw Visual Evidence Synthesis
Images are rendered as $128 \times 128 \times 3$ RGB PIL images containing structured graphical layouts:
- **Benign Visuals**: Clean corporate document headers, soft gradient buttons, uniform borders, low noise.
- **Suspicious Visuals**: Distorted login forms, irregular compression artifacts, suspicious QR codes, high pixel noise.
- **Controlled Noise**: Visual cues are calibrated to yield single-modality CNN AUC in the range $[0.70, 0.82]$, leaving room for cross-modal fusion.

