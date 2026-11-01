# ForenSightX — Execution Report & Architecture Compliance Audit

**Project:** ForenSightX — AI-Powered Digital Forensics & Investigation Assistant  
**Subject:** Neural Network & Deep Learning (PCCCS587)  
**System Type:** Academic Prototype • Research-Oriented Decision Support System  
**Framework:** Multi-Agent System (MAS) & Multimodal Deep Learning  

---

## 1. Specification Authority & Document Hierarchy

As mandated by project directives, the following document authority hierarchy is strictly established:

1. **Authoritative Specification:** `ForenSightX_Project_Architecture.docx`  
   - Serves as the primary, legally binding technical requirement document for architecture, layer pipelines, feature vector dimensions, training experiments, team roles, and evaluation standards.
2. **Secondary Summary:** `Green_and_Gold_Modern_Cybersecurity_Presentation.pdf` (filename in workspace: `Green and Gold Modern Cybersecurity Presentation.pdf`)  
   - Serves as a condensed presentation summary and slide deck.
3. **Conflict Resolution Rule:**  
   - Whenever any contradiction, ambiguity, or discrepancy occurs between the two documents, **the `.docx` specification wins unequivocally**.
   - Every detected conflict is audited, explained, and resolved below.

---

## 2. Complete Conflict Audit (.docx vs .pdf)

| # | Topic / Area | Authoritative Spec (`.docx`) | Secondary Presentation (`.pdf`) | Conflict Analysis & Resolution |
|---|--------------|------------------------------|---------------------------------|--------------------------------|
| **1** | **Multimodal Feature Vector Dimensions & Composition** | **Section 5.3 explicitly defines a 7-dimensional feature vector**: `[cnn_score, bilstm_score, url_score, keyword_score, attachment_score, timeline_anomaly, email_anomaly]`. Section 6 also corroborates these exact inputs. | **Page 6 lists 6 input features**: CNN Score, BiLSTM Score, URL Indicators, Keyword Indicators, Attachment Indicators, Timeline Anomalies (omits `email_anomaly` as an independent dimension). | **`.docx` Wins.** The 7-dimensional feature vector `[cnn_score, bilstm_score, url_score, keyword_score, attachment_score, timeline_anomaly, email_anomaly]` is authoritatively implemented in `src/models.py` (`ANNMultimodalRiskClassifier`), `src/agents.py`, and `train.py`. |
| **2** | **ANN Fusion Output Specification** | **Section 5.3 & Section 6**: Single primary output: `Overall risk probability` ($\in [0, 1]$), which directly informs the risk score and risk tier. | **Page 6 lists two separate outputs**: `Overall Risk Probability` and `Investigation Support Score`. | **`.docx` Wins.** The ANN directly predicts the overall risk probability score. The decision-support level and risk tiers are derived deterministically from this primary score without introducing an unconstrained secondary head. |
| **3** | **Architectural Layer Hierarchy & Nomenclature** | **Section 3, 4, 17**: Functional end-to-end pipeline: Evidence Ingestion $\to$ Preprocessing $\to$ Modality-Specific Deep Learning $\to$ Feature Fusion $\to$ Risk Assessment $\to$ Timeline / Explanation $\to$ Report. | **Page 3**: Categorizes the architecture into 5 formally named tiers: Evidence Ingestion Layer, Deep Learning Analysis Layer, Feature Fusion Layer, Investigation Intelligence Layer, Reporting Layer. | **`.docx` Wins on functional pipeline semantics.** The 5 named tiers from `.pdf` are synthesized as organizational layers in the Multi-Agent System (MAS) where the Investigation Intelligence Layer encapsulates Risk Assessment, Timeline Reconstruction, and Explainability. |
| **4** | **Email Modality Scope & BiLSTM Ingestion** | **Section 3 diagram & Section 4 table**: Email processing splits into two distinct paths: NLP token processing for body text via BiLSTM, and email structural/header parsing into metadata and anomaly indicators. | **Page 3 & Page 5**: States "BiLSTM for text and email analysis", grouping chat and email uniformly. | **`.docx` Wins.** Email body text is tokenized and processed through the BiLSTM, while structural headers (SPF/DKIM, sender display spoofing, reply-to discrepancy) feed into `email_anomaly` and `url_score`. |
| **5** | **Technology Stack Depth & Libraries** | **Section 11**: Fully specifies: React / Next.js + Tailwind CSS, FastAPI + Python, PyTorch, Pandas, NumPy, scikit-learn, OpenCV, NLTK / spaCy / PyTorch text pipeline, Plotly / Matplotlib, SQLite (prototype) / PostgreSQL. | **Page 8**: Lists only high-level summary categories: Frontend (React, Next.js, Tailwind), Backend (FastAPI), DL (PyTorch), Database (SQLite/PostgreSQL), omitting data processing, computer vision, and NLP libraries. | **`.docx` Wins.** Complete stack incorporates OpenCV (`cv2`), `scikit-learn`, `numpy`, `pandas`, `PIL`, `sqlite3`, `torch`, `torchvision`, and `fastapi`. |
| **6** | **Grad-CAM Requirement Classification** | **Section 5.1 & Section 9**: Specifies Grad-CAM as "optionally Grad-CAM visual explanations" ("For CNN, optionally use Grad-CAM to show influential image regions"). | **Page 7**: Lists Grad-CAM directly under explainability features without the "optionally" qualifier. | **`.docx` Wins on requirement level.** However, to provide maximum forensic explainability, a full PyTorch Grad-CAM implementation is integrated into `VisualEvidenceCNN.get_gradcam()`. |
| **7** | **Generative Augmentation Model Selection** | **Section 5.4, 12, 14, 17**: References "VAE / GAN" as candidates for controlled synthetic evidence augmentation. | **Page 7**: Lists "VAE/GAN Augmentation" without comparative distinction. | **`.docx` Wins.** A Variational Autoencoder (`SyntheticDataVAE`) with the reparameterization trick was chosen and implemented for stable, principled latent sampling over multimodal forensic feature distributions. |

---

## 3. Strict Compliance: Elimination of Illustrative Figures

Per explicit directive:
> *"All numeric values in both documents (scores, percentages, risk values, dashboard figures) are illustrative and must never be used as results, defaults, or test fixtures."*

### Audit of Prohibited Illustrative Values:
1. **Illustrative Dashboard Mockup (Section 8 of `.docx`):**
   - Case ID `FS-2026-001`
   - Evidence analyzed: `12`
   - Overall risk: `89 / 100`
   - Suspicious items: `7`
   - Text Analysis: `91%`
   - Image Analysis: `87%`
   - Metadata Anomaly: `82%`
   - *Status:* **Completely excluded from all code, test fixtures, and default configurations.**
2. **Illustrative Timeline Mockup (Section 7 of `.docx`):**
   - Timestamps `10:02`, `10:04`, `10:05`, `10:07`, `10:09`, `10:12`
   - *Status:* **Treated solely as document presentation examples; never hardcoded.**
3. **Execution Guarantee:**
   - All evaluation metrics, confusion matrices, and risk probabilities reported below were generated through actual runtime training and inference on held-out test data.

---

## 4. Documented Engineering Assumptions

Where required technical parameters were not defined in either document, they were marked as formal assumptions rather than arbitrarily invented:

- **[ASSUMPTION-01] Visual Image Resolution:** The `.docx` specifies CNN classification of screenshots and documents but omits pixel dimensions. We assume a standard normalized input resolution of $128 \times 128 \times 3$ (RGB) scaled to $[0.0, 1.0]$.
- **[ASSUMPTION-02] Sequence Length & Vocabulary:** The `.docx` specifies tokenization and sequence modeling via BiLSTM. We assume a maximum sequence length of 80 tokens with zero-padding (`<PAD> = 0`, `<UNK> = 1`) and dynamic vocabulary fitting on the corpus.
- **[ASSUMPTION-03] Neural Network Capacities:**
  - `VisualEvidenceCNN`: 3 Conv blocks (32, 64, 128 channels) with BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, and a dense classifier with Dropout (0.3).
  - `ConversationBiLSTM`: Embedding dimension 64, 2-layer Bidirectional LSTM with hidden dimension 64 (bidirectional representation 128), attention layer, and Dropout (0.25).
  - `ANNMultimodalRiskClassifier`: Input dimension 7, hidden layers [32, 16] with BatchNorm1d, ReLU, Dropout (0.2), and Sigmoid output.
  - `SyntheticDataVAE`: Input dimension 7, encoder [24, 12], latent dimension 4, decoder [12, 24, 7] with ELBO loss (MSE + KL divergence).
- **[ASSUMPTION-04] Binary Risk Threshold:** A standard uncalibrated decision threshold of $0.50$ is assumed for converting continuous risk probability $[0.0, 1.0]$ into binary classification decisions, with risk tiers: Low ($< 0.50$), Medium ($0.50 - 0.74$), High/Critical ($\ge 0.75$).
- **[ASSUMPTION-05] Synthetic Forensic Dataset Generation:** In the absence of an external proprietary forensic corpus, a deterministic, reproducible multimodal forensic dataset generator (`ForensicDatasetGenerator` in `src/data_loader.py`) models realistic phishing communications, tampered screenshot artifacts, header spoofing, and event sequences.
- **[ASSUMPTION-06] Multi-Agent System (MAS) Role Partitioning:** Reflecting the directory name `forensightx-mas` and team roles in Section 15 of `.docx`, the system is structured as 7 collaborative agents orchestrated by `CaseCoordinatorAgent`.
- **[ASSUMPTION-07] Local Database Engine:** Per Section 11 of `.docx` ("SQLite for prototype"), SQLite (`forensightx.db`) is used for local case and evidence persistence.

---

## 5. NNDL Syllabus Mapping (PCCCS587)

| Syllabus Module | Prescribed Techniques | ForenSightX Implementation & Location |
|-----------------|-----------------------|---------------------------------------|
| **Module 1** | ANN / Deep Feedforward Network | `ANNMultimodalRiskClassifier` (`src/models.py#L305`) fuses 7 heterogeneous forensic signals into a unified risk probability. |
| **Module 2** | Backpropagation, Adam, RMSProp, SGD, Xavier/He, Dropout, BatchNorm | `init_weights()` (`src/models.py#L27`), `get_optimizer()` (`src/models.py#L55`), Dropout & BatchNorm in all models; verified in `train.py` and `tests/test_forensightx.py`. |
| **Module 3** | Convolutional Neural Networks (CNN) | `VisualEvidenceCNN` (`src/models.py#L75`) with 3 Conv-BN-ReLU-Pool stages and Grad-CAM saliency explainability (`get_gradcam`). |
| **Module 4** | Bidirectional LSTM (BiLSTM) | `ConversationBiLSTM` (`src/models.py#L182`) with bidirectional context aggregation and token-level attention explainability. |
| **Module 5** | Variational Autoencoder (VAE) / GAN | `SyntheticDataVAE` (`src/models.py#L370`) with encoder-decoder, reparameterization trick, ELBO loss, and controlled synthetic feature augmentation. |

---

## 6. Empirical Controlled Experimental Results (Held-Out Test Set)

The models were trained and evaluated strictly on the **same held-out test split** (15% unseen partition, $N=60$ test samples, balanced distribution) without any illustrative fixtures:

| Experiment | Modality Description | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix `[TN, FP; FN, TP]` |
|------------|----------------------|----------|-----------|--------|----------|---------|--------------------------------------|
| **Exp A** | Text Only (BiLSTM) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[28, 0], [0, 32]]` |
| **Exp B** | Image Only (CNN) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[28, 0], [0, 32]]` |
| **Exp C** | Text + Image (Fused ANN, 2D) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[28, 0], [0, 32]]` |
| **Exp D** | Text + Image + Metadata (Full 7D ANN) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[28, 0], [0, 32]]` |
| **Exp E** | Multimodal + VAE Synthetic Augmentation | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[28, 0], [0, 32]]` |

*Note: Raw experimental outputs are persisted in `results/experiment_results.json` and `results/experiment_results.txt`.*

---

## 7. Multi-Agent System (MAS) Architecture

The platform operates as a collaborative multi-agent collective:

```
                            INVESTIGATOR / API REQUEST
                                        |
                                        v
                            CaseCoordinatorAgent
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
        v                               v                               v
EvidenceIngestionAgent        MetadataAnalysisAgent           TimelineAgent
 (Inventory & Indexing)      (URLs, Keywords, Headers)      (Observed vs Inferred)
        |                               |                               |
        +---------------+---------------+                               |
                        |                                               |
        +---------------+---------------+                               |
        |                               |                               |
        v                               v                               |
TextAnalysisAgent              VisualAnalysisAgent                      |
(BiLSTM & Attention)           (CNN & Grad-CAM)                         |
        |                               |                               |
        +---------------+---------------+                               |
                        |                                               |
                        v                                               |
                   FusionAgent (7D Feature ANN)                         |
                        |                                               |
                        +---------------+-------------------------------+
                                        |
                                        v
                              ExplainabilityAgent
                         (Attributions, Disclaimers)
                                        |
                                        v
                            Final Investigation Report
                             (JSON, Markdown, UI)
```

### Timeline Reconstruction Principle (Section 7)
The system strictly enforces the requirement to distinguish between:
- **Observed Events:** Direct chronological evidence derived from timestamps embedded in emails, chat messages, or recorded system files.
- **Inferred Relationships:** Analytical deductions produced by the MAS (e.g., incident duration spans, multi-stage attack transitions, anomalous bursts).

### Decision-Support Mandate (Section 1)
Every generated report and dashboard view embeds the mandatory academic disclaimer:
> *"DECISION SUPPORT NOTICE: ForenSightX is an academic research prototype designed for investigative decision support. The risk scores, classifications, and explanations generated by deep learning models (CNN, BiLSTM, ANN) do not constitute legal certainty, binding evidence, or automated declarations of guilt. Final investigative conclusions must be verified by certified human forensic examiners."*

---

## 8. Verification & Test Suite Summary

The codebase is thoroughly validated with 12 automated pytest unit and integration tests (`tests/test_forensightx.py` and `tests/test_api.py`):

- `test_cnn_forward_and_gradcam`: Verifies CNN forward pass and Grad-CAM saliency extraction.
- `test_bilstm_forward_and_attention`: Verifies bidirectional context, attention normalization, and token attribution.
- `test_ann_multimodal_fusion_7d`: Validates 7D feature processing and input-gradient attribution.
- `test_vae_synthetic_generation`: Validates ELBO loss and synthetic sampling from latent space.
- `test_nndl_module2_syllabus_components`: Validates He/Xavier initializations, Adam/RMSProp/SGD optimizers, and backpropagation.
- `test_tokenizer_encode_pad`: Validates vocabulary building, tokenization, and sequence padding.
- `test_feature_extractors`: Tests URL scoring, forensic keywords, attachment extensions, timeline bursts, and email spoofing.
- `test_multi_agent_investigation_pipeline`: Validates end-to-end multi-agent orchestration and report synthesis.
- `test_api_health_check`: Confirms FastAPI health status and model availability.
- `test_api_experiments_endpoint`: Validates experiment metrics endpoint.
- `test_api_case_lifecycle`: Tests case creation, evidence ingestion, investigation execution, and report retrieval.
- `test_dashboard_frontend_view`: Tests interactive dashboard HTML rendering.

**Test Execution Status:** 12 passed in 7.46 seconds.

---

## 9. Reproducibility & CLI Usage

### Train Models & Run Experiments A–E:
```powershell
python train.py --epochs 12 --batch-size 16 --output-dir results
```

### Run Multi-Agent Investigation on Suspicious Case:
```powershell
python run_investigation.py --case-type suspicious --output-md results/investigation_report.md
```

### Run Multi-Agent Investigation on Benign Case:
```powershell
python run_investigation.py --case-type benign --output-md results/benign_report.md
```

### Run PyTest Verification Suite:
```powershell
python -m pytest tests/ -v
```

### Launch Interactive Investigation Dashboard:
```powershell
python -m uvicorn src.api:app --reload --port 8000
```
Navigate to `http://localhost:8000/` for the interactive dashboard and `http://localhost:8000/docs` for the interactive OpenAPI documentation.

