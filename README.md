# ForenSightX — Multimodal Deep Learning & Multi-Agent Digital Forensics Intelligence Platform

ForenSightX is an AI-powered digital forensics investigation and decision-support platform designed for incident response teams. It integrates specialized neural representations (CNN, BiLSTM), a 7D multimodal fusion ANN, a conditional generative VAE, and collaborative forensic agents to analyze complex cyber incident cases.

> **LEGAL DECISION-SUPPORT NOTICE:** ForenSightX is a probabilistic decision-support tool designed strictly to assist qualified forensic examiners. Model inferences and risk scores do not constitute automated legal findings or conclusive evidence of liability.

---

## Quickstart: Running the Live Dashboard

```bash
# 1. Start the FastAPI + Uvicorn investigation service
python -m uvicorn src.api:app --port 8000

# 2. Access the interactive web interface in your browser
http://localhost:8000

# 3. (Optional) Run the automated 10-endpoint smoke test suite
python scripts/smoke_demo.py
```

---

## Demo Dashboard Features

1. **Synthetic Benchmark Warning Banner:** Prominently flags synthetic dataset nature per compliance rules.
2. **One-Click Benchmark Samples:**
   - **Load Suspicious Sample:** CEO impersonation, urgent wire fraud request, and tampered workstation screenshot (Evaluates to ~99.6% Critical Risk).
   - **Load Benign Sample:** Routine sprint standup, CI/CD deployment confirmation, and untampered screenshot (Evaluates to ~0.3% Low Risk).
3. **Evidence Ingestion:** Supports interactive upload of digital screenshots, email headers, and chat conversations.
4. **Authoritative 7D Feature Attribution:**
   - CNN Visual Tampering Score
   - BiLSTM Text Semantics Score
   - URL Risk Score
   - Flagged Keyword Density
   - Attachment Risk Score
   - Timeline Anomaly Score
   - Email Header Anomaly Score
5. **Timeline Reconstruction:** Chronological events distinctly labeled as `[OBSERVED]` (raw artifacts) vs `[INFERRED]` (agent causal milestones).
6. **Explainability & Grad-CAM:**
   - Top forensic risk indicators and salient linguistic tokens.
   - Grad-CAM visual heatmaps highlighting tampered image regions with peak activation metrics.
7. **Read-Only Results Tab:** Dynamically queries `results/experiment_results_v10.json` (zero hardcoded numbers) displaying Arms A–E Mean &plusmn; SD AUC and delta CIs (with Arm E vs D explicitly badged as "no benefit").

---

## Repository Organization

```text
forensightx-mas/
│
├── src/                               # Core platform implementation
│   ├── agents.py                      # Multi-Agent Coordinator & specialized agents
│   ├── api.py                         # FastAPI service & interactive dashboard view
│   ├── data_loader.py                 # Forensic tokenizer, case definitions, feature formulas
│   ├── models/                        # Neural architectures (CNN, BiLSTM, ANN, VAE)
│   │   ├── __init__.py                # Model definitions & Grad-CAM implementation
│   │   ├── factory.py                 # Strict model factory & checkpoint loader
│   │   └── vae_trainer.py             # Conditional VAE training & evaluation
│   ├── data/                          # Hard dataset generators & test isolation guards
│   └── utils/                         # Calibration, metrics, and bootstrap utilities
│
├── models_saved/                      # Trained model checkpoints (strict=True)
│   ├── cnn_model.pt                   # VisualEvidenceCNN checkpoint
│   ├── bilstm_model.pt                # ConversationBiLSTM (concat pooling)
│   ├── ann_fusion_model.pt            # ANNMultimodalRiskClassifier (7D fusion)
│   ├── vae_model.pt                   # SyntheticDataVAE checkpoint
│   └── tokenizer_vocab.json           # Trained forensic tokenizer vocabulary
│
├── results/                           # Write-once benchmark results & table ledgers
│   ├── experiment_results_v10.json    # Canonical Iteration 10 results artifact
│   ├── all_tables_v10.md              # Complete statistical tables ledger
│   └── generate_tables_v10.py         # Literal-free table generator (Rule G20)
│
├── docs/                              # Project documentation & demo captures
│   ├── RESULT_STATUS.md               # Cumulative validity status ledger (Iter 1–10)
│   ├── demo/                          # Fallback demo outputs & screenshots
│   │   ├── suspicious_sample_report.json / .md
│   │   ├── benign_sample_report.json / .md
│   │   ├── gradcam_suspicious.png / gradcam_benign.png
│   │   └── dashboard_preview.html
│   └── ForenSightX_Project_Architecture.docx  # Authoritative system specification
│
├── scripts/                           # Utility & verification scripts
│   └── smoke_demo.py                  # 10-endpoint smoke test & artifact generator
│
├── tests/                             # Comprehensive 93-test regression suite
│   ├── test_api.py                    # API integration tests
│   ├── test_forensightx.py            # Model forward hooks & Grad-CAM tests
│   └── test_iteration10_integrity.py  # Integrity, G20 literals, and gate tests
│
└── requirements.txt                   # Pinned dependency manifest
```

---

## Verification & Testing

Run the full pytest test suite across all 14 test modules:

```bash
python -m pytest -q
```

Expected output:
```text
93 passed, 2 warnings in ~14s
```

