# ForenSightX — Multimodal Deep Learning & Multi-Agent Digital Forensics Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Pytest-69%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20%2F%20MIT-lightgrey.svg)]()

> **LEGAL DECISION-SUPPORT NOTICE:** ForenSightX is an academic decision-support platform designed to assist qualified digital forensic examiners and incident responders. All risk classifications, neural activations, and anomaly scores represent probabilistic statistical patterns and **do not constitute automated legal determinations of guilt, liability, or conclusive proof of fraud**.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
  - [1. Visual Evidence CNN](#1-visual-evidence-cnn-image-tampering--forgery)
  - [2. Conversation BiLSTM](#2-conversation-bilstm-linguistic-deception--urgency)
  - [3. 7D Multimodal Fusion ANN](#3-7d-multimodal-fusion-ann-holistic-risk-scoring)
  - [4. Conditional Generative VAE](#4-conditional-generative-vae-synthetic-augmentation)
  - [5. Collaborative Multi-Agent Pipeline](#5-collaborative-multi-agent-pipeline)
- [Interactive Web Dashboard](#interactive-web-dashboard)
- [Benchmark Results & Empirical Findings](#benchmark-results--empirical-findings)
- [Repository Organization](#repository-organization)
- [Installation & Quickstart](#installation--quickstart)
- [Execution & Usage Guide](#execution--usage-guide)
  - [Running the Web Dashboard](#running-the-web-dashboard)
  - [Running the Automated Smoke Demo](#running-the-automated-smoke-demo)
  - [Running Investigations via CLI](#running-investigations-via-cli)
  - [Training Models](#training-models)
  - [Running the Full Benchmark Suite](#running-the-full-benchmark-suite)
- [REST API Reference](#rest-api-reference)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Academic Reference & Syllabus Mapping](#academic-reference--syllabus-mapping)

---

## Overview

Modern digital forensic investigations face heterogeneous multi-source evidence—including chat logs, phishing emails, workstation screenshots, system timestamps, and digital file attachments. Manual cross-examination is labor-intensive and prone to cognitive bias.

**ForenSightX** solves this challenge by unifying:
1. **Specialized Unimodal Neural Networks:** Deep representations extracting spatial manipulation signatures (CNN) and sequential deceptive indicators (BiLSTM).
2. **7D Multimodal Fusion:** An Artificial Neural Network combining vision, language, and structured metadata anomaly metrics.
3. **Generative Modeling:** A Conditional Variational Autoencoder ($\beta$-VAE) exploring synthetic minority oversampling.
4. **Multi-Agent Orchestration:** Autonomous agents handling evidence ingestion, modality analysis, chronological timeline reconstruction, and cross-modality consensus reporting.

All neural network checkpoints (`models_saved/`, ~2.2 MB total) are pre-trained and version-controlled, allowing immediate local deployment without retraining.

---

## System Architecture

```
                       [ Incoming Incident Evidence ]
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
  [ Screenshots / Images ]    [ Text / Emails ]       [ Metadata & Logs ]
           │                         │                         │
           ▼                         ▼                         ▼
   VisualEvidenceCNN       ConversationBiLSTM       Heuristic Extractors
  (Conv2D + BN + GradCAM)  (BiLSTM + Attention)    (URL, Keyword, Headers)
           │                         │                         │
     Visual Score              Text Score               5 Feature Scores
           └─────────────────────────┼─────────────────────────┘
                                     │
                                     ▼
                     [ 7D Multimodal Feature Vector ]
                                     │
                                     ▼
                      ANNMultimodalRiskClassifier
                  (Dense + Dropout + Sigmoid Head)
                                     │
                                     ▼
                         Overall Case Risk Score
                                     │
                                     ▼
                     Multi-Agent Forensic Coordinator
             (Timeline: [OBSERVED] vs [INFERRED] + Report)
```

### 1. Visual Evidence CNN (Image Tampering & Forgery)
- **Architecture:** 3 sequential convolutional blocks (`Conv2D` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D`), followed by Adaptive Average Pooling, dense hidden layer (64 units), Dropout ($p=0.3$), and a Sigmoid classification head.
- **Explainability:** Built-in **Grad-CAM** (Gradient-weighted Class Activation Mapping) targeting the final convolutional layer (`conv3`) to generate pixel-level heatmaps highlighting tampered or forged image regions.
- **Input Dimensions:** 3-channel RGB images resized to $64 \times 64$ pixels.

### 2. Conversation BiLSTM (Linguistic Deception & Urgency)
- **Architecture:** Trainable embedding layer ($D=64$), 2-layer Bidirectional LSTM (`hidden_dim=64`, bidirectional output dimension $128$), and multi-head additive self-attention pooling.
- **Explainability:** Token-level attention weights isolate deceptive phrases, urgency manipulation (e.g., *"wire transfer"*, *"confidential"*, *"immediate action"*), and social engineering markers.
- **Tokenizer:** Custom frequency-based `ForensicTokenizer` with isolated train-only fitting and `[UNK]` handling for zero out-of-vocabulary leakage.

### 3. 7D Multimodal Fusion ANN (Holistic Risk Scoring)
The multimodal risk classifier takes a normalized 7-dimensional feature vector $\mathbf{x} \in [0, 1]^7$:
- **$x_1$ — CNN Visual Score:** Probability of image tampering / forged credential screenshot.
- **$x_2$ — BiLSTM Text Score:** Probability of social engineering or linguistic deception.
- **$x_3$ — URL Risk Score:** Lexical entropy, suspicious TLD detection (`.xyz`, `.top`, `.ru`), and IP-based hostnames.
- **$x_4$ — Flagged Keyword Density:** Normalized frequency of high-risk forensic terms (*wire*, *swift*, *credentials*, *invoice*).
- **$x_5$ — Attachment Risk Score:** Executable extensions (`.exe`, `.scr`, `.vbs`, double extensions) and spoofing heuristics.
- **$x_6$ — Timeline Anomaly Score:** Off-hours transmission, rapid bursts, and irregular inter-message intervals.
- **$x_7$ — Email Header Anomaly Score:** SPF/DKIM validation failures, display name spoofing, and mismatched reply-to headers.

**Network Topology:** Dense(7 $\to$ 32) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.2) $\to$ Dense(32 $\to$ 16) $\to$ ReLU $\to$ Dense(16 $\to$ 1) $\to$ Sigmoid.

### 4. Conditional Generative VAE (Synthetic Augmentation)
- **Architecture:** $\beta$-VAE conditioned on class labels $\mathbf{y} \in \{0, 1\}^2$ with 4-dimensional Gaussian latent space $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$.
- **Free-Bits KL Objective:** Implements minimum capacity per latent dimension ($0.25$ bits) to prevent posterior collapse on low-dimensional forensic tabular manifolds.
- **Controlled Finding:** Rigorous empirical evaluation demonstrates that synthetic VAE augmentation on clean tabular forensic features does **not** improve discrimination over the empirical baseline ($\Delta \text{AUC} = -0.0062$, confirmed no benefit).

### 5. Collaborative Multi-Agent Pipeline
Four specialized forensic agents coordinate case assessment:
1. **Evidence Ingestion Agent:** Hashes artifacts (SHA-256), validates case IDs, and extracts chronological timestamps.
2. **Modality Analysis Agent:** Dispatches artifacts to the CNN and BiLSTM inference pipelines.
3. **Timeline Reconstruction Agent:** Synthesizes chronological timelines distinctly categorizing raw verified events (`[OBSERVED]`) from agent causal deductions (`[INFERRED]`).
4. **Forensic Consensus Coordinator:** Combines 7D neural features, assigns forensic risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), extracts key indicators, and generates audit reports.

---

## Interactive Web Dashboard

ForenSightX features a real-time web dashboard running on FastAPI:

- **Synthetic Benchmark Banner:** Explicitly flags synthetic evaluation datasets per compliance standards.
- **One-Click Benchmark Samples:**
  - **Load Suspicious Sample:** Pre-loads a CEO wire fraud impersonation incident with phishing email, urgent chat, and tampered workstation screenshot (**Evaluates to ~99.6% Risk / CRITICAL**).
  - **Load Benign Sample:** Pre-loads a routine DevOps deployment confirmation and standup discussion (**Evaluates to ~0.3% Risk / LOW**).
- **Evidence Uploader:** Interactive upload for images (PNG/JPG with Grad-CAM preview), chat transcripts, and email headers.
- **7D Feature Radar Breakdown:** Individual feature progress bars showing visual, textual, and metadata anomaly contributions.
- **Grad-CAM Visualizer:** Visual heatmap overlaid onto uploaded images highlighting areas triggering neural tampering detection.
- **Reconstructed Timeline:** Filterable timeline with explicit `[OBSERVED]` and `[INFERRED]` event tags.
- **Dynamic Results Viewer:** Connects directly to `results/experiment_results_v10.json` (zero hardcoded numbers) displaying all 7 experimental arms, AUC statistics, and bootstrap confidence intervals.

---

## Benchmark Results & Empirical Findings

Evaluated on the **hard_v4 benchmark dataset** (400 cases: 280 train, 60 validation, 60 test) with strictly disjoint template families (zero cross-split template leakage) across 10 randomized seeds:

| Arm | Description | Test AUC (Mean ± SD) | Balanced Accuracy | Brier Score |
|:---:|:---|:---:|:---:|:---:|
| **A** | Text-Only BiLSTM | $0.6974 \pm 0.0063$ | $0.6833 \pm 0.0000$ | $0.2117 \pm 0.0016$ |
| **B** | Image-Only CNN | $0.8544 \pm 0.0000$ | $0.7667 \pm 0.0000$ | $0.1264 \pm 0.0000$ |
| **C** | ANN on `[bilstm, cnn]` | $0.8711 \pm 0.0000$ | $0.7833 \pm 0.0000$ | $0.1194 \pm 0.0000$ |
| **D** | **ANN on Full 7D Multimodal** | **$0.9311 \pm 0.0000$** | **$0.8500 \pm 0.0000$** | **$0.0768 \pm 0.0000$** |
| **E** | ANN 7D + VAE Augmentation | $0.9250 \pm 0.0000$ | $0.8500 \pm 0.0000$ | $0.0818 \pm 0.0000$ |
| **A-h** | Heuristics-Only Baseline | $0.7933 \pm 0.0000$ | $0.7333 \pm 0.0000$ | $0.1654 \pm 0.0000$ |
| **S** | Multi-Agent Coordinator | $0.9311 \pm 0.0000$ | $0.8500 \pm 0.0000$ | $0.0768 \pm 0.0000$ |

### Statistical Hypothesis Testing & Delta CIs (95% Bootstrap CI):
1. **Multimodal Fusion Superiority (Arm D vs Arm B):**  
   $$\Delta \text{AUC} = +0.0768 \quad [0.0536, 0.1009] \quad (p < 0.001)$$  
   *Conclusion:* Statistically significant improvement. Multimodal fusion outperforms the best unimodal model by +7.68% AUC.
2. **Tabular Feature Contribution (Arm D vs Arm C):**  
   $$\Delta \text{AUC} = +0.0600 \quad [0.0387, 0.0805] \quad (p < 0.001)$$  
   *Conclusion:* Incorporating the 5 heuristic metadata features significantly improves discrimination over neural vision+language representations alone.
3. **Synthetic VAE Augmentation (Arm E vs Arm D):**  
   $$\Delta \text{AUC} = -0.0062 \quad [-0.0124, 0.0052]$$  
   *Conclusion:* **No benefit.** The 95% confidence interval spans zero, confirming that VAE feature oversampling does not improve performance on clean tabular forensic data.

---

## Repository Organization

```text
Deep-learning/
├── README.md                          # Comprehensive documentation (this file)
├── requirements.txt                   # Frozen, tested dependencies
├── run_investigation.py               # CLI runner for case investigations
├── run_iteration10_investigations.py  # Benchmark execution harness
├── train.py                           # Deep learning model training script
│
├── config/                            # Model hyperparameters and benchmark settings
│   ├── benchmark_config.yaml          # Model dimensions, training epochs, learning rates
│   ├── data_mode.yaml                 # Dataset mode configuration
│   ├── datasets.example.yaml          # Template schema for external datasets
│   └── label_map.example.yaml         # Label mapping schema
│
├── data/                              # Dataset split manifests
│   ├── split_manifest.json            # Base 400-case split manifest
│   └── split_manifest_hard_v4.json    # hard_v4 family-isolated benchmark manifest
│
├── docs/                              # Project documentation & architectural blueprints
│   ├── ARCHITECTURE.md                # Full NNDL syllabus mapping & academic blueprint
│   ├── FEATURES.md                    # Mathematical formulas for all 7 forensic features
│   ├── GENERATOR.md                   # Synthetic dataset generator & family isolation rules
│   ├── RESULT_STATUS.md               # Historical audit trail & control verification
│   ├── BENCHMARK_REPORT_v10.md        # Definitive benchmark execution report
│   └── demo/                          # Fallback sample reports & Grad-CAM visual assets
│       ├── benign_sample_report.json
│       ├── benign_sample_report.md
│       ├── suspicious_sample_report.json
│       ├── suspicious_sample_report.md
│       ├── gradcam_benign.png
│       ├── gradcam_suspicious.png
│       └── dashboard_preview.html
│
├── models_saved/                      # Pre-trained neural weights (~2.2MB total)
│   ├── cnn_model.pt                   # VisualEvidenceCNN checkpoint (1.4 MB)
│   ├── bilstm_model.pt                # ConversationBiLSTM checkpoint (768 KB)
│   ├── ann_fusion_model.pt            # ANNMultimodalRiskClassifier checkpoint (10 KB)
│   ├── ann_augmented_model.pt         # Augmented ANN model checkpoint (10 KB)
│   ├── vae_model.pt                   # SyntheticDataVAE checkpoint (10 KB)
│   └── tokenizer_vocab.json           # Train-fitted tokenizer vocabulary (4 KB)
│
├── results/                           # Benchmark evaluation results (queried by API)
│   ├── experiment_results_v10.json    # Complete benchmark run statistics across 10 seeds
│   ├── all_tables_v10.md              # Markdown tables generated from JSON
│   └── generate_tables_v10.py         # Table emitter script (zero numeric literals)
│
├── scripts/                           # Utilities & verification scripts
│   └── smoke_demo.py                  # 10-endpoint automated smoke verification
│
├── src/                               # Core platform source code
│   ├── __init__.py
│   ├── agents.py                      # Multi-agent investigation pipeline & coordinator
│   ├── api.py                         # FastAPI backend + interactive dashboard UI
│   ├── data_loader.py                 # Tokenizer, feature extractors, dataset loaders
│   ├── data/
│   │   ├── adapters/                  # Evidence adapters for text, emails, and images
│   │   ├── case_builder.py            # Multimodal case pairing & split generation
│   │   ├── hard_generator.py          # hard_v4 benchmark generator with family isolation
│   │   ├── test_guard.py              # Single-use test evaluation guard
│   │   └── validate_config.py         # Config file schema validator
│   ├── models/
│   │   ├── __init__.py                # CNN, BiLSTM, 7D Fusion ANN, VAE architectures
│   │   ├── factory.py                 # Strict model loader factory
│   │   └── vae_trainer.py             # Conditional VAE trainer with free-bits loss
│   └── utils/
│       ├── __init__.py
│       └── metrics.py                 # Confusion matrices, AUC, bootstrap CIs
│
└── tests/                             # Comprehensive test suite (69 tests)
    ├── test_api.py                    # Endpoints, dashboard, file uploads, validation (5 tests)
    ├── test_forensightx.py            # Neural networks, Grad-CAM, agents, pipeline (8 tests)
    ├── test_regression.py             # Autograd, batchnorm, formulas, LOFO, VAE math (46 tests)
    └── test_benchmark_integrity.py    # Benchmark v10 integrity, weights, test guard (10 tests)
```

---

## Installation & Quickstart

### Prerequisites
- **Python:** 3.10 or 3.11 recommended
- **Operating System:** Windows, macOS, or Linux
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/Jyotirmoy-006/Deep-learning.git
cd Deep-learning
```

### 2. Set Up a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Execution & Usage Guide

### Running the Web Dashboard
Launch the interactive web server:
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to:
```
http://localhost:8000
```
- Click **"Load suspicious sample"** to test an incident with ~99.6% fraud risk.
- Click **"Load benign sample"** to test a legitimate communication with ~0.3% risk.
- View the **Grad-CAM visual heatmap**, **7D feature decomposition**, and **Chronological Timeline**.
- Click the **"Results" tab** to inspect benchmark statistics queried live from `results/experiment_results_v10.json`.

### Running the Automated Smoke Demo
Verify all 10 platform endpoints and generate sample outputs:
```bash
python scripts/smoke_demo.py
```
This runs an end-to-end audit: checks health, loads samples, runs investigations, verifies Grad-CAM output, tests evidence upload, and stores fallback artifacts in `docs/demo/`.

### Running Investigations via CLI
Investigate a single case file from the command line:
```bash
python run_investigation.py --case docs/demo/suspicious_sample_report.json
```

### Training Models
Train unimodal and multimodal models from scratch:
```bash
# Train CNN, BiLSTM, and Multimodal ANN on benchmark splits
python train.py --config config/benchmark_config.yaml
```

### Running the Full Benchmark Suite
Execute the complete 10-seed benchmark evaluation across Arms A–E:
```bash
python run_iteration10_investigations.py
```

---

## REST API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serves the interactive dashboard HTML interface. |
| `GET` | `/api/health` | Health check verifying model status and loaded checkpoints. |
| `GET` | `/api/results_v10` | Returns live benchmark evaluation statistics (Arms A–E, AUC, CIs). |
| `GET` | `/api/experiments` | Returns experiment metadata and configuration parameters. |
| `GET` | `/api/cases` | Lists all active investigation cases. |
| `POST` | `/api/cases` | Creates a new investigation case. |
| `POST` | `/api/cases/load_sample` | Loads a pre-built benchmark sample (`suspicious` or `benign`). |
| `POST` | `/api/cases/{case_id}/investigate` | Runs the multi-agent pipeline and returns the 7D risk report. |
| `POST` | `/api/cases/{case_id}/upload_evidence` | Ingests new evidence (image base64 URI, text, or email headers). |
| `GET` | `/api/cases/{case_id}/gradcam/{evidence_id}` | Returns a PNG Grad-CAM heatmap visualization for an image. |

---

## Testing & Quality Assurance

ForenSightX maintains a comprehensive test suite of **69 automated tests** covering neural modeling, mathematical formulas, autograd mechanics, benchmark integrity, and API contracts.

Run the test suite:
```bash
python -m pytest -q
```
Expected output:
```text
.....................................................................    [100%]
69 passed, 2 warnings in 10.92s
```

### Test Suite Structure
1. `tests/test_api.py` (5 tests): Validates FastAPI endpoints, dashboard view, case lifecycles, and rejection of invalid case IDs.
2. `tests/test_forensightx.py` (8 tests): Tests CNN forward pass & Grad-CAM tensor gradients, BiLSTM attention pooling, 7D ANN multimodal fusion, VAE generation, and multi-agent coordination.
3. `tests/test_regression.py` (46 tests): Tests feature extraction math, BatchNorm evaluation with batch size 1, leaf tensor safety in autograd, strict checkpoint loading, VAE closed-form parameter math, and dataset adapter purity.
4. `tests/test_benchmark_integrity.py` (10 tests): Strictly enforces benchmark integrity rules—verifies that all printed tables derive dynamically from results JSON without hardcoded literals, asserts test guard single-use isolation, and validates base-model competence gates.

---

## Academic Reference & Syllabus Mapping

This platform maps directly to key concepts in **Neural Networks & Deep Learning (NNDL)**:

| NNDL Concept | Implementation in ForenSightX | Source Reference |
|:---|:---|:---|
| **Module 1: Deep Feedforward Networks** | 7D Multimodal Risk Classifier ANN with dense layers and sigmoid activation | [`src/models/__init__.py`](src/models/__init__.py) |
| **Module 2: Optimization & Regularization** | Backpropagation, Adam/RMSProp/SGD sweeps, Xavier/He initialization, BatchNorm, Dropout | [`train.py`](train.py), [`src/models/__init__.py`](src/models/__init__.py) |
| **Module 3: Convolutional Neural Networks** | 3-block CNN with feature maps, max pooling, and Grad-CAM interpretability | [`VisualEvidenceCNN`](src/models/__init__.py) |
| **Module 4: Recurrent & Sequence Models** | Bidirectional LSTM with multi-head self-attention pooling | [`ConversationBiLSTM`](src/models/__init__.py) |
| **Module 5: Generative Models** | Conditional $\beta$-VAE with latent reparameterization and free-bits loss | [`SyntheticDataVAE`](src/models/__init__.py) |
| **Explainable AI (XAI)** | Grad-CAM visual heatmaps + token attention attribution | [`src/models/__init__.py`](src/models/__init__.py) |
