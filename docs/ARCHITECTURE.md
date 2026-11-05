# ForenSightX — Project Architecture & Implementation Blueprint

FORENSIGHTX

AI-Powered Digital Forensics & Investigation Assistant

Neural Network & Deep Learning (PCCCS587)Project Architecture & Implementation Blueprint

Core concept: Analyze heterogeneous digital evidence using CNN, BiLSTM and ANN, then study whether multimodal fusion and VAE/GAN-based data augmentation improve digital-fraud investigation support.

Academic prototype • Research-oriented • Decision-support system

1. Project Overview

ForenSightX is a multimodal digital-forensics intelligence assistant designed as an academic prototype. It accepts digital evidence such as chat logs, emails, screenshots/images and metadata, analyzes each evidence modality with an appropriate deep-learning model, combines the resulting features, estimates an overall risk score, reconstructs a probable evidence timeline and produces an explainable investigation report.

The system is decision support, not an autonomous legal authority. It should identify patterns and evidence indicators rather than declare that a person is guilty or that an item is legally proven to be fraudulent.

Primary Objectives

Analyze chat and email evidence with sequence-based deep learning.

Analyze visual evidence with CNN-based classification.

Fuse multimodal features into an ANN-based risk classifier.

Demonstrate backpropagation, optimizers, initialization and regularization experimentally.

Use VAE/GAN-based synthetic data augmentation under controlled experiments.

Reconstruct a timestamped evidence timeline.

Provide interpretable indicators and a structured investigation report.

2. NNDL Syllabus Mapping

NNDL Module

Technique

ForenSightX Role

Module 1

ANN / Deep Feedforward Network

Multimodal feature fusion and overall risk classification

Module 2

Backpropagation, Adam, RMSProp, SGD, Xavier/He, Dropout, BatchNorm

Training optimization, initialization and regularization experiments

Module 3

CNN

Image, screenshot and document-evidence analysis

Module 4

BiLSTM

Chat/email sequence analysis and contextual classification

Module 5

VAE / GAN

Synthetic-data generation and augmentation experiments

3. High-Level System Architecture

INVESTIGATOR / USER                                  |                                  v                           CASE CREATION                                  |                                  v                         EVIDENCE INGESTION                                  |                 +----------------+----------------+                 |                |                |                 v                v                v              TEXT/CHAT         IMAGE            EMAIL                 |                |                |                 v                v                v          NLP PREPROCESS     IMAGE PREP      EMAIL PARSING                 |                |                |                 v                v                |              BiLSTM             CNN              |                 |                |                |                 +----------------+----------------+                                  |                                  v                           FEATURE FUSION                                  |                                  v                                 ANN                                  |                         +--------+--------+                         |                 |                         v                 v                     RISK SCORE         TIMELINE                         |                 |                         +--------+--------+                                  |                                  v                           EXPLAINABILITY                                  |                                  v                         INVESTIGATION REPORT

Architecture Principle

The project is a pipeline rather than five disconnected model demonstrations: evidence ingestion → preprocessing → modality-specific deep learning → feature fusion → risk assessment → timeline/explanation → report.

4. Evidence Ingestion Layer

Evidence Type

Examples

Processing

Text / Chat

Chat logs, messages

Cleaning, tokenization, sequence preparation

Email

Subject, body, URLs, attachments

Parsing, text features and metadata extraction

Image

Screenshots, scanned documents

Resize, normalize and augmentation

Metadata

Timestamps, event records

Parsing, normalization and chronological sorting

5. Model Architecture

5.1 CNN — Visual Evidence Analyzer

Input → image preprocessing → convolution blocks → pooling → dense layer → classification.

The CNN should be trained on a clearly defined visual classification task, such as suspicious versus non-suspicious screenshots or manipulated versus non-manipulated images. The exact classes must be determined from the selected dataset.

Output: class probability, confidence score and optionally Grad-CAM visual explanations.

5.2 BiLSTM — Conversation Intelligence

Input text → tokenization → embedding → forward/backward LSTM → concatenation → dense layer → classification.

The BiLSTM analyzes sequence context in chat/email text and classifies samples into predefined categories such as benign, suspicious or phishing/scam-related.

5.3 ANN — Multimodal Risk Classifier

The ANN receives structured outputs rather than raw text/images. Example inputs are CNN suspicion probability, BiLSTM probability, suspicious-keyword count, URL indicators, attachment indicators and timeline anomaly score.

Example feature vector:[cnn_score, bilstm_score, url_score, keyword_score, attachment_score, timeline_anomaly, email_anomaly]                 |                 v              ANN / MLP                 |                 v          Overall risk probability

5.4 VAE / GAN — Synthetic Data Augmentation

A generative model can be trained on an appropriate training subset to create synthetic examples for controlled augmentation experiments. The synthetic data are not presented as real evidence and are not used to fabricate investigation findings.

The research comparison is: baseline training without augmentation versus training with synthetic augmentation, evaluated on the same held-out test set.

6. Feature Fusion — Core System Component

CNN visual score            BiLSTM text score             Email / URL indicators         +--> FEATURE VECTOR --> ANN --> RISK SCOREKeyword / language features   /Timeline anomaly              /Metadata indicators           /

Feature fusion is the main system-level contribution. It allows the team to test whether heterogeneous evidence provides a stronger signal than any individual modality.

7. Timeline Reconstruction

Events extracted from the uploaded evidence are normalized into timestamped records and sorted chronologically. The interface should clearly distinguish observed events from inferred relationships.

10:02  Suspicious email received10:04  Link interaction recorded10:05  Credential / OTP request10:07  Conversation begins10:09  Suspicious request detected10:12  Transaction / event recorded

8. Investigation Dashboard

FORENSIGHTX — CASE FS-2026-001Evidence analyzed: 12Overall risk:       89 / 100Suspicious items:    7TEXT ANALYSIS       91%IMAGE ANALYSIS      87%METADATA ANOMALY    82%KEY INDICATORS✓ Credential request✓ Urgency language✓ Suspicious URL✓ Visual anomaly✓ Abnormal event sequence

The values above are illustrative interface examples only. The final project must display only values obtained from actual experiments.

9. Explainability

Display the indicators that contributed to a high-risk result.

For CNN, optionally use Grad-CAM to show influential image regions.

For text, show extracted linguistic indicators or important tokens.

Display model confidence and evidence-level scores separately.

Avoid presenting an AI prediction as legal proof or certainty.

10. Final Investigation Report

Case ID and evidence inventory.

Individual model outputs and confidence values.

Overall risk score.

Chronological timeline.

Key evidence indicators.

Model limitations and uncertainty.

A clear statement that AI output is decision support.

11. Recommended Technology Stack

Layer

Technology

Frontend

React / Next.js + Tailwind CSS

Backend

FastAPI + Python

Deep Learning

PyTorch

Data Processing

Pandas, NumPy, scikit-learn

Image Processing

OpenCV

NLP

NLTK / spaCy / PyTorch text pipeline

Visualization

Plotly / Matplotlib

Database

SQLite for prototype; PostgreSQL if required

12. Development Plan

Phase

Work

Deliverable

1

Problem definition, literature review, dataset selection

Research specification

2

Preprocessing and baseline models

Clean datasets + baselines

3

CNN development

Working visual classifier

4

BiLSTM development

Working text classifier

5

ANN fusion model

Multimodal risk classifier

6

VAE/GAN augmentation

Synthetic-data experiment

7

Timeline + explainability

Investigation layer

8

FastAPI + dashboard

Integrated prototype

9

Evaluation + ablation

Results and graphs

10

Documentation + paper + presentation

Final submission

13. Evaluation Strategy

Use accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices. Accuracy alone is not sufficient, especially when the classes are imbalanced.

The most valuable controlled experiments are:

Experiment A: Text onlyExperiment B: Image onlyExperiment C: Text + ImageExperiment D: Text + Image + MetadataExperiment E: Best multimodal model + synthetic-data augmentation

Use the same held-out test set for the comparisons whenever possible. This makes the project experimentally defensible.

14. Publication / Research Direction

Do not frame the paper simply as 'an AI fraud detector'. The stronger research framing is a study of multimodal evidence fusion and data augmentation for digital-fraud investigation support.

<b>Primary research question:</b> Does multimodal evidence fusion improve classification compared with single-modality deep learning models?

<b>Secondary research question:</b> Can VAE/GAN-based synthetic-data augmentation improve robustness when labeled forensic data are limited?

A publication-oriented report should document datasets, preprocessing, architectures, hyperparameters, baselines, ablation studies, evaluation metrics, limitations and reproducibility details.

15. Team Division — 10 Members

Members

Responsibility

1–2

CNN / image evidence analysis

3–4

BiLSTM / NLP / sequence analysis

5

ANN / feature fusion / risk classifier

6

VAE/GAN / augmentation experiments

7

Datasets, preprocessing and evaluation

8

FastAPI backend and database

9

Frontend/dashboard

10

Integration, testing, documentation and deployment

16. Recommended MVP Scope

The first complete version should contain four core components: CNN visual classification, BiLSTM text classification, ANN multimodal risk scoring, and a VAE/GAN augmentation experiment. Timeline visualization and explainability should sit on top of these components.

Avoid real-world interception, account access, encryption breaking, private-device tracking, police databases or automated declarations of guilt. Use public or synthetic datasets and present the system as an academic research and decision-support prototype.

17. Final Architecture in One View

FORENSIGHTX                              |                       DIGITAL EVIDENCE                              |            +-----------------+-----------------+            |                 |                 |            v                 v                 v          TEXT              IMAGE             EMAIL            |                 |                 |         BiLSTM              CNN          NLP / Metadata            |                 |                 |            +-----------------+-----------------+                              |                       FEATURE FUSION                              |                             ANN                              |                       +------+------+                       |             |                       v             v                   RISK SCORE     TIMELINE                       |             |                       +------+------+                              |                       EXPLAINABILITY                              |                              v                       FINAL REPORTParallel training research pipeline:Real data -> VAE/GAN -> synthetic augmentation          -> retraining -> robustness comparison

18. Final Project Positioning

ForenSightX — A Multimodal Deep Learning Framework for Digital Evidence Analysis and Investigation Support.

The project's strength is not the number of neural networks. Its strength is the coherent workflow in which different evidence modalities are analyzed independently, fused into a measurable risk assessment and presented with timeline and explainability layers.

Important Note

All accuracy, confidence, risk and timeline values shown in this document are illustrative examples. Final documentation must contain only results generated from the team's actual experiments and clearly identified datasets.