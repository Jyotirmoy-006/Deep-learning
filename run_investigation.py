"""
ForenSightX - Multi-Agent Digital Forensics Investigation Runner
Authoritative Reference: ForenSightX_Project_Architecture.docx Section 3, 7, 8, 10
Executes end-to-end investigation workflow through specialized agents:
  Evidence Ingestion -> Modality DL -> Multimodal Fusion -> Timeline -> Explainability -> Report
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

import torch
from PIL import Image

from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
)
from src.data_loader import (
    EvidenceItem,
    ForensicCase,
    ForensicTokenizer,
    ForensicDatasetGenerator,
)
from src.agents import CaseCoordinatorAgent


def load_or_init_models(models_dir: str = "models_saved") -> Tuple[VisualEvidenceCNN, ConversationBiLSTM, ANNMultimodalRiskClassifier, ForensicTokenizer]:
    tokenizer = ForensicTokenizer(max_length=80)
    vocab_path = os.path.join(models_dir, "tokenizer_vocab.json")

    # Fit default vocabulary if not saved
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer.word2idx = json.load(f)
            tokenizer.idx2word = {int(v): k for k, v in tokenizer.word2idx.items()}
    else:
        # Fit on baseline generator
        gen = ForensicDatasetGenerator(seed=42)
        texts, _, _, _ = gen.generate_multimodal_dataset(num_samples=100)
        tokenizer.fit_on_texts(texts)

    cnn_model = VisualEvidenceCNN(in_channels=3, num_classes=1)
    bilstm_model = ConversationBiLSTM(vocab_size=max(tokenizer.vocab_size, 100), hidden_dim=64)
    ann_model = ANNMultimodalRiskClassifier(input_dim=7, hidden_dim1=32, hidden_dim2=16)

    # Load weights if available
    cnn_path = os.path.join(models_dir, "cnn_model.pt")
    bilstm_path = os.path.join(models_dir, "bilstm_model.pt")
    ann_path = os.path.join(models_dir, "ann_fusion_model.pt")

    if os.path.exists(cnn_path):
        cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
    if os.path.exists(bilstm_path):
        bilstm_model.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
    if os.path.exists(ann_path):
        ann_model.load_state_dict(torch.load(ann_path, map_location="cpu"))

    cnn_model.eval()
    bilstm_model.eval()
    ann_model.eval()

    return cnn_model, bilstm_model, ann_model, tokenizer


def create_sample_case(case_type: str = "suspicious") -> ForensicCase:
    gen = ForensicDatasetGenerator(seed=99)
    now = datetime.now()

    if case_type.lower() == "suspicious":
        case = ForensicCase(
            case_id="CASE-FS-2026-SUSP",
            title="Suspected CEO Impersonation & Wire Fraud Breach",
            created_at=now.isoformat(),
            investigator="Lead Forensic Agent",
        )
        # Evidence 1: Phishing email
        case.evidence_items.append(
            EvidenceItem(
                evidence_id="EVID-001",
                evidence_type="email",
                timestamp=(now - timedelta(minutes=45)).isoformat(),
                content={
                    "sender": "ceo-desk@free-secure-mail.xyz",
                    "display_name": "Chief Executive Officer",
                    "reply_to": "attacker-c2@dark-relay.top",
                    "subject": "URGENT: Immediate Confidential Wire Transfer Required",
                    "body": "Immediate action required: Wire transfer $45,000 to the attached vendor account before 5 PM to avoid penalty fees. Send OTP immediately.",
                    "attachments": ["vendor_invoice.pdf.exe"],
                },
                metadata={"urls": ["http://192.168.1.105/auth-wire"]},
                is_observed=True,
            )
        )
        # Evidence 2: Chat message
        case.evidence_items.append(
            EvidenceItem(
                evidence_id="EVID-002",
                evidence_type="chat",
                timestamp=(now - timedelta(minutes=38)).isoformat(),
                content="Please send the one-time password (OTP) sent to your mobile phone right now to verify wire transfer compliance.",
                metadata={"sender": "Unknown External"},
                is_observed=True,
            )
        )
        # Evidence 3: Tampered visual screenshot
        img = gen.generate_image_sample(is_suspicious=True)
        case.evidence_items.append(
            EvidenceItem(
                evidence_id="EVID-003",
                evidence_type="image",
                timestamp=(now - timedelta(minutes=30)).isoformat(),
                content=img,
                metadata={"source": "Workstation Screenshot", "resolution": "128x128"},
                is_observed=True,
            )
        )
    else:
        case = ForensicCase(
            case_id="CASE-FS-2026-BENIGN",
            title="Routine IT Maintenance & Standup Audit",
            created_at=now.isoformat(),
            investigator="Lead Forensic Agent",
        )
        case.evidence_items.append(
            EvidenceItem(
                evidence_id="EVID-101",
                evidence_type="email",
                timestamp=(now - timedelta(minutes=60)).isoformat(),
                content={
                    "sender": "lead-dev@company.internal",
                    "display_name": "Dev Team Lead",
                    "reply_to": "lead-dev@company.internal",
                    "subject": "Project deployment went smoothly",
                    "body": "Project deployment went smoothly, all unit tests passed in CI/CD. The quarterly review presentation slides are ready.",
                    "attachments": ["quarterly_review.pdf"],
                },
                metadata={"urls": []},
                is_observed=True,
            )
        )
        case.evidence_items.append(
            EvidenceItem(
                evidence_id="EVID-102",
                evidence_type="chat",
                timestamp=(now - timedelta(minutes=50)).isoformat(),
                content="Are we still on for the 2 PM sprint planning meeting? Team lunch is scheduled for Friday at noon in the cafeteria.",
                metadata={"sender": "Colleague"},
                is_observed=True,
            )
        )
        img = gen.generate_image_sample(is_suspicious=False)
        case.evidence_items.append(
            EvidenceItem(
                evidence_id="EVID-103",
                evidence_type="image",
                timestamp=(now - timedelta(minutes=40)).isoformat(),
                content=img,
                metadata={"source": "Dashboard Snapshot", "resolution": "128x128"},
                is_observed=True,
            )
        )

    return case


def main():
    parser = argparse.ArgumentParser(description="ForenSightX Multi-Agent Investigation Runner")
    parser.add_argument(
        "--case-type",
        choices=["suspicious", "benign"],
        default="suspicious",
        help="Type of sample forensic case to simulate",
    )
    parser.add_argument(
        "--output-md",
        default="results/investigation_report.md",
        help="Path for markdown report",
    )
    parser.add_argument(
        "--output-json",
        default="results/investigation_report.json",
        help="Path for JSON report",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    print(f"[*] Initializing ForenSightX Multi-Agent System...")
    cnn_model, bilstm_model, ann_model, tokenizer = load_or_init_models()

    coordinator = CaseCoordinatorAgent(
        cnn_model=cnn_model,
        bilstm_model=bilstm_model,
        ann_model=ann_model,
        tokenizer=tokenizer,
    )

    print(f"[*] Assembling sample case ({args.case_type.upper()})...")
    case = create_sample_case(case_type=args.case_type)
    print(f"[*] Case ID: {case.case_id} with {len(case.evidence_items)} evidence items.")

    print(f"[*] Commencing collaborative multi-agent investigation...")
    report = coordinator.investigate(case)

    # Save outputs
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"[+] Investigation complete!")
    print(f"    Overall Risk Score: {report.overall_risk_score * 100:.2f}% ({report.risk_tier} RISK)")
    print(f"    Report (Markdown): {args.output_md}")
    print(f"    Report (JSON):     {args.output_json}")
    print("\n" + "=" * 70)
    print("INVESTIGATION SUMMARY (DECISION-SUPPORT NOTICE: Not Legal Proof)")
    print("=" * 70)
    for ind in report.key_evidence_indicators:
        print(f" [!] {ind}")
    print(f"\nTimeline Events Reconstructed: {len(report.chronological_timeline)}")
    for event in report.chronological_timeline:
        print(f"  [{event.timestamp}] ({event.event_type.upper()}) {event.description}")
    print("=" * 70)


if __name__ == "__main__":
    main()

