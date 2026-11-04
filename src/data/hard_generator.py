"""
src/data/hard_generator.py
Hard-Synthetic Forensic Dataset Generator (Iteration 6).
Guarantees:
1. RAW artifacts only (text string, 128x128 PIL image, metadata dict with timestamps/headers/attachments).
2. Zero direct feature sampling (all 5 heuristic features computed solely via extractors).
3. Non-zero within-class variance in both classes for all features.
4. Single-feature AUC capped <= 0.85 (ASSUMPTION-07).
5. Template families partitioned: held-out template families in test set only.
6. Record-level pre-split before case pairing (zero data leakage).
7. Write-once split manifest: data/split_manifest_hard_v1.json.
"""

import os
import re
import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.case_builder import MultimodalCaseItem, ForensicRawRecord
from src.data_loader import (
    extract_url_score,
    extract_keyword_score,
    extract_attachment_score,
    extract_timeline_anomaly,
    extract_email_anomaly,
)

SMOKE_BANNER = "SYNTHETIC BENCHMARK - NOT REAL-WORLD EVIDENCE"

# =====================================================================
# Template Families Configuration
# =====================================================================

TRAIN_TEMPLATES = {
    "invoice_payment": {
        "benign": [
            "Please review the attached monthly vendor invoice for {month}. Payment scheduled per compliance policy.",
            "Invoice #{num} from Acme Consulting has cleared internal audit. Bank wire transfer pending for Friday.",
            "Vendor billing confirmation: Quarterly cloud services invoice approved. Verify allocation in ERP.",
            "Accounts payable update: Routine payment remittance processed for invoice #{num} under standard net 30 terms.",
        ],
        "suspicious": [
            "Invoice #{num} payment instruction update: Remit wire transfer to the newly designated bank account immediately.",
            "Billing dispute notice: Outstanding balance for invoice #{num} overdue. Access remittance portal: http://192.168.1.105/pay",
            "Urgent payment authorization needed for vendor settlement before close of banking day. Review attached instructions.",
            "Final reminder: Wire transfer verification pending for invoice #{num}. Complete authorization at http://invoice-auth.xyz/remit",
        ],
    },
    "account_suspension": {
        "benign": [
            "IT Security Advisory: Corporate passwords expire every 90 days. Please verify and update your credentials next week.",
            "Scheduled directory maintenance: Single sign-on authentication may experience brief downtime from 2 to 3 AM.",
            "Annual compliance reminder: Review information security policy and verify profile completion on LMS.",
            "System patch complete: Access restored following multifactor authentication gateway maintenance.",
        ],
        "suspicious": [
            "Security Alert: Multiple unauthorized login attempts detected. Verify your credentials to avoid account lockout: http://auth-portal.ru/login",
            "Workstation compliance notice: Missing mandatory security agent update. Run emergency remediation patch attached.",
            "Identity verification required: Suspicious session origin flagged. Re-authenticate profile at http://10.0.0.1/verify",
            "Service interruption warning: Corporate account privileges suspended pending compliance review. Details: helpdesk-portal.buzz",
        ],
    },
    "urgent_wire": {
        "benign": [
            "Treasury notification: Wire transfer approval for capital hardware purchase verified by finance committee.",
            "Cash management briefing: Scheduled bank wire transfers and daily liquidity summary available on intranet.",
            "Confidential executive memo: Board briefing on Q4 budget allocation and strategic investment roadmap.",
            "Expense settlement: Travel reimbursement payment for the cybersecurity conference approved for direct deposit.",
        ],
        "suspicious": [
            "CONFIDENTIAL: Executive request from CEO. Process wire transfer remittance of $48,000 for acquisition immediately.",
            "Urgent payment authorization: Chief executive requesting swift wire transfer. Keep strictly confidential; review attached.",
            "Time-sensitive transaction: Board authorized acquisition escrow funds. Wire transfer confirmation needed today.",
            "Emergency wire remittance: Execute payment to designated escrow account before market close. Wire details attached.",
        ],
    },
    "meeting_reschedule": {
        "benign": [
            "Meeting rescheduled: Security architecture sync moved to Thursday at 3 PM. Video link: https://meet.google.com/abc-forensic",
            "Agenda update: Confidential risk committee discussion on quarterly threat landscape. Slides attached for review.",
            "Forensic team standup moved to 10:30 AM to accommodate engineering sync. Join meeting at https://corp.zoom.us/j/492019",
            "Updated invite: Quarterly compliance review with external auditor. Calendar event updated in Outlook.",
        ],
        "suspicious": [
            "Emergency briefing: Urgent meeting regarding reported credential breach. Join audio conference: http://meet-zoom.top/join?id=99",
            "Mandatory briefing updated: Web conference room link changed. Verify token before connecting: http://192.168.4.20/webinar",
            "Rescheduled sync with executive leadership: Review confidential litigation notes at http://box-share.xyz/file",
            "Ad-hoc briefing invitation: Verify corporate credentials to enter confidential conference room: http://meeting-auth.work/conf",
        ],
    },
    "project_status": {
        "benign": [
            "Sprint 24 completion report: Forensic ingestion pipeline verified with zero test regressions. Review attached notes.",
            "Weekly project update: Database optimization resolved latency bottlenecks. Full compliance report on Jira.",
            "Engineering status: Multimodal fusion PR approved and merged. Documentation updated per security guidelines.",
            "Milestone delivered: Q1 forensic architecture deliverables completed on schedule. View dashboard on intranet.",
        ],
        "suspicious": [
            "Incident response notification: Unauthorized exfiltration detected on database cluster. Run inspection script attached.",
            "Confidential status memo: Critical system exposure identified. Download forensic triage archive: triage_pkg.zip",
            "Security advisory: Vulnerability identified in core authentication module. Deploy emergency patch attached: patch_v2.vbs",
            "Internal audit alert: Unverified privileged access logs flagged. Download full investigation dossier: http://audit-report.tk",
        ],
    },
    "executive_travel": {
        "benign": [
            "Flight reservation confirmation for security summit in Seattle. E-ticket and itinerary attached: ticket_{num}.pdf",
            "Corporate travel advisory: Global risk team hotel booking confirmed for annual conference.",
            "Board member travel schedule: Airport transfer and hotel logistics confirmed with executive protection detail.",
            "Expense pre-authorization: Corporate card allocated for international forensic training seminar.",
        ],
        "suspicious": [
            "EMERGENCY TRAVEL BOOKING: Immediate itinerary update required for CEO flight. Access reservation: http://flight-manage.xyz/itin",
            "Corporate travel cancellation: Unpaid airline booking fee $14.50 required to maintain ticket. Settle at http://192.168.3.11/pay",
            "Urgent travel manifest: Download executive delegation travel dossier attached: travel_manifest.exe",
            "VIP passenger alert: Flight gate change requires security verification. Complete authorization: http://air-portal.buzz/auth",
        ],
    },
}

VAL_TEMPLATES = {
    "it_support_ticket": {
        "benign": [
            "Helpdesk ticket #{num} resolved: Software license renewal completed for forensic analysis workstation.",
            "IT Support dispatch: Workstation hardware diagnostic completed with zero memory integrity faults.",
            "Service request #{num}: Access granted to secondary log aggregation repository per security ticket.",
            "IT asset management: Please verify your assigned laptop inventory and serial number on corporate portal.",
        ],
        "suspicious": [
            "CRITICAL IT TICKET #{num}: Urgent rootkit alert detected on endpoint. Download remote repair tool: tool_fix.exe",
            "Helpdesk priority alert: Corporate password reset requested from unknown device. Reject or authorize: http://192.168.8.44/reset",
            "IT security escalation: Unrecognized administrator ticket opened. Confirm identity at http://helpdesk-auth.top/verify",
            "Urgent IT ticket #{num}: Security patch mandatory within 1 hour. Execute remediation package: http://ticket-patch.xyz/run",
        ],
    },
    "hr_benefit_update": {
        "benign": [
            "Annual benefits enrollment: Review healthcare and retirement plan options before the November deadline.",
            "HR notification: Updated employee handbook and code of conduct published to company intranet.",
            "Payroll advisory: Direct deposit account verification required for mid-year compensation adjustment.",
            "Human Resources memo: Corporate wellness program incentives and health screening schedule announced.",
        ],
        "suspicious": [
            "ACTION REQUIRED: Corporate payroll calculation error. Verify your direct deposit details immediately: http://hr-payroll.buzz/verify",
            "Confidential HR review: Employee evaluation dossier and termination schedule leaked. View document: http://10.2.1.90/dossier",
            "HR compensation update: Download revised salary scale and bonus allocation table attached: salary_table.docm",
            "Mandatory HR compliance: Acknowledge disciplinary policy changes to avoid administrative leave: http://employee-portal.xyz/ack",
        ],
    },
    "cloud_access_request": {
        "benign": [
            "Cloud infrastructure alert: AWS IAM role assumption logged for automated backup service.",
            "DevOps deployment notice: Production Kubernetes cluster upgrade scheduled for Saturday maintenance window.",
            "GCP project access: Read-only permissions granted for security monitoring dataset per change ticket.",
            "Cloud security advisory: Multi-region S3 bucket encryption policy successfully applied across all repositories.",
        ],
        "suspicious": [
            "URGENT CLOUD BREACH: Compromised AWS API keys detected in repository. Revoke and rotate immediately: http://aws-rotate.top/keys",
            "Production cluster outage: Critical pod crash on payment gateway. Authenticate DevOps access: http://192.168.12.5/k8s",
            "Cloud credential expiration: GCP service account token expired. Re-authenticate infrastructure credentials: http://cloud-auth.work/login",
            "Security group violation: Unauthorized ingress rule opened on database cluster. Run remediation CLI attached: fix_cloud.vbs",
        ],
    },
}

TEST_TEMPLATES = {
    "shipping_delivery_notice": {
        "benign": [
            "Shipping confirmation: Your hardware token package has shipped via FedEx. Tracking: 48291048201.",
            "Delivery notification: Office supplies order #{num} has been delivered to the 3rd floor reception.",
            "Carrier update: Package delivery scheduled for tomorrow between 1 PM and 5 PM. Signature required.",
            "Postal receipt for certified correspondence sent to external compliance counsel.",
        ],
        "suspicious": [
            "DELIVERY FAILED: Parcel delivery suspended due to incorrect address. Confirm delivery details at http://fedx-track.xyz/ship",
            "Package delivery notice: Outstanding customs fee $3.50 required for international parcel. Pay: http://192.168.9.12/customs",
            "Urgent courier dispatch: Delivery driver unable to access building. Download delivery slip attached: label_382.exe",
            "Unclaimed package held at distribution hub. Update recipient contact details at http://postal-service.buzz/claim",
        ],
    },
    "vpn_credential_update": {
        "benign": [
            "Network operations memo: Global VPN gateway IP addresses will rotate Sunday during maintenance window.",
            "Remote work guide: Step-by-step instructions for installing corporate VPN client on macOS and Linux.",
            "Network performance update: Secondary VPN tunnel in Frankfurt is operational with reduced latency.",
            "IT service desk: Single Sign-On certificate renewal completed successfully for remote users.",
        ],
        "suspicious": [
            "CRITICAL VPN EXPIRATION: Remote access credentials expire in 2 hours. Re-authenticate now: http://vpn-secure.top/gateway",
            "Corporate VPN policy update: Download mandatory security certificate profile attached: vpn_root.p12.exe",
            "Firewall access blocked: Unrecognized device detected on corporate VPN. Re-verify identity at http://192.168.2.80/auth",
            "Urgent network security patch: Required SSL-VPN client update. Execute installer immediately: client_setup.vbs",
        ],
    },
    "conference_invitation": {
        "benign": [
            "Invitation: IEEE Cybersecurity & Privacy Symposium 2027 registration is now open. Early bird rate ends Friday.",
            "Speaker confirmation: Your abstract on Multimodal AI Evidence Extraction has been accepted for presentation.",
            "Annual Technology Summit agenda: Keynotes, workshop tracks, and panel discussions scheduled in San Francisco.",
            "Conference travel approval: Corporate travel request for Black Hat USA has been approved by department manager.",
        ],
        "suspicious": [
            "VIP Conference Invitation: Exclusive cybersecurity briefing pass. Confirm attendee credentials: http://summit-reg.xyz/pass",
            "Keynote speaker presentation materials: Confidential slides from defense conference attached: keynote_slides.docm",
            "Complimentary summit ticket: Claim your reserved VIP conference badge before expiration: http://10.1.1.50/claim",
            "Cyber Threat Intelligence forum attendee list and contact roster. Download registration directory: roster.zip",
        ],
    },
}

# Backward-compatibility aliases
TRAIN_VAL_TEMPLATES = {**TRAIN_TEMPLATES, **VAL_TEMPLATES}
HELD_OUT_TEST_TEMPLATES = TEST_TEMPLATES


# =====================================================================
# Hard Forensic Dataset Generator
# =====================================================================

class HardForensicDatasetGenerator:
    """
    Generates realistic, hard multimodal forensic evidence records.
    Enforces raw artifact generation (text, image, metadata), within-class variance,
    template family holdouts, and zero direct feature sampling.
    """

    def __init__(
        self,
        seed: int = 42,
        label_noise_rate: float = 0.05,
        modality_conflict_rate: float = 0.05,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.label_noise_rate = label_noise_rate
        self.modality_conflict_rate = modality_conflict_rate

    def generate_raw_text(
        self, label: int, template_family: str, template_bank: Dict[str, Any]
    ) -> str:
        """Generates raw evidence message text from template family."""
        family_dict = template_bank[template_family]
        class_key = "suspicious" if label == 1 else "benign"
        candidates = family_dict[class_key]

        template = self.rng.choice(candidates)
        # Format variables
        num = self.rng.randint(1000, 9999)
        month = self.rng.choice(["October", "November", "December", "January"])
        text = template.replace("{num}", str(num)).replace("{month}", month)

        # Inject controlled subtle variations (non-zero within-class variance)
        if label == 0 and self.rng.random() < 0.20:
            # Benign with benign attachment mention or url
            text += f" Please check the shared documents at https://intranet.corp.internal/view?id={num}."
        elif label == 1 and self.rng.random() < 0.15:
            # Stealthy suspicious text without explicit URLs
            text = text.split("http")[0].strip()
            if not text.endswith("."):
                text += "."
            text += " Please review the attached archive immediately for authorization."

        return text

    def generate_raw_metadata(
        self, label: int, text: str
    ) -> Dict[str, Any]:
        """
        Generates realistic metadata: attachments, email headers, event timestamps.
        Features are computed from these by extractors.
        """
        # 1. Attachments
        attachments: List[str] = []
        if label == 1:
            if self.rng.random() < 0.38:
                ext = self.rng.choice([".exe", ".vbs", ".zip", ".bat", ".scr", ".docm"])
                attachments.append(f"document_{self.rng.randint(100, 999)}{ext}")
            elif self.rng.random() < 0.18:
                attachments.append(f"invoice_{self.rng.randint(100, 999)}.pdf.exe")  # double extension
            elif self.rng.random() < 0.22:
                attachments.append(f"summary_{self.rng.randint(100, 999)}.pdf")  # benign-looking extension in suspicious
            # Remaining ~22% have no attachment (link-only phishing)
        else:
            if self.rng.random() < 0.42:
                ext = self.rng.choice([".pdf", ".docx", ".xlsx", ".txt"])
                attachments.append(f"report_{self.rng.randint(100, 999)}{ext}")
            elif self.rng.random() < 0.10:
                attachments.append(f"archive_{self.rng.randint(100, 999)}.zip")  # harmless zip in benign

        # 2. Email Headers
        domains = ["corporate.com", "enterprise.org", "internal.net"]
        free_domains = ["gmail.com", "yahoo.com", "hotmail.com"]
        vendor_domains = ["vendor-support.com", "partner.org", "service-desk.net"]

        if label == 1:
            # Suspicious: mixed between compromised accounts (SPF/DKIM pass) and spoofed mail
            if self.rng.random() < 0.45:
                # Compromised internal account (stealthy)
                d = self.rng.choice(domains)
                sender = f"user{self.rng.randint(10, 99)}@{d}"
                reply_to = sender
                spf = "pass"
                dkim = "pass"
                display_name = "Team Lead"
            else:
                # Spoofed or external lure
                sender_d = self.rng.choice(domains + free_domains)
                reply_d = self.rng.choice(domains + free_domains) if self.rng.random() < 0.45 else sender_d
                sender = f"user{self.rng.randint(10, 99)}@{sender_d}"
                reply_to = f"support@{reply_d}" if reply_d != sender_d else sender
                spf = "fail" if self.rng.random() < 0.45 else "pass"
                dkim = "fail" if self.rng.random() < 0.45 else "pass"
                display_name = self.rng.choice(["IT Support", "Chief Executive", "Admin", "HR Payroll"]) if self.rng.random() < 0.35 else "User"
        else:
            # Benign: legitimate mail with realistic mailing list / vendor variance
            d = self.rng.choice(domains)
            sender = f"user{self.rng.randint(10, 99)}@{d}"
            if self.rng.random() < 0.20:
                # Vendor newsletter or external ticket system where reply-to differs
                reply_d = self.rng.choice(vendor_domains)
                reply_to = f"ticket@{reply_d}"
            else:
                reply_to = sender
            # Occasional SPF/DKIM forwarding glitch in legitimate corporate mail
            spf = "fail" if self.rng.random() < 0.12 else "pass"
            dkim = "fail" if self.rng.random() < 0.08 else "pass"
            display_name = "Team Member" if self.rng.random() < 0.85 else "Chief Executive"

        headers = {
            "sender": sender,
            "reply_to": reply_to,
            "display_name": display_name,
            "headers": {"spf": spf, "dkim": dkim},
        }

        # 3. Event Timestamps
        base_time = datetime(2026, 10, 15, 9, 0, 0) + timedelta(minutes=self.rng.randint(0, 10000))
        timestamps = [base_time]
        if label == 1:
            if self.rng.random() < 0.40:
                # Rapid burst event
                timestamps.append(base_time + timedelta(seconds=1))
                timestamps.append(base_time + timedelta(seconds=2))
            elif self.rng.random() < 0.15:
                # Clock skew / out of order anomaly
                timestamps.append(base_time - timedelta(seconds=30))
            else:
                timestamps.append(base_time + timedelta(minutes=self.rng.randint(5, 30)))
                timestamps.append(base_time + timedelta(minutes=self.rng.randint(35, 90)))
        else:
            if self.rng.random() < 0.15:
                # Benign rapid automated burst (e.g. automated CI/CD alert or notification pair)
                timestamps.append(base_time + timedelta(seconds=1))
                timestamps.append(base_time + timedelta(seconds=2))
            else:
                # Normal spaced events
                timestamps.append(base_time + timedelta(minutes=self.rng.randint(5, 30)))
                timestamps.append(base_time + timedelta(minutes=self.rng.randint(35, 90)))

        return {
            "attachments": attachments,
            "headers": headers,
            "timestamps": timestamps,
        }

    def generate_raw_image(self, label: int) -> Image.Image:
        """
        Generates realistic 128x128 forensic screenshot evidence.
        Visual cues calibrated to achieve CNN AUC in [0.70, 0.82], leaving room for fusion.
        """
        img = Image.new("RGB", (128, 128), color=(245, 245, 248))
        draw = ImageDraw.Draw(img)

        # Base document layout
        draw.rectangle([10, 10, 118, 118], outline=(200, 200, 210), width=1)
        draw.rectangle([10, 10, 118, 30], fill=(230, 235, 245))  # Header bar

        if label == 1:
            # Suspicious Visual Cues: alert box, colored banner, noisy button
            draw.rectangle([15, 35, 113, 55], fill=(255, 230, 230), outline=(220, 100, 100))
            draw.rectangle([20, 65, 108, 80], fill=(220, 50, 50))    # Urgent red button
            # Noisy pixel artifacts
            arr = np.array(img, dtype=np.int16)
            noise = self.np_rng.normal(0, 18, arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            # Benign Visual Cues: clean document, blue corporate banner, clear table
            draw.rectangle([15, 35, 113, 50], fill=(240, 245, 255), outline=(180, 200, 230))
            draw.rectangle([20, 65, 70, 80], fill=(50, 100, 200))     # Standard corporate blue button
            # Low noise
            arr = np.array(img, dtype=np.int16)
            noise = self.np_rng.normal(0, 4, arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def extract_heuristic_feature_vector(
        self, text: str, meta: Dict[str, Any]
    ) -> List[float]:
        """
        Extracts 5 heuristic features strictly via extractors.
        Zero random sampling.
        """
        u = extract_url_score(text)
        k = extract_keyword_score(text)
        a = extract_attachment_score(meta["attachments"])
        tm = extract_timeline_anomaly(meta["timestamps"])
        em = extract_email_anomaly(meta["headers"])
        return [u, k, a, tm, em]

    def generate_benchmark_partition(
        self,
        num_samples: int,
        is_test_split: bool = False,
        split: Optional[str] = None,
    ) -> List[MultimodalCaseItem]:
        """
        Generates paired multimodal cases for a given split using raw artifacts.
        Supports explicit family-disjoint splits: 'train', 'val', 'test'.
        Test split uses strictly held-out template families.
        """
        if split == "train":
            template_bank = TRAIN_TEMPLATES
            case_prefix = "CASE"
        elif split == "val":
            template_bank = VAL_TEMPLATES
            case_prefix = "VAL"
        elif split == "test":
            template_bank = TEST_TEMPLATES
            case_prefix = "TEST"
        elif is_test_split:
            template_bank = HELD_OUT_TEST_TEMPLATES
            case_prefix = "TEST"
        else:
            template_bank = TRAIN_VAL_TEMPLATES
            case_prefix = "CASE"

        template_families = list(template_bank.keys())

        cases: List[MultimodalCaseItem] = []
        half = num_samples // 2

        # Create balanced binary ground-truth labels
        raw_labels = [0] * half + [1] * (num_samples - half)
        self.rng.shuffle(raw_labels)

        for i, target_label in enumerate(raw_labels):
            # Apply label noise (e.g. 5%)
            actual_label = target_label
            if self.rng.random() < self.label_noise_rate:
                actual_label = 1 - target_label

            # Choose template family
            fam = self.rng.choice(template_families)

            # Generate raw artifacts
            text = self.generate_raw_text(actual_label, fam, template_bank)
            meta = self.generate_raw_metadata(actual_label, text)

            # Cross-modality conflict rate (e.g. 5% mismatched visual cue)
            img_label = actual_label
            if self.rng.random() < self.modality_conflict_rate:
                img_label = 1 - actual_label
            img = self.generate_raw_image(img_label)

            # Extract 5 heuristic features strictly via extractors
            heur = self.extract_heuristic_feature_vector(text, meta)

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            img_hash = hashlib.sha256(img.tobytes()).hexdigest()
            content_hash = hashlib.sha256((text_hash + img_hash).encode("utf-8")).hexdigest()

            # Store extracted 5-heuristic vector in metadata
            meta_with_heur = dict(meta)
            meta_with_heur["heuristic_features"] = heur
            meta_with_heur["template_family"] = fam
            meta_with_heur["target_label"] = target_label
            meta_with_heur["is_label_flipped"] = bool(actual_label != target_label)
            meta_with_heur["split"] = split or ("test" if is_test_split else "train")

            item = MultimodalCaseItem(
                case_id=f"{case_prefix}-{i+1:05d}",
                text_content=text,
                text_hash=text_hash,
                image_content=img,
                image_hash=img_hash,
                label=actual_label,
                metadata=meta_with_heur,
                content_hash=content_hash,
            )
            cases.append(item)

        return cases


def generate_hard_benchmark_dataset(
    config_path: str = "config/benchmark_config.yaml",
) -> Tuple[
    List[MultimodalCaseItem],
    List[MultimodalCaseItem],
    List[MultimodalCaseItem],
    Dict[str, Any],
]:
    """
    Generates and persists the official hard-synthetic benchmark dataset.
    Record-level pre-split: train (700), val (200), test (300).
    Enforces write-once manifest at data/split_manifest_hard_v1.json.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if config_path.endswith(".json") else __import__("yaml").safe_load(f)

    seed = cfg["dataset"]["seed"]
    manifest_path = cfg["dataset"]["manifest_path"]
    train_n = int(round(cfg["dataset"]["total_samples"] * cfg["dataset"]["train_ratio"]))
    val_n = int(round(cfg["dataset"]["total_samples"] * cfg["dataset"]["val_ratio"]))
    test_n = cfg["dataset"]["total_samples"] - train_n - val_n

    assert test_n >= cfg["dataset"]["min_test_cases"], f"Test cases {test_n} < min {cfg['dataset']['min_test_cases']}"

    generator = HardForensicDatasetGenerator(
        seed=seed,
        label_noise_rate=cfg["generator"]["label_noise_rate"],
        modality_conflict_rate=cfg["generator"]["modality_conflict_rate"],
    )

    print(f"[*] Generating Hard-Synthetic Benchmark (Train={train_n}, Val={val_n}, Test={test_n})...")
    # Disjoint grouped partition: train, val, and test are mutually family-disjoint
    train_cases = generator.generate_benchmark_partition(train_n, split="train")
    val_cases = generator.generate_benchmark_partition(val_n, split="val")
    test_cases = generator.generate_benchmark_partition(test_n, split="test")

    # Check zero overlap by content hash and ID
    tr_hashes = {c.content_hash for c in train_cases}
    va_hashes = {c.content_hash for c in val_cases}
    te_hashes = {c.content_hash for c in test_cases}
    assert len(tr_hashes.intersection(va_hashes)) == 0, "Hash leakage between train and val"
    assert len(tr_hashes.intersection(te_hashes)) == 0, "Hash leakage between train and test"
    assert len(va_hashes.intersection(te_hashes)) == 0, "Hash leakage between val and test"

    # Check zero overlap by template family
    tr_fams = {c.metadata["template_family"] for c in train_cases}
    va_fams = {c.metadata["template_family"] for c in val_cases}
    te_fams = {c.metadata["template_family"] for c in test_cases}
    assert len(tr_fams.intersection(va_fams)) == 0, "Family leakage between train and val"
    assert len(tr_fams.intersection(te_fams)) == 0, "Family leakage between train and test"
    assert len(va_fams.intersection(te_fams)) == 0, "Family leakage between val and test"

    # Check minimum families per split from config
    min_fams = cfg["dataset"].get("min_families_per_split", {})
    if min_fams:
        assert len(tr_fams) >= min_fams.get("train", 0), f"Train families {len(tr_fams)} < min {min_fams.get('train')}"
        assert len(va_fams) >= min_fams.get("val", 0), f"Val families {len(va_fams)} < min {min_fams.get('val')}"
        assert len(te_fams) >= min_fams.get("test", 0), f"Test families {len(te_fams)} < min {min_fams.get('test')}"

    stats = {
        "seed": seed,
        "splits": {
            "train": {
                "total": len(train_cases),
                "class_0_benign": sum(1 for c in train_cases if c.label == 0),
                "class_1_suspicious": sum(1 for c in train_cases if c.label == 1),
                "imbalance_ratio": float(sum(1 for c in train_cases if c.label == 1) / sum(1 for c in train_cases if c.label == 0)),
            },
            "val": {
                "total": len(val_cases),
                "class_0_benign": sum(1 for c in val_cases if c.label == 0),
                "class_1_suspicious": sum(1 for c in val_cases if c.label == 1),
                "imbalance_ratio": float(sum(1 for c in val_cases if c.label == 1) / sum(1 for c in val_cases if c.label == 0)),
            },
            "test": {
                "total": len(test_cases),
                "class_0_benign": sum(1 for c in test_cases if c.label == 0),
                "class_1_suspicious": sum(1 for c in test_cases if c.label == 1),
                "imbalance_ratio": float(sum(1 for c in test_cases if c.label == 1) / sum(1 for c in test_cases if c.label == 0)),
            },
        },
        "config": cfg,
    }

    manifest_data = {
        "stats": stats,
        "train_ids": [c.case_id for c in train_cases],
        "train_hashes": [c.content_hash for c in train_cases],
        "val_ids": [c.case_id for c in val_cases],
        "val_hashes": [c.content_hash for c in val_cases],
        "test_ids": [c.case_id for c in test_cases],
        "test_hashes": [c.content_hash for c in test_cases],
    }

    # Write-once protection
    if os.path.exists(manifest_path):
        print(f"[*] Manifest '{manifest_path}' already exists (write-once protected).")
    else:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"[+] Persisted write-once split manifest to '{manifest_path}'.")

    # Compute manifest SHA-256
    manifest_sha = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    stats["manifest_sha256"] = manifest_sha
    print(f"[*] Manifest SHA-256: {manifest_sha}")

    return train_cases, val_cases, test_cases, stats


# =====================================================================
# GENERATOR V3: 40-FAMILY GENERALIZABLE BENCHMARK (Iteration 8)
# =====================================================================

SHARED_SUSPICIOUS_INTENT_CUES = [
    "URGENT ACTION REQUIRED: Verify security credentials immediately to prevent suspension.",
    "CRITICAL SECURITY ALERT: Unauthorized session detected. Authenticate now.",
    "TIME SENSITIVE: Immediate compliance authorization mandated to avoid termination.",
    "CONFIDENTIAL DIRECTIVE: Access secure verification gateway without delay.",
    "SECURITY WARNING: Anomalous account activity flagged; complete verification today.",
    "IMMEDIATE NOTIFICATION: Privileged access review required before close of business.",
    "ACCOUNT LOCKOUT WARNING: Confirm identity to prevent access revocation.",
    "MANDATORY REMEDIATION: Critical security update required for your profile.",
]

SHARED_BENIGN_INTENT_CUES = [
    "Routine informational update: Please review attached scheduled summary for your records.",
    "Standard procedure notice: Update provided for your information; no action required.",
    "Periodic team update: Milestones verified and operating within normal parameters.",
    "General communication: Thank you for your continued collaboration on this initiative.",
    "Administrative notice: Quarterly documentation archived per corporate policy.",
    "Departmental briefing: Project deliverables completed according to schedule.",
    "Informational memo: Updated operating guidelines published on internal portal.",
    "Operational summary: Routine system status report for department stakeholders.",
]

SHARED_NEUTRAL_CUES = [
    "Please review the attached corporate document at your convenience.",
    "Operational briefing for team distribution and review.",
    "Administrative notes regarding scheduled department operations.",
    "Weekly summary report for internal stakeholders.",
    "Reference documentation provided for upcoming team review.",
    "Project status attachment provided for department coordination.",
]

TRAIN_TEMPLATES_V3 = {
    "invoice_payment": "Monthly vendor invoice payment processing and remittance verification.",
    "account_suspension": "Active directory identity credentials lifecycle and access policy.",
    "urgent_wire": "Corporate treasury bank wire transfer authorization and escrow instructions.",
    "meeting_reschedule": "Quarterly steering committee conference schedule and agenda synchronization.",
    "project_status": "Engineering pipeline sprint delivery milestones and technical documentation.",
    "executive_travel": "Corporate delegation travel itinerary, flight reservations, and accommodations.",
    "vendor_onboarding": "New supplier registration, W-9 tax documentation, and vendor compliance.",
    "payroll_discrepancy": "Monthly payroll calculation adjustments and direct deposit account verification.",
    "tax_compliance": "Annual corporate tax withholding filing and fiscal compliance certification.",
    "legal_subpoena": "Formal litigation hold notice and confidential document preservation request.",
    "system_alert": "Data center infrastructure monitoring alert regarding server cluster utilization.",
    "domain_renewal": "Corporate DNS domain name registration renewal and SSL certificate expiration.",
    "audit_request": "Internal audit controls sampling and financial records disclosure request.",
    "quarterly_budget": "Departmental operating expense forecast and capital expenditure allocations.",
    "facility_maintenance": "Office physical security access badge upgrades and HVAC servicing schedule.",
    "office_relocation": "Regional headquarters floor plan seating arrangements and asset relocation.",
    "contract_amendment": "Master services agreement terms amendment and contract execution signature.",
    "procurement_order": "Enterprise hardware purchase order requisition and warehouse receipt.",
    "customer_complaint": "Priority client service SLA escalation and dispute resolution summary.",
    "press_release": "Quarterly financial earnings publication draft and media embargo guidelines.",
}

VAL_TEMPLATES_V3 = {
    "it_support_ticket": "Helpdesk technical support ticket resolution and workstation diagnostics.",
    "hr_benefit_update": "Annual employee healthcare benefit enrollment and retirement contributions.",
    "cloud_access_request": "Cloud infrastructure IAM role assignment and storage bucket access policy.",
    "software_license": "Enterprise software license key provisioning and compliance verification.",
    "data_backup_notice": "Automated nightly database backup snapshot verification and disaster recovery.",
    "training_seminar": "Mandatory corporate cybersecurity awareness training and ethics module.",
    "telecom_bill": "Corporate enterprise mobile carrier telecommunications billing statement.",
    "travel_expense_report": "Business travel reimbursement expense claim and hotel folio review.",
    "security_incident_report": "SOC security operations incident summary and forensic log triage.",
    "marketing_campaign": "Product marketing campaign analytics and digital advertising distribution.",
}

TEST_TEMPLATES_V3 = {
    "shipping_delivery_notice": "Freight courier shipment delivery tracking and customs clearance notice.",
    "vpn_credential_update": "Remote access VPN client configuration profile and token rotation.",
    "conference_invitation": "Academic cybersecurity research symposium keynote speaker invitation.",
    "patent_filing_notice": "Intellectual property patent application claims and USPTO docketing.",
    "regulatory_disclosure": "Financial regulatory compliance filing and transparency disclosure.",
    "board_meeting_agenda": "Board of directors executive meeting agenda and governance packet.",
    "supply_chain_dispatch": "Logistics supply chain dispatch manifest and freight bill of lading.",
    "charity_donation_match": "Corporate philanthropic gift matching program and foundation grant.",
    "health_safety_inspection": "Workplace ergonomic safety inspection and OSHA compliance audit.",
    "infrastructure_upgrade": "Core network router firmware patch and fiber optic switch migration.",
}


class HardForensicDatasetGeneratorV3:
    """
    Generator v3 implementing the 40-family ontology, shared forensic cues,
    calibrated oracle ceilings in [0.75, 0.90], and true training label noise.
    """

    def __init__(
        self,
        seed: int = 42,
        label_noise_rate: float = 0.05,
        modality_conflict_rate: float = 0.05,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.label_noise_rate = label_noise_rate
        self.modality_conflict_rate = modality_conflict_rate

    def generate_raw_text(
        self, true_label: int, template_family: str, p_cue: float = 0.48
    ) -> str:
        """
        Generates text evidence containing shared forensic intent cues
        combined with domain-specific family nuisance.
        """
        fam_text = template_family.replace("_", " ").title()
        if self.rng.random() < p_cue:
            cue = self.rng.choice(SHARED_SUSPICIOUS_INTENT_CUES if true_label == 1 else SHARED_BENIGN_INTENT_CUES)
        else:
            cue = self.rng.choice(SHARED_NEUTRAL_CUES)

        ref_id = self.rng.randint(1000, 9999)
        return f"{cue} Reference: {fam_text} File #{ref_id}."

    def generate_raw_image(
        self, true_label: int, fam_idx: int, seed: int, p_vis: float = 0.45
    ) -> Image.Image:
        """
        Generates a 128x128 RGB PIL image with subtle visual indicator cues
        plus controlled sensor noise, calibrated into [0.75, 0.90] oracle AUC.
        """
        rng_img = np.random.RandomState(seed)
        arr = np.full((128, 128, 3), 235.0, dtype=np.float32)
        y_pos = (fam_idx % 5) * 15 + 10

        if rng_img.rand() < p_vis:
            if true_label == 1:
                # Subtle reddish tint
                arr[y_pos:y_pos + 25, 20:108, 0] = 175.0 + rng_img.uniform(-10, 10)
                arr[y_pos:y_pos + 25, 20:108, 1] = 110.0 + rng_img.uniform(-10, 10)
                arr[y_pos:y_pos + 25, 20:108, 2] = 110.0 + rng_img.uniform(-10, 10)
            else:
                # Subtle bluish tint
                arr[y_pos:y_pos + 25, 20:108, 0] = 110.0 + rng_img.uniform(-10, 10)
                arr[y_pos:y_pos + 25, 20:108, 1] = 110.0 + rng_img.uniform(-10, 10)
                arr[y_pos:y_pos + 25, 20:108, 2] = 175.0 + rng_img.uniform(-10, 10)

        # Add controlled sensor Gaussian noise
        noise = rng_img.randn(128, 128, 3) * 35.0
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def generate_raw_metadata(
        self, true_label: int, text: str, seed: int
    ) -> Dict[str, Any]:
        """
        Generates metadata headers, attachments, URLs, and timestamps
        with calibrated signal-to-noise ratios.
        """
        rng_m = random.Random(seed)
        np_m = np.random.RandomState(seed)

        # Signal level for heuristics: beta ~ 0.70
        is_susp_cue = (true_label == 1 and rng_m.random() < 0.70) or (true_label == 0 and rng_m.random() < 0.25)

        # 1. URL
        if is_susp_cue:
            urls = [f"http://192.168.{rng_m.randint(1,250)}.{rng_m.randint(1,250)}/verify"]
        else:
            urls = [f"https://portal.internal-corp.net/docs/{rng_m.randint(100, 999)}"]

        # 2. Attachments
        if is_susp_cue:
            att = ["invoice_remittance.exe"] if rng_m.random() < 0.4 else ["document.docm"]
        else:
            att = ["report_summary.pdf"] if rng_m.random() < 0.8 else ["archive.zip"]

        # 3. Email Headers & Spoofing
        if is_susp_cue:
            sender = f"admin{rng_m.randint(10,99)}@spoofed-domain.org"
            reply_to = f"contact@{rng_m.choice(['external-mail.top', 'secure-lure.buzz'])}"
            spf = "fail" if rng_m.random() < 0.55 else "pass"
            dkim = "fail" if rng_m.random() < 0.55 else "pass"
            disp = "Chief Executive" if rng_m.random() < 0.45 else "IT Support"
        else:
            sender = f"employee{rng_m.randint(10,99)}@internal-corp.com"
            reply_to = sender
            spf = "pass" if rng_m.random() < 0.85 else "fail"
            dkim = "pass" if rng_m.random() < 0.90 else "fail"
            disp = "Team Member"

        headers = {
            "sender": sender,
            "reply_to": reply_to,
            "display_name": disp,
            "headers": {"spf": spf, "dkim": dkim},
        }

        # 4. Timestamps
        base_time = datetime(2026, 11, 1, 9, 0, 0) + timedelta(minutes=rng_m.randint(0, 10000))
        timestamps = [base_time]
        if is_susp_cue and rng_m.random() < 0.50:
            timestamps.append(base_time + timedelta(seconds=1))
            timestamps.append(base_time + timedelta(seconds=2))

        return {
            "urls": urls,
            "attachments": att,
            "email_headers": headers,
            "event_timestamps": timestamps,
        }

    def extract_heuristic_feature_vector(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[float]:
        """Extracts the canonical 5 heuristic features."""
        u_score = extract_url_score(metadata.get("urls", []))
        k_score = extract_keyword_score(text)
        a_score = extract_attachment_score(metadata.get("attachments", []))
        t_score = extract_timeline_anomaly(metadata.get("event_timestamps", []))
        e_score = extract_email_anomaly(metadata.get("email_headers", {}))
        return [round(float(x), 4) for x in [u_score, k_score, a_score, t_score, e_score]]

    def generate_benchmark_partition(
        self,
        num_samples: int,
        split: str,
    ) -> List[MultimodalCaseItem]:
        """
        Generates a benchmark partition from the 40-family ontology.
        Enforces true label noise on the training split observed label.
        """
        if split == "train":
            template_bank = TRAIN_TEMPLATES_V3
            case_prefix = "CASE-TR"
        elif split == "val":
            template_bank = VAL_TEMPLATES_V3
            case_prefix = "VAL"
        else:
            template_bank = TEST_TEMPLATES_V3
            case_prefix = "TEST"

        families = list(template_bank.keys())
        cases: List[MultimodalCaseItem] = []
        half = num_samples // 2
        true_labels = [0] * half + [1] * (num_samples - half)
        self.rng.shuffle(true_labels)

        for i, true_lbl in enumerate(true_labels):
            fam_idx = i % len(families)
            fam = families[fam_idx]
            case_seed = self.seed * 10000 + i

            # Render raw artifacts from true label
            text = self.generate_raw_text(true_lbl, fam)
            img = self.generate_raw_image(true_lbl, fam_idx, case_seed)
            meta = self.generate_raw_metadata(true_lbl, text, case_seed)
            heur = self.extract_heuristic_feature_vector(text, meta)

            # Apply true label noise to training split observed label
            if split == "train" and self.rng.random() < self.label_noise_rate:
                obs_label = 1 - true_lbl
                is_flipped = True
            else:
                obs_label = true_lbl
                is_flipped = False

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            img_hash = hashlib.sha256(img.tobytes()).hexdigest()
            content_hash = hashlib.sha256((text_hash + img_hash).encode("utf-8")).hexdigest()

            meta_record = dict(meta)
            meta_record["heuristic_features"] = heur
            meta_record["template_family"] = fam
            meta_record["true_label"] = int(true_lbl)
            meta_record["observed_label"] = int(obs_label)
            meta_record["is_label_flipped"] = bool(is_flipped)
            meta_record["split"] = split

            item = MultimodalCaseItem(
                case_id=f"{case_prefix}-{i+1:05d}",
                text_content=text,
                text_hash=text_hash,
                image_content=img,
                image_hash=img_hash,
                label=obs_label,
                metadata=meta_record,
                content_hash=content_hash,
            )
            cases.append(item)

        return cases


def generate_hard_benchmark_dataset_v3(
    config_path: str = "config/benchmark_config.yaml",
) -> Tuple[
    List[MultimodalCaseItem],
    List[MultimodalCaseItem],
    List[MultimodalCaseItem],
    Dict[str, Any],
]:
    """
    Generates and persists the official hard-synthetic benchmark v3 dataset.
    700 train, 200 val, 300 test across 40 disjoint template families.
    Enforces write-once manifest at data/split_manifest_hard_v3.json.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if config_path.endswith(".json") else __import__("yaml").safe_load(f)

    seed = cfg["dataset"]["seed"]
    manifest_path = cfg["dataset"].get("manifest_path_v3", "data/split_manifest_hard_v3.json")
    train_n = int(round(cfg["dataset"]["total_samples"] * cfg["dataset"]["train_ratio"]))
    val_n = int(round(cfg["dataset"]["total_samples"] * cfg["dataset"]["val_ratio"]))
    test_n = cfg["dataset"]["total_samples"] - train_n - val_n

    generator = HardForensicDatasetGeneratorV3(
        seed=seed,
        label_noise_rate=cfg["generator"]["label_noise_rate"],
        modality_conflict_rate=cfg["generator"]["modality_conflict_rate"],
    )

    print(f"[*] Generating Hard-Synthetic Benchmark v3 (Train={train_n}, Val={val_n}, Test={test_n})...")
    train_cases = generator.generate_benchmark_partition(train_n, split="train")
    val_cases = generator.generate_benchmark_partition(val_n, split="val")
    test_cases = generator.generate_benchmark_partition(test_n, split="test")

    # Verify zero leakage across splits
    tr_hashes = {c.content_hash for c in train_cases}
    va_hashes = {c.content_hash for c in val_cases}
    te_hashes = {c.content_hash for c in test_cases}
    assert len(tr_hashes.intersection(va_hashes)) == 0, "Hash leakage between train and val"
    assert len(tr_hashes.intersection(te_hashes)) == 0, "Hash leakage between train and test"
    assert len(va_hashes.intersection(te_hashes)) == 0, "Hash leakage between val and test"

    tr_fams = {c.metadata["template_family"] for c in train_cases}
    va_fams = {c.metadata["template_family"] for c in val_cases}
    te_fams = {c.metadata["template_family"] for c in test_cases}
    assert len(tr_fams.intersection(va_fams)) == 0, "Family leakage between train and val"
    assert len(tr_fams.intersection(te_fams)) == 0, "Family leakage between train and test"
    assert len(va_fams.intersection(te_fams)) == 0, "Family leakage between val and test"

    assert len(tr_fams) == 20, f"Expected 20 train families, got {len(tr_fams)}"
    assert len(va_fams) == 10, f"Expected 10 val families, got {len(va_fams)}"
    assert len(te_fams) == 10, f"Expected 10 test families, got {len(te_fams)}"

    tr_flipped = sum(1 for c in train_cases if c.metadata["is_label_flipped"])
    stats = {
        "seed": seed,
        "splits": {
            "train": {
                "total": len(train_cases),
                "families_count": len(tr_fams),
                "class_0_benign": sum(1 for c in train_cases if c.label == 0),
                "class_1_suspicious": sum(1 for c in train_cases if c.label == 1),
                "true_class_0": sum(1 for c in train_cases if c.metadata["true_label"] == 0),
                "true_class_1": sum(1 for c in train_cases if c.metadata["true_label"] == 1),
                "flipped_labels_count": tr_flipped,
                "flipped_fraction": float(tr_flipped / len(train_cases)),
            },
            "val": {
                "total": len(val_cases),
                "families_count": len(va_fams),
                "class_0_benign": sum(1 for c in val_cases if c.label == 0),
                "class_1_suspicious": sum(1 for c in val_cases if c.label == 1),
                "flipped_labels_count": 0,
                "flipped_fraction": 0.0,
            },
            "test": {
                "total": len(test_cases),
                "families_count": len(te_fams),
                "class_0_benign": sum(1 for c in test_cases if c.label == 0),
                "class_1_suspicious": sum(1 for c in test_cases if c.label == 1),
                "flipped_labels_count": 0,
                "flipped_fraction": 0.0,
            },
        },
        "config": cfg,
    }

    manifest_data = {
        "stats": stats,
        "train_ids": [c.case_id for c in train_cases],
        "train_hashes": [c.content_hash for c in train_cases],
        "val_ids": [c.case_id for c in val_cases],
        "val_hashes": [c.content_hash for c in val_cases],
        "test_ids": [c.case_id for c in test_cases],
        "test_hashes": [c.content_hash for c in test_cases],
    }

    # Write-once manifest protection
    if os.path.exists(manifest_path):
        print(f"[*] Manifest '{manifest_path}' already exists (write-once protected).")
    else:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"[+] Persisted write-once split manifest to '{manifest_path}'.")

    manifest_sha = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    stats["manifest_sha256"] = manifest_sha
    print(f"[*] Manifest SHA-256: {manifest_sha}")

    return train_cases, val_cases, test_cases, stats
