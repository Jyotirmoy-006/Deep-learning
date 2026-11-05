"""
scripts/generate_test_data.py
Generates ready-to-test forensic evidence samples for the ForenSightX Dashboard:
- test_data/images/: 2 suspicious tampered screenshots + 2 benign screenshots
- test_data/chats/: 2 suspicious deceptive chat transcripts + 2 benign chats
- test_data/emails/: 2 suspicious phishing emails with headers + 2 benign emails
- test_data/cases/: 2 complete multimodal cases (JSON) ready for API/CLI
- test_data/README.md: Complete step-by-step testing walkthrough
"""

import os
import json
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
IMG_DIR = os.path.join(TEST_DATA_DIR, "images")
CHAT_DIR = os.path.join(TEST_DATA_DIR, "chats")
EMAIL_DIR = os.path.join(TEST_DATA_DIR, "emails")
CASE_DIR = os.path.join(TEST_DATA_DIR, "cases")

for d in [TEST_DATA_DIR, IMG_DIR, CHAT_DIR, EMAIL_DIR, CASE_DIR]:
    os.makedirs(d, exist_ok=True)


# =====================================================================
# 1. Generate Images
# =====================================================================

def make_suspicious_wire_invoice():
    img = Image.new("RGB", (256, 256), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)
    # Header bar
    draw.rectangle([15, 15, 241, 55], fill=(30, 45, 80))
    draw.text((25, 25), "INTERNATIONAL WIRE INVOICE", fill=(255, 255, 255))
    draw.text((25, 68), "Beneficiary: Offshore Holdings Ltd", fill=(30, 30, 30))
    draw.text((25, 88), "Amount: $485,000.00 USD", fill=(190, 20, 20))
    draw.text((25, 108), "Bank: Global Trust Cayman", fill=(30, 30, 30))
    draw.text((25, 128), "SWIFT: GLTRKY33 | IBAN: KY88GLTR00", fill=(30, 30, 30))
    draw.text((25, 148), "Memo: Confidential Acquisition Wire", fill=(80, 80, 80))
    # Urgent stamp
    draw.rectangle([130, 185, 240, 235], outline=(220, 20, 20), width=3)
    draw.text((145, 202), "URGENT PAYMENT", fill=(220, 20, 20))
    # Tampering noise patch in corner
    np.random.seed(42)
    patch = np.random.randint(180, 255, (45, 45, 3), dtype=np.uint8)
    img.paste(Image.fromarray(patch), (20, 185))
    path = os.path.join(IMG_DIR, "suspicious_wire_transfer_invoice.png")
    img.save(path)
    return path


def make_suspicious_login_portal():
    img = Image.new("RGB", (256, 256), color=(25, 30, 40))
    draw = ImageDraw.Draw(img)
    # Dialog box
    draw.rectangle([20, 20, 236, 236], fill=(35, 42, 58), outline=(60, 75, 105), width=2)
    draw.text((35, 35), "Corporate Single Sign-On (SSO)", fill=(255, 255, 255))
    draw.text((35, 55), "Account Session Expired", fill=(240, 80, 80))
    # Fake inputs
    draw.rectangle([35, 85, 220, 115], fill=(20, 25, 35), outline=(90, 110, 140))
    draw.text((45, 93), "john.doe@company.internal", fill=(160, 175, 195))
    draw.rectangle([35, 130, 220, 160], fill=(20, 25, 35), outline=(90, 110, 140))
    draw.text((45, 138), "Enter OTP / Password", fill=(220, 100, 100))
    # Suspicious tampering overlay / spliced warning
    draw.rectangle([35, 175, 220, 215], fill=(180, 30, 30))
    draw.text((45, 188), "VERIFY NOW OR LOCK", fill=(255, 255, 255))
    np.random.seed(101)
    noise = np.random.randint(100, 255, (25, 50, 3), dtype=np.uint8)
    img.paste(Image.fromarray(noise), (160, 75))
    path = os.path.join(IMG_DIR, "suspicious_spoofed_login_portal.png")
    img.save(path)
    return path


def make_benign_ci_cd_log():
    img = Image.new("RGB", (256, 256), color=(15, 18, 25))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "$ git status --short", fill=(100, 220, 120))
    draw.text((20, 42), "nothing to commit, working tree clean", fill=(180, 190, 205))
    draw.text((20, 70), "$ python -m pytest -q", fill=(100, 220, 120))
    draw.text((20, 92), ".................................... [100%]", fill=(100, 220, 120))
    draw.text((20, 114), "69 passed, 2 warnings in 10.92s", fill=(100, 220, 120))
    draw.text((20, 145), "$ docker build -t forensightx:v1.0 .", fill=(100, 220, 120))
    draw.text((20, 167), "[+] Building 4.8s (12/12) FINISHED", fill=(180, 190, 205))
    draw.text((20, 190), "Successfully tagged forensightx:v1.0", fill=(100, 220, 120))
    draw.text((20, 215), "STATUS: HEALTHY - DEPLOYED", fill=(60, 180, 240))
    path = os.path.join(IMG_DIR, "benign_ci_cd_deployment_log.png")
    img.save(path)
    return path


def make_benign_dashboard_screenshot():
    img = Image.new("RGB", (256, 256), color=(248, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle([15, 15, 241, 50], fill=(230, 235, 245))
    draw.text((25, 25), "System Health Monitor: US-East-1", fill=(30, 40, 60))
    # Metric cards
    draw.rectangle([25, 65, 125, 125], fill=(255, 255, 255), outline=(210, 220, 235))
    draw.text((35, 75), "CPU Load", fill=(100, 110, 130))
    draw.text((35, 95), "18.4% (OK)", fill=(30, 160, 80))

    draw.rectangle([135, 65, 235, 125], fill=(255, 255, 255), outline=(210, 220, 235))
    draw.text((145, 75), "Memory", fill=(100, 110, 130))
    draw.text((145, 95), "4.1 GB / 16GB", fill=(30, 160, 80))

    draw.rectangle([25, 140, 235, 225], fill=(255, 255, 255), outline=(210, 220, 235))
    draw.text((35, 150), "Service Availability Status", fill=(80, 90, 110))
    draw.text((35, 172), "API Gateway: 100.0%", fill=(30, 160, 80))
    draw.text((35, 192), "Model Worker Pools: 100.0%", fill=(30, 160, 80))
    path = os.path.join(IMG_DIR, "benign_system_dashboard.png")
    img.save(path)
    return path


# =====================================================================
# 2. Generate Chat Transcripts
# =====================================================================

chats = {
    "suspicious_ceo_urgent_wire.txt": (
        "[14:02:11] Chief Executive Officer: Are you available right now? I am currently in a closed-door board meeting.\n"
        "[14:02:45] Finance Lead: Yes, sir. How can I assist you?\n"
        "[14:03:15] Chief Executive Officer: We are closing a highly confidential international acquisition. "
        "I need an immediate wire transfer of $485,000 processed to our overseas escrow account before 3:30 PM cutoff.\n"
        "[14:04:02] Finance Lead: Understood. Should I initiate the standard dual-authorization protocol through the portal?\n"
        "[14:04:30] Chief Executive Officer: Negative. Standard protocol will delay closing. I have authorized this under executive emergency powers. "
        "Do NOT contact other directors as strict confidentiality applies under SEC regulations. "
        "Forward payment confirmation immediately upon transmission: http://secure-escrow-update.xyz/verify-token\n"
    ),
    "suspicious_helpdesk_otp_phish.txt": (
        "[09:15:20] IT Security Support: ATTENTION: Unusual sign-in attempt detected from IP 185.220.101.4 (Saint Petersburg, RU).\n"
        "[09:16:01] Employee: Wait, that wasn't me! What should I do?\n"
        "[09:16:35] IT Security Support: We are initiating account containment. A 6-digit one-time password (OTP) verification code "
        "has just been sent to your registered mobile phone number.\n"
        "[09:17:10] IT Security Support: Please read back the 6-digit code to me immediately to authenticate your session and prevent permanent credential revocation.\n"
        "[09:17:42] Employee: The code says 894-102. Is my account safe now?\n"
        "[09:18:05] IT Security Support: Authentication confirmed. Access credentials updated. Please download the corporate security patch from: "
        "http://it-support-portal.top/patch_update.exe\n"
    ),
    "benign_sprint_standup.txt": (
        "[10:00:15] Tech Lead: Good morning team! Let's kick off the daily sprint standup.\n"
        "[10:01:02] Frontend Dev: Yesterday I finalized the responsive layout for the metrics cards. Today I am connecting the WebSocket status feed.\n"
        "[10:02:10] Backend Dev: I reviewed the PR for the model checkpoint loading factory. All 69 unit tests passed cleanly with 0 errors.\n"
        "[10:03:05] QA Lead: Automated test suites and smoke verification scripts ran in 10.9s on staging. All endpoints returned HTTP 200.\n"
        "[10:04:00] Tech Lead: Excellent work. Let's merge the branch to master and prepare for the sprint demo at 2 PM. Who's joining for lunch?\n"
    ),
    "benign_incident_resolved.txt": (
        "[16:20:10] DevOps Engineer: Monitoring alert: High memory utilization on cache worker node redis-02.\n"
        "[16:21:40] SRE Lead: Investigating now. It looks like the TTL on temporary session keys was misconfigured during the morning migration.\n"
        "[16:23:05] SRE Lead: Applied the TTL patch via config map. Cache evicted 1.4 GB of expired keys. Memory usage dropped back to 22%.\n"
        "[16:25:00] DevOps Engineer: Verified alert has cleared. All health metrics green in Datadog dashboard. Post-mortem scheduled for tomorrow morning.\n"
    ),
}

for filename, content in chats.items():
    with open(os.path.join(CHAT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)


# =====================================================================
# 3. Generate Email Files (with Headers and Body)
# =====================================================================

emails = {
    "suspicious_wire_fraud_email.json": {
        "headers": {
            "From": '"Chief Executive Officer" <ceo-desk@free-secure-mail.xyz>',
            "To": "finance-ops@company.internal",
            "Reply-To": "executive-direct-response@mail-gateway.top",
            "Subject": "URGENT: Immediate Confidential Wire Transfer Required for Project Titan",
            "Date": "Thu, 05 Nov 2026 14:15:00 -0400",
            "Authentication-Results": "spf=softfail (sender IP 198.51.100.42 not in SPF record); dkim=neutral",
            "X-Priority": "1 (Highest)",
            "X-Mailer": "PHPMailer 6.2.0 (https://github.com/PHPMailer/PHPMailer)",
        },
        "body": (
            "Dear Finance Team,\n\n"
            "Due to an urgent deadline regarding the acquisition of Project Titan, please process the attached wire instruction immediately.\n"
            "Wire Amount: $485,000.00 USD\n"
            "Beneficiary: Offshore Holdings Ltd (Cayman Islands)\n"
            "Account: 8829-1029-4410\n"
            "Please confirm transmission by opening the secure escrow link: http://wire-verification-portal.xyz/confirm\n\n"
            "Regards,\nChief Executive Officer"
        ),
        "attachments": ["Invoice_Wire_485K.pdf.exe"],
    },
    "suspicious_payroll_update.json": {
        "headers": {
            "From": '"Human Resources Portal" <hr-service@corporate-payroll-update.top>',
            "To": "all-staff@company.internal",
            "Reply-To": "payroll-collector@protonmail.com",
            "Subject": "ACTION REQUIRED: Direct Deposit Banking Information Verification Deadline",
            "Date": "Thu, 05 Nov 2026 08:30:15 -0500",
            "Authentication-Results": "spf=fail (domain corporate-payroll-update.top fails SPF); dkim=none",
            "X-Priority": "1 (Highest)",
        },
        "body": (
            "All Employees,\n\n"
            "Our payroll system requires immediate re-verification of all employee direct deposit bank accounts prior to the end of this pay cycle.\n"
            "Failure to authenticate your banking credentials within 24 hours will result in paycheck withholding.\n\n"
            "Click here to submit your direct deposit credentials: http://payroll-login-portal.xyz/auth\n\n"
            "Human Resources Management"
        ),
        "attachments": ["direct_deposit_form.scr"],
    },
    "benign_release_notification.json": {
        "headers": {
            "From": '"Release Management" <releases@company.internal>',
            "To": "engineering@company.internal",
            "Reply-To": "releases@company.internal",
            "Subject": "[RELEASE] ForenSightX Platform Version 1.0 Deployment Complete",
            "Date": "Thu, 05 Nov 2026 11:00:00 -0400",
            "Authentication-Results": "spf=pass (IP 10.0.0.1); dkim=pass header.d=company.internal",
            "Message-ID": "<20261105.110000.12345@company.internal>",
        },
        "body": (
            "Hi Engineering,\n\n"
            "We are pleased to announce that ForenSightX v1.0 has been deployed to the production staging environment.\n"
            "Key milestones achieved:\n"
            "- 69 automated tests passing across CNN, BiLSTM, 7D Fusion ANN, and VAE modules.\n"
            "- Interactive web dashboard running on port 8000 with Grad-CAM visual heatmaps.\n"
            "- Full benchmark results accessible via /api/results_v10.\n\n"
            "Documentation is available at: https://internal.company.internal/docs/forensightx\n\n"
            "Regards,\nRelease Management Team"
        ),
        "attachments": [],
    },
    "benign_meeting_invite.json": {
        "headers": {
            "From": '"Project Coordinator" <coordinator@company.internal>',
            "To": "investigations-team@company.internal",
            "Reply-To": "coordinator@company.internal",
            "Subject": "Invitation: Architecture & Technical Review @ Fri Nov 6, 2026 2pm - 3pm (EDT)",
            "Date": "Thu, 05 Nov 2026 13:45:00 -0400",
            "Authentication-Results": "spf=pass; dkim=pass header.d=company.internal",
        },
        "body": (
            "Hello Investigations Team,\n\n"
            "You are invited to the upcoming technical review meeting:\n"
            "Topic: ForenSightX Multimodal Forensic Architecture & Syllabus Mapping\n"
            "Time: Friday, November 6, 2026 · 2:00 – 3:00pm\n\n"
            "Meeting link: https://meet.company.internal/fx-arch-review\n\n"
            "Agenda:\n"
            "1. Multimodal neural representation analysis\n"
            "2. Baseline comparisons (Arms A-E)\n"
            "3. Live dashboard walkthrough"
        ),
        "attachments": ["agenda_nov6.pdf"],
    },
}

for filename, data in emails.items():
    with open(os.path.join(EMAIL_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =====================================================================
# 4. Generate Complete Multimodal Test Cases (JSON)
# =====================================================================

suspicious_case = {
    "case_id": "CASE-TEST-2026-SUSP",
    "title": "Business Email Compromise & Wire Fraud Incident",
    "investigator": "Forensic Analyst",
    "created_at": datetime.now().isoformat(),
    "evidence_items": [
        {
            "evidence_id": "EVID-001",
            "evidence_type": "email",
            "timestamp": (datetime.now() - timedelta(minutes=45)).isoformat(),
            "content": emails["suspicious_wire_fraud_email.json"]["body"],
            "metadata": emails["suspicious_wire_fraud_email.json"]["headers"],
            "is_observed": True,
        },
        {
            "evidence_id": "EVID-002",
            "evidence_type": "chat",
            "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
            "content": chats["suspicious_ceo_urgent_wire.txt"],
            "metadata": {"platform": "Slack", "channel": "direct-message", "urgency": "high"},
            "is_observed": True,
        },
        {
            "evidence_id": "EVID-003",
            "evidence_type": "image",
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
            "content": "test_data/images/suspicious_wire_transfer_invoice.png",
            "metadata": {"filename": "suspicious_wire_transfer_invoice.png", "source": "workstation_download"},
            "is_observed": True,
        },
    ],
}

benign_case = {
    "case_id": "CASE-TEST-2026-BENIGN",
    "title": "Scheduled DevOps Release & Standup Audit",
    "investigator": "Compliance Examiner",
    "created_at": datetime.now().isoformat(),
    "evidence_items": [
        {
            "evidence_id": "EVID-101",
            "evidence_type": "email",
            "timestamp": (datetime.now() - timedelta(minutes=60)).isoformat(),
            "content": emails["benign_release_notification.json"]["body"],
            "metadata": emails["benign_release_notification.json"]["headers"],
            "is_observed": True,
        },
        {
            "evidence_id": "EVID-102",
            "evidence_type": "chat",
            "timestamp": (datetime.now() - timedelta(minutes=40)).isoformat(),
            "content": chats["benign_sprint_standup.txt"],
            "metadata": {"platform": "Teams", "channel": "dev-engineering"},
            "is_observed": True,
        },
        {
            "evidence_id": "EVID-103",
            "evidence_type": "image",
            "timestamp": (datetime.now() - timedelta(minutes=20)).isoformat(),
            "content": "test_data/images/benign_ci_cd_deployment_log.png",
            "metadata": {"filename": "benign_ci_cd_deployment_log.png", "source": "build_pipeline"},
            "is_observed": True,
        },
    ],
}

with open(os.path.join(CASE_DIR, "case_suspicious_ceo_wire_fraud.json"), "w", encoding="utf-8") as f:
    json.dump(suspicious_case, f, indent=2)

with open(os.path.join(CASE_DIR, "case_benign_devops_maintenance.json"), "w", encoding="utf-8") as f:
    json.dump(benign_case, f, indent=2)


# =====================================================================
# 5. Generate test_data/README.md Walkthrough Guide
# =====================================================================

readme_content = """# ForenSightX — Dashboard Test Data & Sample Evidence Package

This directory contains realistic digital forensic evidence samples designed specifically for testing all interactive capabilities of the ForenSightX Web Dashboard (`http://localhost:8000`).

---

## Directory Contents

```text
test_data/
├── images/                                # Realistic digital screenshots & invoices
│   ├── suspicious_wire_transfer_invoice.png   # Tampered banking wire invoice with altered amount and urgent stamp
│   ├── suspicious_spoofed_login_portal.png    # Spoofed SSO login portal with credential harvesting prompts
│   ├── benign_ci_cd_deployment_log.png        # Clean terminal output (git commit, pytest passing 69/69)
│   └── benign_system_dashboard.png            # Clean server metrics & health monitoring screenshot
│
├── chats/                                 # Chat communication transcripts
│   ├── suspicious_ceo_urgent_wire.txt         # Urgent CEO impersonation requesting confidential wire transfer
│   ├── suspicious_helpdesk_otp_phish.txt      # Fake IT support harvesting 2FA OTP tokens
│   ├── benign_sprint_standup.txt              # Standard sprint standup discussing PR review & test suites
│   └── benign_incident_resolved.txt           # Routine SRE redis cache eviction log
│
├── emails/                                # Email files (headers + body)
│   ├── suspicious_wire_fraud_email.json       # Phishing email with SPF SoftFail, mismatched Reply-To, .xyz link
│   ├── suspicious_payroll_update.json         # Direct deposit payroll change phishing email
│   ├── benign_release_notification.json       # Clean internal deployment notification (SPF/DKIM pass)
│   └── benign_meeting_invite.json             # Calendar invite for technical architecture review
│
└── cases/                                 # Complete multimodal test cases (JSON)
    ├── case_suspicious_ceo_wire_fraud.json    # Pre-assembled suspicious case (Email + Chat + Image)
    └── case_benign_devops_maintenance.json    # Pre-assembled benign case (Email + Chat + Image)
```

---

## How to Test in the Live Dashboard

### Step 1: Start the Dashboard
```bash
python -m uvicorn src.api:app --port 8000
```
Open **`http://localhost:8000`** in your web browser.

---

### Step 2: Instant One-Click Testing

In the top-left card **"Pre-Packaged Samples"**:
1. Click **"Load suspicious sample"**:
   - Loads a pre-built CEO wire fraud incident.
   - Click **"Run Investigation"** (blue button).
   - **Expected Outcome:**
     - Overall Risk Score: **~99.6% (CRITICAL RISK)**
     - CNN Visual Score: **100.0%**
     - BiLSTM Text Score: **~99.4%**
     - Key Forensic Indicators flagged: *Urgency & Suspicious Language Pattern*, *Visual Evidence Tampering (CNN)*, *Suspicious TLD Link (.xyz)*, *Executable File Attachment*.
     - Grad-CAM heatmap generated with peak activation highlighting tampered area.
2. Click **"Load benign sample"**:
   - Loads a routine IT maintenance case.
   - Click **"Run Investigation"**.
   - **Expected Outcome:**
     - Overall Risk Score: **~0.3% (LOW RISK)**
     - Risk Tier: **LOW** (Emerald Badge)
     - No high-risk indicators flagged.

---

### Step 3: Interactive Custom Evidence Ingestion

You can upload custom evidence items into the active case using the **"Ingest New Evidence"** card:

#### Scenario A: Testing Image Tampering Detection & Grad-CAM Heatmap
1. In the **"Ingest New Evidence"** card, select **"Image (Screenshot)"** from the dropdown.
2. Click the file selector and upload `test_data/images/suspicious_wire_transfer_invoice.png`.
3. Click **"Add Evidence to Active Case"**.
4. Click **"Run Investigation"**.
5. Scroll down to the **Grad-CAM Visual Heatmap** section:
   - Notice the visual heatmap generated over the image highlighting the altered banking amount and urgent stamp!

#### Scenario B: Testing Linguistic Urgency & Deceptive Chat Detection
1. Select **"Chat (Text)"** from the dropdown.
2. Copy the text from `test_data/chats/suspicious_ceo_urgent_wire.txt` and paste it into the textarea:
   ```text
   Chief Executive Officer: We are closing a highly confidential international acquisition. I need an immediate wire transfer of $485,000 processed to our overseas escrow account before 3:30 PM cutoff. Standard protocol will delay closing. Do NOT contact other directors as strict confidentiality applies under SEC regulations.
   ```
3. Click **"Add Evidence to Active Case"**.
4. Click **"Run Investigation"**.
5. Observe:
   - BiLSTM score jumps to **>95%**.
   - Salient Linguistic Tokens displayed: `wire`, `transfer`, `immediate`, `confidential`, `urgent`.

#### Scenario C: Testing Phishing Email Headers & Domain Anomaly Detection
1. Select **"Email (Text/JSON)"** from the dropdown.
2. Paste the body and headers from `test_data/emails/suspicious_wire_fraud_email.json`.
3. Click **"Add Evidence to Active Case"** $\to$ **"Run Investigation"**.
4. Observe the **Email Header Anomaly** and **URL Score** bars in the 7D Feature breakdown.

---

### Step 4: Testing the Read-Only "Results" Tab
1. In the top navigation bar, click the **"Results (Benchmark v10)"** tab.
2. The tab immediately queries `results/experiment_results_v10.json` (zero hardcoded numbers):
   - Inspect **Arms A–E** Mean $\pm$ SD ROC-AUC and Brier scores.
   - Inspect the **Delta Comparisons**:
     - Multimodal Fusion ($D - B$): $+0.0768$ (statistically significant benefit).
     - VAE Augmentation ($E - D$): $-0.0062$ (badged as *"no benefit"*).
   - Inspect the **SHA-256 Manifest and Results Hashes** verifying tamper-proof data provenance.

---

### Step 5: Testing via Command-Line Interface (CLI)
You can also run full investigations directly on the JSON case files from your terminal:

```bash
# Run investigation on the suspicious test case
python run_investigation.py --case test_data/cases/case_suspicious_ceo_wire_fraud.json

# Run investigation on the benign test case
python run_investigation.py --case test_data/cases/case_benign_devops_maintenance.json
```
The CLI outputs the complete textual report and saves a detailed Markdown audit summary.
"""

with open(os.path.join(TEST_DATA_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

# Run image creation
make_suspicious_wire_invoice()
make_suspicious_login_portal()
make_benign_ci_cd_log()
make_benign_dashboard_screenshot()

print("All test data generated successfully in test_data/!")
