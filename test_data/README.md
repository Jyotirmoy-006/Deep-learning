# ForenSightX — Dashboard Test Data & Sample Evidence Package

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
3. Click **"Add Evidence to Active Case"** $	o$ **"Run Investigation"**.
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
