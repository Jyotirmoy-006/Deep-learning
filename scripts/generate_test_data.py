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

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
IMG_DIR = os.path.join(TEST_DATA_DIR, "images")
CHAT_DIR = os.path.join(TEST_DATA_DIR, "chats")
EMAIL_DIR = os.path.join(TEST_DATA_DIR, "emails")
CASE_DIR = os.path.join(TEST_DATA_DIR, "cases")

for d in [TEST_DATA_DIR, IMG_DIR, CHAT_DIR, EMAIL_DIR, CASE_DIR]:
    os.makedirs(d, exist_ok=True)


from src.data_loader import ForensicDatasetGenerator

# =====================================================================
# 1. Generate Realistic Forensic Visual Evidence Images (512x512)
# =====================================================================

def safe_save(img, path):
    tmp = path + "_tmp.png"
    img.save(tmp, format="PNG")
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)


def make_suspicious_wire_invoice():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    # Red alert bar
    d.rectangle([20, 20, 492, 60], fill=(210, 45, 45))
    d.text((35, 30), "SWIFT FIN-103 WIRE VOUCHER - TRANSACTION #88219-X", fill=(255, 255, 255))
    
    # Header info
    d.text((35, 70), "CITIBANK GLOBAL TRANSACTION SERVICES | NEW YORK / CAYMAN", fill=(40, 50, 70))
    
    # Altered/Tampered alert box
    d.rectangle([40, 95, 472, 285], outline=(150, 0, 0), fill=(255, 230, 230), width=2)
    d.text((55, 110), "BENEFICIARY: Offshore Escrow Holdings Cayman Ltd", fill=(150, 0, 0))
    d.text((55, 135), "BANK: CITIBANK INT. N.A. (CAYMAN ISLANDS BRANCH)", fill=(50, 50, 50))
    d.text((55, 160), "IBAN: KY88CITI00019283749281 | SWIFT: CITIUS33KY", fill=(50, 50, 50))
    d.text((55, 185), "ORDERING CUSTOMER: Apex Corporate Holdings LLC", fill=(50, 50, 50))
    d.text((55, 215), "REMITTANCE AMOUNT: $485,000.00 USD (CLEARED)", fill=(180, 20, 20))
    d.text((55, 245), "TAMPERING FLAG: [SPLICED BENEFICIARY & ROUTING DETECTED]", fill=(200, 20, 20))
    
    # Dense document metadata
    d.line([(40, 310), (472, 310)], fill=(200, 210, 220), width=2)
    d.text((45, 325), "Value Date: 05-NOV-2026 14:22:15 UTC | Channel: Automated Fedwire", fill=(70, 80, 95))
    d.text((45, 350), "Intermediary Clearing: CHIPS Participant #00441-NYC", fill=(70, 80, 95))
    d.text((45, 375), "Dual-Control Authorization: OVERRIDDEN BY EXECUTIVE DIRECTIVE", fill=(160, 40, 40))
    d.text((45, 400), "Originating Workstation IP: 198.51.100.24 (External Proxy)", fill=(70, 80, 95))
    d.text((45, 425), "Remittance Memo: Confidential Acquisition Bridge Funding", fill=(70, 80, 95))
    d.text((45, 455), "AUDIT STATUS: CRITICAL - HIGH VALUE ESCROW ANOMALY", fill=(180, 20, 20))
    
    arr = np.array(img)
    noise = np.random.randint(0, 45, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    p = os.path.join(IMG_DIR, "suspicious_wire_transfer_invoice.png")
    safe_save(Image.fromarray(arr), p)
    return p


def make_suspicious_login_portal():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 60], fill=(210, 45, 45))
    d.text((35, 30), "SECURITY ALERT: UNUSUAL SSO SIGN-IN DETECTED", fill=(255, 255, 255))
    
    d.rectangle([25, 70, 487, 100], fill=(230, 235, 242), outline=(190, 200, 215))
    d.text((35, 78), "https://login.microsoftonline.corporate-sso-auth.xyz/oauth2/token", fill=(180, 30, 30))
    
    d.rectangle([40, 115, 472, 305], outline=(150, 0, 0), fill=(255, 230, 230), width=2)
    d.text((55, 125), "MICROSOFT AZURE AD / CORPORATE IDENTITY ACCESS", fill=(150, 0, 0))
    d.text((55, 150), "Your enterprise session has expired due to foreign login from IP 185.220.101.4", fill=(60, 60, 60))
    
    d.rectangle([55, 180, 455, 210], fill=(255, 255, 255), outline=(150, 150, 150))
    d.text((65, 188), "Username: john.doe@company.internal", fill=(80, 80, 80))
    
    d.rectangle([55, 220, 455, 250], fill=(255, 255, 255), outline=(150, 150, 150))
    d.text((65, 228), "Password: ********************", fill=(180, 20, 20))
    
    d.rectangle([55, 260, 455, 290], fill=(255, 255, 255), outline=(180, 30, 30), width=2)
    d.text((65, 268), "Enter 6-Digit SMS OTP Code: [ 8 9 4 - 1 0 2 ]", fill=(200, 20, 20))
    
    d.text((45, 330), "Failure to authenticate credentials immediately will result in access lock.", fill=(180, 20, 20))
    d.text((45, 360), "Phishing Indicators Detected: Spoofed .xyz domain, fake OAuth redirection.", fill=(70, 80, 95))
    d.text((45, 390), "Certificate Status: INVALID SELF-SIGNED SSL CERTIFICATE", fill=(180, 30, 30))
    d.text((45, 420), "Corporate IT Policy: Never enter phone OTP codes on unverified portals.", fill=(70, 80, 95))
    d.text((45, 450), "STATUS: HARVESTED CREDENTIALS TRANSMITTED TO C2 SERVER", fill=(180, 20, 20))
    
    arr = np.array(img)
    noise = np.random.randint(0, 45, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    p = os.path.join(IMG_DIR, "suspicious_spoofed_login_portal.png")
    safe_save(Image.fromarray(arr), p)
    return p


def make_suspicious_ransomware_demand():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 60], fill=(210, 45, 45))
    d.text((35, 30), "BLACK-CAT LOCKER V4.1 - SYSTEM VOLUMES ENCRYPTED", fill=(255, 255, 255))
    
    d.rectangle([40, 80, 472, 275], outline=(150, 0, 0), fill=(255, 230, 230), width=2)
    d.text((55, 95), "ALL YOUR CORPORATE SERVERS, DATABASES AND BACKUPS ARE ENCRYPTED", fill=(180, 20, 20))
    d.text((55, 125), "RANSOM DEMAND: 2.50000000 BITCOIN (~$175,000 USD)", fill=(180, 20, 20))
    d.text((55, 155), "DEPOSIT BTC TO: bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq", fill=(40, 40, 40))
    d.text((55, 185), "DEADLINE: 23 HOURS 45 MINUTES (COUNTDOWN RUNNING)", fill=(200, 20, 20))
    d.text((55, 215), "TOR CONTACT URL: http://blackcat7x7x7x29.onion/negotiation", fill=(70, 80, 95))
    d.text((55, 245), "WARNING: Attempting recovery without private key will destroy keys.", fill=(150, 0, 0))
    
    d.rectangle([40, 295, 472, 470], fill=(240, 242, 248), outline=(200, 205, 215))
    d.text((55, 310), "FORENSIC INCIDENT SUMMARY REPORT (AUTOMATED TELEMETRY)", fill=(60, 70, 85))
    d.text((55, 335), "Initial Ingress Vector: Compromised VPN Gateway (CVE-2024-3400)", fill=(80, 90, 105))
    d.text((55, 360), "Encryption Standard: ChaCha20-Poly1305 + RSA-4096 Master Key", fill=(80, 90, 105))
    d.text((55, 385), "Exfiltrated Volume: 42.8 GB Customer Financial PII Archives", fill=(180, 20, 20))
    d.text((55, 410), "Shadow Copies: System Volume Information Deletion Confirmed", fill=(180, 20, 20))
    d.text((55, 435), "INCIDENT CLASSIFICATION: SEVERITY-1 ENTERPRISE RANSOMWARE", fill=(180, 20, 20))
    
    arr = np.array(img)
    noise = np.random.randint(0, 45, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    p = os.path.join(IMG_DIR, "suspicious_ransomware_demand.png")
    safe_save(Image.fromarray(arr), p)
    return p


def make_suspicious_fake_security_alert():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 60], fill=(210, 45, 45))
    d.text((35, 30), "WINDOWS DEFENDER SECURITY CENTER: CRITICAL TROJAN ALERT", fill=(255, 255, 255))
    
    d.rectangle([40, 80, 472, 290], outline=(150, 0, 0), fill=(255, 230, 230), width=2)
    d.text((55, 95), "THREAT DETECTED: Trojan:Win32/Spy.AgentTesla.D6", fill=(180, 20, 20))
    d.text((55, 125), "ALERT LEVEL: Severe | Status: Active Memory Infiltration", fill=(180, 20, 20))
    d.text((55, 155), "IMMEDIATE ACTION REQUIRED: Call Microsoft Certified Support Hotline:", fill=(40, 40, 40))
    d.text((55, 185), "TOLL-FREE HELPDESK: 1-800-555-0199 (CODE #8841-SEC)", fill=(180, 20, 20))
    d.text((55, 215), "Do NOT restart workstation - Private Banking & Passwords at risk!", fill=(160, 30, 30))
    d.text((55, 245), "Malicious Payload Origin: C:\\Users\\Public\\svchost_updater.exe", fill=(70, 80, 95))
    
    d.rectangle([40, 310, 472, 470], fill=(240, 242, 248), outline=(200, 205, 215))
    d.text((55, 325), "FORENSIC SOC DECEPTION INDICATORS", fill=(60, 70, 85))
    d.text((55, 350), "Browser Lock Technique: HTML5 Fullscreen API with cursor trap", fill=(80, 90, 105))
    d.text((55, 375), "Phone Number Attribution: VoIP Routing associated with Tech Support Scams", fill=(180, 20, 20))
    d.text((55, 400), "Audio Synthesizer: Continuous deceptive alarm playback detected", fill=(80, 90, 105))
    d.text((55, 430), "STATUS: MALICIOUS SOCIAL ENGINEERING / TECH SUPPORT HOAX", fill=(180, 20, 20))
    
    arr = np.array(img)
    noise = np.random.randint(0, 45, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    p = os.path.join(IMG_DIR, "suspicious_fake_security_alert.png")
    safe_save(Image.fromarray(arr), p)
    return p


def make_suspicious_tampered_executive_memo():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 60], fill=(210, 45, 45))
    d.text((35, 30), "CONFIDENTIAL EXECUTIVE MEMORANDUM - BOARD AUDIT", fill=(255, 255, 255))
    
    d.text((35, 75), "FROM: Office of the Chief Executive Officer | TO: Treasury Management", fill=(50, 60, 75))
    
    d.rectangle([40, 105, 472, 290], outline=(150, 0, 0), fill=(255, 230, 230), width=2)
    d.text((55, 120), "SUBJECT: STRICT CONFIDENTIAL DISBURSEMENT AUTHORIZATION", fill=(180, 20, 20))
    d.text((55, 145), "Disburse immediately: $1,450,000.00 USD to Cayman Trust Account", fill=(180, 20, 20))
    d.text((55, 170), "Mandatory Directive: Bypass routine multi-party verification gates.", fill=(150, 0, 0))
    d.text((55, 195), "Rationale: Competitive bidding embargo on Project Sovereign.", fill=(50, 50, 50))
    d.text((55, 225), "Executive Signature: [DIGITALLY COPIED / SPLICED STAMP DETECTED]", fill=(180, 20, 20))
    d.text((55, 255), "Discrepancy: Signer Public Key mismatch against Enterprise PKI", fill=(180, 20, 20))
    
    d.line([(40, 310), (472, 310)], fill=(200, 210, 220), width=2)
    d.text((45, 325), "Document Forensic Inspection: Optical Character Inconsistencies", fill=(70, 80, 95))
    d.text((45, 350), "Font Kerning Variance: +4.2pt in disbursement paragraph (tampered block)", fill=(180, 20, 20))
    d.text((45, 375), "PDF Metadata Creator: LibreOffice 7.4 / Converted via unknown utility", fill=(70, 80, 95))
    d.text((45, 400), "Original Creation Timestamp: 2026-11-05 13:44:12 UTC", fill=(70, 80, 95))
    d.text((45, 430), "FORENSIC VERDICT: HIGH CERTAINTY DOCUMENT FORGERY / BEC VECTOR", fill=(180, 20, 20))
    
    arr = np.array(img)
    noise = np.random.randint(0, 45, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    p = os.path.join(IMG_DIR, "suspicious_tampered_executive_memo.png")
    safe_save(Image.fromarray(arr), p)
    return p


def make_benign_ci_cd_log():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 55], fill=(70, 130, 180))
    d.text((35, 28), "TERMINAL: devops@cluster-prod-01: ~/forensightx-repo", fill=(255, 255, 255))
    
    d.rectangle([20, 55, 492, 480], fill=(20, 22, 28))
    
    d.ellipse([30, 68, 42, 80], fill=(240, 90, 80))
    d.ellipse([48, 68, 60, 80], fill=(240, 190, 60))
    d.ellipse([66, 68, 78, 80], fill=(80, 200, 100))
    
    d.text((35, 95), "devops@k8s-prod:~/forensightx$ git status", fill=(100, 220, 130))
    d.text((35, 120), "On branch master. Your branch is up to date with origin/master.", fill=(200, 210, 225))
    d.text((35, 145), "nothing to commit, working tree clean", fill=(160, 175, 195))
    
    d.text((35, 180), "devops@k8s-prod:~/forensightx$ python -m pytest -q", fill=(100, 220, 130))
    d.text((35, 205), "................................................................. [100%]", fill=(100, 220, 130))
    d.text((35, 230), "69 passed, 2 warnings in 10.92s (Unit & Regression Passed)", fill=(100, 220, 130))
    
    d.text((35, 265), "devops@k8s-prod:~/forensightx$ docker compose ps", fill=(100, 220, 130))
    d.text((35, 290), "NAME                IMAGE               COMMAND             STATUS", fill=(130, 145, 170))
    d.text((35, 315), "forensightx-api     forensightx:v1.0    uvicorn src.api     Up (healthy)", fill=(200, 210, 225))
    d.text((35, 340), "forensightx-db      sqlite3:v3.42       sqlite3 init        Up (clean)", fill=(200, 210, 225))
    
    d.text((35, 380), "devops@k8s-prod:~/forensightx$ curl -s http://localhost:8000/api/health", fill=(100, 220, 130))
    d.text((35, 405), "{\"status\":\"healthy\",\"service\":\"ForenSightX Forensics Engine\"}", fill=(240, 200, 100))
    
    d.text((35, 440), "devops@k8s-prod:~/forensightx$ exit", fill=(100, 220, 130))
    
    p = os.path.join(IMG_DIR, "benign_ci_cd_deployment_log.png")
    safe_save(img, p)
    return p


def make_benign_dashboard_screenshot():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 55], fill=(70, 130, 180))
    d.text((35, 28), "AWS CLOUDWATCH INFRASTRUCTURE HEALTH - US-EAST-1", fill=(255, 255, 255))
    
    d.rectangle([25, 65, 245, 160], fill=(255, 255, 255), outline=(210, 220, 235))
    d.text((35, 75), "CPU UTILIZATION (AVG)", fill=(100, 110, 130))
    d.text((35, 95), "18.4% (NORMAL)", fill=(30, 150, 70))
    d.text((35, 125), "Max Spike: 24.1% at 14:00 UTC", fill=(130, 140, 155))
    
    d.rectangle([265, 65, 485, 160], fill=(255, 255, 255), outline=(210, 220, 235))
    d.text((275, 75), "MEMORY ALLOCATION", fill=(100, 110, 130))
    d.text((275, 95), "4.12 GB / 16.0 GB (25.7%)", fill=(30, 150, 70))
    d.text((275, 125), "Buffer Cache: 1.8 GB Clean", fill=(130, 140, 155))
    
    d.rectangle([25, 175, 485, 305], fill=(255, 255, 255), outline=(210, 220, 235))
    d.text((35, 185), "SERVICE AVAILABILITY & HEALTH MATRIX", fill=(70, 80, 100))
    d.text((35, 210), "API Gateway (FastAPI): 100.0% Uptime | p99 Latency: 18ms", fill=(40, 140, 60))
    d.text((35, 235), "BiLSTM Linguistic Inference Pool: 100.0% Healthy", fill=(40, 140, 60))
    d.text((35, 260), "Visual Evidence CNN Worker Nodes: 100.0% Healthy", fill=(40, 140, 60))
    d.text((35, 285), "7D Multimodal Risk ANN Classifier: 100.0% Operational", fill=(40, 140, 60))
    
    for y in range(325, 480, 30):
        d.rectangle([25, y, 485, y+20], fill=(235, 240, 248))
    d.text((35, 330), "Database Replica Status: Synced (SQLite WAL mode active)", fill=(80, 95, 115))
    d.text((35, 360), "Network Ingress / Egress: 42.1 MB/s clean traffic", fill=(80, 95, 115))
    d.text((35, 390), "Security Group Audit: 0 open malicious ingress ports", fill=(40, 140, 60))
    d.text((35, 420), "Automated Compliance Gate G2.1: PASSED & VALIDATED", fill=(40, 140, 60))
    d.text((35, 450), "Cluster Status: ALL SYSTEMS FULLY OPERATIONAL", fill=(30, 150, 70))
    
    p = os.path.join(IMG_DIR, "benign_system_dashboard.png")
    safe_save(img, p)
    return p


def make_benign_slack_team_chat():
    size = (512, 512)
    img = Image.new("RGB", size, color=(26, 29, 33))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 55], fill=(70, 130, 180))
    d.text((35, 28), "SLACK: #general-engineering | Core Platform Team", fill=(255, 255, 255))
    
    d.rectangle([20, 55, 140, 480], fill=(18, 20, 24))
    d.text((28, 70), "CHANNELS", fill=(140, 150, 170))
    d.text((28, 95), "# general", fill=(180, 190, 210))
    d.text((28, 120), "# engineering", fill=(255, 255, 255))
    d.text((28, 145), "# releases", fill=(180, 190, 210))
    d.text((28, 170), "# standup", fill=(180, 190, 210))
    d.text((28, 210), "DIRECT MSGS", fill=(140, 150, 170))
    d.text((28, 235), "@ Sarah Chen", fill=(180, 190, 210))
    d.text((28, 260), "@ David Kim", fill=(180, 190, 210))
    d.text((28, 285), "@ Alex Mercer", fill=(180, 190, 210))
    
    d.rectangle([140, 55, 492, 480], fill=(26, 29, 33))
    
    d.text((155, 75), "Sarah Chen (Tech Lead) [10:01 AM]", fill=(120, 170, 230))
    d.text((155, 95), "Good morning team! Standup time. Let's do a quick sync.", fill=(220, 225, 230))
    
    d.text((155, 130), "Alex Mercer (ML Engineer) [10:02 AM]", fill=(120, 170, 230))
    d.text((155, 150), "All 69 unit tests passed cleanly in 10.9s on staging.", fill=(220, 225, 230))
    d.text((155, 170), "BiLSTM and CNN checkpoints loaded via factory with strict=True.", fill=(220, 225, 230))
    
    d.text((155, 205), "David Kim (Frontend) [10:04 AM]", fill=(120, 170, 230))
    d.text((155, 225), "New tactical dark forensic theme deployed and verified.", fill=(220, 225, 230))
    d.text((155, 245), "Grad-CAM heatmaps render smoothly at 60fps.", fill=(220, 225, 230))
    
    d.text((155, 280), "Sarah Chen (Tech Lead) [10:05 AM]", fill=(120, 170, 230))
    d.text((155, 300), "Awesome work everyone. Sprint demo at 2 PM is approved.", fill=(220, 225, 230))
    d.text((155, 320), "Who is joining for lunch at 12:30 PM at the cafeteria?", fill=(220, 225, 230))
    
    d.rectangle([155, 360, 475, 440], fill=(34, 38, 44), outline=(50, 56, 66))
    d.text((165, 375), "Reactions: thumbs-up (6), checkmark (4), coffee (5)", fill=(160, 175, 195))
    d.text((165, 405), "Status: Verified Routine Internal Communication", fill=(100, 200, 120))
    
    p = os.path.join(IMG_DIR, "benign_slack_team_chat.png")
    safe_save(img, p)
    return p


def make_benign_vendor_receipt_pdf():
    size = (512, 512)
    img = Image.new("RGB", size, color=(32, 33, 36))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 55], fill=(70, 130, 180))
    d.text((35, 28), "ADOBE ACROBAT: AWS_OCTOBER_INVOICE_99210.PDF", fill=(255, 255, 255))
    
    d.rectangle([20, 55, 492, 85], fill=(44, 46, 51), outline=(60, 64, 72))
    d.text((35, 63), "File   Edit   View   Tools | Page 1 of 1 | Zoom: 100% | Verified Signed", fill=(180, 190, 205))
    
    d.rectangle([40, 95, 472, 470], fill=(24, 25, 28), outline=(50, 55, 65))
    
    d.text((60, 110), "AMAZON WEB SERVICES BILLING STATEMENT", fill=(220, 225, 235))
    d.text((60, 135), "Invoice Number: #AWS-2026-99210-INV | Date: October 31, 2026", fill=(150, 160, 175))
    d.text((60, 160), "Account ID: 9928-1029-4410 (Corporate Dev Environment)", fill=(150, 160, 175))
    d.line([(60, 185), (452, 185)], fill=(60, 65, 75), width=2)
    
    d.text((60, 195), "SERVICE DESCRIPTION                  USAGE        AMOUNT", fill=(180, 190, 205))
    d.line([(60, 215), (452, 215)], fill=(50, 55, 65), width=1)
    d.text((60, 225), "Amazon EC2 Compute Instances         744.00 Hrs   $24.80 USD", fill=(160, 175, 195))
    d.text((60, 250), "Amazon S3 Standard Storage           500.00 GB    $11.20 USD", fill=(160, 175, 195))
    d.text((60, 275), "Amazon CloudWatch Metrics & Logs     10 Metrics   $ 6.50 USD", fill=(160, 175, 195))
    d.line([(60, 305), (452, 305)], fill=(50, 55, 65), width=1)
    
    d.text((60, 320), "TOTAL CHARGES DUE:                                $42.50 USD", fill=(220, 225, 235))
    
    d.rectangle([250, 360, 440, 420], outline=(80, 180, 100), width=2)
    d.text((275, 375), "PAID IN FULL", fill=(100, 200, 120))
    d.text((260, 395), "AUTO-PAY: VISA #****-8812", fill=(100, 200, 120))
    
    d.text((60, 440), "Status: Verified Legitimate Corporate Vendor Billing", fill=(100, 200, 120))
    
    p = os.path.join(IMG_DIR, "benign_vendor_receipt_pdf.png")
    safe_save(img, p)
    return p


def make_benign_vscode_editor_diff():
    size = (512, 512)
    img = Image.new("RGB", size, color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    
    d.rectangle([20, 20, 492, 55], fill=(70, 130, 180))
    d.text((35, 28), "VISUAL STUDIO CODE: test_regression.py - ForenSightX", fill=(255, 255, 255))
    
    d.rectangle([20, 55, 492, 480], fill=(30, 30, 30))
    
    d.rectangle([20, 55, 110, 480], fill=(37, 37, 38))
    d.text((25, 65), "EXPLORER", fill=(140, 140, 140))
    d.text((25, 85), "> src/", fill=(180, 180, 180))
    d.text((25, 105), "v tests/", fill=(220, 220, 220))
    d.text((32, 125), "test_reg.py", fill=(255, 255, 255))
    d.text((25, 145), "> models/", fill=(180, 180, 180))
    d.text((25, 165), "> docs/", fill=(180, 180, 180))
    
    d.text((120, 65), "1  import pytest", fill=(180, 140, 200))
    d.text((120, 85), "2  from src.models import build_model", fill=(180, 140, 200))
    d.text((120, 105), "3  ", fill=(200, 200, 200))
    d.text((120, 125), "4  def test_multimodal_fusion_forward():", fill=(220, 220, 140))
    d.text((120, 145), "5      model = build_model({'type': 'ann_7d'})", fill=(150, 200, 250))
    d.text((120, 165), "6      x = torch.zeros((1, 7))", fill=(150, 200, 250))
    d.text((120, 185), "7      out = model(x)", fill=(150, 200, 250))
    d.text((120, 205), "8      assert out.shape == (1, 1)", fill=(200, 140, 140))
    d.text((120, 225), "9      assert 0.0 <= out.item() <= 1.0", fill=(200, 140, 140))
    
    d.rectangle([110, 260, 492, 480], fill=(24, 24, 24))
    d.text((120, 270), "TERMINAL  PROBLEMS  OUTPUT  DEBUG CONSOLE", fill=(140, 140, 140))
    d.line([(110, 290), (492, 290)], fill=(50, 50, 50), width=1)
    
    d.text((120, 305), "PS C:\\workspace\\forensightx> python -m pytest -q", fill=(220, 220, 220))
    d.text((120, 330), "................................................................. [100%]", fill=(80, 200, 120))
    d.text((120, 360), "69 passed, 2 warnings in 10.92s", fill=(80, 200, 120))
    d.text((120, 390), "Status: 100% Regression Suite Passing Cleanly", fill=(80, 200, 120))
    d.text((120, 420), "Branch: master (working tree clean, zero uncommitted changes)", fill=(160, 175, 195))
    
    p = os.path.join(IMG_DIR, "benign_vscode_editor_diff.png")
    safe_save(img, p)
    return p


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
├── images/                                # Realistic digital screenshots & forensic visual artifacts
│   ├── suspicious_wire_transfer_invoice.png   # Tampered banking wire invoice with altered amount and urgent stamp
│   ├── suspicious_spoofed_login_portal.png    # Spoofed SSO login portal with credential harvesting prompts
│   ├── suspicious_ransomware_demand.png       # LockBit ransomware demand screen demanding 2.5 BTC
│   ├── suspicious_fake_security_alert.png     # Fake Windows Defender / CrowdStrike trojan alert scam
│   ├── suspicious_tampered_executive_memo.png # Spliced CEO signature on confidential acquisition memo
│   ├── benign_ci_cd_deployment_log.png        # Clean terminal output (git commit, pytest passing 69/69)
│   ├── benign_system_dashboard.png            # Clean server metrics & health monitoring screenshot
│   ├── benign_slack_team_chat.png             # Clean Slack developer channel standup conversation
│   ├── benign_vendor_receipt_pdf.png          # Clean Adobe Acrobat AWS hosting statement (PAID)
│   └── benign_vscode_editor_diff.png          # Clean Visual Studio Code regression test window
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
make_suspicious_ransomware_demand()
make_suspicious_fake_security_alert()
make_suspicious_tampered_executive_memo()

make_benign_ci_cd_log()
make_benign_dashboard_screenshot()
make_benign_slack_team_chat()
make_benign_vendor_receipt_pdf()
make_benign_vscode_editor_diff()

print("All 10 test images and case data generated successfully in test_data/!")

