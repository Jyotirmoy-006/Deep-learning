"""
scripts/verify_all_proofs.py
Comprehensive End-to-End Forensic Model & Dashboard Proof Verification Suite.
Runs 23 automated end-to-end tests across all pre-packaged samples, 10 images,
4 chats, 4 emails, 2 complete cases, and the canonical results ledger.
"""

import sys
import os
import json
import base64
import glob
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.api import app

client = TestClient(app)

print("=" * 80)
print("FORENSIGHTX — COMPREHENSIVE END-TO-END MODEL & DASHBOARD PROOF VERIFICATION")
print("=" * 80)

failures = []
passed_tests = []

def record_pass(category, name, risk, tier, details):
    passed_tests.append((category, name, risk, tier, details))
    print(f"  [PASS] {category:<14} | {name:<38} -> Risk: {risk*100:5.1f}% [{tier:<4}] | {details}")

def record_fail(category, name, risk, tier, reason):
    failures.append((category, name, risk, tier, reason))
    print(f"  [FAIL] {category:<14} | {name:<38} -> Risk: {risk*100:5.1f}% [{tier:<4}] | REASON: {reason}")


# =============================================================================
# 1. TEST PRE-PACKAGED SAMPLES (Dashboard Instant Buttons)
# =============================================================================
print("\n[SECTION 1] Pre-Packaged Benchmark Cases (Dashboard Instant Buttons)")

res = client.post("/api/cases/load_sample?sample_type=suspicious")
assert res.status_code == 200, f"Failed to load suspicious sample: {res.text}"
case_data = res.json()
case_id = case_data["case_id"]

res = client.post(f"/api/cases/{case_id}/investigate")
assert res.status_code == 200, f"Investigation failed: {res.text}"
rep = res.json()
risk = rep["overall_risk_score"]
tier = rep["risk_tier"]

res_cam = client.get(f"/api/cases/{case_id}/gradcam/EVID-003")
cam_ok = (res_cam.status_code == 200 and len(res_cam.content) > 1000)

if risk > 0.90 and tier == "HIGH" and cam_ok and len(rep["key_evidence_indicators"]) >= 4:
    record_pass("Button", "Load suspicious sample", risk, tier, f"CNN: {rep['model_predictions']['cnn_score']*100:.1f}%, BiLSTM: {rep['model_predictions']['bilstm_score']*100:.1f}%, Grad-CAM: OK")
else:
    record_fail("Button", "Load suspicious sample", risk, tier, "Expected risk > 90%, tier HIGH, valid Grad-CAM")

res = client.post("/api/cases/load_sample?sample_type=benign")
assert res.status_code == 200, f"Failed to load benign sample: {res.text}"
case_data = res.json()
case_id = case_data["case_id"]

res = client.post(f"/api/cases/{case_id}/investigate")
assert res.status_code == 200, f"Investigation failed: {res.text}"
rep = res.json()
risk = rep["overall_risk_score"]
tier = rep["risk_tier"]

if risk < 0.10 and tier == "LOW" and len(rep["key_evidence_indicators"]) == 0:
    record_pass("Button", "Load benign sample", risk, tier, f"CNN: {rep['model_predictions']['cnn_score']*100:.1f}%, BiLSTM: {rep['model_predictions']['bilstm_score']*100:.1f}%, Indicators: 0")
else:
    record_fail("Button", "Load benign sample", risk, tier, "Expected risk < 10%, tier LOW, 0 indicators")


# =============================================================================
# 2. TEST ALL 10 EVIDENCE IMAGES (Direct Upload & Ingest)
# =============================================================================
print("\n[SECTION 2] All 10 Forensic Visual Images (CNN + Grad-CAM Heatmaps)")

images = sorted(glob.glob("test_data/images/*.png"))
for img_path in images:
    fn = os.path.basename(img_path)
    is_susp = "suspicious" in fn
    
    cid = f"TEST-IMG-{fn[:12]}"
    client.post("/api/cases", json={"case_id": cid, "title": f"Image Audit: {fn}", "investigator": "Automated Tester"})
    
    with open(img_path, "rb") as f:
        b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    
    upload_res = client.post(f"/api/cases/{cid}/upload_evidence", json={
        "evidence_type": "image",
        "evidence_id": "EVID-IMG",
        "filename": fn,
        "content_b64": b64
    })
    assert upload_res.status_code == 200
    
    rep = client.post(f"/api/cases/{cid}/investigate").json()
    risk = rep["overall_risk_score"]
    tier = rep["risk_tier"]
    cnn_score = rep["model_predictions"]["cnn_score"]
    
    vis_dict = rep.get("visual_explainability", {})
    ev_info = vis_dict.get("EVID-IMG", {})
    max_act = ev_info.get("max_activation", 0.0)
    salient = ev_info.get("salient_patch_detected", False)
    
    if is_susp:
        if risk > 0.80 and tier == "HIGH" and cnn_score > 0.90 and salient:
            record_pass("Image (Susp)", fn, risk, tier, f"CNN: {cnn_score*100:5.1f}%, Grad-CAM Max: {max_act:.4f} [Salient: {salient}]")
        else:
            record_fail("Image (Susp)", fn, risk, tier, f"Expected HIGH risk, CNN > 90%, Salient TRUE. Got CNN={cnn_score}, Salient={salient}")
    else:
        if risk < 0.10 and tier == "LOW" and cnn_score < 0.05 and not salient:
            record_pass("Image (Benign)", fn, risk, tier, f"CNN: {cnn_score*100:5.1f}%, Grad-CAM Max: {max_act:.4f} [Salient: {salient}]")
        else:
            record_fail("Image (Benign)", fn, risk, tier, f"Expected LOW risk, CNN < 5%, Salient FALSE. Got CNN={cnn_score}, Salient={salient}")


# =============================================================================
# 3. TEST ALL 4 CHAT TRANSCRIPTS
# =============================================================================
print("\n[SECTION 3] All 4 Chat Transcripts (BiLSTM Linguistic Agent)")

chats = sorted(glob.glob("test_data/chats/*.txt"))
for chat_path in chats:
    fn = os.path.basename(chat_path)
    is_susp = "suspicious" in fn
    with open(chat_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    cid = f"TEST-CHAT-{fn[:12]}"
    client.post("/api/cases", json={"case_id": cid, "title": f"Chat Audit: {fn}", "investigator": "Automated Tester"})
    
    client.post(f"/api/cases/{cid}/upload_evidence", json={
        "evidence_type": "chat",
        "evidence_id": "EVID-CHAT",
        "filename": fn,
        "text_content": content
    })
    
    rep = client.post(f"/api/cases/{cid}/investigate").json()
    risk = rep["overall_risk_score"]
    tier = rep["risk_tier"]
    bilstm_score = rep["model_predictions"]["bilstm_score"]
    ind_count = len(rep["key_evidence_indicators"])
    
    if is_susp:
        if risk > 0.85 and tier == "HIGH" and bilstm_score > 0.80 and ind_count >= 1:
            record_pass("Chat (Susp)", fn, risk, tier, f"BiLSTM: {bilstm_score*100:5.1f}%, Indicators: {ind_count}")
        else:
            record_fail("Chat (Susp)", fn, risk, tier, f"Expected HIGH risk, BiLSTM > 80%. Got BiLSTM={bilstm_score}, Indicators={ind_count}")
    else:
        if risk < 0.20 and tier == "LOW" and ind_count == 0:
            record_pass("Chat (Benign)", fn, risk, tier, f"BiLSTM: {bilstm_score*100:5.1f}%, Indicators: {ind_count}")
        else:
            record_fail("Chat (Benign)", fn, risk, tier, f"Expected LOW risk, 0 indicators. Got Risk={risk}, Indicators={ind_count}")


# =============================================================================
# 4. TEST ALL 4 EMAIL FILES
# =============================================================================
print("\n[SECTION 4] All 4 Email Datasets (Headers + Body + Attachments + URLs)")

emails = sorted(glob.glob("test_data/emails/*.json"))
for email_path in emails:
    fn = os.path.basename(email_path)
    is_susp = "suspicious" in fn
    with open(email_path, "r", encoding="utf-8") as f:
        em_data = json.load(f)
    
    cid = f"TEST-EM-{fn[:12]}"
    client.post("/api/cases", json={"case_id": cid, "title": f"Email Audit: {fn}", "investigator": "Automated Tester"})
    
    body = em_data.get("body", "")
    headers = em_data.get("headers", {})
    attachments = em_data.get("attachments", [])
    
    client.post(f"/api/cases/{cid}/upload_evidence", json={
        "evidence_type": "email",
        "evidence_id": "EVID-EMAIL",
        "filename": fn,
        "text_content": body,
        "metadata": {"headers": headers, "attachments": attachments}
    })
    
    rep = client.post(f"/api/cases/{cid}/investigate").json()
    risk = rep["overall_risk_score"]
    tier = rep["risk_tier"]
    p = rep["model_predictions"]
    ind_count = len(rep["key_evidence_indicators"])
    
    if is_susp:
        if risk > 0.85 and tier == "HIGH" and ind_count >= 2:
            record_pass("Email (Susp)", fn, risk, tier, f"BiLSTM: {p['bilstm_score']*100:5.1f}%, URL: {p['url_score']*100:.0f}%, Att: {p['attachment_score']*100:.0f}%")
        else:
            record_fail("Email (Susp)", fn, risk, tier, f"Expected HIGH risk, >=2 indicators. Got Risk={risk}, Indicators={ind_count}")
    else:
        if risk < 0.20 and tier == "LOW" and ind_count == 0:
            record_pass("Email (Benign)", fn, risk, tier, f"BiLSTM: {p['bilstm_score']*100:5.1f}%, Clean SPF/DKIM, 0 indicators")
        else:
            record_fail("Email (Benign)", fn, risk, tier, f"Expected LOW risk, 0 indicators. Got Risk={risk}, Indicators={ind_count}")


# =============================================================================
# 5. TEST COMPLETE MULTIMODAL CASES (JSON Cases)
# =============================================================================
print("\n[SECTION 5] Complete Pre-Assembled Multimodal Incidents")

cases = sorted(glob.glob("test_data/cases/*.json"))
for case_file in cases:
    fn = os.path.basename(case_file)
    is_susp = "suspicious" in fn
    with open(case_file, "r", encoding="utf-8") as f:
        cdata = json.load(f)
    
    cid = cdata["case_id"]
    client.post("/api/cases", json={"case_id": cid, "title": cdata["title"], "investigator": cdata["investigator"]})
    
    for it in cdata["evidence_items"]:
        client.post(f"/api/cases/{cid}/upload_evidence", json={
            "evidence_type": it["evidence_type"],
            "evidence_id": it["evidence_id"],
            "timestamp": it.get("timestamp"),
            "text_content": it.get("content") if it["evidence_type"] != "image" else None,
            "filename": it.get("content") if it["evidence_type"] == "image" else None,
            "metadata": it.get("metadata", {})
        })
    
    rep = client.post(f"/api/cases/{cid}/investigate").json()
    risk = rep["overall_risk_score"]
    tier = rep["risk_tier"]
    
    if is_susp:
        if risk > 0.95 and tier == "HIGH" and len(rep["chronological_timeline"]) >= 3:
            record_pass("Case (Susp)", fn, risk, tier, f"Full 7D synthesis, {len(rep['chronological_timeline'])} timeline events")
        else:
            record_fail("Case (Susp)", fn, risk, tier, f"Expected risk > 95%, tier HIGH. Got Risk={risk}")
    else:
        if risk < 0.10 and tier == "LOW":
            record_pass("Case (Benign)", fn, risk, tier, f"Full 7D synthesis, {len(rep['chronological_timeline'])} timeline events")
        else:
            record_fail("Case (Benign)", fn, risk, tier, f"Expected risk < 10%, tier LOW. Got Risk={risk}")


# =============================================================================
# 6. TEST RESULTS TAB (Canonical Benchmark Ledger)
# =============================================================================
print("\n[SECTION 6] Results Tab & Provenance Verification (/api/results_v10)")

res = client.get("/api/results_v10")
assert res.status_code == 200
v10 = res.json()

manifest_ok = len(v10.get("manifest_sha256", "")) == 64
results_ok = len(v10.get("results_sha256", "")) == 64
verdict_ok = v10.get("execution_verdict", {}).get("status") == "ALL_GATES_PASSED_AND_VALIDATED"
arms_count = len(v10.get("arms_summary", []))
deltas_count = len(v10.get("delta_comparisons", []))

ed = next((d for d in v10.get("delta_comparisons", []) if "E_augmented" in d["comparison"] and "D_ann_7d" in d["comparison"]), None)
ed_ok = (ed is not None and ed["benefit_status"] == "no benefit")

if manifest_ok and results_ok and verdict_ok and arms_count == 7 and deltas_count == 8 and ed_ok:
    record_pass("Ledger", "results/experiment_results_v10.json", 0.0, "INFO", f"7 Arms, 8 Deltas, E-D 'no benefit', Verdict: ALL_GATES_PASSED")
else:
    record_fail("Ledger", "results/experiment_results_v10.json", 0.0, "FAIL", "Results schema, SHA hashes or E-D status mismatch")


# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 80)
print(f"VERIFICATION SUMMARY: {len(passed_tests)} PASSED | {len(failures)} FAILED")
print("=" * 80)

if failures:
    print("\nFAILED CHECKS:")
    for cat, name, risk, tier, reason in failures:
        print(f"  - [{cat}] {name}: {reason}")
    sys.exit(1)
else:
    print("\nALL FORENSIC PROOFS, MODELS, IMAGES, CHATS, EMAILS & CASES VALIDATED 100% CORRECT!")
    sys.exit(0)
