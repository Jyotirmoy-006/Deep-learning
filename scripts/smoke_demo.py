"""
ForenSightX - Smoke Demo Verification Script
Hits every REST endpoint, validates model inference, and persists fallback reports
plus Grad-CAM screenshots in docs/demo/.
"""

import os
import sys
import json
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from src.api import app as fastapi_app
from run_investigation import create_sample_case, load_or_init_models
from src.agents import CaseCoordinatorAgent

DEMO_DIR = os.path.join("docs", "demo")
os.makedirs(DEMO_DIR, exist_ok=True)


def run_smoke_demo():
    print("=" * 70)
    print("ForenSightX - Running Smoke Demo & Endpoint Verification")
    print("=" * 70)

    client = TestClient(fastapi_app)
    results_ledger = {}

    # 1. Health Check
    print("\n[1/10] Testing GET /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print(f"  -> Status: {health_data['status']}")
    print(f"  -> Models Loaded: {health_data['models_loaded']}")
    print(f"  -> Missing Checkpoints: {health_data['missing_checkpoints']}")
    assert len(health_data["missing_checkpoints"]) == 0, "Missing model checkpoints detected!"
    results_ledger["health"] = "PASS"

    # 2. Serve Dashboard HTML
    print("\n[2/10] Testing GET / (Dashboard HTML)...")
    res = client.get("/")
    assert res.status_code == 200
    assert "SYNTHETIC BENCHMARK - NOT REAL-WORLD EVIDENCE" in res.text
    assert "Load suspicious sample" in res.text
    assert "Load benign sample" in res.text
    assert "Run Investigation" in res.text
    assert "Decision Support Notice" in res.text or "Decision Support" in res.text
    print("  -> Dashboard HTML loaded and contains all mandatory elements.")
    # Save standalone dashboard preview
    with open(os.path.join(DEMO_DIR, "dashboard_preview.html"), "w", encoding="utf-8") as f:
        f.write(res.text)
    results_ledger["dashboard_html"] = "PASS"

    # 3. Read-Only Results v10 Endpoint
    print("\n[3/10] Testing GET /api/results_v10...")
    res = client.get("/api/results_v10")
    assert res.status_code == 200
    v10_data = res.json()
    print(f"  -> Manifest SHA-256: {v10_data['manifest_sha256'][:16]}...")
    print(f"  -> Results SHA-256: {v10_data['results_sha256']}")
    print(f"  -> Execution Verdict: {v10_data['execution_verdict']}")
    print(f"  -> Arms Reported: {len(v10_data['arms_summary'])}")
    print(f"  -> Delta Comparisons: {len(v10_data['delta_comparisons'])}")
    
    # Check that E-D is marked 'no benefit'
    ed_comp = next((d for d in v10_data["delta_comparisons"] if "E_augmented" in d["comparison"] and "D_ann_7d" in d["comparison"]), None)
    assert ed_comp is not None, "E vs D comparison missing from results_v10!"
    print(f"  -> E vs D Delta: {ed_comp['point_estimate_delta_auc']} [{ed_comp['ci_lower_auc']}, {ed_comp['ci_upper_auc']}] -> {ed_comp['benefit_status']}")
    assert ed_comp["benefit_status"] == "no benefit", "E vs D must be stated as 'no benefit'!"
    results_ledger["results_v10"] = "PASS"

    # 4. Backward-Compatible /api/experiments
    print("\n[4/10] Testing GET /api/experiments...")
    res = client.get("/api/experiments")
    assert res.status_code == 200
    assert res.json()["status"] == "completed"
    results_ledger["experiments_endpoint"] = "PASS"

    # 5. Case Creation & Listing
    print("\n[5/10] Testing Case Creation & Listing...")
    case_test_id = f"CASE-SMOKE-{int(datetime.now().timestamp())}"
    res = client.post("/api/cases", json={
        "case_id": case_test_id,
        "title": "Automated Smoke Test Case",
        "investigator": "CI Test Lead"
    })
    assert res.status_code in [201, 200]
    res_list = client.get("/api/cases")
    assert res_list.status_code == 200
    cases_found = [c["case_id"] for c in res_list.json()["cases"]]
    assert case_test_id in cases_found
    print(f"  -> Case {case_test_id} successfully created and verified in listing.")
    results_ledger["cases_crud"] = "PASS"

    # 6. Load Suspicious Sample Case
    print("\n[6/10] Testing POST /api/cases/load_sample?sample_type=suspicious...")
    res = client.post("/api/cases/load_sample?sample_type=suspicious")
    assert res.status_code == 200
    susp_info = res.json()
    susp_case_id = susp_info["case_id"]
    print(f"  -> Loaded suspicious case {susp_case_id} with {susp_info['evidence_count']} evidence items.")
    results_ledger["load_suspicious_sample"] = "PASS"

    # 7. Investigate Suspicious Sample Case
    print(f"\n[7/10] Testing POST /api/cases/{susp_case_id}/investigate...")
    res_inv = client.post(f"/api/cases/{susp_case_id}/investigate")
    assert res_inv.status_code == 200
    susp_report = res_inv.json()
    print(f"  -> Overall Risk Score: {susp_report['overall_risk_score'] * 100:.2f}% ({susp_report['risk_tier']})")
    print(f"  -> 7D Feature Vector: {susp_report['feature_vector_7d']}")
    print(f"  -> Key Indicators: {len(susp_report['key_evidence_indicators'])}")
    print(f"  -> Timeline Events: {len(susp_report['chronological_timeline'])}")
    
    # Save fallback outputs in docs/demo/
    susp_json_path = os.path.join(DEMO_DIR, "suspicious_sample_report.json")
    with open(susp_json_path, "w", encoding="utf-8") as f:
        json.dump(susp_report, f, indent=2)

    susp_md_path = os.path.join(DEMO_DIR, "suspicious_sample_report.md")
    with open(susp_md_path, "w", encoding="utf-8") as f:
        f.write(f"# ForenSightX Investigation Report: {susp_case_id}\n\n")
        f.write(f"**Title:** {susp_report.get('title')}\n")
        f.write(f"**Risk Score:** {susp_report['overall_risk_score'] * 100:.2f}%\n")
        f.write(f"**Risk Tier:** {susp_report['risk_tier']}\n\n")
        f.write("## 7D Feature Scores\n")
        for k, v in susp_report["model_predictions"].items():
            f.write(f"- **{k}:** {v:.4f}\n")
        f.write("\n## Chronological Timeline\n")
        for ev in susp_report["chronological_timeline"]:
            f.write(f"- `[{ev['event_type'].upper()}]` {ev['timestamp']}: {ev['description']}\n")
        f.write("\n## Key Forensic Indicators\n")
        for ind in susp_report["key_evidence_indicators"]:
            f.write(f"- ⚠️ {ind}\n")
        f.write(f"\n> **Legal Notice:** {susp_report['legal_disclaimer']}\n")
    print(f"  -> Saved reports to {susp_json_path} and {susp_md_path}")
    results_ledger["investigate_suspicious"] = "PASS"

    # 8. Grad-CAM Image Endpoint & Screenshot
    print(f"\n[8/10] Testing GET /api/cases/{susp_case_id}/gradcam/EVID-003...")
    res_cam = client.get(f"/api/cases/{susp_case_id}/gradcam/EVID-003")
    assert res_cam.status_code == 200, f"Gradcam failed: {res_cam.text}"
    assert res_cam.headers["content-type"] == "image/png"
    gradcam_path = os.path.join(DEMO_DIR, "gradcam_suspicious.png")
    with open(gradcam_path, "wb") as f:
        f.write(res_cam.content)
    print(f"  -> Saved Grad-CAM screenshot to {gradcam_path} ({len(res_cam.content)} bytes).")
    results_ledger["gradcam_screenshot"] = "PASS"

    # 9. Load & Investigate Benign Sample Case
    print("\n[9/10] Testing Benign Sample Case Workflow...")
    res_b = client.post("/api/cases/load_sample?sample_type=benign")
    assert res_b.status_code == 200
    benign_case_id = res_b.json()["case_id"]
    res_inv_b = client.post(f"/api/cases/{benign_case_id}/investigate")
    assert res_inv_b.status_code == 200
    benign_report = res_inv_b.json()
    print(f"  -> Benign Risk Score: {benign_report['overall_risk_score'] * 100:.2f}% ({benign_report['risk_tier']})")
    assert benign_report["overall_risk_score"] < 0.20, "Benign case risk score unexpectedly high!"

    # Save benign report
    benign_json_path = os.path.join(DEMO_DIR, "benign_sample_report.json")
    with open(benign_json_path, "w", encoding="utf-8") as f:
        json.dump(benign_report, f, indent=2)

    benign_md_path = os.path.join(DEMO_DIR, "benign_sample_report.md")
    with open(benign_md_path, "w", encoding="utf-8") as f:
        f.write(f"# ForenSightX Investigation Report: {benign_case_id}\n\n")
        f.write(f"**Title:** {benign_report.get('title')}\n")
        f.write(f"**Risk Score:** {benign_report['overall_risk_score'] * 100:.2f}%\n")
        f.write(f"**Risk Tier:** {benign_report['risk_tier']}\n\n")
        f.write("## 7D Feature Scores\n")
        for k, v in benign_report["model_predictions"].items():
            f.write(f"- **{k}:** {v:.4f}\n")
        f.write(f"\n> **Legal Notice:** {benign_report['legal_disclaimer']}\n")
    print(f"  -> Saved benign report to {benign_json_path} and {benign_md_path}")

    # Save benign Grad-CAM screenshot
    res_cam_b = client.get(f"/api/cases/{benign_case_id}/gradcam/EVID-103")
    if res_cam_b.status_code == 200:
        gradcam_b_path = os.path.join(DEMO_DIR, "gradcam_benign.png")
        with open(gradcam_b_path, "wb") as f:
            f.write(res_cam_b.content)
        print(f"  -> Saved benign Grad-CAM screenshot to {gradcam_b_path}")
    results_ledger["benign_workflow"] = "PASS"

    # 10. Manual Upload Ingestion Endpoint
    print("\n[10/10] Testing POST /api/cases/{case_id}/upload_evidence...")
    res_up = client.post(f"/api/cases/{susp_case_id}/upload_evidence", json={
        "evidence_type": "chat",
        "evidence_id": "EVID-MANUAL-01",
        "text_content": "Examiner manual observation: suspected remote shell beacon detected on host.",
        "metadata": {"examiner": "Lead Forensic Lead", "source": "Console"}
    })
    assert res_up.status_code == 200
    assert res_up.json()["evidence_id"] == "EVID-MANUAL-01"
    print("  -> Upload evidence recorded successfully.")
    results_ledger["upload_evidence"] = "PASS"

    # Summary
    print("\n" + "=" * 70)
    print("ALL 10 SMOKE DEMO ENDPOINTS VERIFIED SUCCESSFULLY:")
    for k, v in results_ledger.items():
        print(f"  - {k:25s}: {v}")
    print(f"Fallback outputs saved in: {os.path.abspath(DEMO_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_demo()
