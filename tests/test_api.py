"""
ForenSightX - API Integration Test Suite
Verifies FastAPI endpoints:
  - Health check
  - Case creation & listing
  - Evidence ingestion
  - Multi-agent investigation trigger & reporting
  - Experimental results endpoint
  - Dashboard frontend view
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_api_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_api_experiments_endpoint():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["completed", "pending"]


def test_api_case_lifecycle():
    case_id = f"TEST-API-CASE-{uuid.uuid4().hex[:8]}"

    # 1. Create Case
    create_res = client.post(
        "/api/cases",
        json={"case_id": case_id, "title": "API Test Phishing Case", "investigator": "Test Agent"},
    )
    assert create_res.status_code in [201, 400]

    # 2. Add Evidence
    evid_res = client.post(
        f"/api/cases/{case_id}/evidence",
        json={
            "evidence_id": "EV-01",
            "evidence_type": "chat",
            "timestamp": "2026-11-01T10:00:00",
            "content": "URGENT: Please verify your credentials immediately",
            "metadata": {"source": "Slack"},
            "is_observed": True,
        },
    )
    assert evid_res.status_code == 200

    # 3. Investigate Case
    inv_res = client.post(f"/api/cases/{case_id}/investigate")
    assert inv_res.status_code == 200
    report = inv_res.json()
    assert report["case_id"] == case_id
    assert "overall_risk_score" in report
    assert "chronological_timeline" in report
    assert "legal_disclaimer" in report

    # 4. Fetch Stored Report
    rep_res = client.get(f"/api/cases/{case_id}/report")
    assert rep_res.status_code == 200
    assert rep_res.json()["case_id"] == case_id


def test_dashboard_frontend_view():
    res = client.get("/")
    assert res.status_code == 200
    assert "FORENSIGHTX" in res.text
    assert "Decision Support Only" in res.text
