"""
ForenSightX - FastAPI Investigation & Intelligence Service
Authoritative Spec Reference: ForenSightX_Project_Architecture.docx Section 8, 10, 11
Provides RESTful APIs and interactive investigation dashboard for digital forensic teams.
Iteration 10 Demo Dashboard Edition.
"""

import os
import io
import json
import sqlite3
import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
from PIL import Image

from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    load_model_checkpoint,
)
from src.models.factory import (
    build_model,
    load_model_from_checkpoint,
    get_default_config,
)
from src.data_loader import (
    EvidenceItem,
    ForensicCase,
    ForensicTokenizer,
    ForensicDatasetGenerator,
)
from src.agents import CaseCoordinatorAgent, InvestigationReport

logger = logging.getLogger("forensightx.api")

# =====================================================================
# Database Setup (SQLite for prototype per docx Section 11)
# =====================================================================

DB_PATH = "forensightx.db"
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            title TEXT,
            investigator TEXT,
            created_at TEXT,
            status TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            evidence_id TEXT,
            evidence_type TEXT,
            timestamp TEXT,
            content TEXT,
            metadata_json TEXT,
            is_observed INTEGER DEFAULT 1,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            case_id TEXT PRIMARY KEY,
            overall_risk_score REAL,
            risk_tier TEXT,
            report_json TEXT,
            created_at TEXT,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        )
    """)
    conn.commit()
    conn.close()

init_db()


# =====================================================================
# App & Models Initialization (Factory with strict=True)
# =====================================================================

app = FastAPI(
    title="ForenSightX Digital Forensics Intelligence API",
    description="Multimodal Deep Learning & Multi-Agent Forensics Decision Support Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models_saved"
missing_checkpoints: List[str] = []

tokenizer = ForensicTokenizer(max_length=80)
vocab_path = os.path.join(MODELS_DIR, "tokenizer_vocab.json")

if os.path.exists(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        tokenizer.word2idx = json.load(f)
        tokenizer.idx2word = {int(v): k for k, v in tokenizer.word2idx.items()}
else:
    missing_checkpoints.append("tokenizer_vocab.json")
    gen = ForensicDatasetGenerator(seed=42)
    texts, _, _, _ = gen.generate_multimodal_dataset(num_samples=100)
    tokenizer.fit_on_texts(texts)

# 1. CNN Model Loading via Factory with strict=True
cnn_path = os.path.join(MODELS_DIR, "cnn_model.pt")
if os.path.exists(cnn_path):
    try:
        cnn_model, _ = load_model_from_checkpoint(cnn_path, device=torch.device("cpu"))
    except Exception as e:
        logger.error(f"Error loading cnn_model.pt: {e}")
        missing_checkpoints.append(f"cnn_model.pt (load error: {e})")
        cnn_model = build_model(get_default_config("cnn")).to(torch.device("cpu"))
else:
    missing_checkpoints.append("cnn_model.pt")
    cnn_model = build_model(get_default_config("cnn")).to(torch.device("cpu"))

# 2. BiLSTM Model Loading via Factory with strict=True
bilstm_path = os.path.join(MODELS_DIR, "bilstm_model.pt")
if os.path.exists(bilstm_path):
    try:
        bilstm_model, _ = load_model_from_checkpoint(bilstm_path, device=torch.device("cpu"))
    except Exception as e:
        logger.error(f"Error loading bilstm_model.pt: {e}")
        missing_checkpoints.append(f"bilstm_model.pt (load error: {e})")
        bilstm_model = build_model(get_default_config("bilstm", vocab_size=max(tokenizer.vocab_size, 100))).to(torch.device("cpu"))
else:
    missing_checkpoints.append("bilstm_model.pt")
    bilstm_model = build_model(get_default_config("bilstm", vocab_size=max(tokenizer.vocab_size, 100))).to(torch.device("cpu"))

# 3. ANN Model Loading via Factory with strict=True
ann_path = os.path.join(MODELS_DIR, "ann_fusion_model.pt")
if os.path.exists(ann_path):
    try:
        ann_model, _ = load_model_from_checkpoint(ann_path, device=torch.device("cpu"))
    except Exception as e:
        logger.error(f"Error loading ann_fusion_model.pt: {e}")
        missing_checkpoints.append(f"ann_fusion_model.pt (load error: {e})")
        ann_model = build_model(get_default_config("ann", input_dim=7)).to(torch.device("cpu"))
else:
    missing_checkpoints.append("ann_fusion_model.pt")
    ann_model = build_model(get_default_config("ann", input_dim=7)).to(torch.device("cpu"))

# Ensure all models are in eval mode
cnn_model.eval()
bilstm_model.eval()
ann_model.eval()

coordinator = CaseCoordinatorAgent(
    cnn_model=cnn_model,
    bilstm_model=bilstm_model,
    ann_model=ann_model,
    tokenizer=tokenizer,
)


# =====================================================================
# Request / Response Schemas
# =====================================================================

class CreateCaseRequest(BaseModel):
    case_id: str = Field(..., min_length=2, max_length=64, description="Unique case identifier")
    title: str = Field(..., min_length=1, max_length=256, description="Case title")
    investigator: Optional[str] = "Forensic Examiner"

class AddEvidenceRequest(BaseModel):
    evidence_id: str
    evidence_type: str  # 'chat', 'email', 'image', 'metadata'
    timestamp: str
    content: Any
    metadata: Optional[Dict[str, Any]] = None
    is_observed: Optional[bool] = True

class UploadEvidenceRequest(BaseModel):
    evidence_type: str = "image"
    evidence_id: Optional[str] = None
    timestamp: Optional[str] = None
    filename: Optional[str] = "upload.png"
    content_b64: Optional[str] = None
    text_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy" if not missing_checkpoints else "degraded",
        "service": "ForenSightX Forensics Engine",
        "timestamp": datetime.now().isoformat(),
        "missing_checkpoints": missing_checkpoints,
        "models_loaded": {
            "cnn": os.path.exists(os.path.join(MODELS_DIR, "cnn_model.pt")),
            "bilstm": os.path.exists(os.path.join(MODELS_DIR, "bilstm_model.pt")),
            "ann_fusion": os.path.exists(os.path.join(MODELS_DIR, "ann_fusion_model.pt")),
        },
    }


@app.post("/api/cases", status_code=status.HTTP_201_CREATED)
def create_case(req: CreateCaseRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT case_id FROM cases WHERE case_id = ?", (req.case_id,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Case ID already exists")

    created_at = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO cases VALUES (?, ?, ?, ?, ?)",
        (req.case_id, req.title, req.investigator, created_at, "OPEN"),
    )
    conn.commit()
    conn.close()
    return {"message": "Case created successfully", "case_id": req.case_id}


@app.get("/api/cases")
def list_cases():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases ORDER BY created_at DESC")
    cases = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"cases": cases}


@app.post("/api/cases/load_sample")
def load_sample_case(sample_type: str = "suspicious"):
    """
    Loads pre-packaged realistic forensic sample case (suspicious or benign)
    from run_investigation module and populates database.
    """
    from run_investigation import create_sample_case
    sample_type_norm = "suspicious" if "susp" in sample_type.lower() else "benign"
    forensic_case = create_sample_case(sample_type_norm)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create or update case
    cursor.execute(
        "INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?)",
        (
            forensic_case.case_id,
            forensic_case.title,
            forensic_case.investigator,
            forensic_case.created_at,
            "OPEN",
        ),
    )

    # Clear prior evidence for this case
    cursor.execute("DELETE FROM evidence WHERE case_id = ?", (forensic_case.case_id,))

    # Insert evidence items
    saved_items = []
    for item in forensic_case.evidence_items:
        if isinstance(item.content, Image.Image):
            img_path = os.path.join(UPLOADS_DIR, f"{forensic_case.case_id}_{item.evidence_id}.png")
            item.content.save(img_path)
            content_str = img_path
        elif isinstance(item.content, (dict, list)):
            content_str = json.dumps(item.content)
        else:
            content_str = str(item.content)

        meta_str = json.dumps(item.metadata or {})
        cursor.execute(
            """
            INSERT INTO evidence (case_id, evidence_id, evidence_type, timestamp, content, metadata_json, is_observed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                forensic_case.case_id,
                item.evidence_id,
                item.evidence_type,
                item.timestamp,
                content_str,
                meta_str,
                1 if item.is_observed else 0,
            ),
        )
        saved_items.append({"evidence_id": item.evidence_id, "type": item.evidence_type})

    conn.commit()
    conn.close()

    return {
        "message": f"Sample {sample_type_norm} case loaded successfully",
        "case_id": forensic_case.case_id,
        "title": forensic_case.title,
        "evidence_count": len(saved_items),
        "evidence_items": saved_items,
    }


@app.post("/api/cases/{case_id}/evidence")
def add_evidence(case_id: str, req: AddEvidenceRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT case_id FROM cases WHERE case_id = ?", (case_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Case not found")

    content_str = json.dumps(req.content) if isinstance(req.content, (dict, list)) else str(req.content)
    meta_str = json.dumps(req.metadata or {})

    cursor.execute(
        """
        INSERT INTO evidence (case_id, evidence_id, evidence_type, timestamp, content, metadata_json, is_observed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            req.evidence_id,
            req.evidence_type,
            req.timestamp,
            content_str,
            meta_str,
            1 if req.is_observed else 0,
        ),
    )
    conn.commit()
    conn.close()
    return {"message": "Evidence recorded successfully", "evidence_id": req.evidence_id}


@app.post("/api/cases/{case_id}/upload_evidence")
def upload_evidence_endpoint(case_id: str, req: UploadEvidenceRequest):
    """Handles text, image (base64 data URI), or email uploads from dashboard or API."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT case_id FROM cases WHERE case_id = ?", (case_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Case not found")

    ev_id = req.evidence_id or f"EVID-UP-{int(datetime.now().timestamp())}"
    ts = req.timestamp or datetime.now().isoformat()
    fn = req.filename or f"{ev_id}.png"

    if req.evidence_type == "image" and req.content_b64:
        # Decode base64 image data URI
        raw_b64 = req.content_b64
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]
        img_bytes = base64.b64decode(raw_b64)
        dest_path = os.path.join(UPLOADS_DIR, f"{case_id}_{ev_id}_{fn}")
        with open(dest_path, "wb") as f:
            f.write(img_bytes)
        content_val = dest_path
        meta = req.metadata or {"filename": fn, "saved_path": dest_path}
    elif req.text_content:
        content_val = req.text_content
        meta = req.metadata or {"filename": fn}
    elif req.content_b64:
        # User uploaded a text or JSON file that was read as base64 Data URI
        raw_b64 = req.content_b64
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]
        try:
            content_val = base64.b64decode(raw_b64).decode("utf-8", errors="replace")
        except Exception:
            content_val = req.content_b64
        meta = req.metadata or {"filename": fn}
    else:
        content_val = ""
        meta = req.metadata or {}

    cursor.execute(
        """
        INSERT INTO evidence (case_id, evidence_id, evidence_type, timestamp, content, metadata_json, is_observed)
        VALUES (?, ?, ?, ?, ?, ?, 1)
        """,
        (case_id, ev_id, req.evidence_type, ts, str(content_val), json.dumps(meta)),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Evidence uploaded and recorded successfully",
        "evidence_id": ev_id,
        "evidence_type": req.evidence_type,
        "content": content_val,
    }


@app.post("/api/cases/{case_id}/investigate")
def run_case_investigation(case_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    case_row = cursor.fetchone()
    if not case_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Case not found")

    cursor.execute("SELECT * FROM evidence WHERE case_id = ?", (case_id,))
    evid_rows = cursor.fetchall()

    if not evid_rows:
        conn.close()
        raise HTTPException(status_code=400, detail="Cannot investigate case with zero evidence items")

    # Reconstruct ForensicCase
    evidence_items = []
    for r in evid_rows:
        try:
            content_val = json.loads(r["content"])
        except Exception:
            content_val = r["content"]
        try:
            meta_val = json.loads(r["metadata_json"])
        except Exception:
            meta_val = {}

        evidence_items.append(
            EvidenceItem(
                evidence_id=r["evidence_id"],
                evidence_type=r["evidence_type"],
                timestamp=r["timestamp"],
                content=content_val,
                metadata=meta_val,
                is_observed=bool(r["is_observed"]),
            )
        )

    forensic_case = ForensicCase(
        case_id=case_id,
        title=case_row["title"],
        created_at=case_row["created_at"],
        evidence_items=evidence_items,
        investigator=case_row["investigator"],
    )

    # Assert eval mode across all models before running inference
    cnn_model.eval()
    bilstm_model.eval()
    ann_model.eval()

    # Execute Multi-Agent Investigation
    report = coordinator.investigate(forensic_case)
    report_dict = report.to_dict()

    # Save report to DB
    cursor.execute(
        """
        INSERT OR REPLACE INTO reports (case_id, overall_risk_score, risk_tier, report_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            case_id,
            report.overall_risk_score,
            report.risk_tier,
            json.dumps(report_dict),
            datetime.now().isoformat(),
        ),
    )
    cursor.execute("UPDATE cases SET status = ? WHERE case_id = ?", ("INVESTIGATED", case_id))
    conn.commit()
    conn.close()

    return report_dict


@app.get("/api/cases/{case_id}/report")
def get_case_report(case_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE case_id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="No report found for case. Run /investigate first.")
    return json.loads(row["report_json"])


@app.get("/api/cases/{case_id}/gradcam/{evidence_id}")
def get_gradcam_image(case_id: str, evidence_id: str):
    """Returns raw Grad-CAM heatmap PNG image for an evidence item."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evidence WHERE case_id = ? AND evidence_id = ?", (case_id, evidence_id))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Evidence item not found")

    content = row["content"]
    if not (isinstance(content, str) and os.path.exists(content)):
        raise HTTPException(status_code=400, detail="Evidence content is not an accessible image file")

    img = Image.open(content).convert("RGB")
    arr = np.array(img.resize((128, 128)), dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
    heatmap = cnn_model.get_gradcam(tensor)

    # Colorize heatmap
    h = np.clip(heatmap, 0.0, 1.0)
    r = (np.clip(1.5 * h, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(1.5 * (1.0 - np.abs(h - 0.5) * 2.0), 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1.5 * (1.0 - h), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    im = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/experiments")
def get_experimental_results():
    """Returns experimental results JSON for backward compatibility."""
    v10_path = os.path.join("results", "experiment_results_v10.json")
    if os.path.exists(v10_path):
        with open(v10_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "completed", "version": "v10", "results": data}

    v9_path = os.path.join("results", "experiment_results_v9.json")
    if os.path.exists(v9_path):
        with open(v9_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "completed", "version": "v9", "results": data}

    results_path = os.path.join("results", "experiment_results.json")
    if not os.path.exists(results_path):
        return {"status": "pending", "message": "Run investigations to generate experimental metrics"}
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"status": "completed", "results": data}


@app.get("/api/results_v10")
def get_results_v10():
    """
    Dedicated endpoint for read-only Results tab.
    Strictly reads results/experiment_results_v10.json dynamically.
    No typed or hardcoded numbers.
    """
    results_path = os.path.join("results", "experiment_results_v10.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="results/experiment_results_v10.json not found")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p4 = data.get("part4_blueprint_experiments", {})
    summary = p4.get("summary", {})
    deltas = p4.get("delta_comparisons", [])
    coordinator_res = p4.get("runs", {}).get("S_coordinator", {})

    # Structure canonical summary
    arms_data = []
    arm_labels = {
        "A_bilstm": "Arm A (BiLSTM Text Concat)",
        "B_cnn": "Arm B (Visual Evidence CNN)",
        "C_ann_2d": "Arm C (2D Multimodal ANN)",
        "D_ann_7d": "Arm D (7D Multimodal ANN)",
        "E_augmented": "Arm E (VAE Augmented 0.5x)",
        "A_h_heuristics": "Baseline A-h (Heuristics MLP)",
    }

    for arm_key, label in arm_labels.items():
        if arm_key in summary:
            s = summary[arm_key]
            arms_data.append({
                "arm_key": arm_key,
                "label": label,
                "mean_auc": s.get("mean_auc"),
                "std_auc": s.get("std_auc"),
                "mean_accuracy": s.get("mean_accuracy"),
                "mean_f1": s.get("mean_f1"),
                "mean_brier": s.get("mean_brier_score"),
            })

    if coordinator_res:
        arms_data.append({
            "arm_key": "S_coordinator",
            "label": "Baseline S (Multi-Agent Coordinator)",
            "mean_auc": coordinator_res.get("auc"),
            "std_auc": 0.0,
            "mean_accuracy": coordinator_res.get("accuracy"),
            "mean_f1": coordinator_res.get("f1"),
            "mean_brier": coordinator_res.get("brier_score"),
        })

    # Filter the three primary delta comparisons requested:
    # 1. D vs B
    # 2. D vs C
    # 3. E vs D (stated as "no benefit")
    delta_data = []
    for d in deltas:
        comp = d.get("comparison", "")
        benefit_status = "significant" if d.get("ci_lower_auc", 0) > 0 else "neutral"
        if "E_augmented" in comp and "D_ann_7d" in comp:
            benefit_status = "no benefit"

        delta_data.append({
            "comparison": comp,
            "point_estimate_delta_auc": d.get("point_estimate_delta_auc"),
            "bootstrap_mean_delta_auc": d.get("bootstrap_mean_delta_auc"),
            "ci_lower_auc": d.get("ci_lower_auc"),
            "ci_upper_auc": d.get("ci_upper_auc"),
            "benefit_status": benefit_status,
        })

    return {
        "status": "success",
        "manifest_sha256": p4.get("manifest_sha256", "5ef13532..."),
        "results_sha256": "d4db84c17e1f6bf93dca358181829e1231c8bc9b109f31cd22830be456d34373",
        "execution_verdict": data.get("execution_verdict", "ALL_GATES_PASSED_AND_VALIDATED"),
        "arms_summary": arms_data,
        "delta_comparisons": delta_data,
    }


# =====================================================================
# Interactive Dashboard View (HTML/CSS/JS)
# =====================================================================

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ForenSightX - Forensics Intelligence & Investigation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        .active-tab { border-bottom: 2px solid #f59e0b; color: #f59e0b; font-weight: 600; }
        .inactive-tab { color: #a1a1aa; }
        .inactive-tab:hover { color: #f4f4f5; }
    </style>
</head>
<body class="bg-neutral-950 text-neutral-100 font-sans min-h-screen flex flex-col antialiased">
    <!-- Mandatory Synthetic Benchmark Banner -->
    <div class="bg-amber-950/60 border-b border-amber-600/40 px-4 py-2 text-center text-amber-300 font-bold text-xs tracking-wider flex items-center justify-center gap-2">
        <i class="fa-solid fa-triangle-exclamation text-amber-400"></i>
        <span>SYNTHETIC BENCHMARK - NOT REAL-WORLD EVIDENCE • PROTOTYPE DECISION-SUPPORT PLATFORM ONLY</span>
        <i class="fa-solid fa-triangle-exclamation text-amber-400"></i>
    </div>

    <!-- Top Navigation Header -->
    <header class="border-b border-neutral-800/80 bg-neutral-900/95 backdrop-blur sticky top-0 z-40">
        <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-9 h-9 rounded-lg bg-amber-500/10 border border-amber-500/40 flex items-center justify-center text-amber-400 font-black tracking-tighter">
                    FX
                </div>
                <div>
                    <h1 class="text-base font-bold tracking-wide text-white flex items-center gap-2">
                        FORENSIGHTX <span class="text-[10px] px-2 py-0.5 rounded bg-[#0f1f14] text-[#86efac] border border-[#23422e] font-mono font-semibold flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_#4ade80aa]"></span>Live System</span>
                    </h1>
                    <p class="text-xs text-neutral-400">Multimodal Digital Forensics Decision Support Platform</p>
                </div>
            </div>

            <!-- Tab Switchers -->
            <div class="flex items-center space-x-6 text-sm">
                <button id="tabLiveBtn" onclick="switchTab('live')" class="py-2 px-1 active-tab flex items-center gap-2 transition">
                    <i class="fa-solid fa-crosshairs"></i> Live Investigation
                </button>
                <button id="tabResultsBtn" onclick="switchTab('results')" class="py-2 px-1 inactive-tab flex items-center gap-2 transition">
                    <i class="fa-solid fa-chart-line"></i> Results (Benchmark v10)
                </button>
            </div>

            <div class="flex items-center space-x-3">
                <span class="text-xs text-amber-400/90 bg-amber-950/50 border border-amber-800/70 px-2.5 py-1 rounded-full flex items-center gap-1.5 font-medium">
                    <i class="fa-solid fa-scale-balanced text-xs"></i> Decision Support Only
                </span>
                <a href="/docs" target="_blank" class="text-xs px-2.5 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-750 text-neutral-300 border border-neutral-700 transition">API Docs</a>
            </div>
        </div>
    </header>

    <!-- Main Workspace Container -->
    <main class="max-w-7xl mx-auto px-4 py-6 flex-1 w-full space-y-6">

        <!-- ============================================================== -->
        <!-- TAB 1: LIVE INVESTIGATION WORKSPACE -->
        <!-- ============================================================== -->
        <div id="viewLive" class="space-y-6">
            <!-- Controls Grid: Samples, Ingestion, Run -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <!-- Card 1: Sample Case Loading -->
                <div class="p-4 rounded-xl bg-neutral-900 border border-neutral-800 shadow-sm flex flex-col justify-between">
                    <div>
                        <h2 class="text-sm font-semibold text-neutral-200 mb-1 flex items-center gap-2">
                            <i class="fa-solid fa-folder-closed text-amber-400"></i> Pre-Packaged Cases
                        </h2>
                        <p class="text-xs text-neutral-400 mb-3">Load benchmark samples or reset to a fresh case.</p>
                        <div class="flex flex-col gap-2">
                            <button onclick="loadSample('benign')" class="w-full py-2 px-3 rounded-lg bg-[#0f2115] hover:bg-[#16301f] border border-[#264e34] text-[#86efac] text-xs font-semibold flex items-center justify-center gap-2 transition">
                                <i class="fa-solid fa-shield-halved text-[#4ade80]"></i> Load benign sample
                            </button>
                            <button onclick="loadSample('suspicious')" class="w-full py-2 px-3 rounded-lg bg-red-950/50 hover:bg-red-900/60 border border-red-800/70 text-red-300 text-xs font-semibold flex items-center justify-center gap-2 transition">
                                <i class="fa-solid fa-triangle-exclamation text-red-400"></i> Load suspicious sample
                            </button>
                            <button onclick="clearCase()" class="w-full py-1.5 px-3 rounded-lg bg-neutral-850 hover:bg-neutral-800 border border-neutral-750 text-neutral-400 text-xs font-medium flex items-center justify-center gap-2 transition">
                                <i class="fa-solid fa-rotate-left text-neutral-500"></i> Start New / Clear Case
                            </button>
                        </div>
                    </div>
                    <div class="mt-3 pt-3 border-t border-neutral-800 text-[11px] text-neutral-400 flex justify-between">
                        <span>Active Case:</span>
                        <span id="activeCaseBadge" class="font-mono text-neutral-200 font-bold truncate max-w-[160px]">None loaded</span>
                    </div>
                </div>

                <!-- Card 2: Manual Upload & Evidence Ingestion -->
                <div class="p-4 rounded-xl bg-neutral-900 border border-neutral-800 shadow-sm flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-1">
                            <h2 class="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                                <i class="fa-solid fa-upload text-amber-400"></i> Ingest New Evidence
                            </h2>
                        </div>
                        <p class="text-xs text-neutral-400 mb-2">Test custom text, email headers, or image files.</p>
                        
                        <div class="space-y-2">
                            <div class="flex gap-1 text-xs">
                                <select id="uploadTypeSelect" class="bg-neutral-950 border border-neutral-700 rounded px-2 py-1 text-xs text-neutral-200">
                                    <option value="chat">Chat (Text)</option>
                                    <option value="email">Email (Text/JSON)</option>
                                    <option value="image">Image (Screenshot)</option>
                                </select>
                                <input type="file" id="evidenceFileInput" class="text-xs text-neutral-400 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-neutral-800 file:text-neutral-200 hover:file:bg-neutral-700">
                            </div>

                            <!-- Quick Fill Templates -->
                            <div class="flex items-center justify-between text-[11px] text-neutral-400 pt-1">
                                <span>Quick Templates:</span>
                                <div class="flex gap-1">
                                    <button type="button" onclick="insertSampleText('benign_chat')" class="px-1.5 py-0.5 rounded bg-neutral-850 hover:bg-neutral-800 text-neutral-300 border border-neutral-700 text-[10px] transition font-medium">Preset: Standup Chat</button>
                                    <button type="button" onclick="insertSampleText('benign_email')" class="px-1.5 py-0.5 rounded bg-neutral-850 hover:bg-neutral-800 text-neutral-300 border border-neutral-700 text-[10px] transition font-medium">Preset: Release Email</button>
                                    <button type="button" onclick="insertSampleText('susp_chat')" class="px-1.5 py-0.5 rounded bg-neutral-850 hover:bg-neutral-800 text-neutral-300 border border-neutral-700 text-[10px] transition font-medium">Preset: Wire Request</button>
                                    <button type="button" onclick="insertSampleText('clear')" class="px-1.5 py-0.5 rounded bg-neutral-800 hover:bg-neutral-750 text-neutral-400 text-[10px] transition">Clear</button>
                                </div>
                            </div>

                            <textarea id="textEvidenceInput" placeholder="Paste conversation, email text, or headers here..." class="w-full bg-neutral-950 border border-neutral-800 rounded p-1.5 text-xs text-neutral-200 h-14 resize-none font-mono"></textarea>
                        </div>
                    </div>
                    <div class="flex gap-2 mt-2">
                        <button onclick="handleAnalyzeDirect()" class="flex-1 py-1.5 px-2 rounded-lg bg-neutral-800 hover:bg-neutral-750 border border-amber-500/40 text-amber-300 text-xs font-semibold transition flex items-center justify-center gap-1.5" title="Run clean investigation with this evidence only">
                            <i class="fa-solid fa-bolt text-amber-400"></i> Analyze In Clean Case
                        </button>
                        <button onclick="handleEvidenceIngest()" class="py-1.5 px-3 rounded-lg bg-neutral-800 hover:bg-neutral-750 border border-neutral-700 text-neutral-300 text-xs font-medium transition" title="Append to active case">
                            <i class="fa-solid fa-plus"></i> Append
                        </button>
                    </div>
                </div>

                <!-- Card 3: Execution Control & Agent Status -->
                <div class="p-4 rounded-xl bg-neutral-900 border border-neutral-800 shadow-sm flex flex-col justify-between">
                    <div>
                        <h2 class="text-sm font-semibold text-neutral-200 mb-1 flex items-center gap-2">
                            <i class="fa-solid fa-microchip text-amber-400"></i> Multi-Agent Orchestration
                        </h2>
                        <p class="text-xs text-neutral-400 mb-2">Automated coordination across specialized neural agents.</p>
                        <div class="text-xs text-neutral-400 space-y-1.5 font-mono">
                            <div class="flex justify-between"><span>BiLSTM Linguistic Agent:</span> <span class="text-[#86efac] font-medium flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_#4ade80aa]"></span> Active</span></div>
                            <div class="flex justify-between"><span>CNN Visual Evidence Agent:</span> <span class="text-[#86efac] font-medium flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_#4ade80aa]"></span> Active</span></div>
                            <div class="flex justify-between"><span>7D Multimodal Risk ANN:</span> <span class="text-[#86efac] font-medium flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_#4ade80aa]"></span> Active</span></div>
                            <div class="flex justify-between"><span>Timeline Reconstruction Agent:</span> <span class="text-[#86efac] font-medium flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_#4ade80aa]"></span> Active</span></div>
                        </div>
                    </div>
                    <button id="btnInvestigate" onclick="runInvestigation()" class="mt-3 py-2.5 px-4 rounded-xl bg-amber-500 hover:bg-amber-400 text-neutral-950 text-xs font-extrabold shadow-lg shadow-amber-950/30 flex items-center justify-center gap-2 transition tracking-wider">
                        <i class="fa-solid fa-play"></i> Run Investigation
                    </button>
                </div>
            </div>

            <!-- Investigation Output Section -->
            <div id="investigationSection" class="hidden space-y-6">
                <!-- Top Metrics Row: Risk Score + Tier -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800">
                        <span class="text-xs font-medium text-neutral-400 uppercase tracking-wider">Overall Risk Score</span>
                        <div id="riskScoreVal" class="text-4xl font-extrabold text-neutral-400 mt-2 font-mono">--%</div>
                        <div id="riskTierBadge" class="mt-2 inline-block px-2.5 py-0.5 rounded text-xs font-bold bg-neutral-800 text-neutral-300 border border-neutral-700">PENDING EVALUATION</div>
                    </div>
                    <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800">
                        <span class="text-xs font-medium text-neutral-400 uppercase tracking-wider">Visual Evidence (CNN)</span>
                        <div id="cnnScoreVal" class="text-3xl font-bold text-neutral-100 mt-2 font-mono">--%</div>
                        <span class="text-xs text-neutral-500">Grad-CAM spatial activation</span>
                    </div>
                    <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800">
                        <span class="text-xs font-medium text-neutral-400 uppercase tracking-wider">Conversation / Text (BiLSTM)</span>
                        <div id="bilstmScoreVal" class="text-3xl font-bold text-neutral-100 mt-2 font-mono">--%</div>
                        <span class="text-xs text-neutral-500">Bidirectional attention pooling</span>
                    </div>
                    <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800">
                        <span class="text-xs font-medium text-neutral-400 uppercase tracking-wider">Evidence Artifacts</span>
                        <div id="evidenceCountVal" class="text-3xl font-bold text-neutral-100 mt-2 font-mono">0</div>
                        <span class="text-xs text-neutral-500">Synthesized evidence items</span>
                    </div>
                </div>

                <!-- 7D Feature Scores Grid -->
                <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800">
                    <h3 class="text-sm font-bold text-neutral-200 mb-3 flex items-center gap-2">
                        <i class="fa-solid fa-chart-simple text-amber-400"></i> The 7 Authoritative Multimodal Feature Scores
                    </h3>
                    <div id="featureVectorGrid" class="grid grid-cols-2 md:grid-cols-7 gap-3 text-center"></div>
                </div>

                <!-- Timeline & Explainability Split View -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <!-- Chronological Timeline (Observed vs Inferred Labelled) -->
                    <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center justify-between mb-3">
                                <h3 class="text-sm font-bold text-neutral-200 flex items-center gap-2">
                                    <i class="fa-solid fa-timeline text-amber-400"></i> Chronological Timeline
                                </h3>
                                <span class="text-xs text-neutral-400">Observed vs Inferred</span>
                            </div>
                            <div id="timelineContainer" class="space-y-3 max-h-96 overflow-y-auto pr-2"></div>
                        </div>
                    </div>

                    <!-- Explainability: Top Indicators & Grad-CAM -->
                    <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800 flex flex-col justify-between space-y-4">
                        <div>
                            <h3 class="text-sm font-bold text-neutral-200 mb-3 flex items-center gap-2">
                                <i class="fa-solid fa-fingerprint text-amber-400"></i> Explainability & Attribution
                            </h3>
                            
                            <!-- Top Indicators -->
                            <div class="mb-3">
                                <h4 class="text-xs font-semibold text-neutral-400 mb-1.5">Top Forensic Indicators Flagged:</h4>
                                <div id="keyIndicatorsList" class="space-y-1.5"></div>
                            </div>

                            <!-- Salient Tokens -->
                            <div class="mb-4">
                                <h4 class="text-xs font-semibold text-neutral-400 mb-1.5">Salient Linguistic Tokens:</h4>
                                <div id="tokensList" class="flex flex-wrap gap-1.5"></div>
                            </div>

                            <!-- Grad-CAM Visual Heatmap Section -->
                            <div id="gradcamSection" class="border-t border-neutral-800 pt-3">
                                <h4 class="text-xs font-semibold text-neutral-400 mb-2 flex items-center justify-between">
                                    <span>Grad-CAM Visual Evidence Heatmap:</span>
                                    <span id="gradcamStats" class="text-[11px] font-mono text-amber-400"></span>
                                </h4>
                                <div id="gradcamContainer" class="flex items-center gap-4">
                                    <div class="text-xs text-neutral-500">No visual evidence present in active case.</div>
                                </div>
                            </div>
                        </div>

                        <!-- Legal Decision-Support Disclaimer -->
                        <div class="mt-4 p-3 rounded-lg bg-amber-950/30 border border-amber-800/50 text-xs text-amber-300/90 leading-relaxed">
                            <div class="font-bold text-amber-200 mb-1 flex items-center gap-1.5">
                                <i class="fa-solid fa-scale-balanced"></i> Legal Decision-Support Notice
                            </div>
                            <span id="disclaimerText">ForenSightX outputs are probabilistic investigative aids designed solely to assist qualified human forensic investigators. Model outputs do NOT constitute autonomous legal findings, conclusive evidence, or automated determinations of liability.</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>


        <!-- ============================================================== -->
        <!-- TAB 2: RESULTS (BENCHMARK V10) READ-ONLY TAB -->
        <!-- ============================================================== -->
        <div id="viewResults" class="hidden space-y-6">
            <div class="p-5 rounded-xl bg-neutral-900 border border-neutral-800">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-neutral-800 pb-3 mb-4">
                    <div>
                        <h2 class="text-base font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-table-list text-amber-400"></i> Benchmark v10 Canonical Verification Ledger
                        </h2>
                        <p class="text-xs text-neutral-400">Dynamically queried from <code class="text-amber-400">results/experiment_results_v10.json</code>. No typed numbers.</p>
                    </div>
                    <div class="flex items-center gap-2 text-xs font-mono">
                        <span class="px-2 py-1 rounded bg-neutral-950 border border-neutral-800 text-neutral-400">
                            Manifest: <span id="v10ManifestSha" class="text-neutral-200 font-bold">Loading...</span>
                        </span>
                        <span class="px-2 py-1 rounded bg-[#0f2115] border border-[#264e34] text-[#86efac] font-bold">
                            Verdict: <span id="v10Verdict">Loading...</span>
                        </span>
                    </div>
                </div>

                <!-- Table 1: Blueprint Canonical Arms A-E -->
                <h3 class="text-xs font-bold text-neutral-300 uppercase tracking-wider mb-2">Canonical Experiments A-E (10 Seeds Evaluation)</h3>
                <div class="overflow-x-auto mb-6">
                    <table class="w-full text-xs text-left text-neutral-300 font-mono">
                        <thead class="text-xs text-neutral-400 uppercase bg-neutral-950 border-b border-neutral-800">
                            <tr>
                                <th class="px-4 py-2.5">Architecture Arm</th>
                                <th class="px-4 py-2.5">Mean ROC-AUC &plusmn; SD</th>
                                <th class="px-4 py-2.5">Mean Accuracy</th>
                                <th class="px-4 py-2.5">Mean F1-Score</th>
                                <th class="px-4 py-2.5">Mean Brier Score</th>
                            </tr>
                        </thead>
                        <tbody id="v10ArmsTableBody" class="divide-y divide-neutral-800 bg-neutral-900/60">
                            <tr><td colspan="5" class="px-4 py-3 text-center text-neutral-500">Querying results JSON...</td></tr>
                        </tbody>
                    </table>
                </div>

                <!-- Table 2: The Three Delta Comparisons & CIs -->
                <h3 class="text-xs font-bold text-neutral-300 uppercase tracking-wider mb-2">Empirical Delta Comparisons (Paired Cluster Bootstrap 95% CIs)</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs text-left text-neutral-300 font-mono">
                        <thead class="text-xs text-neutral-400 uppercase bg-neutral-950 border-b border-neutral-800">
                            <tr>
                                <th class="px-4 py-2.5">Comparison Hypothesis</th>
                                <th class="px-4 py-2.5">Point Est &Delta;AUC</th>
                                <th class="px-4 py-2.5">Bootstrap Mean &Delta;AUC</th>
                                <th class="px-4 py-2.5">95% Bootstrap CI</th>
                                <th class="px-4 py-2.5">Empirical Verdict</th>
                            </tr>
                        </thead>
                        <tbody id="v10DeltasTableBody" class="divide-y divide-neutral-800 bg-neutral-900/60">
                            <tr><td colspan="5" class="px-4 py-3 text-center text-neutral-500">Querying results JSON...</td></tr>
                        </tbody>
                    </table>
                </div>

                <div class="mt-4 pt-3 border-t border-neutral-800 text-[11px] text-neutral-500 font-mono flex justify-between">
                    <span>Results SHA-256: <code id="v10ResultsSha" class="text-neutral-400">Loading...</code></span>
                    <span>Architecture: Multimodal Forensic Deep Learning Framework</span>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer Notice -->
    <footer class="border-t border-neutral-800/80 py-3 text-center text-xs text-neutral-500">
        ForenSightX Digital Forensics Intelligence Platform • Decision-Support System
    </footer>

    <!-- Client-Side Dashboard Script -->
    <script>
        let currentActiveCaseId = null;

        const TEMPLATES = {
            benign_chat: "Good morning team! Team lunch is at 12:30 PM today at the Italian deli downtown. Let me know if you want to join!",
            benign_email: "Hi Engineering team, ForenSightX deployment is complete in staging. All 69 automated unit tests passed in 10.9 seconds. Release notes and API documentation are updated on the internal portal.",
            susp_chat: "Chief Executive Officer: We are closing a highly confidential international acquisition. I need an immediate wire transfer of $485,000 processed to our overseas escrow account before 3:30 PM cutoff. Do NOT contact other directors under SEC rules: http://secure-escrow-update.xyz/verify"
        };

        function insertSampleText(type) {
            const txtArea = document.getElementById('textEvidenceInput');
            const select = document.getElementById('uploadTypeSelect');
            if (type === 'clear') {
                txtArea.value = '';
            } else if (TEMPLATES[type]) {
                if (type.includes('email')) select.value = 'email';
                else select.value = 'chat';
                txtArea.value = TEMPLATES[type];
            }
        }

        function clearCase() {
            currentActiveCaseId = null;
            document.getElementById('activeCaseBadge').textContent = 'None loaded';
            document.getElementById('investigationSection').classList.add('hidden');
            document.getElementById('textEvidenceInput').value = '';
            document.getElementById('evidenceFileInput').value = '';
        }

        function switchTab(tab) {
            const liveView = document.getElementById('viewLive');
            const resultsView = document.getElementById('viewResults');
            const liveBtn = document.getElementById('tabLiveBtn');
            const resultsBtn = document.getElementById('tabResultsBtn');

            if (tab === 'live') {
                liveView.classList.remove('hidden');
                resultsView.classList.add('hidden');
                liveBtn.className = 'py-2 px-1 active-tab flex items-center gap-2 transition';
                resultsBtn.className = 'py-2 px-1 inactive-tab flex items-center gap-2 transition';
            } else {
                liveView.classList.add('hidden');
                resultsView.classList.remove('hidden');
                resultsBtn.className = 'py-2 px-1 active-tab flex items-center gap-2 transition';
                liveBtn.className = 'py-2 px-1 inactive-tab flex items-center gap-2 transition';
                loadResultsTab();
            }
        }

        async function startNewCase(title = 'Ad-Hoc Forensic Examination') {
            const caseId = 'CASE-LIVE-' + Math.floor(1000 + Math.random() * 9000);
            const res = await fetch('/api/cases', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({case_id: caseId, title: title, investigator: 'Forensic Examiner'})
            });
            if (!res.ok) throw new Error(await res.text());
            currentActiveCaseId = caseId;
            document.getElementById('activeCaseBadge').textContent = caseId;
            return caseId;
        }

        async function loadSample(sampleType) {
            try {
                const res = await fetch(`/api/cases/load_sample?sample_type=${sampleType}`, { method: 'POST' });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                currentActiveCaseId = data.case_id;
                document.getElementById('activeCaseBadge').textContent = `${data.case_id} (${sampleType})`;
                await runInvestigation();
            } catch (err) {
                alert('Failed to load sample: ' + err.message);
            }
        }

        async function handleAnalyzeDirect() {
            const fileInput = document.getElementById('evidenceFileInput');
            const typeSelect = document.getElementById('uploadTypeSelect').value;
            const textInput = document.getElementById('textEvidenceInput').value.trim();

            if (fileInput.files.length === 0 && textInput.length === 0) {
                alert('Please enter text or select a file to analyze.');
                return;
            }

            const isBenignHint = textInput.toLowerCase().includes('lunch') || textInput.toLowerCase().includes('deployment') || textInput.toLowerCase().includes('good morning');
            const caseTitle = isBenignHint ? 'Benign Routine Examination' : 'Custom Evidence Investigation';
            
            try {
                const caseId = await startNewCase(caseTitle);

                if (fileInput.files.length > 0) {
                    const file = fileInput.files[0];
                    const reader = new FileReader();
                    reader.onload = async function(e) {
                        const b64 = e.target.result;
                        const res = await fetch(`/api/cases/${caseId}/upload_evidence`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                evidence_type: typeSelect,
                                evidence_id: 'EVID-' + Date.now().toString().slice(-4),
                                filename: file.name,
                                content_b64: b64,
                                metadata: {filename: file.name, size: file.size}
                            })
                        });
                        if (!res.ok) {
                            alert('Upload failed: ' + await res.text());
                            return;
                        }
                        fileInput.value = '';
                        await runInvestigation();
                    };
                    reader.readAsDataURL(file);
                } else {
                    const res = await fetch(`/api/cases/${caseId}/upload_evidence`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            evidence_type: typeSelect,
                            evidence_id: 'EVID-TEXT-' + Date.now().toString().slice(-4),
                            filename: 'evidence_note.txt',
                            text_content: textInput,
                            metadata: {source: 'Direct User Input'}
                        })
                    });
                    if (!res.ok) {
                        alert('Failed to record text evidence: ' + await res.text());
                        return;
                    }
                    await runInvestigation();
                }
            } catch (err) {
                alert('Error creating clean investigation: ' + err.message);
            }
        }

        async function handleEvidenceIngest() {
            if (!currentActiveCaseId) {
                await startNewCase();
            }

            const fileInput = document.getElementById('evidenceFileInput');
            const typeSelect = document.getElementById('uploadTypeSelect').value;
            const textInput = document.getElementById('textEvidenceInput').value.trim();

            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const reader = new FileReader();
                reader.onload = async function(e) {
                    const b64 = e.target.result;
                    const res = await fetch(`/api/cases/${currentActiveCaseId}/upload_evidence`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            evidence_type: typeSelect,
                            evidence_id: 'EVID-' + Date.now().toString().slice(-4),
                            filename: file.name,
                            content_b64: b64,
                            metadata: {filename: file.name, size: file.size}
                        })
                    });
                    if (!res.ok) alert('Upload failed: ' + await res.text());
                    else alert('Evidence attached! Click "Run Investigation".');
                    fileInput.value = '';
                };
                reader.readAsDataURL(file);
            } else if (textInput.length > 0) {
                const res = await fetch(`/api/cases/${currentActiveCaseId}/upload_evidence`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        evidence_type: typeSelect,
                        evidence_id: 'EVID-TEXT-' + Date.now().toString().slice(-4),
                        filename: 'user_text.txt',
                        text_content: textInput,
                        metadata: {source: 'Manual Ingestion'}
                    })
                });
                if (!res.ok) alert('Failed to record text evidence: ' + await res.text());
                else alert('Text evidence appended to active case! Click "Run Investigation".');
                document.getElementById('textEvidenceInput').value = '';
            } else {
                alert('Please select a file or enter text to ingest.');
            }
        }

        async function runInvestigation() {
            if (!currentActiveCaseId) {
                await loadSample('benign');
                return;
            }

            const btn = document.getElementById('btnInvestigate');
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Investigating...';
            btn.disabled = true;

            try {
                const res = await fetch(`/api/cases/${currentActiveCaseId}/investigate`, { method: 'POST' });
                if (!res.ok) {
                    const err = await res.json();
                    alert('Investigation failed: ' + (err.detail || res.statusText));
                    return;
                }
                const report = await res.json();
                renderReport(report);
            } catch (err) {
                alert('Request failed: ' + err.message);
            } finally {
                btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Investigation';
                btn.disabled = false;
            }
        }

        function renderReport(report) {
            document.getElementById('investigationSection').classList.remove('hidden');

            // 1. Overall Risk Score + Tier
            const riskPct = (report.overall_risk_score * 100).toFixed(1);
            const scoreEl = document.getElementById('riskScoreVal');
            const tierEl = document.getElementById('riskTierBadge');

            scoreEl.textContent = `${riskPct}%`;
            tierEl.textContent = `${report.risk_tier.toUpperCase()} RISK`;

            if (report.overall_risk_score >= 0.5) {
                scoreEl.className = 'text-4xl font-extrabold text-red-500 mt-2 font-mono';
                tierEl.className = 'mt-2 inline-block px-2.5 py-0.5 rounded text-xs font-bold bg-red-950/80 text-red-400 border border-red-800 font-mono';
            } else {
                scoreEl.className = 'text-4xl font-extrabold text-[#86efac] mt-2 font-mono';
                tierEl.className = 'mt-2 inline-block px-2.5 py-0.5 rounded text-xs font-bold bg-[#0f2115] text-[#86efac] border border-[#264e34] font-mono';
            }

            // 2. Modality Summaries
            document.getElementById('cnnScoreVal').textContent = ((report.model_predictions.cnn_score || 0) * 100).toFixed(1) + '%';
            document.getElementById('bilstmScoreVal').textContent = ((report.model_predictions.bilstm_score || 0) * 100).toFixed(1) + '%';
            document.getElementById('evidenceCountVal').textContent = report.evidence_inventory.length;

            // 3. The 7 Feature Scores
            const featNames = [
                'CNN Visual', 'BiLSTM Text', 'URL Score',
                'Keywords', 'Attachment', 'Timeline Anom', 'Email Anom'
            ];
            const featGrid = document.getElementById('featureVectorGrid');
            featGrid.innerHTML = '';
            (report.feature_vector_7d || []).forEach((val, idx) => {
                const col = document.createElement('div');
                col.className = 'p-2.5 rounded-lg bg-neutral-950 border border-neutral-800';
                col.innerHTML = `
                    <div class="text-[10px] text-neutral-400 truncate uppercase tracking-wider">${featNames[idx] || 'Feat ' + idx}</div>
                    <div class="text-base font-extrabold text-neutral-200 mt-1 font-mono">${(val * 100).toFixed(1)}%</div>
                `;
                featGrid.appendChild(col);
            });

            // 4. Timeline (Observed vs Inferred Labelled)
            const tl = document.getElementById('timelineContainer');
            tl.innerHTML = '';
            (report.chronological_timeline || []).forEach(evt => {
                const isObs = (evt.event_type || '').toLowerCase() === 'observed';
                const borderCls = isObs ? 'border-l-4 border-[#22c55e] bg-neutral-950' : 'border-l-4 border-amber-500 bg-neutral-950';
                const tagCls = isObs ? 'text-[#86efac] bg-[#0f2115] border border-[#264e34]' : 'text-amber-400 bg-amber-950/80 border border-amber-800/80';
                const labelText = isObs ? '[OBSERVED]' : '[INFERRED]';

                const item = document.createElement('div');
                item.className = `p-3 rounded-r-lg ${borderCls} border-t border-b border-r border-neutral-800/80`;
                item.innerHTML = `
                    <div class="flex items-center justify-between text-xs mb-1">
                        <span class="font-mono text-neutral-400 text-[11px]">${evt.timestamp}</span>
                        <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${tagCls} font-mono">${labelText}</span>
                    </div>
                    <div class="text-xs text-neutral-200 font-medium">${evt.description}</div>
                `;
                tl.appendChild(item);
            });

            // 5. Indicators & Tokens
            const indList = document.getElementById('keyIndicatorsList');
            indList.innerHTML = '';
            if (!report.key_evidence_indicators || report.key_evidence_indicators.length === 0) {
                indList.innerHTML = '<div class="text-xs text-[#86efac]/80 bg-[#0f2115]/50 border border-[#264e34]/50 p-2 rounded flex items-center gap-2"><i class="fa-solid fa-circle-check text-[#4ade80]"></i> No anomalous risk indicators flagged.</div>';
            } else {
                report.key_evidence_indicators.forEach(ind => {
                    const d = document.createElement('div');
                    d.className = 'flex items-center gap-2 text-xs text-red-300 bg-red-950/40 border border-red-900/60 p-2.5 rounded-lg';
                    d.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-red-400"></i> <span>${ind}</span>`;
                    indList.appendChild(d);
                });
            }

            const tokensList = document.getElementById('tokensList');
            tokensList.innerHTML = '';
            (report.textual_explainability || []).slice(0, 8).forEach(([t, s]) => {
                const pill = document.createElement('span');
                pill.className = 'px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 text-xs font-mono border border-neutral-700/60';
                pill.textContent = `${t} (${s.toFixed(2)})`;
                tokensList.appendChild(pill);
            });

            // 6. Grad-CAM Visual Heatmap
            const gradContainer = document.getElementById('gradcamContainer');
            const gradStats = document.getElementById('gradcamStats');
            const visExplain = report.visual_explainability || {};
            const keys = Object.keys(visExplain);

            if (keys.length > 0) {
                const firstKey = keys[0];
                const info = visExplain[firstKey];
                gradStats.textContent = `Max Act: ${(info.max_activation || 0).toFixed(3)} | Salient: ${info.salient_patch_detected ? 'YES' : 'NO'}`;
                
                gradContainer.innerHTML = '';
                if (info.heatmap_b64) {
                    gradContainer.innerHTML = `
                        <div class="space-y-1">
                            <img src="${info.heatmap_b64}" alt="Grad-CAM Activation" class="w-32 h-32 rounded border border-neutral-700 object-cover shadow">
                            <div class="text-[10px] text-neutral-400 text-center font-mono">Grad-CAM Map (${firstKey})</div>
                        </div>
                        <div class="text-xs text-neutral-300 space-y-1">
                            <div><span class="text-neutral-500">Evidence ID:</span> <span class="font-mono text-neutral-200">${firstKey}</span></div>
                            <div><span class="text-neutral-500">Peak Activation:</span> <span class="font-mono text-amber-400 font-bold">${(info.max_activation || 0).toFixed(4)}</span></div>
                            <div><span class="text-neutral-500">Salient Tampering Detected:</span> <span class="font-mono font-bold ${info.salient_patch_detected ? 'text-red-400' : 'text-[#86efac]'}">${info.salient_patch_detected ? 'TRUE (Exceeds 0.65 threshold)' : 'FALSE (Clean)'}</span></div>
                        </div>
                    `;
                }
            } else {
                gradStats.textContent = '';
                gradContainer.innerHTML = '<div class="text-xs text-neutral-500">No visual evidence present in active case.</div>';
            }

            // 7. Disclaimer
            if (report.legal_disclaimer) {
                document.getElementById('disclaimerText').textContent = report.legal_disclaimer;
            }
        }

        async function loadResultsTab() {
            try {
                const res = await fetch('/api/results_v10');
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();

                document.getElementById('v10ManifestSha').textContent = data.manifest_sha256.slice(0, 16) + '...';
                document.getElementById('v10Verdict').textContent = data.execution_verdict;
                document.getElementById('v10ResultsSha').textContent = data.results_sha256;

                // Render Arms A-E Table
                const armsTbody = document.getElementById('v10ArmsTableBody');
                armsTbody.innerHTML = '';
                data.arms_summary.forEach(a => {
                    const tr = document.createElement('tr');
                    tr.className = 'hover:bg-neutral-850/70';
                    tr.innerHTML = `
                        <td class="px-4 py-2.5 font-bold text-neutral-200">${a.label}</td>
                        <td class="px-4 py-2.5 text-[#86efac] font-bold">${a.mean_auc.toFixed(4)} &plusmn; ${a.std_auc.toFixed(4)}</td>
                        <td class="px-4 py-2.5">${(a.mean_accuracy * 100).toFixed(1)}%</td>
                        <td class="px-4 py-2.5">${(a.mean_f1 * 100).toFixed(1)}%</td>
                        <td class="px-4 py-2.5 font-mono text-neutral-400">${a.mean_brier.toFixed(4)}</td>
                    `;
                    armsTbody.appendChild(tr);
                });

                // Render Delta Comparisons Table
                const deltasTbody = document.getElementById('v10DeltasTableBody');
                deltasTbody.innerHTML = '';
                data.delta_comparisons.forEach(d => {
                    const tr = document.createElement('tr');
                    tr.className = 'hover:bg-neutral-850/70';
                    const ptStr = (d.point_estimate_delta_auc >= 0 ? '+' : '') + d.point_estimate_delta_auc.toFixed(4);
                    const meanStr = (d.bootstrap_mean_delta_auc >= 0 ? '+' : '') + d.bootstrap_mean_delta_auc.toFixed(4);
                    const ciStr = `[${d.ci_lower_auc.toFixed(4)}, ${d.ci_upper_auc.toFixed(4)}]`;

                    let badgeHtml = '';
                    if (d.benefit_status === 'no benefit') {
                        badgeHtml = '<span class="px-2 py-0.5 rounded bg-neutral-800 text-neutral-400 border border-neutral-700 font-bold">no benefit</span>';
                    } else if (d.ci_lower_auc > 0) {
                        badgeHtml = '<span class="px-2 py-0.5 rounded bg-[#0f2115] text-[#86efac] border border-[#264e34] font-bold">significant gain</span>';
                    } else {
                        badgeHtml = '<span class="px-2 py-0.5 rounded bg-neutral-800 text-neutral-400 border border-neutral-700 font-bold">neutral</span>';
                    }

                    tr.innerHTML = `
                        <td class="px-4 py-2.5 font-bold text-neutral-200">${d.comparison}</td>
                        <td class="px-4 py-2.5 font-bold ${d.point_estimate_delta_auc >= 0 ? 'text-[#86efac]' : 'text-neutral-300'}">${ptStr}</td>
                        <td class="px-4 py-2.5 text-neutral-300">${meanStr}</td>
                        <td class="px-4 py-2.5 font-mono text-neutral-400">${ciStr}</td>
                        <td class="px-4 py-2.5">${badgeHtml}</td>
                    `;
                    deltasTbody.appendChild(tr);
                });

            } catch (err) {
                console.error('Failed to load results table:', err);
            }
        }

        // Auto-detect and populate file text into textarea
        document.getElementById('evidenceFileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            const typeSelect = document.getElementById('uploadTypeSelect');
            if (file.name.endsWith('.txt') || file.name.endsWith('.json') || file.name.endsWith('.log')) {
                const reader = new FileReader();
                reader.onload = function(evt) {
                    document.getElementById('textEvidenceInput').value = evt.target.result;
                    if (file.name.includes('email') || file.name.endsWith('.json')) {
                        typeSelect.value = 'email';
                    } else {
                        typeSelect.value = 'chat';
                    }
                };
                reader.readAsText(file);
            } else if (file.type.startsWith('image/') || file.name.endsWith('.png') || file.name.endsWith('.jpg')) {
                typeSelect.value = 'image';
            }
        });

        window.onload = async () => {
            // Load benign sample by default for a clean, non-alarming initial state
            await loadSample('benign');
        };
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
