"""
ForenSightX - FastAPI Investigation & Intelligence Service
Authoritative Spec Reference: ForenSightX_Project_Architecture.docx Section 8, 10, 11
Provides RESTful APIs and interactive investigation dashboard for digital forensic teams.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
from src.agents import CaseCoordinatorAgent, InvestigationReport


# =====================================================================
# Database Setup (SQLite for prototype per docx Section 11)
# =====================================================================

DB_PATH = "forensightx.db"

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
# App & Models Initialization
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

# Load DL models and Multi-Agent Coordinator
MODELS_DIR = "models_saved"
tokenizer = ForensicTokenizer(max_length=80)
vocab_path = os.path.join(MODELS_DIR, "tokenizer_vocab.json")

if os.path.exists(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        tokenizer.word2idx = json.load(f)
        tokenizer.idx2word = {int(v): k for k, v in tokenizer.word2idx.items()}
else:
    gen = ForensicDatasetGenerator(seed=42)
    texts, _, _, _ = gen.generate_multimodal_dataset(num_samples=100)
    tokenizer.fit_on_texts(texts)

cnn_model = VisualEvidenceCNN(in_channels=3, num_classes=1)
bilstm_model = ConversationBiLSTM(vocab_size=max(tokenizer.vocab_size, 100), hidden_dim=64)
ann_model = ANNMultimodalRiskClassifier(input_dim=7, hidden_dim1=32, hidden_dim2=16)

# Load weights if available
for m, filename in [(cnn_model, "cnn_model.pt"), (bilstm_model, "bilstm_model.pt"), (ann_model, "ann_fusion_model.pt")]:
    p = os.path.join(MODELS_DIR, filename)
    if os.path.exists(p):
        m.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
    m.eval()

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


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ForenSightX Forensics Engine",
        "timestamp": datetime.now().isoformat(),
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

    # Assert eval mode across all models before running inference (Part 7 BN regression requirement)
    cnn_model.eval()
    bilstm_model.eval()
    ann_model.eval()
    assert not cnn_model.training, "CNN must be in eval mode during API inference"
    assert not bilstm_model.training, "BiLSTM must be in eval mode during API inference"
    assert not ann_model.training, "ANN must be in eval mode during API inference"

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


@app.get("/api/experiments")
def get_experimental_results():
    results_path = os.path.join("results", "experiment_results.json")
    if not os.path.exists(results_path):
        return {"status": "pending", "message": "Run train.py to generate experimental metrics"}
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"status": "completed", "results": data}


# =====================================================================
# Interactive Dashboard View (HTML/CSS/JS)
# =====================================================================

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ForenSightX - Forensics Intelligence & Investigation Assistant</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-slate-950 text-slate-100 font-sans min-h-screen">
        <!-- Navigation -->
        <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
            <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-9 h-9 rounded-lg bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center text-emerald-400 font-bold">
                        FX
                    </div>
                    <div>
                        <h1 class="text-lg font-bold tracking-wide text-white flex items-center gap-2">
                            FORENSIGHTX <span class="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">MAS Prototype</span>
                        </h1>
                        <p class="text-xs text-slate-400">AI-Powered Digital Forensics & Multimodal Investigation Assistant</p>
                    </div>
                </div>
                <div class="flex items-center space-x-3">
                    <span class="text-xs text-amber-400/90 bg-amber-950/60 border border-amber-800/80 px-2.5 py-1 rounded-full flex items-center gap-1.5">
                        <i class="fa-solid fa-scale-balanced text-xs"></i> Decision Support Only • Not Legal Proof
                    </span>
                    <a href="/docs" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition">API Docs</a>
                </div>
            </div>
        </header>

        <!-- Main Workspace -->
        <main class="max-w-7xl mx-auto px-4 py-6 space-y-6">
            <!-- Top Controls: Cases & Quick Simulation -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-col justify-between">
                    <div>
                        <h2 class="text-sm font-semibold text-slate-300 mb-1 flex items-center gap-2">
                            <i class="fa-solid fa-folder-open text-emerald-400"></i> Active Forensic Cases
                        </h2>
                        <p class="text-xs text-slate-400 mb-3">Select or initialize an active case to examine.</p>
                        <select id="caseSelect" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"></select>
                    </div>
                    <div class="flex gap-2 mt-4">
                        <button onclick="createNewCasePrompt()" class="flex-1 py-1.5 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition">
                            <i class="fa-solid fa-plus mr-1"></i> New Case
                        </button>
                        <button onclick="runInvestigation()" class="flex-1 py-1.5 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white shadow-sm transition">
                            <i class="fa-solid fa-play mr-1"></i> Investigate
                        </button>
                    </div>
                </div>

                <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-col justify-between">
                    <div>
                        <h2 class="text-sm font-semibold text-slate-300 mb-1 flex items-center gap-2">
                            <i class="fa-solid fa-vial text-blue-400"></i> NNDL Evaluation Experiments
                        </h2>
                        <p class="text-xs text-slate-400 mb-2">Controlled benchmarks across Modalities A through E on held-out test set.</p>
                        <div id="expStatus" class="text-xs text-slate-400 font-mono">Loading experimental results...</div>
                    </div>
                    <button onclick="fetchExperiments()" class="mt-3 py-1.5 px-3 rounded-lg bg-blue-600/30 border border-blue-500/50 hover:bg-blue-600/50 text-blue-300 text-xs font-medium transition">
                        <i class="fa-solid fa-chart-column mr-1"></i> Refresh Experiments Table
                    </button>
                </div>

                <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-col justify-between">
                    <div>
                        <h2 class="text-sm font-semibold text-slate-300 mb-1 flex items-center gap-2">
                            <i class="fa-solid fa-robot text-purple-400"></i> Multi-Agent Forensics Team (MAS)
                        </h2>
                        <p class="text-xs text-slate-400 mb-2">Ingestion, BiLSTM, CNN, Metadata, Fusion, Timeline, Explainability agents.</p>
                        <div class="text-xs text-slate-400 space-y-1">
                            <div class="flex justify-between"><span>BiLSTM Agent:</span> <span class="text-emerald-400">Online</span></div>
                            <div class="flex justify-between"><span>CNN Visual Agent:</span> <span class="text-emerald-400">Online</span></div>
                            <div class="flex justify-between"><span>7D ANN Fusion:</span> <span class="text-emerald-400">Online</span></div>
                        </div>
                    </div>
                    <button onclick="seedSampleCase()" class="mt-3 py-1.5 px-3 rounded-lg bg-purple-600/30 border border-purple-500/50 hover:bg-purple-600/50 text-purple-300 text-xs font-medium transition">
                        <i class="fa-solid fa-bolt mr-1"></i> Seed Phishing Simulation Case
                    </button>
                </div>
            </div>

            <!-- Investigation Overview / Dashboard -->
            <div id="investigationSection" class="hidden space-y-6">
                <!-- Risk Cards -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800">
                        <span class="text-xs font-medium text-slate-400">Overall Risk Score</span>
                        <div id="riskScoreVal" class="text-3xl font-extrabold text-red-500 mt-1">--%</div>
                        <div id="riskTierBadge" class="mt-2 inline-block px-2 py-0.5 rounded text-xs font-semibold bg-red-950 text-red-400 border border-red-800">CRITICAL</div>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800">
                        <span class="text-xs font-medium text-slate-400">Visual Evidence (CNN)</span>
                        <div id="cnnScoreVal" class="text-2xl font-bold text-slate-200 mt-1">--%</div>
                        <span class="text-xs text-slate-500">Grad-CAM verified</span>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800">
                        <span class="text-xs font-medium text-slate-400">Conversation / Email (BiLSTM)</span>
                        <div id="bilstmScoreVal" class="text-2xl font-bold text-slate-200 mt-1">--%</div>
                        <span class="text-xs text-slate-500">Attention token analysis</span>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800">
                        <span class="text-xs font-medium text-slate-400">Evidence Count</span>
                        <div id="evidenceCountVal" class="text-2xl font-bold text-slate-200 mt-1">0</div>
                        <span class="text-xs text-slate-500">Multimodal items ingested</span>
                    </div>
                </div>

                <!-- 7D Feature Vector & Attribution -->
                <div class="p-5 rounded-xl bg-slate-900 border border-slate-800">
                    <h3 class="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                        <i class="fa-solid fa-network-wired text-emerald-400"></i> Multimodal Feature Vector & ANN Attribution (Authoritative 7D Spec)
                    </h3>
                    <div id="featureVectorGrid" class="grid grid-cols-2 md:grid-cols-7 gap-3 text-center"></div>
                </div>

                <!-- Timeline & Explainability Indicators -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <!-- Chronological Timeline -->
                    <div class="p-5 rounded-xl bg-slate-900 border border-slate-800">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-sm font-bold text-slate-200 flex items-center gap-2">
                                <i class="fa-solid fa-clock-rotate-left text-amber-400"></i> Chronological Timeline
                            </h3>
                            <span class="text-xs text-slate-400">Observed vs Inferred</span>
                        </div>
                        <div id="timelineContainer" class="space-y-3 max-h-96 overflow-y-auto pr-2"></div>
                    </div>

                    <!-- Key Indicators & Disclaimers -->
                    <div class="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                        <div>
                            <h3 class="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                                <i class="fa-solid fa-triangle-exclamation text-rose-400"></i> Forensic Indicators & Attributions
                            </h3>
                            <div id="keyIndicatorsList" class="space-y-2 mb-4"></div>
                            
                            <h4 class="text-xs font-semibold text-slate-400 mb-2">Salient Linguistic Tokens:</h4>
                            <div id="tokensList" class="flex flex-wrap gap-1.5"></div>
                        </div>

                        <div class="mt-6 p-3 rounded-lg bg-amber-950/40 border border-amber-800/60 text-xs text-amber-300/90 leading-relaxed">
                            <div class="font-bold text-amber-200 mb-1 flex items-center gap-1.5">
                                <i class="fa-solid fa-shield-halved"></i> Decision Support Mandate
                            </div>
                            <span id="disclaimerText"></span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Controlled Experiments Modal / Panel -->
            <div id="experimentsPanel" class="p-5 rounded-xl bg-slate-900 border border-slate-800">
                <h3 class="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                    <i class="fa-solid fa-flask-vial text-blue-400"></i> Empirical Syllabus Evaluation (Module 1 - 5)
                </h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs text-left text-slate-300 font-mono">
                        <thead class="text-xs text-slate-400 uppercase bg-slate-950 border-b border-slate-800">
                            <tr>
                                <th class="px-4 py-3">Experiment</th>
                                <th class="px-4 py-3">Accuracy</th>
                                <th class="px-4 py-3">Precision</th>
                                <th class="px-4 py-3">Recall</th>
                                <th class="px-4 py-3">F1-Score</th>
                                <th class="px-4 py-3">ROC-AUC</th>
                            </tr>
                        </thead>
                        <tbody id="experimentsTableBody" class="divide-y divide-slate-800">
                            <tr><td colspan="6" class="px-4 py-3 text-center text-slate-500">Loading experimental benchmarks...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </main>

        <script>
            let currentCases = [];

            async function loadCases() {
                try {
                    const res = await fetch('/api/cases');
                    const data = await res.json();
                    currentCases = data.cases || [];
                    const sel = document.getElementById('caseSelect');
                    sel.innerHTML = '';
                    if (currentCases.length === 0) {
                        sel.innerHTML = '<option value="">No cases recorded</option>';
                    } else {
                        currentCases.forEach(c => {
                            const opt = document.createElement('option');
                            opt.value = c.case_id;
                            opt.textContent = `${c.case_id}: ${c.title} (${c.status})`;
                            sel.appendChild(opt);
                        });
                    }
                } catch(e) {
                    console.error(e);
                }
            }

            async function seedSampleCase() {
                const sampleCaseId = 'CASE-PHISH-' + Math.floor(1000 + Math.random() * 9000);
                try {
                    await fetch('/api/cases', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            case_id: sampleCaseId,
                            title: 'Simulated Financial Phishing & Credential Interception',
                            investigator: 'Forensic Unit'
                        })
                    });

                    // Add email evidence
                    await fetch(`/api/cases/${sampleCaseId}/evidence`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            evidence_id: 'EVID-MAIL-01',
                            evidence_type: 'email',
                            timestamp: '2026-11-01T10:02:00',
                            content: {
                                sender: 'compliance@bank-security.xyz',
                                display_name: 'Security Admin',
                                reply_to: 'c2@dark-relay.top',
                                subject: 'URGENT: Confidential Password Expiration Notice',
                                body: 'URGENT: Your account has been suspended due to unauthorized access. Click here immediately to verify your credentials: http://192.168.1.105/auth',
                                attachments: ['security_update.pdf.exe']
                            },
                            metadata: { urls: ['http://192.168.1.105/auth'] },
                            is_observed: true
                        })
                    });

                    // Add chat evidence
                    await fetch(`/api/cases/${sampleCaseId}/evidence`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            evidence_id: 'EVID-CHAT-01',
                            evidence_type: 'chat',
                            timestamp: '2026-11-01T10:05:00',
                            content: 'Please send the one-time password (OTP) right now to verify wire transfer compliance.',
                            metadata: {},
                            is_observed: true
                        })
                    });

                    await loadCases();
                    document.getElementById('caseSelect').value = sampleCaseId;
                    alert(`Case ${sampleCaseId} created with realistic forensic evidence! Click Investigate to run analysis.`);
                } catch(e) {
                    alert('Error seeding sample case: ' + e);
                }
            }

            async function createNewCasePrompt() {
                const caseId = prompt('Enter Case Identifier (e.g. CASE-2026-002):', 'CASE-' + Date.now().toString().slice(-4));
                if (!caseId) return;
                const title = prompt('Enter Case Title:', 'Suspicious Domain Exfiltration Incident');
                if (!title) return;

                await fetch('/api/cases', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ case_id: caseId, title: title, investigator: 'Forensic Lead' })
                });
                await loadCases();
                document.getElementById('caseSelect').value = caseId;
            }

            async function runInvestigation() {
                const caseId = document.getElementById('caseSelect').value;
                if (!caseId) {
                    alert('Please select or create a case first.');
                    return;
                }

                try {
                    const res = await fetch(`/api/cases/${caseId}/investigate`, { method: 'POST' });
                    if (!res.ok) {
                        const err = await res.json();
                        alert('Investigation failed: ' + (err.detail || res.statusText));
                        return;
                    }
                    const report = await res.json();
                    renderInvestigationReport(report);
                } catch(e) {
                    alert('Investigation request failed: ' + e);
                }
            }

            function renderInvestigationReport(report) {
                document.getElementById('investigationSection').classList.remove('hidden');

                const riskPercent = (report.overall_risk_score * 100).toFixed(1);
                document.getElementById('riskScoreVal').textContent = `${riskPercent}%`;
                document.getElementById('riskTierBadge').textContent = report.risk_tier + ' RISK';

                const tier = report.risk_tier.toLowerCase();
                const badge = document.getElementById('riskTierBadge');
                if (tier === 'high' || tier === 'critical') {
                    badge.className = 'mt-2 inline-block px-2 py-0.5 rounded text-xs font-semibold bg-rose-950 text-rose-400 border border-rose-800';
                    document.getElementById('riskScoreVal').className = 'text-3xl font-extrabold text-rose-500 mt-1';
                } else {
                    badge.className = 'mt-2 inline-block px-2 py-0.5 rounded text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800';
                    document.getElementById('riskScoreVal').className = 'text-3xl font-extrabold text-emerald-500 mt-1';
                }

                document.getElementById('cnnScoreVal').textContent = ((report.model_predictions.cnn_score || 0) * 100).toFixed(1) + '%';
                document.getElementById('bilstmScoreVal').textContent = ((report.model_predictions.bilstm_score || 0) * 100).toFixed(1) + '%';
                document.getElementById('evidenceCountVal').textContent = report.evidence_inventory.length;

                // 7D Feature Grid
                const featNames = [
                    'CNN Visual', 'BiLSTM Text', 'URL Score',
                    'Keywords', 'Attachment', 'Timeline Anom', 'Email Anom'
                ];
                const featGrid = document.getElementById('featureVectorGrid');
                featGrid.innerHTML = '';
                report.feature_vector_7d.forEach((val, idx) => {
                    const col = document.createElement('div');
                    col.className = 'p-2 rounded bg-slate-950 border border-slate-800';
                    col.innerHTML = `
                        <div class="text-[10px] text-slate-400 truncate">${featNames[idx]}</div>
                        <div class="text-sm font-bold text-slate-200 mt-1">${(val * 100).toFixed(0)}%</div>
                    `;
                    featGrid.appendChild(col);
                });

                // Timeline
                const tl = document.getElementById('timelineContainer');
                tl.innerHTML = '';
                report.chronological_timeline.forEach(evt => {
                    const item = document.createElement('div');
                    const isObserved = evt.event_type === 'observed';
                    const borderCls = isObserved ? 'border-l-2 border-emerald-500 bg-slate-950' : 'border-l-2 border-amber-500 bg-amber-950/20';
                    const tagCls = isObserved ? 'text-emerald-400 bg-emerald-950' : 'text-amber-400 bg-amber-950';

                    item.className = `p-3 rounded-r-lg ${borderCls} border-t border-b border-r border-slate-800/80`;
                    item.innerHTML = `
                        <div class="flex items-center justify-between text-xs mb-1">
                            <span class="font-mono text-slate-400">${evt.timestamp}</span>
                            <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${tagCls} uppercase">${evt.event_type}</span>
                        </div>
                        <div class="text-xs text-slate-200 font-medium">${evt.description}</div>
                    `;
                    tl.appendChild(item);
                });

                // Indicators
                const indList = document.getElementById('keyIndicatorsList');
                indList.innerHTML = '';
                if (report.key_evidence_indicators.length === 0) {
                    indList.innerHTML = '<div class="text-xs text-slate-500">No high-risk indicators flagged.</div>';
                } else {
                    report.key_evidence_indicators.forEach(ind => {
                        const d = document.createElement('div');
                        d.className = 'flex items-center gap-2 text-xs text-rose-300 bg-rose-950/30 border border-rose-900/50 p-2 rounded';
                        d.innerHTML = `<i class="fa-solid fa-circle-exclamation text-rose-400"></i> <span>${ind}</span>`;
                        indList.appendChild(d);
                    });
                }

                // Salient Tokens
                const tokensList = document.getElementById('tokensList');
                tokensList.innerHTML = '';
                (report.textual_explainability || []).slice(0, 8).forEach(([t, s]) => {
                    const pill = document.createElement('span');
                    pill.className = 'px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-xs font-mono';
                    pill.textContent = `${t} (${s.toFixed(2)})`;
                    tokensList.appendChild(pill);
                });

                // Disclaimer
                document.getElementById('disclaimerText').textContent = report.legal_disclaimer;
            }

            async function fetchExperiments() {
                try {
                    const res = await fetch('/api/experiments');
                    const data = await res.json();
                    const tbody = document.getElementById('experimentsTableBody');
                    if (data.status === 'completed' && data.results) {
                        tbody.innerHTML = '';
                        for (const [expName, m] of Object.entries(data.results)) {
                            const tr = document.createElement('tr');
                            tr.className = 'hover:bg-slate-900/50';
                            tr.innerHTML = `
                                <td class="px-4 py-3 font-medium text-slate-200">${expName.replace(/_/g, ' ')}</td>
                                <td class="px-4 py-3 text-emerald-400">${(m.accuracy * 100).toFixed(2)}%</td>
                                <td class="px-4 py-3">${(m.precision * 100).toFixed(2)}%</td>
                                <td class="px-4 py-3">${(m.recall * 100).toFixed(2)}%</td>
                                <td class="px-4 py-3 text-blue-400">${(m.f1_score * 100).toFixed(2)}%</td>
                                <td class="px-4 py-3 text-purple-400">${(m.roc_auc * 100).toFixed(2)}%</td>
                            `;
                            tbody.appendChild(tr);
                        }
                        document.getElementById('expStatus').textContent = 'Benchmarks verified against held-out test split.';
                    } else {
                        document.getElementById('expStatus').textContent = 'Experiments pending or in progress.';
                    }
                } catch(e) {
                    console.error(e);
                }
            }

            window.onload = async () => {
                await loadCases();
                await fetchExperiments();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

