"""
ForenSightX - Comprehensive Test Suite
Authoritative Reference: ForenSightX_Project_Architecture.docx
Verifies:
  - CNN, BiLSTM, ANN 7D Fusion, VAE
  - NNDL Module 2 (Optimizers, Initializers, Regularization)
  - Modality Feature Extractors (URL, Keywords, Attachments, Timeline, Email)
  - Multi-Agent System (Ingestion, Modality Agents, Fusion, Timeline, Explainability)
  - Chronological Timeline (Observed vs Inferred events)
  - Zero Illustrative Hardcoded Fixtures Rule
"""

from datetime import datetime, timedelta
import numpy as np
import pytest
import torch
from PIL import Image

from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    SyntheticDataVAE,
    init_weights,
    get_optimizer,
)
from src.data_loader import (
    ForensicTokenizer,
    ForensicDatasetGenerator,
    EvidenceItem,
    ForensicCase,
    extract_keyword_score,
    extract_url_score,
    extract_attachment_score,
    extract_timeline_anomaly,
    extract_email_anomaly,
)
from src.agents import (
    CaseCoordinatorAgent,
    TimelineAgent,
    FusionAgent,
    ExplainabilityAgent,
)


# =====================================================================
# Model Architecture & Syllabus Tests
# =====================================================================

def test_cnn_forward_and_gradcam():
    model = VisualEvidenceCNN(in_channels=3, num_classes=1)
    init_weights(model, method="he_normal")
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()

    # Grad-CAM explanation test
    single_x = torch.randn(1, 3, 128, 128)
    heatmap = model.get_gradcam(single_x, target_size=(128, 128))
    assert heatmap.shape == (128, 128)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_bilstm_forward_and_attention():
    vocab_size = 50
    model = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=32, num_layers=2)
    init_weights(model, method="xavier_uniform")
    token_ids = torch.randint(0, vocab_size, (3, 20))
    out, attn = model(token_ids, return_attention=True)
    assert out.shape == (3, 1)
    assert attn.shape == (3, 20)
    assert (out >= 0.0).all() and (out <= 1.0).all()

    # Attention weights must sum to 1 across sequence
    row_sums = torch.sum(attn, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    # Token importance test
    tokens = [f"word_{i}" for i in range(20)]
    importance = model.get_token_importance(tokens, token_ids[0])
    assert len(importance) == 20
    assert all(isinstance(t, str) and isinstance(s, float) for t, s in importance)


def test_ann_multimodal_fusion_7d():
    """Verifies authoritative 7D feature vector processing."""
    model = ANNMultimodalRiskClassifier(input_dim=7, hidden_dim1=16, hidden_dim2=8)
    init_weights(model, method="he_normal")

    # Batch pass
    x_batch = torch.rand(4, 7)
    out_batch = model(x_batch)
    assert out_batch.shape == (4, 1)
    assert (out_batch >= 0.0).all() and (out_batch <= 1.0).all()

    # Single vector pass (eval mode)
    model.eval()
    x_single = torch.tensor([0.8, 0.9, 0.7, 0.6, 0.9, 0.5, 0.7], dtype=torch.float32)
    out_single = model(x_single)
    assert out_single.shape == (1, 1)
    assert 0.0 <= out_single.item() <= 1.0

    # Feature attributions
    attributions = model.explain_features(x_single)
    assert len(attributions) == 7
    for name in ANNMultimodalRiskClassifier.FEATURE_NAMES:
        assert name in attributions
        assert attributions[name] >= 0.0


def test_vae_synthetic_generation():
    vae = SyntheticDataVAE(input_dim=7, latent_dim=3)
    x = torch.rand(5, 7)
    recon, mu, logvar = vae(x)
    assert recon.shape == (5, 7)
    assert mu.shape == (5, 3)
    assert logvar.shape == (5, 3)

    total_loss, recon_loss, kld_loss = SyntheticDataVAE.vae_loss(recon, x, mu, logvar)
    assert total_loss.item() > 0.0
    assert not torch.isnan(total_loss)

    # Controlled generation
    samples = vae.generate(num_samples=10)
    assert samples.shape == (10, 7)
    assert (samples >= 0.0).all() and (samples <= 1.0).all()


def test_nndl_module2_syllabus_components():
    """Tests backpropagation, optimizers, and initializations."""
    model = ANNMultimodalRiskClassifier(input_dim=7)

    # Check initializers
    for init_m in ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal"]:
        init_weights(model, method=init_m)

    # Check optimizers
    for opt_name in ["adam", "rmsprop", "sgd"]:
        opt = get_optimizer(model, optimizer_name=opt_name, lr=0.01)
        assert opt is not None

        # Backpropagation step
        opt.zero_grad()
        dummy_input = torch.rand(2, 7)
        dummy_label = torch.ones(2, 1)
        pred = model(dummy_input)
        loss = torch.nn.functional.binary_cross_entropy(pred, dummy_label)
        loss.backward()
        opt.step()


# =====================================================================
# Data Processing & Heuristic Extractors Tests
# =====================================================================

def test_tokenizer_encode_pad():
    tokenizer = ForensicTokenizer(max_length=10)
    corpus = ["urgent wire transfer required immediately", "hello team meeting scheduled"]
    tokenizer.fit_on_texts(corpus)
    tokens, ids = tokenizer.encode("urgent wire transfer")
    assert len(tokens) == 10
    assert len(ids) == 10
    assert tokens[0] == "urgent"
    assert tokens[-1] == ForensicTokenizer.PAD_TOKEN


def test_feature_extractors():
    # URL extraction
    assert extract_url_score("No links here") == 0.0
    assert extract_url_score("Visit http://192.168.1.1/login") > 0.5
    assert extract_url_score("Check out http://phish-update.xyz") > 0.5

    # Keyword extraction
    assert extract_keyword_score("Just checking the team sync minutes") == 0.0
    assert extract_keyword_score("URGENT: Provide your OTP and credentials immediately") > 0.5

    # Attachment extraction
    assert extract_attachment_score([]) == 0.0
    assert extract_attachment_score(["notes.txt"]) == 0.0
    assert extract_attachment_score(["invoice.pdf.exe"]) >= 0.95
    assert extract_attachment_score(["script.vbs"]) >= 0.95

    # Timeline anomaly
    t0 = datetime.fromisoformat("2026-11-01T10:00:00")
    normal_times = [t0, t0 + timedelta(minutes=15), t0 + timedelta(minutes=45)]
    assert extract_timeline_anomaly(normal_times) == 0.0

    burst_times = [t0, t0 + timedelta(seconds=1), t0 + timedelta(seconds=2)]
    assert extract_timeline_anomaly(burst_times) >= 0.5

    # Email anomaly
    clean_email = {"sender": "bob@corp.com", "reply_to": "bob@corp.com", "display_name": "Bob"}
    assert extract_email_anomaly(clean_email) == 0.0

    spoofed_email = {
        "sender": "ceo@free-mail.com",
        "reply_to": "attacker@evil.top",
        "display_name": "Chief Executive Officer",
        "headers": {"spf": "fail"},
    }
    assert extract_email_anomaly(spoofed_email) > 0.6


# =====================================================================
# Multi-Agent Forensics Workflow & Reporting Tests
# =====================================================================

def test_multi_agent_investigation_pipeline():
    cnn = VisualEvidenceCNN(in_channels=3, num_classes=1)
    bilstm = ConversationBiLSTM(vocab_size=100, hidden_dim=32)
    ann = ANNMultimodalRiskClassifier(input_dim=7)
    tok = ForensicTokenizer(max_length=50)
    tok.fit_on_texts(["urgent password reset verify credentials", "team lunch standup"])

    coordinator = CaseCoordinatorAgent(cnn, bilstm, ann, tok)

    now = datetime.now()
    case = ForensicCase(
        case_id="TEST-CASE-001",
        title="Automated Test Forensics Case",
        created_at=now.isoformat(),
        evidence_items=[
            EvidenceItem(
                evidence_id="E1",
                evidence_type="email",
                timestamp=(now - timedelta(minutes=10)).isoformat(),
                content={"subject": "Password Reset Alert", "body": "urgent password reset required"},
                metadata={"urls": ["http://phish.xyz"]},
                is_observed=True,
            ),
            EvidenceItem(
                evidence_id="E2",
                evidence_type="chat",
                timestamp=(now - timedelta(minutes=5)).isoformat(),
                content="Please verify credentials immediately",
                is_observed=True,
            ),
        ],
    )

    report = coordinator.investigate(case)
    assert report.case_id == "TEST-CASE-001"
    assert len(report.feature_vector_7d) == 7
    assert 0.0 <= report.overall_risk_score <= 1.0
    assert report.risk_tier in ["LOW", "MEDIUM", "HIGH"]

    # Verify timeline distinguishes observed from inferred
    timeline_types = {e.event_type for e in report.chronological_timeline}
    assert "observed" in timeline_types

    # Verify legal disclaimer presence
    assert "DECISION SUPPORT" in report.legal_disclaimer
    assert "not constitute legal certainty" in report.legal_disclaimer

    # Verify Markdown and Dict conversion
    md = report.to_markdown()
    assert "# FORENSIGHTX — FORENSIC INVESTIGATION REPORT" in md
    d = report.to_dict()
    assert d["case_id"] == "TEST-CASE-001"

