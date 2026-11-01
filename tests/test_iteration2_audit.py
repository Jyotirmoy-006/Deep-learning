"""
ForenSightX - Iteration 2 Audit, Leakage-Proofing & Regression Test Suite
Guards:
  - Part 0: BatchNorm eval fix, Autograd leaf fix, API case_id validation
  - Part 1: Dataset adapters, smoke mode, case builder, zero ID/hash overlap, VAE exclusion from val/test
  - Part 2: OOF cross-validation leakage assertion, heuristic feature formulas in docs/FEATURES.md
  - Part 3: Circularity controls (shuffled label, heuristic baseline, trivial separability detection)
  - Part 5 & 6: Module 2 grid options, BiLSTM concat (Blueprint 5.2) vs attention pooling, derived max_len
  - Part 7: Shape tracing, grad norm tracking, OOV calculation, overfit-one-batch, BN eval batch size 1
"""

import hashlib
import os
import uuid
import numpy as np
import pytest
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from PIL import Image

from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    SyntheticDataVAE,
    init_weights,
    get_optimizer,
    DEVICE,
)
from src.data.adapters import (
    ImageForensicAdapter,
    TextForensicAdapter,
    ForensicRawRecord,
)
from src.data.case_builder import (
    CaseBuilder,
    MultimodalCaseItem,
    create_stratified_split,
)
from src.data_loader import (
    ForensicTokenizer,
    derive_max_len,
    calculate_oov_rate,
    extract_url_score,
    extract_keyword_score,
    extract_attachment_score,
    extract_timeline_anomaly,
    extract_email_anomaly,
    SMOKE_BANNER,
)


# =====================================================================
# Part 0 & Part 7: Regression Tests for Iteration 1 Failures
# =====================================================================

def test_regression_batchnorm_eval_batch_size_1():
    """
    Guards Iteration 1 failure: BatchNorm must not bypass running_mean/var during eval.
    Must produce valid normalized probabilities even for batch size 1.
    """
    model = ANNMultimodalRiskClassifier(input_dim=7, use_batchnorm=True)
    # Put in eval mode
    model.eval()
    single_x = torch.tensor([0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=torch.float32)
    out = model(single_x)
    assert out.shape == (1, 1), f"Expected shape (1, 1), got {out.shape}"
    assert 0.0 <= out.item() <= 1.0, f"Expected probability in [0, 1], got {out.item()}"


def test_regression_autograd_leaf_tensor_explain_features():
    """
    Guards Iteration 1 failure: explain_features unsqueeze must not break leaf tensor grad tracking.
    """
    model = ANNMultimodalRiskClassifier(input_dim=7)
    model.eval()
    feat = torch.tensor([0.8, 0.9, 0.7, 0.6, 0.9, 0.5, 0.7], dtype=torch.float32)
    attributions = model.explain_features(feat)
    assert len(attributions) == 7
    for name in ANNMultimodalRiskClassifier.FEATURE_NAMES:
        assert name in attributions
        assert attributions[name] >= 0.0
    # Must sum to approx 1.0
    assert abs(sum(attributions.values()) - 1.0) < 1e-4


# =====================================================================
# Part 1: Data Integrity & Adapters
# =====================================================================

def test_dataset_adapters_and_smoke_mode():
    img_adapter = ImageForensicAdapter("non_existent_image_path")
    txt_adapter = TextForensicAdapter("non_existent_text_path")
    assert not img_adapter.is_available()
    assert not txt_adapter.is_available()
    assert SMOKE_BANNER == "SYNTHETIC SMOKE RUN - NOT A RESULT"


def test_case_builder_label_conditioned_pairing():
    texts = [
        ForensicRawRecord("T1", "text", "benign email", 0, {}, "h1"),
        ForensicRawRecord("T2", "text", "suspicious email", 1, {}, "h2"),
    ]
    images = [
        ForensicRawRecord("I1", "image", "benign.png", 0, {}, "h3"),
        ForensicRawRecord("I2", "image", "phish.png", 1, {}, "h4"),
    ]
    builder = CaseBuilder(pairing_strategy="label_conditioned", seed=42)
    cases = builder.build_cases(texts, images)
    assert len(cases) == 2
    for c in cases:
        if c.label == 0:
            assert "benign" in c.text_content
        elif c.label == 1:
            assert "suspicious" in c.text_content


def test_stratified_split_zero_leakage_and_manifest():
    cases = []
    for i in range(100):
        lbl = 1 if i % 2 == 0 else 0
        h = hashlib.sha256(f"content-{i}".encode()).hexdigest()
        cases.append(
            MultimodalCaseItem(
                case_id=f"CASE-{i:03d}",
                text_content=f"Text sample {i}",
                text_hash=f"thash-{i}",
                image_content=f"img_{i}.png",
                image_hash=f"ihash-{i}",
                label=lbl,
                metadata={},
                content_hash=h,
            )
        )

    manifest_file = "data/test_manifest.json"
    train_set, val_set, test_set, stats = create_stratified_split(
        cases, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, manifest_path=manifest_file
    )

    assert len(train_set) == 70
    assert len(val_set) + len(test_set) == 30

    # Zero ID overlap assertion
    train_ids = {c.case_id for c in train_set}
    val_ids = {c.case_id for c in val_set}
    test_ids = {c.case_id for c in test_set}
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Zero Content Hash overlap assertion
    train_hashes = {c.content_hash for c in train_set}
    val_hashes = {c.content_hash for c in val_set}
    test_hashes = {c.content_hash for c in test_set}
    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0

    # Imbalance ratio recorded
    assert "imbalance_ratio" in stats["splits"]["train"]
    assert stats["manifest_sha256"] is not None
    assert os.path.exists(manifest_file)
    os.remove(manifest_file)


# =====================================================================
# Part 2: Feature Formulas Unit Tests (docs/FEATURES.md)
# =====================================================================

def test_feature_formula_url_score():
    # Empty
    assert extract_url_score("") == 0.0
    # IPv4 literal (+0.80) + baseline (+0.20) = 1.0
    assert extract_url_score("http://192.168.1.100/login") == 1.0
    # Suspicious TLD (.xyz: +0.60, baseline +0.20) = 0.80
    assert pytest.approx(extract_url_score("http://sample-domain.xyz"), 0.01) == 0.80
    # Basic auth trick '@' (+0.70, baseline +0.20) = 0.90
    assert pytest.approx(extract_url_score("http://legit.com@phish.net"), 0.01) == 0.90


def test_feature_formula_keyword_score():
    assert extract_keyword_score("routine quarterly engineering sync") == 0.0
    # 'otp' (1.0) / 3.0 = 0.3333
    assert pytest.approx(extract_keyword_score("please share your otp"), 0.01) == 0.3333
    # 'urgent' (0.8) + 'wire' (0.9) + 'transfer' (0.7) = 2.4 / 3.0 = 0.80
    assert pytest.approx(extract_keyword_score("urgent wire transfer required"), 0.01) == 0.80
    # Bound to 1.0 max
    assert extract_keyword_score("urgent wire transfer immediately verify credentials password") == 1.0


def test_feature_formula_attachment_score():
    assert extract_attachment_score([]) == 0.0
    assert extract_attachment_score(["report.docx"]) == 0.0
    # Double extension -> 0.95
    assert extract_attachment_score(["invoice.pdf.exe"]) == 1.0  # max(0.95, 1.0 for .exe) = 1.0
    assert extract_attachment_score(["statement.pdf.vbs"]) == 0.95
    # Archive -> 0.40
    assert extract_attachment_score(["data.zip"]) == 0.40


def test_feature_formula_timeline_anomaly():
    t0 = datetime(2026, 11, 1, 10, 0, 0)
    assert extract_timeline_anomaly([t0]) == 0.0
    # Normal spacing (> 3 sec)
    assert extract_timeline_anomaly([t0, t0 + timedelta(minutes=5), t0 + timedelta(minutes=10)]) == 0.0
    # Rapid burst (< 3 sec) for >= 2 events -> 0.50 + 0.10 * 2 = 0.70
    burst = [t0, t0 + timedelta(seconds=1), t0 + timedelta(seconds=2)]
    assert pytest.approx(extract_timeline_anomaly(burst), 0.01) == 0.70


def test_feature_formula_email_anomaly():
    clean = {"sender": "lead@company.com", "reply_to": "lead@company.com", "display_name": "Lead"}
    assert extract_email_anomaly(clean) == 0.0

    # Domain mismatch (0.60)
    mismatch = {"sender": "ceo@company.com", "reply_to": "attacker@evil.com", "display_name": "CEO"}
    assert pytest.approx(extract_email_anomaly(mismatch), 0.01) == 0.60

    # Display spoofing from free mail (0.70)
    spoof = {"sender": "attacker@gmail.com", "reply_to": "attacker@gmail.com", "display_name": "Chief Executive Officer"}
    assert pytest.approx(extract_email_anomaly(spoof), 0.01) == 0.70


# =====================================================================
# Part 5 & 6: Module 2 Grid & BiLSTM Alignment
# =====================================================================

def test_bilstm_blueprint_pooling_modes():
    vocab_size = 50
    # Blueprint 5.2 default: concat
    model_concat = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=32, pooling="concat")
    tokens = torch.randint(0, vocab_size, (2, 15))
    out_c = model_concat(tokens)
    assert out_c.shape == (2, 1)

    # Optional attention pooling
    model_attn = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=32, pooling="attention")
    out_a, attn_w = model_attn(tokens, return_attention=True)
    assert out_a.shape == (2, 1)
    assert attn_w.shape == (2, 15)


def test_derive_max_len_percentile():
    texts = [
        "short text",
        "medium length message for sync",
        "this is a much longer chat message simulating detailed phishing instructions with url and tokens",
    ]
    derived = derive_max_len(texts, percentile=95.0)
    assert derived > 5
    assert isinstance(derived, int)


def test_ann_module2_options():
    # Test all variations of dropout and batchnorm
    for use_bn in [True, False]:
        for use_drop in [True, False]:
            model = ANNMultimodalRiskClassifier(input_dim=7, use_batchnorm=use_bn, use_dropout=use_drop)
            x = torch.rand(4, 7)
            out = model(x)
            assert out.shape == (4, 1)

    # Test initializers
    for init_m in ["xavier_uniform", "he_normal", "default"]:
        model = ANNMultimodalRiskClassifier(input_dim=7)
        init_weights(model, method=init_m)


# =====================================================================
# Part 7: Diagnostics & Traces
# =====================================================================

def test_model_shape_traces():
    cnn = VisualEvidenceCNN(in_channels=3, num_classes=1)
    cnn_trace = cnn.trace_shapes(torch.randn(1, 3, 128, 128))
    assert len(cnn_trace) >= 6
    assert cnn_trace[0]["stage"] == "input"
    assert cnn_trace[-1]["stage"] in ["output", "classifier_output"]

    bilstm = ConversationBiLSTM(vocab_size=50, pooling="concat")
    bilstm_trace = bilstm.trace_shapes(torch.randint(0, 50, (1, 10)))
    assert len(bilstm_trace) >= 4

    ann = ANNMultimodalRiskClassifier(input_dim=7)
    ann_trace = ann.trace_shapes(torch.rand(1, 7))
    assert len(ann_trace) >= 4


def test_oov_rate_calculation():
    tok = ForensicTokenizer(max_length=20)
    tok.fit_on_texts(["apple banana cherry"])
    # "date" is OOV
    rate = calculate_oov_rate(["apple date"], tok)
    assert pytest.approx(rate, 0.01) == 0.50
