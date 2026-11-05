"""
ForenSightX - Multi-Agent System (MAS) Investigation Architecture
Authoritative Spec Reference: ForenSightX_Project_Architecture.docx Section 3, 7, 8, 9, 10
Implements collaborative specialized agents for forensic evidence analysis,
multimodal fusion, timeline reconstruction, explainability, and reporting.
"""

import os
import io
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
    extract_url_score,
    extract_keyword_score,
    extract_attachment_score,
    extract_email_anomaly,
    extract_timeline_anomaly,
)


@dataclass
class TimelineEvent:
    timestamp: str
    description: str
    source_evidence_id: str
    event_type: str        # 'observed' or 'inferred'
    risk_level: str        # 'low', 'medium', 'high'
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationReport:
    case_id: str
    title: str
    investigator: str
    timestamp_generated: str
    evidence_inventory: List[Dict[str, Any]]
    model_predictions: Dict[str, float]
    feature_vector_7d: List[float]
    overall_risk_score: float  # Scale: 0.0 to 1.0 (or 0 to 100%)
    risk_tier: str             # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    chronological_timeline: List[TimelineEvent]
    key_evidence_indicators: List[str]
    feature_attributions: Dict[str, float]
    visual_explainability: Dict[str, Any]
    textual_explainability: List[Tuple[str, float]]
    limitations_and_uncertainty: List[str]
    legal_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "investigator": self.investigator,
            "timestamp_generated": self.timestamp_generated,
            "evidence_inventory": self.evidence_inventory,
            "model_predictions": self.model_predictions,
            "feature_vector_7d": self.feature_vector_7d,
            "overall_risk_score": self.overall_risk_score,
            "risk_tier": self.risk_tier,
            "chronological_timeline": [e.__dict__ for e in self.chronological_timeline],
            "key_evidence_indicators": self.key_evidence_indicators,
            "feature_attributions": self.feature_attributions,
            "visual_explainability": self.visual_explainability,
            "textual_explainability": self.textual_explainability,
            "limitations_and_uncertainty": self.limitations_and_uncertainty,
            "legal_disclaimer": self.legal_disclaimer,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# FORENSIGHTX — FORENSIC INVESTIGATION REPORT",
            f"**Case ID:** `{self.case_id}`  |  **Date:** {self.timestamp_generated}  |  **Investigator:** {self.investigator}",
            f"**Case Title:** {self.title}",
            "",
            "## 1. Executive Summary & Risk Assessment",
            f"- **Overall Risk Score:** **{self.overall_risk_score * 100:.2f}%** ({self.risk_tier} RISK)",
            f"- **Evidence Items Analyzed:** {len(self.evidence_inventory)}",
            "",
            "### Modality Score Breakdown:",
            f"- Visual Evidence Suspicion (CNN): {self.model_predictions.get('cnn_score', 0.0) * 100:.2f}%",
            f"- Conversation/Text Suspicion (BiLSTM): {self.model_predictions.get('bilstm_score', 0.0) * 100:.2f}%",
            f"- Embedded URL Risk: {self.model_predictions.get('url_score', 0.0) * 100:.2f}%",
            f"- Forensic Keyword Weight: {self.model_predictions.get('keyword_score', 0.0) * 100:.2f}%",
            f"- Attachment Threat Indicator: {self.model_predictions.get('attachment_score', 0.0) * 100:.2f}%",
            f"- Timeline Anomaly Index: {self.model_predictions.get('timeline_anomaly', 0.0) * 100:.2f}%",
            f"- Email Structural Anomaly: {self.model_predictions.get('email_anomaly', 0.0) * 100:.2f}%",
            "",
            "## 2. Key Forensic Indicators",
        ]
        if self.key_evidence_indicators:
            for ind in self.key_evidence_indicators:
                lines.append(f"- [x] {ind}")
        else:
            lines.append("- No acute anomalous indicators detected.")

        lines.extend([
            "",
            "## 3. Chronological Evidence Timeline",
            "*Observed events reflect raw evidence records; Inferred relationships represent reconstructed sequence logic.*",
            "",
            "| Timestamp | Type | Status | Source ID | Event Description |",
            "|-----------|------|--------|-----------|-------------------|",
        ])

        for event in self.chronological_timeline:
            status_tag = f"**{event.event_type.upper()}**"
            lines.append(
                f"| {event.timestamp} | {event.risk_level.upper()} | {status_tag} | `{event.source_evidence_id}` | {event.description} |"
            )

        lines.extend([
            "",
            "## 4. Explainability & Model Attribution",
            "### Top Attributing Multimodal Features to Risk Prediction:",
        ])
        for feat, weight in sorted(self.feature_attributions.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{feat}**: attribution weight = `{weight:.4f}`")

        if self.textual_explainability:
            lines.extend([
                "",
                "### Salient Linguistic & Keyword Tokens (BiLSTM Attention):",
            ])
            for token, score in self.textual_explainability[:10]:
                lines.append(f"- Token `'{token}'`: attention score = `{score:.4f}`")

        lines.extend([
            "",
            "## 5. Model Limitations, Uncertainty & Academic Scope",
        ])
        for lim in self.limitations_and_uncertainty:
            lines.append(f"- {lim}")

        lines.extend([
            "",
            "## 6. Legal & Ethics Disclaimer",
            f"> {self.legal_disclaimer}",
            "",
            "---",
            "*Report generated by ForenSightX Multi-Agent Forensics Assistant (NNDL Academic Prototype)*",
        ])
        return "\n".join(lines)


# =====================================================================
# Agent Definitions
# =====================================================================

class BaseForensicAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @abstractmethod
    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        pass


class EvidenceIngestionAgent(BaseForensicAgent):
    """Ingests, validates, and indexes heterogeneous digital evidence."""

    def __init__(self):
        super().__init__(
            name="EvidenceIngestionAgent",
            role="Evidence Intake, Normalization & Inventory Management",
        )

    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        inventory = []
        for idx, item in enumerate(case.evidence_items):
            inventory.append({
                "item_index": idx,
                "evidence_id": item.evidence_id,
                "evidence_type": item.evidence_type,
                "timestamp": item.timestamp,
                "is_observed": item.is_observed,
                "summary": str(item.content)[:80] + ("..." if len(str(item.content)) > 80 else ""),
            })
        return {"inventory": inventory, "total_items": len(case.evidence_items)}


class TextAnalysisAgent(BaseForensicAgent):
    """Performs sequence deep-learning analysis on chats and emails via BiLSTM."""

    def __init__(self, model: ConversationBiLSTM, tokenizer: ForensicTokenizer):
        super().__init__(
            name="TextAnalysisAgent",
            role="Conversation & Email Sequence Intelligence (BiLSTM)",
        )
        self.model = model
        self.tokenizer = tokenizer

    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        text_items = [
            item for item in case.evidence_items if item.evidence_type in ["chat", "email"]
        ]
        if not text_items:
            return {"bilstm_score": 0.0, "token_importance": []}

        scores = []
        all_attributions: List[Tuple[str, float]] = []

        self.model.eval()
        for item in text_items:
            content_str = (
                item.content if isinstance(item.content, str) else str(item.content.get("body", ""))
            )
            tokens, token_ids = self.tokenizer.encode(content_str)
            token_ids_tensor = token_ids.unsqueeze(0)

            with torch.no_grad():
                pred, attn = self.model(token_ids_tensor, return_attention=True)
                score = float(pred.squeeze().item())
                scores.append(score)

            # Extract token attention
            attributions = self.model.get_token_importance(tokens, token_ids_tensor)
            all_attributions.extend(attributions)

        # Average BiLSTM score across text items
        avg_score = float(np.mean(scores)) if scores else 0.0
        # Consolidate token scores
        token_map: Dict[str, float] = {}
        for t, s in all_attributions:
            token_map[t] = max(token_map.get(t, 0.0), s)
        sorted_tokens = sorted(token_map.items(), key=lambda x: x[1], reverse=True)

        return {"bilstm_score": avg_score, "token_importance": sorted_tokens}


def _heatmap_to_base64_png(heatmap: np.ndarray) -> str:
    """Converts a (H, W) float heatmap in [0, 1] into a base64 encoded PNG data URI."""
    h = np.clip(heatmap, 0.0, 1.0)
    r = (np.clip(1.5 * h, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(1.5 * (1.0 - np.abs(h - 0.5) * 2.0), 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1.5 * (1.0 - h), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    im = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class VisualAnalysisAgent(BaseForensicAgent):
    """Analyzes image/screenshot evidence via CNN and computes Grad-CAM explainability."""

    def __init__(self, model: VisualEvidenceCNN):
        super().__init__(
            name="VisualAnalysisAgent",
            role="Visual Evidence & Tampering Classifier (CNN)",
        )
        self.model = model

    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        img_items = [
            item for item in case.evidence_items if item.evidence_type == "image"
        ]
        if not img_items:
            return {"cnn_score": 0.0, "gradcam_summary": None}

        scores = []
        gradcam_results = {}

        self.model.eval()
        for item in img_items:
            img = item.content
            if isinstance(img, str):  # File path
                try:
                    img = Image.open(img).convert("RGB")
                except Exception:
                    continue
            elif not isinstance(img, Image.Image):
                continue

            arr = np.array(img.resize((128, 128)), dtype=np.float32) / 255.0
            tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)

            with torch.no_grad():
                pred = self.model(tensor)
                score = float(pred.squeeze().item())
                scores.append(score)

            # Grad-CAM explanation
            gradcam_map = self.model.get_gradcam(tensor)
            gradcam_results[item.evidence_id] = {
                "max_activation": float(np.max(gradcam_map)),
                "mean_activation": float(np.mean(gradcam_map)),
                "salient_patch_detected": bool(np.max(gradcam_map) > 0.65),
                "heatmap_b64": _heatmap_to_base64_png(gradcam_map),
            }

        avg_cnn = float(np.mean(scores)) if scores else 0.0
        return {"cnn_score": avg_cnn, "gradcam_summary": gradcam_results}


class MetadataAnalysisAgent(BaseForensicAgent):
    """Extracts non-visual and structural indicators: URLs, keywords, attachments, emails, timestamps."""

    def __init__(self):
        super().__init__(
            name="MetadataAnalysisAgent",
            role="Forensic Indicators & Metadata Heuristics Extractor",
        )

    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        all_texts = []
        all_urls = []
        all_attachments = []
        timestamps = []
        email_anomalies = []

        for item in case.evidence_items:
            # Parse timestamps
            try:
                dt = datetime.fromisoformat(item.timestamp)
                timestamps.append(dt)
            except Exception:
                pass

            if item.evidence_type in ["chat", "email"]:
                content_str = (
                    item.content if isinstance(item.content, str) else str(item.content.get("body", ""))
                )
                all_texts.append(content_str)
                if isinstance(item.content, dict):
                    email_anomalies.append(extract_email_anomaly(item.content))
                    for att in item.content.get("attachments", []):
                        all_attachments.append(att)

            if "attachments" in item.metadata:
                all_attachments.extend(item.metadata["attachments"])
            if "urls" in item.metadata:
                all_urls.extend(item.metadata["urls"])

        full_corpus = " ".join(all_texts)
        kw_score = extract_keyword_score(full_corpus)
        url_score = extract_url_score(all_urls if all_urls else full_corpus)
        att_score = extract_attachment_score(all_attachments)
        time_anom = extract_timeline_anomaly(timestamps)
        email_anom = float(np.mean(email_anomalies)) if email_anomalies else 0.0

        return {
            "keyword_score": kw_score,
            "url_score": url_score,
            "attachment_score": att_score,
            "timeline_anomaly": time_anom,
            "email_anomaly": email_anom,
        }


class FusionAgent(BaseForensicAgent):
    """
    Multimodal Feature Fusion & Risk Scoring via ANN.
    Assembles the authoritative 7-dimensional feature vector:
      [cnn_score, bilstm_score, url_score, keyword_score, attachment_score, timeline_anomaly, email_anomaly]
    """

    def __init__(self, model: ANNMultimodalRiskClassifier):
        super().__init__(
            name="FusionAgent",
            role="Multimodal Feature Fusion & Overall Risk Classifier (ANN)",
        )
        self.model = model

    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        cnn_score = float(context.get("cnn_score", 0.0))
        bilstm_score = float(context.get("bilstm_score", 0.0))
        url_score = float(context.get("url_score", 0.0))
        kw_score = float(context.get("keyword_score", 0.0))
        att_score = float(context.get("attachment_score", 0.0))
        time_anom = float(context.get("timeline_anomaly", 0.0))
        email_anom = float(context.get("email_anomaly", 0.0))

        feat_vector_7d = [
            cnn_score,
            bilstm_score,
            url_score,
            kw_score,
            att_score,
            time_anom,
            email_anom,
        ]

        feat_tensor = torch.tensor(feat_vector_7d, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            risk_pred = self.model(feat_tensor)
            overall_risk = float(risk_pred.squeeze().item())

        attributions = self.model.explain_features(feat_tensor)

        # Risk tiering
        if overall_risk >= 0.75:
            tier = "HIGH"
        elif overall_risk >= 0.50:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return {
            "feature_vector_7d": feat_vector_7d,
            "overall_risk_score": overall_risk,
            "risk_tier": tier,
            "feature_attributions": attributions,
        }


class TimelineAgent(BaseForensicAgent):
    """Reconstructs timestamped events chronologically and distinguishes observed from inferred relationships."""

    def __init__(self):
        super().__init__(
            name="TimelineAgent",
            role="Chronological Timeline Reconstruction & Sequence Verification",
        )

    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        events: List[TimelineEvent] = []

        for item in case.evidence_items:
            # Determine risk indicator of individual item
            desc = ""
            risk = "low"
            if item.evidence_type == "chat":
                desc = f"Chat communication recorded: '{str(item.content)[:60]}...'"
                if extract_keyword_score(str(item.content)) > 0.4:
                    risk = "high"
            elif item.evidence_type == "email":
                subj = item.content.get("subject", "No Subject") if isinstance(item.content, dict) else "Email"
                desc = f"Email event logged: Subject '{subj}'"
                if context.get("email_anomaly", 0.0) > 0.4 or context.get("url_score", 0.0) > 0.4:
                    risk = "high"
                else:
                    risk = "medium"
            elif item.evidence_type == "image":
                desc = f"Visual artifact/screenshot recorded (`{item.evidence_id}`)"
                if context.get("cnn_score", 0.0) > 0.5:
                    risk = "high"
            elif item.evidence_type == "metadata":
                desc = f"System metadata event: {item.metadata.get('action', 'Access log')}"

            events.append(
                TimelineEvent(
                    timestamp=item.timestamp,
                    description=desc,
                    source_evidence_id=item.evidence_id,
                    event_type="observed" if item.is_observed else "inferred",
                    risk_level=risk,
                    details={"evidence_type": item.evidence_type},
                )
            )

        # Add inferred sequence relationship if rapid events or suspicious sequence occurs
        sorted_events = sorted(events, key=lambda x: x.timestamp)
        if len(sorted_events) >= 2:
            try:
                t1 = datetime.fromisoformat(sorted_events[0].timestamp)
                t2 = datetime.fromisoformat(sorted_events[-1].timestamp)
                inferred_event = TimelineEvent(
                    timestamp=sorted_events[-1].timestamp,
                    description=f"Inferred Incident Trajectory: span of {(t2 - t1).total_seconds():.1f}s between initiation and final logged event.",
                    source_evidence_id="INFERRED-MAS-01",
                    event_type="inferred",
                    risk_level="high" if context.get("overall_risk_score", 0.0) > 0.5 else "low",
                )
                sorted_events.append(inferred_event)
            except Exception:
                pass

        return {"timeline": sorted_events}


class ExplainabilityAgent(BaseForensicAgent):
    """Compiles explainability indicators, confidence disclosures, and academic/legal disclaimers."""

    LEGAL_DISCLAIMER = (
        "DECISION SUPPORT NOTICE: ForenSightX is an academic research prototype designed for "
        "investigative decision support. The risk scores, classifications, and explanations generated "
        "by deep learning models (CNN, BiLSTM, ANN) do not constitute legal certainty, binding evidence, "
        "or automated declarations of guilt. Final investigative conclusions must be verified by certified "
        "human forensic examiners."
    )

    def __init__(self):
        super().__init__(
            name="ExplainabilityAgent",
            role="Forensic Explainability, Indicator Aggregator & Ethics Gate",
        )

    def process(self, case: ForensicCase, context: Dict[str, Any]) -> Dict[str, Any]:
        indicators = []
        if context.get("bilstm_score", 0.0) > 0.5:
            indicators.append("Urgency & Suspicious Language Pattern (BiLSTM)")
        if context.get("keyword_score", 0.0) > 0.35:
            indicators.append("Credential / OTP / Wire Request Keywords Detected")
        if context.get("url_score", 0.0) > 0.3:
            indicators.append("Suspicious URL / Phishing Domain Target")
        if context.get("cnn_score", 0.0) > 0.5:
            indicators.append("Visual Tampering / Suspicious Image Artifacts (CNN)")
        if context.get("attachment_score", 0.0) > 0.4:
            indicators.append("High-Risk Attachment File Extension")
        if context.get("timeline_anomaly", 0.0) > 0.4:
            indicators.append("Abnormal Rapid Event Clustering / Sequence Anomaly")
        if context.get("email_anomaly", 0.0) > 0.4:
            indicators.append("Email Header / SPF / Display Spoofing Discrepancy")

        limitations = [
            "Neural network predictions are probabilistic estimates conditioned on training corpus distribution.",
            "Visual CNN analysis may show false alarms on novel document templates or low-resolution scans.",
            "BiLSTM sequence classification may be influenced by domain-specific jargon or rare acronyms.",
            "Feature fusion ANN synthesizes multi-source heuristic and deep-learning indicators but requires corroboration.",
        ]

        return {
            "key_indicators": indicators,
            "limitations": limitations,
            "legal_disclaimer": self.LEGAL_DISCLAIMER,
        }


# =====================================================================
# Multi-Agent Coordinator
# =====================================================================

class CaseCoordinatorAgent:
    """
    Master orchestrator for the ForenSightX Multi-Agent System (MAS).
    Coordinates ingestion, modality analyses, fusion, timeline, explainability, and report authoring.
    """

    def __init__(
        self,
        cnn_model: VisualEvidenceCNN,
        bilstm_model: ConversationBiLSTM,
        ann_model: ANNMultimodalRiskClassifier,
        tokenizer: ForensicTokenizer,
    ):
        self.ingestion_agent = EvidenceIngestionAgent()
        self.text_agent = TextAnalysisAgent(bilstm_model, tokenizer)
        self.visual_agent = VisualAnalysisAgent(cnn_model)
        self.metadata_agent = MetadataAnalysisAgent()
        self.fusion_agent = FusionAgent(ann_model)
        self.timeline_agent = TimelineAgent()
        self.explainability_agent = ExplainabilityAgent()

    def investigate(self, case: ForensicCase) -> InvestigationReport:
        context: Dict[str, Any] = {}

        # 1. Ingestion
        ingestion_res = self.ingestion_agent.process(case, context)
        context.update(ingestion_res)

        # 2. Modality Deep Learning & Metadata Analysis
        text_res = self.text_agent.process(case, context)
        visual_res = self.visual_agent.process(case, context)
        meta_res = self.metadata_agent.process(case, context)

        context.update(text_res)
        context.update(visual_res)
        context.update(meta_res)

        # 3. Multimodal Feature Fusion (7D ANN)
        fusion_res = self.fusion_agent.process(case, context)
        context.update(fusion_res)

        # 4. Timeline Reconstruction
        timeline_res = self.timeline_agent.process(case, context)
        context.update(timeline_res)

        # 5. Explainability & Disclaimers
        explain_res = self.explainability_agent.process(case, context)
        context.update(explain_res)

        # 6. Structured Final Report
        report = InvestigationReport(
            case_id=case.case_id,
            title=case.title,
            investigator=case.investigator,
            timestamp_generated=datetime.now().isoformat(),
            evidence_inventory=context["inventory"],
            model_predictions={
                "cnn_score": context.get("cnn_score", 0.0),
                "bilstm_score": context.get("bilstm_score", 0.0),
                "url_score": context.get("url_score", 0.0),
                "keyword_score": context.get("keyword_score", 0.0),
                "attachment_score": context.get("attachment_score", 0.0),
                "timeline_anomaly": context.get("timeline_anomaly", 0.0),
                "email_anomaly": context.get("email_anomaly", 0.0),
            },
            feature_vector_7d=context["feature_vector_7d"],
            overall_risk_score=context["overall_risk_score"],
            risk_tier=context["risk_tier"],
            chronological_timeline=context["timeline"],
            key_evidence_indicators=context["key_indicators"],
            feature_attributions=context["feature_attributions"],
            visual_explainability=context.get("gradcam_summary") or {},
            textual_explainability=context.get("token_importance", []),
            limitations_and_uncertainty=context["limitations"],
            legal_disclaimer=context["legal_disclaimer"],
        )

        return report

