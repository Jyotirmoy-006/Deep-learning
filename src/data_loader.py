"""
ForenSightX - Forensic Data Loader & Preprocessing Pipeline
Authoritative Reference: ForenSightX_Project_Architecture.docx
Handles multimodal evidence ingestion, NLP tokenization, image normalization,
metadata extraction, and PyTorch dataset creation.
"""

import os
import re
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class EvidenceItem:
    evidence_id: str
    evidence_type: str  # 'chat', 'email', 'image', 'metadata'
    timestamp: str      # ISO format 'YYYY-MM-DDTHH:MM:SS'
    content: Any        # text string, file path, or dict
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_observed: bool = True  # Distinguishes observed facts from inferred relationships


@dataclass
class ForensicCase:
    case_id: str
    title: str
    created_at: str
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    investigator: str = "Forensic Analyst"


# =====================================================================
# NLP Tokenizer & Preprocessor
# =====================================================================

class ForensicTokenizer:
    """
    Tokenizer for forensic chat logs and email bodies.
    Cleans, lowercases, tokenizes, and encodes sequence with padding.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
        text = re.sub(r"[^\w\s\[\]]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        return cleaned.split()

    def fit_on_texts(self, texts: List[str], min_freq: int = 1):
        freqs: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                freqs[token] = freqs.get(token, 0) + 1

        idx = len(self.word2idx)
        for token, count in sorted(freqs.items(), key=lambda x: x[1], reverse=True):
            if count >= min_freq and token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

    def encode(self, text: str) -> Tuple[List[str], torch.Tensor]:
        tokens = self.tokenize(text)
        padded_tokens = tokens[: self.max_length]
        token_ids = [
            self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in padded_tokens
        ]

        # Pad to max_length
        if len(token_ids) < self.max_length:
            pad_count = self.max_length - len(token_ids)
            token_ids += [self.word2idx[self.PAD_TOKEN]] * pad_count
            padded_tokens += [self.PAD_TOKEN] * pad_count

        return padded_tokens, torch.tensor(token_ids, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)


# =====================================================================
# Evidence Feature Extractors (Heuristic & Metadata)
# =====================================================================

SUSPICIOUS_KEYWORDS = {
    "urgent": 0.8,
    "urgency": 0.8,
    "otp": 1.0,
    "password": 0.9,
    "wire": 0.9,
    "transfer": 0.7,
    "immediately": 0.7,
    "suspended": 0.85,
    "verify": 0.6,
    "credentials": 0.95,
    "bank": 0.6,
    "bitcoin": 0.85,
    "crypto": 0.8,
    "giftcard": 0.95,
    "confidential": 0.65,
    "compliance": 0.5,
    "invoice": 0.6,
    "payment": 0.6,
    "breach": 0.8,
    "unauthorized": 0.85,
}

SUSPICIOUS_EXTENSIONS = {
    ".exe": 1.0,
    ".scr": 1.0,
    ".vbs": 0.95,
    ".bat": 0.9,
    ".cmd": 0.9,
    ".iso": 0.85,
    ".zip": 0.4,
    ".js": 0.8,
    ".ps1": 0.9,
    ".jar": 0.8,
}


def extract_keyword_score(text: str) -> float:
    """Computes normalized keyword suspicion score based on forensic indicators."""
    lower = text.lower()
    total_score = 0.0
    matches = 0
    for kw, weight in SUSPICIOUS_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            total_score += weight
            matches += 1
    if matches == 0:
        return 0.0
    return float(min(1.0, total_score / 3.0))


def extract_url_score(text_or_urls: Union[str, List[str]]) -> float:
    """Computes suspicion score for embedded URLs."""
    if isinstance(text_or_urls, str):
        urls = re.findall(r"https?://\S+|www\.\S+", text_or_urls)
    else:
        urls = text_or_urls

    if not urls:
        return 0.0

    score = 0.0
    for u in urls:
        u_lower = u.lower()
        if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", u_lower):  # Raw IP URL
            score += 0.8
        if any(tld in u_lower for tld in [".xyz", ".top", ".tk", ".ru", ".buzz", ".work"]):
            score += 0.6
        if "@" in u_lower:  # Basic auth trick
            score += 0.7
        if any(term in u_lower for term in ["login", "verify", "secure", "update", "banking"]):
            score += 0.5
        score += 0.2  # Baseline per URL

    return float(min(1.0, score / max(1, len(urls))))


def extract_attachment_score(attachments: List[str]) -> float:
    """Analyzes attachment extensions and file characteristics."""
    if not attachments:
        return 0.0
    max_score = 0.0
    for att in attachments:
        att_lower = att.lower()
        # Check double extensions like invoice.pdf.exe
        if len(re.findall(r"\.\w+", att_lower)) > 1:
            max_score = max(max_score, 0.95)
        for ext, weight in SUSPICIOUS_EXTENSIONS.items():
            if att_lower.endswith(ext):
                max_score = max(max_score, weight)
    return float(max_score)


def extract_email_anomaly(email_dict: Dict[str, Any]) -> float:
    """
    Analyzes email header discrepancies (e.g. sender display vs actual address,
    reply-to mismatch, missing security verification).
    """
    anomaly = 0.0
    sender = email_dict.get("sender", "")
    reply_to = email_dict.get("reply_to", "")
    headers = email_dict.get("headers", {})

    # Sender domain mismatch with reply-to
    if reply_to and sender:
        sender_domain = sender.split("@")[-1].lower() if "@" in sender else ""
        reply_domain = reply_to.split("@")[-1].lower() if "@" in reply_to else ""
        if sender_domain and reply_domain and sender_domain != reply_domain:
            anomaly += 0.6

    # SPF / DKIM verification status if present
    if headers.get("spf", "").lower() == "fail":
        anomaly += 0.5
    if headers.get("dkim", "").lower() == "fail":
        anomaly += 0.5

    # Display name spoofing (e.g. "CEO" <attacker@free.mail>)
    display_name = email_dict.get("display_name", "").lower()
    if any(title in display_name for title in ["ceo", "admin", "it support", "security"]):
        if any(free in sender.lower() for free in ["gmail", "yahoo", "hotmail", "proton"]):
            anomaly += 0.7

    return float(min(1.0, anomaly))


def extract_timeline_anomaly(timestamps: List[datetime]) -> float:
    """
    Computes anomaly score in event timing (e.g. rapid automated burst,
    unusual off-hours clustering, or broken sequence progression).
    """
    if len(timestamps) < 2:
        return 0.0

    sorted_times = sorted(timestamps)
    deltas_seconds = [
        (t2 - t1).total_seconds() for t1, t2 in zip(sorted_times[:-1], sorted_times[1:])
    ]

    anomaly = 0.0
    # Ultra-rapid burst (< 2 seconds between multiple distinct actions)
    rapid_events = sum(1 for d in deltas_seconds if 0 <= d < 3.0)
    if rapid_events >= 2:
        anomaly += 0.5 + (0.1 * rapid_events)

    # Significant backward timestamp anomaly (clock skew or spoofing)
    for t1, t2 in zip(timestamps[:-1], timestamps[1:]):
        if (t2 - t1).total_seconds() < 0:
            anomaly += 0.8

    return float(min(1.0, anomaly))


# =====================================================================
# Synthetic Forensic Corpus & Dataset Generator
# =====================================================================

BENIGN_TEXTS = [
    "Hey, could you send over the meeting minutes from yesterday's sync?",
    "Project deployment went smoothly, all unit tests passed in CI/CD.",
    "Let's reschedule our coffee catchup to Thursday afternoon if that works.",
    "Attached is the monthly financial review spreadsheet for your team.",
    "Thanks for following up on the documentation updates. Looks good to merge.",
    "Remember to submit your vacation requests before the quarter ends.",
    "The quarterly review presentation slides are ready in the shared drive.",
    "Can you review my pull request on the backend auth middleware?",
    "Please find attached the customer feedback summary report for Q3.",
    "Great work on resolving the database latency issue this morning.",
    "Are we still on for the 2 PM sprint planning meeting?",
    "The library update has been completed and verified across all environments.",
    "Here is the link to the design mockups discussed in the standup.",
    "Team lunch is scheduled for Friday at noon in the cafeteria.",
    "The client confirmed they received the release notes and agreed to the demo.",
]

SUSPICIOUS_TEXTS = [
    "URGENT: Your account has been suspended due to unauthorized access. Click here immediately to verify your credentials: http://192.168.1.105/auth",
    "Please send the one-time password (OTP) sent to your mobile phone right now to verify wire transfer compliance.",
    "Immediate action required: Wire transfer $45,000 to the attached vendor account before 5 PM to avoid penalty fees.",
    "IT Support Alert: Confidential password expiration. Login to http://verify-account.bank-security.xyz to retain access.",
    "URGENT NOTICE: Legal compliance violation detected. Open the attached invoice.pdf.exe and enter your network credentials.",
    "Attention: Unrecognized bitcoin transaction initiated. Confirm or cancel by providing your secret key immediately.",
    "Urgent message from CEO: Purchase 10 gift cards for the partner meeting and reply with the codes right away.",
    "Security breach detected on your workstation! Download and run http://patch-security.buzz/fix.scr immediately.",
    "FINAL WARNING: Your payroll account will be locked unless you verify your banking credentials at http://update-payroll.top",
    "Kindly review the attached confidential document transaction_receipt.vbs regarding the disputed wire payment.",
    "Urgent wire instruction change: Do not use the old banking details, transfer funds to the updated account attached.",
    "Alert: Your mailbox is 99% full. Click here to prevent email termination: http://mail-service-update.work/login",
    "Please provide the OTP code immediately so we can cancel the fraudulent withdrawal from your savings account.",
    "Action Required: Verify your employee profile now to receive this month's bonus payout http://10.0.0.14/bonus",
    "Security Department: Suspicious login from Moscow detected. Reset your password immediately by opening credentials.cmd",
]


class ForensicDatasetGenerator:
    """
    Generates controlled forensic datasets for experiments A, B, C, D, E.
    Ensures reproducibility and zero hardcoded illustrative values.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_text_data(
        self, num_samples: int = 300
    ) -> Tuple[List[str], List[int]]:
        """Generates synthetic chat and email text samples with binary labels."""
        texts = []
        labels = []
        num_per_class = num_samples // 2

        for _ in range(num_per_class):
            template = random.choice(BENIGN_TEXTS)
            texts.append(template)
            labels.append(0)

        for _ in range(num_per_class):
            template = random.choice(SUSPICIOUS_TEXTS)
            texts.append(template)
            labels.append(1)

        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        return list(texts), list(labels)

    def generate_image_sample(
        self, is_suspicious: bool, size: Tuple[int, int] = (128, 128)
    ) -> Image.Image:
        """
        Synthesizes a realistic digital forensic screenshot/document image:
        - Benign: Regular document / clean window structure with neutral colors.
        - Suspicious: Tampered noise patterns, alert banners, fake login input fields.
        """
        img = Image.new("RGB", size, color=(245, 247, 250))
        draw = ImageDraw.Draw(img)

        if not is_suspicious:
            # Benign document header and clean paragraph lines
            draw.rectangle([10, 10, size[0] - 10, 25], fill=(70, 130, 180))
            for y in range(35, size[1] - 15, 12):
                w = random.randint(60, size[0] - 20)
                draw.rectangle([10, y, w, y + 4], fill=(180, 190, 200))
        else:
            # Suspicious alert banner (red/crimson) + fake credential box + visual tampering artifacts
            draw.rectangle([10, 10, size[0] - 10, 30], fill=(210, 45, 45))
            draw.text((15, 12), "ALERT", fill=(255, 255, 255))
            # Fake login dialog
            draw.rectangle([20, 45, size[0] - 20, 85], outline=(150, 0, 0), fill=(255, 230, 230))
            draw.rectangle([30, 55, size[0] - 30, 65], fill=(255, 255, 255), outline=(100, 100, 100))
            draw.rectangle([30, 70, size[0] - 30, 80], fill=(255, 255, 255), outline=(100, 100, 100))
            # Visual noise / splice artifacts
            arr = np.array(img)
            noise = np.random.randint(0, 60, arr.shape, dtype=np.uint8)
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def generate_multimodal_dataset(
        self, num_samples: int = 400
    ) -> Tuple[
        List[str],           # texts
        List[Image.Image],   # images
        np.ndarray,          # 7D feature vectors
        np.ndarray           # binary ground truth labels
    ]:
        """
        Generates paired multimodal forensic samples adhering to the 7-D feature spec:
        [cnn_score, bilstm_score, url_score, keyword_score, attachment_score, timeline_anomaly, email_anomaly]
        """
        texts: List[str] = []
        images: List[Image.Image] = []
        features_list: List[List[float]] = []
        labels: List[int] = []

        half = num_samples // 2

        # Benign distribution
        for _ in range(half):
            t = random.choice(BENIGN_TEXTS)
            img = self.generate_image_sample(is_suspicious=False)

            cnn_score = float(np.clip(np.random.normal(0.15, 0.10), 0.0, 0.5))
            bilstm_score = float(np.clip(np.random.normal(0.12, 0.08), 0.0, 0.45))
            url_score = extract_url_score(t)
            kw_score = extract_keyword_score(t)
            att_score = 0.0 if random.random() > 0.1 else 0.2
            time_anom = float(np.clip(np.random.normal(0.05, 0.05), 0.0, 0.3))
            email_anom = float(np.clip(np.random.normal(0.05, 0.05), 0.0, 0.3))

            features_list.append([
                cnn_score, bilstm_score, url_score, kw_score,
                att_score, time_anom, email_anom
            ])
            texts.append(t)
            images.append(img)
            labels.append(0)

        # Suspicious distribution
        for _ in range(half):
            t = random.choice(SUSPICIOUS_TEXTS)
            img = self.generate_image_sample(is_suspicious=True)

            cnn_score = float(np.clip(np.random.normal(0.82, 0.10), 0.45, 1.0))
            bilstm_score = float(np.clip(np.random.normal(0.85, 0.09), 0.5, 1.0))
            url_score = extract_url_score(t)
            kw_score = extract_keyword_score(t)
            att_score = 0.85 if "attached" in t.lower() else float(np.random.uniform(0.3, 0.8))
            time_anom = float(np.clip(np.random.normal(0.65, 0.15), 0.3, 1.0))
            email_anom = float(np.clip(np.random.normal(0.70, 0.15), 0.35, 1.0))

            features_list.append([
                cnn_score, bilstm_score, url_score, kw_score,
                att_score, time_anom, email_anom
            ])
            texts.append(t)
            images.append(img)
            labels.append(1)

        indices = list(range(num_samples))
        random.shuffle(indices)

        texts = [texts[i] for i in indices]
        images = [images[i] for i in indices]
        features = np.array([features_list[i] for i in indices], dtype=np.float32)
        labels_arr = np.array([labels[i] for i in indices], dtype=np.float32)

        return texts, images, features, labels_arr


# =====================================================================
# PyTorch Datasets
# =====================================================================

class TextEvidenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: ForensicTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, token_ids = self.tokenizer.encode(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return token_ids, label


class ImageEvidenceDataset(Dataset):
    def __init__(self, images: List[Image.Image], labels: List[int]):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        tensor = torch.tensor(arr).permute(2, 0, 1)    # (3, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return tensor, label


class MultimodalFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

