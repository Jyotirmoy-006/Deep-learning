"""
ForenSightX - Case Builder & Stratified Manifest Pipeline
Authoritative Spec Reference: ForenSightX_Project_Architecture.docx Section 4, 13
Implements:
  - Config-driven multimodal pairing (label-conditioned pairing as approved default)
  - Stratified seeded splits (Train / Val / Test)
  - Split manifest generation with SHA-256 integrity verification
  - Zero-leakage verification (no sample ID or content hash overlap)
  - Class distribution and imbalance ratio auditing
"""

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.data.adapters.base import ForensicRawRecord


@dataclass
class MultimodalCaseItem:
    case_id: str
    text_content: str
    text_hash: str
    image_content: Any  # File path or PIL Image
    image_hash: str
    label: int          # 0 = Benign, 1 = Suspicious
    metadata: Dict[str, Any]
    content_hash: str


class CaseBuilder:
    """
    Assembles multimodal cases from raw text and image evidence streams.
    Default Pairing Strategy: 'label_conditioned'
      (Pairs one image record and one text record with the same class label).
      *Limitation Documented:* Cross-modality dependency is simulated under homogeneous class assignment.
    """

    def __init__(
        self,
        pairing_strategy: str = "label_conditioned",
        seed: int = 42,
    ):
        self.pairing_strategy = pairing_strategy
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def build_cases(
        self,
        text_records: List[ForensicRawRecord],
        image_records: List[ForensicRawRecord],
    ) -> List[MultimodalCaseItem]:
        """Pairs text and image records according to configured strategy."""
        if self.pairing_strategy == "label_conditioned":
            return self._build_label_conditioned(text_records, image_records)
        elif self.pairing_strategy == "random":
            return self._build_random_pairing(text_records, image_records)
        else:
            raise ValueError(f"Unknown pairing strategy: {self.pairing_strategy}")

    def _build_label_conditioned(
        self,
        text_records: List[ForensicRawRecord],
        image_records: List[ForensicRawRecord],
    ) -> List[MultimodalCaseItem]:
        texts_0 = [r for r in text_records if r.label == 0]
        texts_1 = [r for r in text_records if r.label == 1]
        imgs_0 = [r for r in image_records if r.label == 0]
        imgs_1 = [r for r in image_records if r.label == 1]

        cases: List[MultimodalCaseItem] = []
        n_0 = min(len(texts_0), len(imgs_0))
        n_1 = min(len(texts_1), len(imgs_1))

        # Benign pairs
        for i in range(n_0):
            t = texts_0[i]
            img = imgs_0[i]
            combined_hash = hashlib.sha256(
                f"{t.content_hash}-{img.content_hash}".encode("utf-8")
            ).hexdigest()
            cases.append(
                MultimodalCaseItem(
                    case_id=f"CASE-0-{i:04d}-{combined_hash[:8]}",
                    text_content=str(t.content),
                    text_hash=t.content_hash,
                    image_content=img.content,
                    image_hash=img.content_hash,
                    label=0,
                    metadata={"pairing_strategy": "label_conditioned", "class": "benign"},
                    content_hash=combined_hash,
                )
            )

        # Suspicious pairs
        for i in range(n_1):
            t = texts_1[i]
            img = imgs_1[i]
            combined_hash = hashlib.sha256(
                f"{t.content_hash}-{img.content_hash}".encode("utf-8")
            ).hexdigest()
            cases.append(
                MultimodalCaseItem(
                    case_id=f"CASE-1-{i:04d}-{combined_hash[:8]}",
                    text_content=str(t.content),
                    text_hash=t.content_hash,
                    image_content=img.content,
                    image_hash=img.content_hash,
                    label=1,
                    metadata={"pairing_strategy": "label_conditioned", "class": "suspicious"},
                    content_hash=combined_hash,
                )
            )

        random.shuffle(cases)
        return cases

    def _build_random_pairing(
        self,
        text_records: List[ForensicRawRecord],
        image_records: List[ForensicRawRecord],
    ) -> List[MultimodalCaseItem]:
        cases: List[MultimodalCaseItem] = []
        n = min(len(text_records), len(image_records))
        shuffled_images = list(image_records)
        random.shuffle(shuffled_images)

        for i in range(n):
            t = text_records[i]
            img = shuffled_images[i]
            label = 1 if (t.label == 1 or img.label == 1) else 0
            combined_hash = hashlib.sha256(
                f"{t.content_hash}-{img.content_hash}".encode("utf-8")
            ).hexdigest()
            cases.append(
                MultimodalCaseItem(
                    case_id=f"CASE-RND-{i:04d}-{combined_hash[:8]}",
                    text_content=str(t.content),
                    text_hash=t.content_hash,
                    image_content=img.content,
                    image_hash=img.content_hash,
                    label=label,
                    metadata={"pairing_strategy": "random"},
                    content_hash=combined_hash,
                )
            )
        return cases


def create_stratified_split(
    cases: List[MultimodalCaseItem],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    manifest_path: str = "data/split_manifest.json",
) -> Tuple[List[MultimodalCaseItem], List[MultimodalCaseItem], List[MultimodalCaseItem], Dict[str, Any]]:
    """
    Splits cases into stratified seeded splits (train/val/test).
    Validates zero ID/hash leakage and records class distribution metrics.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    cases_0 = [c for c in cases if c.label == 0]
    cases_1 = [c for c in cases if c.label == 1]

    rng.shuffle(cases_0)
    rng.shuffle(cases_1)

    def split_class(items: List[MultimodalCaseItem]):
        n_total = len(items)
        n_tr = int(n_total * train_ratio)
        n_va = int(n_total * val_ratio)
        tr = items[:n_tr]
        va = items[n_tr:n_tr + n_va]
        te = items[n_tr + n_va:]
        return tr, va, te

    tr_0, va_0, te_0 = split_class(cases_0)
    tr_1, va_1, te_1 = split_class(cases_1)

    train_set = tr_0 + tr_1
    val_set = va_0 + va_1
    test_set = te_0 + te_1

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    # Integrity Assertions: Zero ID and Zero Content Hash Overlap
    train_ids = {c.case_id for c in train_set}
    val_ids = {c.case_id for c in val_set}
    test_ids = {c.case_id for c in test_set}

    assert len(train_ids.intersection(val_ids)) == 0, "FATAL: Train/Val ID overlap detected!"
    assert len(train_ids.intersection(test_ids)) == 0, "FATAL: Train/Test ID overlap detected!"
    assert len(val_ids.intersection(test_ids)) == 0, "FATAL: Val/Test ID overlap detected!"

    train_hashes = {c.content_hash for c in train_set}
    val_hashes = {c.content_hash for c in val_set}
    test_hashes = {c.content_hash for c in test_set}

    assert len(train_hashes.intersection(val_hashes)) == 0, "FATAL: Train/Val content hash leakage!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "FATAL: Train/Test content hash leakage!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "FATAL: Val/Test content hash leakage!"

    # Class Counts & Imbalance Ratio
    stats = {
        "seed": seed,
        "splits": {
            "train": {
                "total": len(train_set),
                "class_0_benign": len(tr_0),
                "class_1_suspicious": len(tr_1),
                "imbalance_ratio": len(tr_0) / max(1, len(tr_1)),
            },
            "val": {
                "total": len(val_set),
                "class_0_benign": len(va_0),
                "class_1_suspicious": len(va_1),
                "imbalance_ratio": len(va_0) / max(1, len(va_1)),
            },
            "test": {
                "total": len(test_set),
                "class_0_benign": len(te_0),
                "class_1_suspicious": len(te_1),
                "imbalance_ratio": len(te_0) / max(1, len(te_1)),
            },
        },
        "manifest_sha256": None,
    }

    # Persist Manifest
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    manifest_data = {
        "stats": stats,
        "train_ids": [c.case_id for c in train_set],
        "train_hashes": [c.content_hash for c in train_set],
        "val_ids": [c.case_id for c in val_set],
        "val_hashes": [c.content_hash for c in val_set],
        "test_ids": [c.case_id for c in test_set],
        "test_hashes": [c.content_hash for c in test_set],
    }
    encoded = json.dumps(manifest_data, indent=2).encode("utf-8")
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    stats["manifest_sha256"] = manifest_hash
    manifest_data["stats"]["manifest_sha256"] = manifest_hash

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return train_set, val_set, test_set, stats

