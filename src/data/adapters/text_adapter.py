"""
ForenSightX - Text / Email Evidence Dataset Adapter
Adapter for external text forensics datasets (e.g., Enron, SpamAssassin, SMS-Phishing, or JSONL/CSV corpora).
"""

import hashlib
import json
import os
import pandas as pd
from typing import List, Optional

from src.data.adapters.base import BaseDatasetAdapter, ForensicRawRecord


class TextForensicAdapter(BaseDatasetAdapter):
    """
    Adapter for text and email evidence datasets.
    """

    DEFAULT_PATH = os.path.join("data", "raw", "text")

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.environ.get("FORENSIGHTX_TEXT_DATA_DIR", self.DEFAULT_PATH)

    @property
    def dataset_name(self) -> str:
        return "RawTextForensicDataset"

    def is_available(self) -> bool:
        if os.path.isfile(self.data_path):
            return True
        if os.path.isdir(self.data_path):
            valid_exts = {".csv", ".json", ".jsonl", ".txt"}
            for root, _, files in os.walk(self.data_path):
                if any(os.path.splitext(f)[1].lower() in valid_exts for f in files):
                    return True
        return False

    def load_records(self) -> List[ForensicRawRecord]:
        if not self.is_available():
            raise FileNotFoundError(
                f"Text dataset not available at {self.data_path}. Falling back to SMOKE mode."
            )

        records: List[ForensicRawRecord] = []
        # Support CSV, JSONL, or raw text directories
        if os.path.isfile(self.data_path):
            records.extend(self._load_file(self.data_path))
        else:
            for root, _, files in os.walk(self.data_path):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in {".csv", ".json", ".jsonl", ".txt"}:
                        records.extend(self._load_file(os.path.join(root, f)))
        return records

    def _load_file(self, file_path: str) -> List[ForensicRawRecord]:
        records: List[ForensicRawRecord] = []
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(file_path)
            text_col = "text" if "text" in df.columns else df.columns[0]
            label_col = "label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
            for idx, row in df.iterrows():
                content = str(row[text_col])
                lbl = int(row[label_col]) if label_col else 0
                chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                records.append(
                    ForensicRawRecord(
                        record_id=f"TXT-{chash[:12]}",
                        modality="text",
                        content=content,
                        label=lbl,
                        metadata={"source_file": file_path, "row": idx},
                        content_hash=chash,
                    )
                )
        elif ext in {".json", ".jsonl"}:
            with open(file_path, "r", encoding="utf-8") as fp:
                for idx, line in enumerate(fp):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    content = item.get("text", item.get("body", str(item)))
                    lbl = int(item.get("label", 0))
                    chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    records.append(
                        ForensicRawRecord(
                            record_id=f"TXT-{chash[:12]}",
                            modality="text",
                            content=content,
                            label=lbl,
                            metadata={"source_file": file_path, "line": idx},
                            content_hash=chash,
                        )
                    )
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
            chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            lbl = 1 if any(k in file_path.lower() for k in ["spam", "phish", "suspicious"]) else 0
            records.append(
                ForensicRawRecord(
                    record_id=f"TXT-{chash[:12]}",
                    modality="text",
                    content=content,
                    label=lbl,
                    metadata={"source_file": file_path},
                    content_hash=chash,
                )
            )
        return records

