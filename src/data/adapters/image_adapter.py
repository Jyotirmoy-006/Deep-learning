"""
ForenSightX - Image Evidence Dataset Adapter
Adapter for external image forensics datasets (screenshots, tampered documents, splice artifacts).
"""

import hashlib
import os
from typing import List, Optional
from PIL import Image

from src.data.adapters.base import BaseDatasetAdapter, ForensicRawRecord


class ImageForensicAdapter(BaseDatasetAdapter):
    """
    Adapter for real-world visual evidence datasets (e.g., CASIA, Defacto, or forensic screenshots).
    Checks path specified via environment variable or default directory.
    """

    DEFAULT_PATH = os.path.join("data", "raw", "images")

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.environ.get("FORENSIGHTX_IMAGE_DATA_DIR", self.DEFAULT_PATH)

    @property
    def dataset_name(self) -> str:
        return "RawVisualForensicDataset"

    def is_available(self) -> bool:
        if not os.path.isdir(self.data_dir):
            return False
        # Check if contains valid image files
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        for root, _, files in os.walk(self.data_dir):
            if any(os.path.splitext(f)[1].lower() in valid_exts for f in files):
                return True
        return False

    def load_records(self) -> List[ForensicRawRecord]:
        if not self.is_available():
            raise FileNotFoundError(
                f"Image dataset not available at {self.data_dir}. Falling back to SMOKE mode."
            )

        records: List[ForensicRawRecord] = []
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for root, _, files in os.walk(self.data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    file_path = os.path.join(root, f)
                    with open(file_path, "rb") as fp:
                        content_hash = hashlib.sha256(fp.read()).hexdigest()

                    # Label inference from subfolder name (e.g. 'suspicious' / 'benign' or 'tampered' / 'authentic')
                    rel_dir = os.path.relpath(root, self.data_dir).lower()
                    label = 1 if any(k in rel_dir or k in f.lower() for k in ["suspicious", "tampered", "fake", "forged"]) else 0

                    records.append(
                        ForensicRawRecord(
                            record_id=f"IMG-{content_hash[:12]}",
                            modality="image",
                            content=file_path,
                            label=label,
                            metadata={"file_path": file_path, "format": ext},
                            content_hash=content_hash,
                        )
                    )
        return records

