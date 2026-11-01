"""
ForenSightX - Base Dataset Adapter
Defines interface for ingestion of external/raw forensic datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ForensicRawRecord:
    record_id: str
    modality: str  # 'text', 'email', 'image'
    content: Any   # text string, file path, PIL Image
    label: int     # 0 = benign, 1 = suspicious/fraudulent
    metadata: Dict[str, Any]
    content_hash: str


class BaseDatasetAdapter(ABC):
    """Abstract adapter interface for evidence ingestion."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the physical dataset files exist on disk."""
        pass

    @abstractmethod
    def load_records(self) -> List[ForensicRawRecord]:
        """Loads and parses records into unified format."""
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset source."""
        pass

