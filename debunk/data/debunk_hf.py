from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


class DebunkHFLoader:
    """Wraps HuggingFace datasets.load_dataset for Debunk representation.

    Expects splits with columns at least: payload_hex, label, domain.
    """

    def __init__(self, hf_id: str, revision: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        self.hf_id = hf_id
        self.revision = revision
        self.cache_dir = cache_dir

    def load(self) -> Dict[str, any]:
        ds = load_dataset(self.hf_id, revision=self.revision, cache_dir=self.cache_dir)
        return ds


