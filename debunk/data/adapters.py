from __future__ import annotations

from typing import Dict, Iterable, List

from .tokenizer import FeatureTokenizer


class DebunkAdapter:
    """Converts raw rows to token sequences using FeatureTokenizer."""

    def __init__(self, tokenizer: FeatureTokenizer) -> None:
        self.tokenizer = tokenizer

    def to_tokens(self, row: Dict) -> List[int]:
        return self.tokenizer.encode(row)


