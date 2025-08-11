from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficSample:
    def __init__(self, tokens: List[int], label: int, domain: int) -> None:
        self.tokens = tokens
        self.label = label
        self.domain = domain


class TrafficDataset(Dataset):
    """Simple dataset for tokenized traffic samples."""

    def __init__(self, samples: List[TrafficSample], max_len: int) -> None:
        self.samples = samples
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        seq = np.array(s.tokens[: self.max_len], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(seq),
            "label": torch.tensor(s.label, dtype=torch.long),
            "domain": torch.tensor(s.domain, dtype=torch.long),
            "length": torch.tensor(len(seq), dtype=torch.long),
        }

    def num_classes(self) -> int:
        if not self.samples:
            return 1
        max_label = max(s.label for s in self.samples)
        return int(max_label) + 1


