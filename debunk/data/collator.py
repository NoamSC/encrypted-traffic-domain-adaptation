from __future__ import annotations

from typing import Dict, List

import torch


class TrafficCollator:
    """Pads sequences to max length in batch."""

    def __init__(self, pad_value: int = 0) -> None:
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(lengths) > 0 else 0
        input_ids = []
        for b in batch:
            ids = b["input_ids"]
            if ids.ndim == 1:
                ids = ids
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_value, dtype=ids.dtype)])
            input_ids.append(ids)
        input_ids = torch.stack(input_ids, dim=0)
        labels = torch.stack([b["label"] for b in batch], dim=0)
        domains = torch.stack([b["domain"] for b in batch], dim=0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "domains": domains,
            "lengths": lengths,
        }


