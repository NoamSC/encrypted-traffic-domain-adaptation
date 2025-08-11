from __future__ import annotations

from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from ..registry import DATASETS, COLLATORS
from .hf_tls_early2late import build_tls_early2late_splits


class SplitManager:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def build_loaders(self) -> Dict[str, DataLoader]:
        data_cfg = self.cfg.get("data", {})
        source_type = data_cfg.get("source", {}).get("type", "toy")
        collator_fn = COLLATORS.get("traffic")
        if source_type == "hf" and data_cfg.get("shift") == "tls_early2late":
            datasets = build_tls_early2late_splits(self.cfg)
            train_src = datasets["src_train"]
            val_src = datasets["src_val"]
            test_src = datasets["src_test"]
            unlabeled_tgt = datasets["tgt_unlabeled"]
            test_tgt = datasets["tgt_test"]
        else:
            dataset_fn = DATASETS.get(source_type)
            train_src = dataset_fn(self.cfg, split="train_source")
            val_src = dataset_fn(self.cfg, split="val_source")
            test_src = dataset_fn(self.cfg, split="test_source")
            unlabeled_tgt = dataset_fn(self.cfg, split="unlabeled_target")
            test_tgt = dataset_fn(self.cfg, split="test_target")
        collator = collator_fn(self.cfg)
        num_workers = int(self.cfg.get("system", {}).get("num_workers", 0))
        pin_memory = bool(self.cfg.get("system", {}).get("pin_memory", False))
        batch_train = int(self.cfg.get("data", {}).get("batch_sizes", {}).get("train", 64))
        batch_eval = int(self.cfg.get("data", {}).get("batch_sizes", {}).get("eval", 128))
        loaders = {
            "src_train": DataLoader(train_src, batch_size=batch_train, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collator),
            "src_val": DataLoader(val_src, batch_size=batch_eval, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collator),
            "src_test": DataLoader(test_src, batch_size=batch_eval, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collator),
            "tgt_unlabeled": DataLoader(unlabeled_tgt, batch_size=batch_train, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collator),
            "tgt_test": DataLoader(test_tgt, batch_size=batch_eval, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collator),
        }
        return loaders


