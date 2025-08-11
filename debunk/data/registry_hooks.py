from __future__ import annotations

from typing import Any, Dict, List

import os
import pandas as pd
from datasets import load_dataset

from ..registry import DATASETS, COLLATORS
from .dataset import TrafficDataset, TrafficSample
from .collator import TrafficCollator
from .tokenizer import FeatureTokenizer


@COLLATORS.register("traffic")
def build_collator(_: Dict[str, Any]) -> TrafficCollator:
    return TrafficCollator(pad_value=0)


@DATASETS.register("toy")
def build_toy_dataset(cfg: Dict[str, Any], split: str) -> TrafficDataset:
    base = os.path.join("toy_data", f"{split}.csv")
    if not os.path.exists(base):
        # fallback to source naming
        base = os.path.join("toy_data", f"{split}_source.csv")
    df = pd.read_csv(base)
    tokenizer = FeatureTokenizer(preset=cfg.get("tokenizer", {}).get("preset", "debunk_v1"))
    samples: List[TrafficSample] = []
    for _, row in df.iterrows():
        tokens = tokenizer.encode(row.to_dict())
        label = int(row.get("label", 0))
        domain = int(row.get("domain", 0))
        samples.append(TrafficSample(tokens=tokens, label=label, domain=domain))
    max_len = int(cfg.get("data", {}).get("preprocessing", {}).get("max_len", 256))
    return TrafficDataset(samples, max_len=max_len)


@DATASETS.register("local")
def build_local_dataset(cfg: Dict[str, Any], split: str) -> TrafficDataset:
    cache_dir = cfg.get("data", {}).get("source", {}).get("cache_dir", "")
    # Allow mapping our logical split names to common filenames
    name_map = {
        "train_source": "train",
        "val_source": "validation",
        "test_source": "test",
        "unlabeled_target": "unlabeled_target",
        "test_target": "test_target",
    }
    base_name = name_map.get(split, split)
    def _load_csv_or_parquet(name: str) -> pd.DataFrame | None:
        csv_path = os.path.join(cache_dir, f"{name}.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        pq_path = os.path.join(cache_dir, f"{name}.parquet")
        if os.path.exists(pq_path):
            return pd.read_parquet(pq_path)
        return None

    df = _load_csv_or_parquet(base_name)
    # Fallbacks for DA splits if only generic splits exist
    if df is None and base_name in {"unlabeled_target", "test_target"}:
        df = _load_csv_or_parquet("test")
    if df is None and base_name in {"train_source", "val_source", "test_source"}:
        # Should already have been mapped to train/validation/test above, but keep safe fallback
        alt = {"train_source": "train", "val_source": "validation", "test_source": "test"}[base_name]
        df = _load_csv_or_parquet(alt)
    if df is None:
        raise FileNotFoundError(f"Local dataset split not found: {split} (mapped {base_name}) in {cache_dir}")
    tokenizer = FeatureTokenizer(preset=cfg.get("tokenizer", {}).get("preset", "debunk_v1"))
    samples: List[TrafficSample] = []
    for _, row in df.iterrows():
        tokens = tokenizer.encode(row.to_dict())
        label = int(row.get("label", 0))
        domain = int(row.get("domain", 0))
        samples.append(TrafficSample(tokens=tokens, label=label, domain=domain))
    max_len = int(cfg.get("data", {}).get("preprocessing", {}).get("max_len", 256))
    return TrafficDataset(samples, max_len=max_len)


@DATASETS.register("hf")
def build_hf_dataset(cfg: Dict[str, Any], split: str) -> TrafficDataset:
    src_cfg = cfg.get("data", {}).get("source", {})
    hf_id = src_cfg.get("hf_id")
    revision = src_cfg.get("hf_revision")
    cache_dir = src_cfg.get("cache_dir")
    if not hf_id:
        raise ValueError("data.source.hf_id must be set for hf source")
    ds = load_dataset(hf_id, revision=revision, cache_dir=cache_dir)
    # Try a few common mappings
    mapping = {
        "train_source": "train",
        "val_source": "validation",
        "test_source": "test",
        "unlabeled_target": "unlabeled_target" if "unlabeled_target" in ds else "test",
        "test_target": "test_target" if "test_target" in ds else "test",
    }
    split_name = mapping.get(split, split)
    if split_name not in ds:
        raise KeyError(f"Split {split} (resolved {split_name}) not in HF dataset {hf_id}")
    hf_split = ds[split_name]
    tokenizer = FeatureTokenizer(preset=cfg.get("tokenizer", {}).get("preset", "debunk_v1"))
    samples: List[TrafficSample] = []
    for row in hf_split:
        tokens = tokenizer.encode(row)
        label = int(row.get("label", 0))
        domain = int(row.get("domain", 0))
        samples.append(TrafficSample(tokens=tokens, label=label, domain=domain))
    max_len = int(cfg.get("data", {}).get("preprocessing", {}).get("max_len", 256))
    return TrafficDataset(samples, max_len=max_len)


