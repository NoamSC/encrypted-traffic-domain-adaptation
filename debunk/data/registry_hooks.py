from __future__ import annotations

from typing import Any, Dict, List, Tuple

import os
import io
import torch
import pandas as pd
import numpy as np
try:
    import arff as liac_arff  # liac-arff
except Exception:  # pragma: no cover - optional until requirements installed
    liac_arff = None  # type: ignore
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


class TabularDataset:
    """Simple tabular dataset for float features.

    Returns dict with input_ids (float32 vector), label (long), domain (long), length (long=feature_dim).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, domain: int) -> None:
        assert features.ndim == 2, "features must be 2D (N, D)"
        assert features.shape[0] == labels.shape[0]
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.domain = int(domain)
        self._num_classes = int(max(1, int(labels.max()) + 1)) if labels.size > 0 else 1

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.features[idx]
        y = int(self.labels[idx])
        return {
            "input_ids": torch.from_numpy(x),
            "label": torch.tensor(y, dtype=torch.long),
            "domain": torch.tensor(self.domain, dtype=torch.long),
            "length": torch.tensor(self.features.shape[1], dtype=torch.long),
        }

    def num_classes(self) -> int:
        return self._num_classes

    def num_features(self) -> int:
        return int(self.features.shape[1])


def _read_arff_dataframe(path: str) -> pd.DataFrame:
    """Robust ARFF reader with fallbacks.

    - Skips any preamble before the @RELATION line
    - Falls back to CSV parsing if ARFF parsing fails
    """
    # Try ARFF first (if available)
    if liac_arff is not None:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            # Strip BOM and find first @RELATION
            text = text.lstrip("\ufeff")
            lines = text.splitlines()
            start = 0
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("@relation"):
                    start = i
                    break
            arff_text = "\n".join(lines[start:]) if start > 0 else text
            data = liac_arff.load(io.StringIO(arff_text))
            columns = [a[0] for a in data.get("attributes", [])]
            df = pd.DataFrame(data.get("data", []), columns=columns)
            df.replace({"?": np.nan}, inplace=True)
            return df
        except Exception:
            pass
    # Fallback: try CSV
    try:
        return pd.read_csv(path)
    except Exception:
        # Last resort: try semicolon or tab delimiters
        for sep in [";", "\t", " "]:
            try:
                return pd.read_csv(path, sep=sep)
            except Exception:
                continue
    raise RuntimeError(f"Failed to read ARFF/CSV file: {path}")


def _pick_label_column(df: pd.DataFrame) -> str:
    candidates = ["class", "Class", "label", "Label", "Application", "app", "Type", "type"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def _factorize_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        # Try numeric conversion first
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].dtype.kind not in {"i", "u", "f"}:
            # Fallback to string factorization
            codes, _ = pd.factorize(out[col].astype(str), sort=True)
            out[col] = codes
    out = out.fillna(0)
    return out


def _build_iscx_a2_paths(cfg: Dict[str, Any]) -> Tuple[str, str]:
    src_cfg = cfg.get("data", {}).get("source", {})
    root = src_cfg.get("root")
    window = str(src_cfg.get("window", "60s"))
    if root is None:
        raise ValueError("data.source.root must be set for iscx_a2_arff")
    base = os.path.join(root, "CSVs", "Scenario A2-ARFF")
    nonvpn = os.path.join(base, f"TimeBasedFeatures-Dataset-{window}-NO-VPN.arff")
    vpn = os.path.join(base, f"TimeBasedFeatures-Dataset-{window}-VPN.arff")
    return nonvpn, vpn


@DATASETS.register("iscx_a2_arff")
def build_iscx_a2_arff(cfg: Dict[str, Any], split: str) -> TabularDataset:
    nonvpn_path, vpn_path = _build_iscx_a2_paths(cfg)
    if not os.path.exists(nonvpn_path):
        raise FileNotFoundError(f"Non-VPN ARFF not found: {nonvpn_path}")
    if not os.path.exists(vpn_path):
        raise FileNotFoundError(f"VPN ARFF not found: {vpn_path}")

    df_src = _read_arff_dataframe(nonvpn_path)
    df_tgt = _read_arff_dataframe(vpn_path)
    # Label resolution across both domains
    label_col = _pick_label_column(df_src)
    if label_col not in df_tgt.columns:
        label_col = _pick_label_column(df_tgt)
    src_labels = df_src[label_col].astype(str)
    tgt_labels = df_tgt[label_col].astype(str)
    uniq = sorted(set(src_labels.unique()).union(set(tgt_labels.unique())))
    label_to_id = {lab: i for i, lab in enumerate(uniq)}

    def to_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        y = df[label_col].astype(str).map(label_to_id).astype(np.int64)
        X = _factorize_non_numeric(df.drop(columns=[label_col])).to_numpy(dtype=np.float32)
        return X, y.to_numpy(dtype=np.int64)

    X_src, y_src = to_xy(df_src)
    X_tgt, y_tgt = to_xy(df_tgt)

    # Deterministic split for source into train/val/test
    rng = np.random.RandomState(int(cfg.get("seed", 0)))
    n = X_src.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    # Feature standardization fitted on source train
    if n > 0:
        mu = X_src[train_idx].mean(axis=0)
        sigma = X_src[train_idx].std(axis=0)
        sigma[sigma == 0] = 1.0
        X_src = (X_src - mu) / sigma
        X_tgt = (X_tgt - mu) / sigma

    if split == "train_source":
        return TabularDataset(X_src[train_idx], y_src[train_idx], domain=0)
    if split == "val_source":
        return TabularDataset(X_src[val_idx], y_src[val_idx], domain=0)
    if split == "test_source":
        return TabularDataset(X_src[test_idx], y_src[test_idx], domain=0)
    if split in {"unlabeled_target", "test_target"}:
        return TabularDataset(X_tgt, y_tgt, domain=1)

    raise KeyError(f"Unknown split: {split}")

