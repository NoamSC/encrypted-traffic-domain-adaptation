from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict

from .dataset import TrafficDataset, TrafficSample
from .tokenizer import FeatureTokenizer


def _pick_time_key(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "timestamp",
        "time",
        "frame_time_epoch",
        "capture_time",
        "ts",
    ]
    for k in candidates:
        if k in df.columns:
            return k
    return None


def _pick_group_key(df: pd.DataFrame) -> Optional[str]:
    for k in ["flow_id", "session_id", "conn_id", "flow"]:
        if k in df.columns:
            return k
    return None


def _pick_label_key(df: pd.DataFrame) -> Optional[str]:
    preferred = ["label", "label_id", "class", "app", "application", "category", "target"]
    for k in preferred:
        if k in df.columns:
            return k
    # fallback: choose a low-cardinality non-numeric string column
    exclude = {"domain", "payload_hex", "src_port", "dst_port", "length", "protocol", "is_vpn", "num_packets", "timestamp", "time", "frame_time_epoch"}
    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if s.dtype == object or str(s.dtype).startswith("category"):
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= 1000:
                candidates.append((nunique, col))
    candidates.sort()
    return candidates[0][1] if candidates else None


def build_tls_early2late_splits(cfg: Dict[str, Any]) -> Dict[str, TrafficDataset]:
    src_cfg = cfg.get("data", {}).get("source", {})
    hf_id = src_cfg.get("hf_id")
    revision = src_cfg.get("hf_revision")
    cache_dir = src_cfg.get("cache_dir")
    if not hf_id:
        raise ValueError("data.source.hf_id must be set for hf source")
    dsd: DatasetDict = load_dataset(hf_id, revision=revision, cache_dir=cache_dir)

    # Merge all available splits
    dfs: List[pd.DataFrame] = []
    for split_name, split in dsd.items():
        dfs.append(split.to_pandas())
    df = pd.concat(dfs, ignore_index=True)

    # Optional filters
    pre = cfg.get("data", {}).get("preprocessing", {})
    if pre.get("drop_vpn", False) and "is_vpn" in df.columns:
        df = df[df["is_vpn"] == 0]
    if pre.get("protocol_whitelist") and "protocol" in df.columns:
        df = df[df["protocol"].isin(pre.get("protocol_whitelist"))]
    if pre.get("dedup", True):
        gk = _pick_group_key(df)
        if gk:
            df = df.drop_duplicates(subset=[gk])

    tokenizer = FeatureTokenizer(preset=cfg.get("tokenizer", {}).get("preset", "debunk_v1"))

    # Determine label column and build a consistent numeric mapping across all rows
    label_key = _pick_label_key(df)
    if label_key is None:
        # create a dummy label if missing to avoid crashes, but warn via print
        print("[warn] No label column detected; defaulting to 0 for all labels (this will break evaluation)")
        df["label_id"] = 0
    else:
        # If already numeric small-range, just cast; else factorize
        if pd.api.types.is_integer_dtype(df[label_key]) and df[label_key].nunique() <= 1024:
            # ensure non-negative contiguous ids
            unique_vals = sorted(df[label_key].dropna().unique().tolist())
            remap = {v: i for i, v in enumerate(unique_vals)}
            df["label_id"] = df[label_key].map(remap).fillna(0).astype(int)
        else:
            codes, uniques = pd.factorize(df[label_key].astype(str))
            df["label_id"] = codes.astype(int)

    gk = _pick_group_key(df)
    tk = _pick_time_key(df)
    rng = np.random.RandomState(cfg.get("seed", 0))

    if "domain" in df.columns:
        # Use provided domain: 0 = early/source, 1 = late/target
        df_src = df[df["domain"] == 0]
        df_tgt = df[df["domain"] == 1]
    elif tk is not None:
        # Split by time: early vs late based on group-level min time
        if gk is not None:
            grp = df.groupby(gk)[tk].min().sort_values()
            cutoff = int(0.6 * len(grp))
            early_groups = set(grp.index[:cutoff])
            df_src = df[df[gk].isin(early_groups)]
            df_tgt = df[~df[gk].isin(early_groups)]
        else:
            df_sorted = df.sort_values(tk)
            cutoff = int(0.6 * len(df_sorted))
            df_src = df_sorted.iloc[:cutoff]
            df_tgt = df_sorted.iloc[cutoff:]
    else:
        # Fallback: random split by groups to avoid leakage if possible
        if gk is not None:
            groups = df[gk].unique()
            rng.shuffle(groups)
            cutoff = int(0.6 * len(groups))
            early_groups = set(groups[:cutoff])
            df_src = df[df[gk].isin(early_groups)]
            df_tgt = df[~df[gk].isin(early_groups)]
        else:
            idx = np.arange(len(df))
            rng.shuffle(idx)
            cutoff = int(0.6 * len(idx))
            df_src = df.iloc[idx[:cutoff]]
            df_tgt = df.iloc[idx[cutoff:]]

    # Within-domain splits by groups for leakage-free eval
    def split_domain(df_d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if gk is not None and gk in df_d.columns:
            groups = df_d[gk].unique()
            rng.shuffle(groups)
            n = len(groups)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_g = set(groups[:n_train])
            val_g = set(groups[n_train : n_train + n_val])
            test_g = set(groups[n_train + n_val :])
            train_df = df_d[df_d[gk].isin(train_g)]
            val_df = df_d[df_d[gk].isin(val_g)]
            test_df = df_d[df_d[gk].isin(test_g)]
        else:
            idx = np.arange(len(df_d))
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_df = df_d.iloc[idx[:n_train]]
            val_df = df_d.iloc[idx[n_train : n_train + n_val]]
            test_df = df_d.iloc[idx[n_train + n_val :]]
        return train_df, val_df, test_df

    # Debug label distributions (helpful for sanity checks)
    def summarize(df_a: pd.DataFrame, name: str):
        try:
            vc = df_a["label_id"].value_counts().sort_index()
            print(f"[label_dist] {name}: {vc.to_dict()}")
        except Exception:
            pass

    src_train_df, src_val_df, src_test_df = split_domain(df_src)
    tgt_train_df, _, tgt_test_df = split_domain(df_tgt)

    # Optional subsampling for quick experiments
    limit = cfg.get("data", {}).get("limit_per_split")
    if isinstance(limit, int) and limit > 0:
        def _lim(df_d: pd.DataFrame) -> pd.DataFrame:
            if len(df_d) <= limit:
                return df_d
            idx = rng.permutation(len(df_d))[:limit]
            return df_d.iloc[idx]
        src_train_df = _lim(src_train_df)
        src_val_df = _lim(src_val_df)
        src_test_df = _lim(src_test_df)
        tgt_train_df = _lim(tgt_train_df)
        tgt_test_df = _lim(tgt_test_df)

    # Build samples
    def to_samples(df_i: pd.DataFrame, domain_id: int) -> List[TrafficSample]:
        samples: List[TrafficSample] = []
        for _, row in df_i.iterrows():
            tokens = tokenizer.encode(row.to_dict())
            label = int(row.get("label_id", row.get("label", 0)))
            samples.append(TrafficSample(tokens=tokens, label=label, domain=domain_id))
        return samples

    # Log distributions
    summarize(df_src, "src_all")
    summarize(src_train_df, "src_train")
    summarize(src_val_df, "src_val")
    summarize(src_test_df, "src_test")
    summarize(df_tgt, "tgt_all")
    summarize(tgt_train_df, "tgt_unlabeled")
    summarize(tgt_test_df, "tgt_test")

    src_train_samples = to_samples(src_train_df, 0)
    src_val_samples = to_samples(src_val_df, 0)
    src_test_samples = to_samples(src_test_df, 0)
    tgt_unlab_samples = to_samples(tgt_train_df, 1)
    tgt_test_samples = to_samples(tgt_test_df, 1)

    max_len = int(cfg.get("data", {}).get("preprocessing", {}).get("max_len", 256))
    return {
        "src_train": TrafficDataset(src_train_samples, max_len=max_len),
        "src_val": TrafficDataset(src_val_samples, max_len=max_len),
        "src_test": TrafficDataset(src_test_samples, max_len=max_len),
        "tgt_unlabeled": TrafficDataset(tgt_unlab_samples, max_len=max_len),
        "tgt_test": TrafficDataset(tgt_test_samples, max_len=max_len),
    }


