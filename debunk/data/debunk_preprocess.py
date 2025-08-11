from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PreprocessConfig:
    drop_vpn: bool = True
    protocol_whitelist: Optional[List[str]] = None
    dedup: bool = True
    min_packets: int = 5
    max_len: int = 256


def compute_hash(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DebunkPreprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg

    def filter_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.cfg.drop_vpn and "is_vpn" in out.columns:
            out = out[out["is_vpn"] == 0]
        if self.cfg.protocol_whitelist and "protocol" in out.columns:
            out = out[out["protocol"].isin(self.cfg.protocol_whitelist)]
        if self.cfg.dedup and "flow_id" in out.columns:
            out = out.drop_duplicates(subset=["flow_id"])  # rough proxy
        if self.cfg.min_packets and "num_packets" in out.columns:
            out = out[out["num_packets"] >= self.cfg.min_packets]
        return out.reset_index(drop=True)

    def truncate_payload(self, df: pd.DataFrame) -> pd.DataFrame:
        if "payload_hex" in df.columns:
            df = df.copy()
            df["payload_hex"] = df["payload_hex"].astype(str).str.replace(" ", "")
            df["payload_hex"] = df["payload_hex"].str[: 2 * self.cfg.max_len]
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.filter_frame(df)
        df = self.truncate_payload(df)
        return df

    def save(self, df: pd.DataFrame, out_dir: str, name: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        # save metadata
        meta_dir = os.path.join(out_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        with open(os.path.join(meta_dir, "preprocess_config.yaml"), "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump({
                "drop_vpn": self.cfg.drop_vpn,
                "protocol_whitelist": self.cfg.protocol_whitelist,
                "dedup": self.cfg.dedup,
                "min_packets": self.cfg.min_packets,
                "max_len": self.cfg.max_len,
            }, f, sort_keys=False)
        with open(os.path.join(meta_dir, "preprocess_hash.txt"), "w", encoding="utf-8") as f:
            f.write(compute_hash(self.cfg.__dict__))
        return path


