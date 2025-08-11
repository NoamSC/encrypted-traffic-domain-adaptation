from __future__ import annotations

import argparse
import os
import json
import sys
from typing import Optional

import pandas as pd
from datasets import load_dataset

# Ensure package import works when running the script directly from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from debunk.data.debunk_preprocess import DebunkPreprocessor, PreprocessConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-id", type=str, required=True)
    p.add_argument("--hf-revision", type=str, default=None)
    p.add_argument("--shift", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--drop-vpn", action="store_true")
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    ds = load_dataset(args.hf_id, revision=args.hf_revision)
    cfg = PreprocessConfig(drop_vpn=args.drop_vpn, max_len=args.max_len)
    prep = DebunkPreprocessor(cfg)
    meta_dir = os.path.join(args.cache_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump({"hf_id": args.hf_id, "revision": args.hf_revision}, f, indent=2)
    for split in ds.keys():
        df = ds[split].to_pandas()
        df = prep.run(df)
        out_path = prep.save(df, args.cache_dir, split)
        print(f"Saved {split} to {out_path}")


if __name__ == "__main__":
    main()


