from __future__ import annotations

import os
import argparse
import sys

# Ensure Hugging Face caches use /home storage before any downstream imports
os.environ.setdefault("HF_HOME", "/home/anatbr/students/noamshakedc/env/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/anatbr/students/noamshakedc/env/.cache/huggingface/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/anatbr/students/noamshakedc/env/.cache/huggingface/transformers")

from debunk.runner import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True, help="YAML config paths; later override earlier")
    p.add_argument("--method", type=str, required=True, choices=["source_only", "bn_adapt", "dann"])
    p.add_argument("--shift", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="Override keys: a.b.c=value")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()


