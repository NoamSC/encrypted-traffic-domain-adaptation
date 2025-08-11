from __future__ import annotations

import argparse
import sys

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


