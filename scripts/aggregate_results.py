from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, default="runs")
    args = p.parse_args()
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs, "*"))):
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                row = json.load(f)
            row["run_dir"] = run_dir
            rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()


