from __future__ import annotations

import argparse
import itertools


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["source_only", "dann"]) 
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2"]) 
    p.add_argument("--shifts", nargs="+", default=["tls_early2late"]) 
    args = p.parse_args()
    for method, seed, shift in itertools.product(args.methods, args.seeds, args.shifts):
        print(f"python scripts/run_experiment.py --config configs/netmamba_base.yaml configs/data_tls_early2late.yaml configs/train_{'dann' if method=='dann' else 'source_only'}.yaml --method {method} --shift {shift} --seed {seed}")


if __name__ == "__main__":
    main()


