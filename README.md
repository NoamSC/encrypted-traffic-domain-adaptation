Encrypted Traffic Classification with NetMamba and Domain Adaptation

This repository provides a modular, config-driven PyTorch codebase for encrypted-traffic classification using a NetMamba-style backbone and three domain adaptation methods: Source-Only, BN-Adapt, and DANN (Domain-Adversarial Neural Network). It emphasizes reproducibility, clear logging, and ease of experimentation.

Highlights
- NetMambaBackbone: efficient, self-contained Mamba-style sequence encoder (no external Mamba deps).
- Methods: Source-Only, BN-Adapt, and DANN with configurable GRL λ schedules.
- Data: load directly from Hugging Face Datasets or local CSV/Parquet processed by the included script.
- Optional submodule integration with `SmartData-Polito/Debunk_Traffic_Representation` for preprocessing parity.
- Config-driven via YAML with CLI overrides; metrics, checkpoints, TB logs saved per run.

Directory Layout
```
configs/
  netmamba_base.yaml
  data_tls_early2late.yaml
  train_source_only.yaml
  train_bn_adapt.yaml
  train_dann.yaml
  grid_example.yaml

debunk/
  ... (data, models, losses, train, utils, runner)

external/
  Debunk_Traffic_Representation/  # optional git submodule

scripts/
  run_experiment.py
  make_grid.py
  aggregate_results.py
  prepare_debunk_data.py

toy_data/
  train_source.csv
  val_source.csv
  test_source.csv
  unlabeled_target.csv
  test_target.csv
```

Installation
1) Python >= 3.9, PyTorch >= 2.1.
2) Install minimal dependencies:
```
pip install -r requirements.txt
```

Optional: Add the external preprocessing repo as a submodule
```
git submodule add https://github.com/SmartData-Polito/Debunk_Traffic_Representation external/Debunk_Traffic_Representation
git submodule update --init --recursive
```

Data Loading
- HF direct: uses `datasets.load_dataset`. Configure `data.source.type=hf`, set `hf_id`, optional `hf_revision`.
- Local processed: CSV/Parquet loaded from `data.cache_dir` produced by `scripts/prepare_debunk_data.py`.
- Toy data: default config points to `toy_data/` so smoke tests run out-of-the-box.

Preprocessing
Use the helper to filter/deduplicate and export parquet/csv locally. It can capture HF fingerprint, revision, and flags.
```
python scripts/prepare_debunk_data.py \
  --hf-id rigcor7/Debunk_Traffic_Representation \
  --shift tls_early2late --cache-dir data/processed/tls_early2late/ \
  --drop-vpn --max-len 256
```

CLI Usage
HF direct:
```
python scripts/run_experiment.py \
  --config configs/netmamba_base.yaml \
  --method source_only --shift tls_early2late --seed 0 \
  --set data.source.type=hf data.source.hf_id=rigcor7/Debunk_Traffic_Representation
```

Local processed:
```
python scripts/prepare_debunk_data.py \
  --hf-id rigcor7/Debunk_Traffic_Representation \
  --shift tls_early2late --cache-dir data/processed/tls_early2late/ \
  --drop-vpn --max-len 256

python scripts/run_experiment.py \
  --config configs/netmamba_base.yaml \
  --method dann --shift tls_early2late --seed 0 \
  --set data.source.type=local data.cache_dir=data/processed/tls_early2late/
```

Reproducibility and Logging
- TensorBoard scalars: losses, accuracies, λ (DANN), throughput, GPU memory.
- Hparams: full resolved config and data metadata.
- Artifacts per run in `runs/<timestamp>_<method>_<shift>_<seed>/`:
  - `meta/config.yaml`, `meta/args.json`, `meta/preprocess_config.yaml` (if used)
  - `meta/preprocess_hash.txt`, `meta/external_commit.txt` (if submodule detected)
  - `meta/env.txt` with Python/Torch/CUDA + `pip freeze`
  - `dataset.json` with HF id, revision, fingerprint, sizes
  - `label_map.json`
  - `metrics.jsonl` (per-eval), `metrics.json` (final)
  - Checkpoints: `last.pt`, `best_target.pt`

Smoke Tests
The following run on toy data by default:
```
python scripts/run_experiment.py --config configs/netmamba_base.yaml \
  --method source_only --shift tls_early2late --seed 0 --set train.epochs=1

python scripts/run_experiment.py --config configs/netmamba_base.yaml \
  --method dann --shift tls_early2late --seed 0 --set train.epochs=1

tensorboard --logdir runs/
```

Notes
- The NetMamba backbone provided here is a self-contained efficient approximation designed to be stable and linear-time. Its interface is compatible with future swaps to external Mamba implementations.
- BN-Adapt updates batch norm statistics on target-only unlabeled data, leaving weights frozen.
- DANN employs a gradient reversal layer with λ strategies: constant, linear, sigmoid.

License
This project is provided as-is for research and educational use.


