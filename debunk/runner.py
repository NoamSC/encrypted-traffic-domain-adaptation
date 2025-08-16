from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch

from .config import ExperimentConfig
from .utils.seed import set_seed
from .utils.reporting import ensure_run_dir
from .utils.io import save_json
from .data.splits import SplitManager
from .models.backbone import NetMambaBackbone
from .models.classifier import TrafficClassifier
from .models.debug_mlp import DebugMLPClassifier
from .models.dann import DomainDiscriminator, GradientReversal, DannModel
from .train.source_only import SourceOnlyTrainer
from .train.bn_adapt import BNAdaptTrainer
from .train.dann import DannTrainer
from .utils.io import save_json
import subprocess


def build_model(cfg: Dict[str, Any]):
    mcfg = cfg.get("model", {})
    model_name = str(mcfg.get("name", "netmamba")).lower()

    # Resolve number of classes first
    head_cfg = cfg.setdefault("head", {})
    requested_nc = head_cfg.get("num_classes")
    infer_needed = (
        (requested_nc is None)
        or (isinstance(requested_nc, (int, float)) and int(requested_nc) <= 0)
        or (isinstance(requested_nc, str) and str(requested_nc).lower() in {"auto", "infer"})
        or ("num_classes" not in head_cfg)
    )
    if infer_needed:
        try:
            tmp_splits = SplitManager(cfg).build_loaders()
            num_classes = tmp_splits["src_train"].dataset.num_classes()
        except Exception:
            num_classes = int(cfg.get("data", {}).get("num_classes", 2))
        head_cfg["num_classes"] = int(num_classes)
    else:
        num_classes = int(requested_nc)

    if model_name == "debug_mlp":
        # Infer input feature dimension from a small sample
        input_dim: int
        try:
            tmp_splits = SplitManager(cfg).build_loaders()
            ds = tmp_splits["src_train"].dataset
            if hasattr(ds, "num_features"):
                input_dim = int(ds.num_features())
            else:
                sample = next(iter(tmp_splits["src_train"]))
                input_dim = int(sample["input_ids"].shape[-1])
        except Exception:
            input_dim = int(mcfg.get("input_dim", 128))
        hidden_dims = list(mcfg.get("hidden_dims", [128, 64]))
        dropout = float(mcfg.get("dropout", 0.1))
        return DebugMLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, dropout=dropout)

    # Default path: NetMamba backbone classifier
    backbone = NetMambaBackbone(
        d_model=int(mcfg.get("d_model", 256)),
        n_layers=int(mcfg.get("n_layers", 6)),
        dropout=float(mcfg.get("dropout", 0.1)),
        max_len=int(mcfg.get("max_len", 256)),
    )
    head_dropout = float(cfg.get("head", {}).get("dropout", 0.1))
    return TrafficClassifier(backbone=backbone, num_classes=num_classes, head_dropout=head_dropout)


def run_experiment(args: argparse.Namespace) -> None:
    exp = ExperimentConfig.from_args(args)
    cfg = exp.load_and_resolve()
    set_seed(int(cfg.get("seed", args.seed)), deterministic=bool(cfg.get("system", {}).get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ensure_run_dir(exp.method, exp.shift, exp.seed)
    exp.save_meta(run_dir, args)

    # Build data
    splits = SplitManager(cfg).build_loaders()

    # Save dataset metadata
    src_cfg = cfg.get("data", {}).get("source", {})
    ds_meta = {
        "type": src_cfg.get("type", "toy"),
        "hf_id": src_cfg.get("hf_id"),
        "revision": src_cfg.get("hf_revision"),
        "split_sizes": {k: len(v.dataset) for k, v in splits.items()},
    }
    save_json(ds_meta, os.path.join(run_dir, "dataset.json"))

    # Determine number of classes robustly before saving label map
    head_cfg = cfg.setdefault("head", {})
    requested_nc = head_cfg.get("num_classes")
    resolved_num_classes: int
    try:
        resolved_num_classes = int(requested_nc) if requested_nc is not None else int(cfg.get("data", {}).get("num_classes", 5))
    except Exception:
        # If not an int (e.g., "infer"), compute from source train split
        try:
            resolved_num_classes = int(splits["src_train"].dataset.num_classes())
        except Exception:
            resolved_num_classes = int(cfg.get("data", {}).get("num_classes", 5))
    head_cfg["num_classes"] = int(resolved_num_classes)

    # Save label map
    label_map = {str(i): i for i in range(head_cfg["num_classes"])}
    save_json(label_map, os.path.join(run_dir, "label_map.json"))

    # Save external submodule commit if present
    ext_path = os.path.join("external", "Debunk_Traffic_Representation")
    if os.path.isdir(ext_path) and os.path.isdir(os.path.join(ext_path, ".git")):
        try:
            commit = subprocess.check_output(["git", "-C", ext_path, "rev-parse", "HEAD"], text=True).strip()
            with open(os.path.join(run_dir, "external_commit.txt"), "w", encoding="utf-8") as f:
                f.write(commit + "\n")
        except Exception:
            pass

    # Build model (will use the resolved head.num_classes)
    classifier = build_model(cfg)

    # Select trainer
    method = exp.method
    if method == "source_only":
        trainer = SourceOnlyTrainer(cfg, run_dir, device, classifier)
        trainer.train_loop(splits)
    elif method == "bn_adapt":
        trainer = BNAdaptTrainer(cfg, run_dir, device, classifier)
        trainer.train_loop(splits)
    elif method == "dann":
        dann_cfg = cfg.get("dann", {})
        grl = GradientReversal(lambd=float(dann_cfg.get("grl", {}).get("lambda", 1.0)))
        dom_disc = DomainDiscriminator(in_dim=classifier.feature_dim, hidden_dims=dann_cfg.get("discriminator", {}).get("hidden_dims", [256, 128]), dropout=float(dann_cfg.get("discriminator", {}).get("dropout", 0.1)))
        dann_model = DannModel(classifier, dom_disc, grl)
        trainer = DannTrainer(cfg, run_dir, device, dann_model, grl, dom_disc)
        trainer.train_loop(splits)
    else:
        raise ValueError(f"Unknown method: {method}")


