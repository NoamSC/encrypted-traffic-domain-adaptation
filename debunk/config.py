from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def parse_set_overrides(pairs: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        # try to parse json-ish scalars
        try:
            value_parsed = json.loads(value)
        except Exception:
            if value.lower() in {"true", "false"}:
                value_parsed = value.lower() == "true"
            else:
                try:
                    if "." in value:
                        value_parsed = float(value)
                    else:
                        value_parsed = int(value)
                except Exception:
                    value_parsed = value

        # support nested keys a.b.c
        cur = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value_parsed
    return result


@dataclass
class ExperimentConfig:
    config_paths: List[str]
    method: str
    shift: str
    seed: int = 0
    set_overrides: List[str] = field(default_factory=list)

    resolved: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ExperimentConfig":
        return ExperimentConfig(
            config_paths=args.config,
            method=args.method,
            shift=args.shift,
            seed=args.seed,
            set_overrides=args.set or [],
        )

    def load_and_resolve(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for p in self.config_paths:
            cfg = merge_dicts(cfg, load_yaml(p))
        cfg = merge_dicts(cfg, {"method": self.method, "data": {"shift": self.shift}})
        if self.set_overrides:
            overrides = parse_set_overrides(self.set_overrides)
            cfg = merge_dicts(cfg, overrides)
        cfg.setdefault("train", {})
        cfg.setdefault("log", {})
        cfg.setdefault("system", {})
        cfg.setdefault("data", {})
        cfg.setdefault("model", {})
        cfg.setdefault("head", {})
        cfg.setdefault("tokenizer", {})
        self.resolved = cfg
        return cfg

    def save_meta(self, run_dir: str, args: argparse.Namespace) -> None:
        meta_dir = os.path.join(run_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        with open(os.path.join(meta_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(self.resolved, f, sort_keys=False)
        with open(os.path.join(meta_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)


