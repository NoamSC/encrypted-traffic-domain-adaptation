from __future__ import annotations

from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter
import json


def write_hparams(writer: SummaryWriter, cfg: Dict[str, Any]) -> None:
    flat = {}
    def _flatten(prefix: str, d: Dict[str, Any]):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(key, v)
            else:
                flat[key] = v
    _flatten("", cfg)
    # sanitize unsupported types for hparams (lists/dicts/None)
    safe = {}
    for k, v in flat.items():
        if isinstance(v, (int, float, bool, str)):
            safe[k] = v
        else:
            try:
                safe[k] = json.dumps(v, sort_keys=True)
            except Exception:
                safe[k] = str(v)
    writer.add_hparams(safe, {})


