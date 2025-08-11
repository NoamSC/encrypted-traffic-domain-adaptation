from __future__ import annotations

import json
import os
from typing import Any, Dict


def ensure_run_dir(method: str, shift: str, seed: int) -> str:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{ts}_{method}_{shift}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tb"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "meta"), exist_ok=True)
    return run_dir


