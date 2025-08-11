from __future__ import annotations

import os
import platform
import subprocess
from typing import Optional

import torch


def capture_environment(path: str) -> None:
    lines = []
    lines.append(f"python: {platform.python_version()}")
    lines.append(f"torch: {torch.__version__}")
    lines.append(f"cuda: {torch.version.cuda if torch.cuda.is_available() else 'cpu'}")
    try:
        pip = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:
        pip = ""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")
        f.write(pip)


