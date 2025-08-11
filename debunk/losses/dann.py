from __future__ import annotations

import math
from typing import Literal


def grl_lambda(step: int, schedule: Literal["constant", "linear", "sigmoid"], base: float, max_lambda: float, total_steps: int) -> float:
    if schedule == "constant":
        return float(base)
    if schedule == "linear":
        return float(min(max_lambda, base + (max_lambda - base) * step / max(1, total_steps)))
    # sigmoid schedule from DANN paper
    p = step / max(1, total_steps)
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0) * max_lambda


