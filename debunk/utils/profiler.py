from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def measure_time():
    start = time.time()
    yield lambda: time.time() - start


