from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    def __init__(self) -> None:
        self._fns: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._fns[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._fns:
            raise KeyError(f"Unknown registry key: {name}")
        return self._fns[name]


MODELS = Registry()
TOKENIZERS = Registry()
COLLATORS = Registry()
DATASETS = Registry()
TRAINERS = Registry()


