"""Debunk - Encrypted traffic classification and domain adaptation.

Public modules:
- config: ExperimentConfig loader and overrides
- registry: simple registries for components
- runner: CLI entrypoints
"""

from .config import ExperimentConfig, load_yaml, merge_dicts

# register data builders
from .data import registry_hooks as _registry_hooks  # noqa: F401

__all__ = [
    "ExperimentConfig",
    "load_yaml",
    "merge_dicts",
]


