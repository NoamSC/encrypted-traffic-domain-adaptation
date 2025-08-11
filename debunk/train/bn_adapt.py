from __future__ import annotations

from typing import Any, Dict

import os
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from ..losses.bn_adapt import update_bn_stats
from ..utils.checkpoint import save_checkpoint


class BNAdaptTrainer(BaseTrainer):
    def __init__(self, cfg: Dict[str, Any], run_dir: str, device: torch.device, model: nn.Module) -> None:
        super().__init__(cfg, run_dir, device)
        self.model = model.to(device)

    def train_loop(self, loaders: Dict[str, Any]) -> None:
        unlabeled_tgt = loaders["tgt_unlabeled"]
        test_tgt = loaders["tgt_test"]
        test_src = loaders["src_test"]
        self.log_env()
        self.log_hparams(self.model)
        # update BN stats
        update_bn_stats(self.model, unlabeled_tgt, self.device)
        # evaluate
        src_metrics = self.evaluate_classifier(self.model, test_src, split="src_test")
        tgt_metrics = self.evaluate_classifier(self.model, test_tgt, split="tgt_test")
        save_checkpoint(self.model, os.path.join(self.run_dir, "last.pt"))
        save_checkpoint(self.model, os.path.join(self.run_dir, "best_target.pt"))
        self.save_metrics({
            "src_test_acc": src_metrics["acc"],
            "tgt_test_acc": tgt_metrics["acc"],
        }, final=True)


