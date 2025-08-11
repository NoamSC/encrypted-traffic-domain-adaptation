from __future__ import annotations

from typing import Any, Dict

import os
import time
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from ..losses.classification import classification_loss
from ..utils.checkpoint import save_checkpoint


class SourceOnlyTrainer(BaseTrainer):
    def __init__(self, cfg: Dict[str, Any], run_dir: str, device: torch.device, model: nn.Module) -> None:
        super().__init__(cfg, run_dir, device)
        self.model = model.to(device)

    def train_loop(self, loaders: Dict[str, Any]) -> None:
        train_loader = loaders["src_train"]
        val_src = loaders["src_val"]
        test_src = loaders["src_test"]
        test_tgt = loaders["tgt_test"]
        epochs = int(self.cfg.get("train", {}).get("epochs", 1))
        lr = float(self.cfg.get("train", {}).get("lr", 3e-4))
        wd = float(self.cfg.get("train", {}).get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler = None
        self.log_env()
        self.log_hparams(self.model)

        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch in train_loader:
                self.state.global_step += 1
                input_ids = batch["input_ids"].to(self.device)
                lengths = batch.get("lengths")
                if lengths is not None:
                    lengths = lengths.to(self.device)
                labels = batch["labels"].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    logits, _ = self.model(input_ids=input_ids, lengths=lengths)
                    loss = classification_loss(logits, labels)
                self.backward_and_step(loss, optimizer)
                if self.state.global_step % int(self.cfg.get("log", {}).get("log_interval", 50)) == 0:
                    self.writer.add_scalar("loss/train", float(loss.item()), self.state.global_step)

            # evaluate
            src_metrics = self.evaluate_classifier(self.model, val_src, split="src_val")
            tgt_metrics = self.evaluate_classifier(self.model, test_tgt, split="tgt_test")
            # save checkpoints
            save_checkpoint(self.model, os.path.join(self.run_dir, "last.pt"))
            if tgt_metrics["acc"] >= self.state.best_target_acc:
                self.state.best_target_acc = tgt_metrics["acc"]
                save_checkpoint(self.model, os.path.join(self.run_dir, "best_target.pt"))
            self.save_metrics({
                "step": self.state.global_step,
                "epoch": epoch,
                "src_val_acc": src_metrics["acc"],
                "tgt_test_acc": tgt_metrics["acc"],
            })

        final_src = self.evaluate_classifier(self.model, test_src, split="src_test_final")
        final_tgt = self.evaluate_classifier(self.model, test_tgt, split="tgt_test_final")
        self.save_metrics({
            "src_test_acc": final_src["acc"],
            "tgt_test_acc": final_tgt["acc"],
            "best_tgt_acc": self.state.best_target_acc,
        }, final=True)


