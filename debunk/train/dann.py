from __future__ import annotations

from typing import Any, Dict

import os
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from ..losses.classification import classification_loss
from ..losses.dann import grl_lambda
from ..models.dann import GradientReversal
from ..utils.checkpoint import save_checkpoint


class DannTrainer(BaseTrainer):
    def __init__(self, cfg: Dict[str, Any], run_dir: str, device: torch.device, model: nn.Module, grl: GradientReversal, dom_disc: nn.Module) -> None:
        super().__init__(cfg, run_dir, device)
        self.model = model.to(device)
        self.grl = grl
        self.dom_disc = dom_disc.to(device)

    def train_loop(self, loaders: Dict[str, Any]) -> None:
        train_src = loaders["src_train"]
        unlabeled_tgt = loaders["tgt_unlabeled"]
        test_tgt = loaders["tgt_test"]
        val_src = loaders["src_val"]
        epochs = int(self.cfg.get("train", {}).get("epochs", 1))
        lr = float(self.cfg.get("train", {}).get("lr", 3e-4))
        wd = float(self.cfg.get("train", {}).get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.dom_disc.parameters()), lr=lr, weight_decay=wd)
        self.log_env()
        self.log_hparams(self.model)
        dann_cfg = self.cfg.get("dann", {})
        schedule = dann_cfg.get("grl", {}).get("schedule", "sigmoid")
        base_lambda = float(dann_cfg.get("grl", {}).get("lambda", 1.0))
        max_lambda = float(dann_cfg.get("grl", {}).get("max_lambda", 1.0))
        total_steps = int(dann_cfg.get("grl", {}).get("steps", 10000))
        criterion_domain = nn.CrossEntropyLoss()

        src_iter = iter(train_src)
        tgt_iter = iter(unlabeled_tgt)

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.dom_disc.train()
            for _ in range(len(train_src)):
                self.state.global_step += 1
                try:
                    src_batch = next(src_iter)
                except StopIteration:
                    src_iter = iter(train_src)
                    src_batch = next(src_iter)
                try:
                    tgt_batch = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(unlabeled_tgt)
                    tgt_batch = next(tgt_iter)

                # source supervised
                src_input = src_batch["input_ids"].to(self.device)
                src_len = src_batch.get("lengths")
                if src_len is not None:
                    src_len = src_len.to(self.device)
                src_labels = src_batch["labels"].to(self.device)
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    src_logits, src_feat = self.model.classifier(input_ids=src_input, lengths=src_len)
                    sup_loss = classification_loss(src_logits, src_labels)

                # domain adversarial: source
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    lam = grl_lambda(self.state.global_step, schedule, base_lambda, max_lambda, total_steps)
                    self.grl.set_lambda(lam)
                    src_dom_logits = self.dom_disc(self.grl(src_feat))
                    src_dom_labels = torch.zeros(src_dom_logits.size(0), dtype=torch.long, device=self.device)
                    src_dom_loss = criterion_domain(src_dom_logits, src_dom_labels)

                # domain adversarial: target
                tgt_input = tgt_batch["input_ids"].to(self.device)
                tgt_len = tgt_batch.get("lengths")
                if tgt_len is not None:
                    tgt_len = tgt_len.to(self.device)
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    _, tgt_feat = self.model.classifier(input_ids=tgt_input, lengths=tgt_len)
                    tgt_dom_logits = self.dom_disc(self.grl(tgt_feat))
                    tgt_dom_labels = torch.ones(tgt_dom_logits.size(0), dtype=torch.long, device=self.device)
                    tgt_dom_loss = criterion_domain(tgt_dom_logits, tgt_dom_labels)

                loss = sup_loss + (src_dom_loss + tgt_dom_loss)
                self.backward_and_step(loss, optimizer)

                if self.state.global_step % int(self.cfg.get("log", {}).get("log_interval", 50)) == 0:
                    self.writer.add_scalar("loss/train_total", float(loss.item()), self.state.global_step)
                    self.writer.add_scalar("loss/supervised", float(sup_loss.item()), self.state.global_step)
                    self.writer.add_scalar("loss/domain_src", float(src_dom_loss.item()), self.state.global_step)
                    self.writer.add_scalar("loss/domain_tgt", float(tgt_dom_loss.item()), self.state.global_step)
                    self.writer.add_scalar("grl/lambda", float(self.grl.lambd), self.state.global_step)

            # evaluate
            src_metrics = self.evaluate_classifier(self.model, val_src, split="src_val")
            tgt_metrics = self.evaluate_classifier(self.model, test_tgt, split="tgt_test")
            save_checkpoint(self.model, os.path.join(self.run_dir, "last.pt"))
            if tgt_metrics["acc"] >= self.state.best_target_acc:
                self.state.best_target_acc = tgt_metrics["acc"]
                save_checkpoint(self.model, os.path.join(self.run_dir, "best_target.pt"))
            self.save_metrics({
                "step": self.state.global_step,
                "epoch": epoch,
                "src_val_acc": src_metrics["acc"],
                "tgt_test_acc": tgt_metrics["acc"],
                "lambda": float(self.grl.lambd),
            })

        # final metrics
        final_src = self.evaluate_classifier(self.model, val_src, split="src_val_final")
        final_tgt = self.evaluate_classifier(self.model, test_tgt, split="tgt_test_final")
        self.save_metrics({
            "src_val_acc": final_src["acc"],
            "tgt_test_acc": final_tgt["acc"],
            "best_tgt_acc": self.state.best_target_acc,
        }, final=True)


