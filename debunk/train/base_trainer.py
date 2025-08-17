from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..utils.metrics import accuracy_from_logits, confusion_matrix_image
from ..losses.classification import classification_loss
from ..utils.checkpoint import save_checkpoint
from ..utils.tb import write_hparams
from ..utils.env import capture_environment


@dataclass
class TrainState:
    global_step: int = 0
    best_target_acc: float = 0.0


class BaseTrainer:
    def __init__(self, cfg: Dict[str, Any], run_dir: str, device: torch.device) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        use_cuda = (str(cfg.get("system", {}).get("device", "auto")).lower() in {"auto", "cuda", "gpu"}) and torch.cuda.is_available()
        amp_enabled = bool(cfg.get("train", {}).get("amp", False)) and use_cuda
        # Use new torch.amp.GradScaler API; select CUDA explicitly
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        except Exception:
            # Fallback for older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.state = TrainState()
        self.writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    def log_env(self) -> None:
        meta_dir = os.path.join(self.run_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        capture_environment(os.path.join(meta_dir, "env.txt"))

    def log_hparams(self, model: torch.nn.Module) -> None:
        write_hparams(self.writer, self.cfg)

    def step_scheduler(self, scheduler) -> None:
        if scheduler is not None:
            scheduler.step()

    def backward_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.scale(loss).backward()
        grad_clip = float(self.cfg.get("train", {}).get("grad_clip", 0.0))
        if grad_clip and grad_clip > 0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

    def _forward_classifier(self, model: torch.nn.Module, input_ids: torch.Tensor, lengths: torch.Tensor | None):
        if hasattr(model, "classifier"):
            return model.classifier(input_ids=input_ids, lengths=lengths)
        return model(input_ids=input_ids, lengths=lengths)

    def evaluate_classifier(self, model: torch.nn.Module, loader: DataLoader, split: str) -> Dict[str, float]:
        model.eval()
        total, correct = 0, 0
        loss_sum = 0.0
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                lengths = batch.get("lengths")
                if lengths is not None:
                    lengths = lengths.to(self.device)
                labels = batch["labels"].to(self.device)
                logits, _ = self._forward_classifier(model, input_ids=input_ids, lengths=lengths)
                # accumulate loss and accuracy
                batch_loss = classification_loss(logits, labels)
                loss_sum += float(batch_loss.item()) * float(labels.numel())
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.numel()
                all_targets.append(labels.cpu())
                all_preds.append(logits.argmax(dim=1).cpu())
        acc = correct / max(1, total)
        avg_loss = loss_sum / max(1, total)
        self.writer.add_scalar(f"acc/{split}", acc, self.state.global_step)
        self.writer.add_scalar(f"loss/{split}", avg_loss, self.state.global_step)
        # confusion matrix image (force consistent labels)
        import torch as _t

        if all_targets and all_preds:
            num_classes = int(self.cfg.get("head", {}).get("num_classes", self.cfg.get("data", {}).get("num_classes", 5)))
            img = confusion_matrix_image(_t.cat(all_targets), _t.cat(all_preds), num_classes=num_classes)
            self.writer.add_image(f"cm/{split}", img, self.state.global_step, dataformats="HWC")
        return {"acc": acc, "loss": avg_loss}

    def save_metrics(self, metrics: Dict[str, Any], final: bool = False) -> None:
        metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        if final:
            with open(os.path.join(self.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)


