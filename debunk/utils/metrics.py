from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def confusion_matrix_image(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> np.ndarray:
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    cm = cm.astype(np.float32)
    cm = cm / np.maximum(1.0, cm.sum(axis=1, keepdims=True))
    # render as small HWC image
    h, w = cm.shape
    img = (cm * 255.0).clip(0, 255).astype(np.uint8)
    img = np.stack([img, img, img], axis=-1)
    return img


