from __future__ import annotations

import torch


def update_bn_stats(model: torch.nn.Module, data_iter, device: torch.device) -> None:
    model.train()
    # Freeze params
    for p in model.parameters():
        p.requires_grad = False
    # Enable BN layers to update running stats
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
    with torch.no_grad():
        for batch in data_iter:
            input_ids = batch["input_ids"].to(device)
            lengths = batch.get("lengths")
            if lengths is not None:
                lengths = lengths.to(device)
            if hasattr(model, "classifier"):
                model.classifier.backbone(input_ids=input_ids, lengths=lengths)
            else:
                model.backbone(input_ids=input_ids, lengths=lengths)


