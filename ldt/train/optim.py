from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)


def create_adamw(model: torch.nn.Module, config: AdamWConfig | None = None) -> torch.optim.AdamW:
    optimizer_config = config or AdamWConfig()
    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
    )


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
