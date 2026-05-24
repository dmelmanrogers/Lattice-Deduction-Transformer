from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int = 0
    epoch: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: CheckpointMetadata | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "metadata": asdict(metadata or CheckpointMetadata()),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    metadata = payload.get("metadata", {})
    return CheckpointMetadata(**metadata)
