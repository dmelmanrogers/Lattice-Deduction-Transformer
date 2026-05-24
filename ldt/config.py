from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ldt.model import LDTConfig
from ldt.solve import InferenceConfig, LatticeStepConfig, PoolConfig
from ldt.train import LDTLossConfig


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.1


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    domain: str
    model: LDTConfig
    loss: LDTLossConfig = field(default_factory=LDTLossConfig)
    step: LatticeStepConfig = field(default_factory=LatticeStepConfig)
    pool: PoolConfig = field(default_factory=lambda: PoolConfig(pool_size=512))
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text())
    model_data = dict(data["model"])
    if "position_shape" in model_data:
        model_data["position_shape"] = tuple(model_data["position_shape"])
    optimizer_data = dict(data.get("optimizer", {}))
    if "betas" in optimizer_data:
        optimizer_data["betas"] = tuple(optimizer_data["betas"])
    return ExperimentConfig(
        name=data["name"],
        domain=data["domain"],
        model=LDTConfig(**model_data),
        loss=LDTLossConfig(**data.get("loss", {})),
        step=LatticeStepConfig(**data.get("step", {})),
        pool=PoolConfig(**data.get("pool", {"pool_size": 512})),
        inference=_load_inference_config(data.get("inference", {})),
        optimizer=OptimizerConfig(**optimizer_data),
        metadata=data.get("metadata", {}),
    )


def _load_inference_config(data: dict[str, Any]) -> InferenceConfig:
    step_data = data.get("step")
    values = dict(data)
    if step_data is not None:
        values["step"] = LatticeStepConfig(**step_data)
    return InferenceConfig(**values)
