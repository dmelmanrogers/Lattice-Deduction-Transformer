from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ldt.lattice import LatticeState
from ldt.model import LDTOutput
from ldt.solve import (
    LatticeStepConfig,
    OnPolicyTrainingPool,
    PoolUpdateStats,
    lattice_projection_step,
)
from ldt.train.loss import LDTLossConfig, LDTLossOutput, LDTTargets, compute_loss, compute_targets

ExtraFeaturesFn = Callable[[Any, LatticeState], NDArray[np.float32] | Tensor | None]


@dataclass(frozen=True)
class TrainingStepOutput:
    loss: LDTLossOutput
    targets: LDTTargets
    pool_stats: PoolUpdateStats
    model_output: LDTOutput | Any


def run_training_step(
    model: torch.nn.Module,
    pool: OnPolicyTrainingPool,
    *,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    loss_config: LDTLossConfig | None = None,
    step_config: LatticeStepConfig | None = None,
    device: str | torch.device = "cpu",
    extra_features_fn: ExtraFeaturesFn | None = None,
    grad_clip: float | None = None,
) -> TrainingStepOutput:
    """Run one training-ready on-policy step.

    Passing an optimizer performs a single update. Omitting it computes the
    objective, projection, and pool update without changing model parameters,
    which is useful for smoke tests and dry runs.
    """

    batch = pool.sample(batch_size)
    candidates = torch.as_tensor(
        np.array(batch.state.candidates, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    extra_features = _pack_extra_features(batch.entries, batch.state, extra_features_fn, device)

    model.train(True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    model_output = model(candidates, extra_features)
    candidate_logits, cls_logits = _logits_for_loss(model_output)
    targets = compute_targets(
        batch.state,
        batch.solutions,
        last_nonempty_target=_stack_previous_targets(batch.entries),
        device=candidate_logits.device,
    )
    loss = compute_loss(candidate_logits, cls_logits, targets, loss_config)

    if optimizer is not None:
        loss.total.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    step_result = lattice_projection_step(
        batch.state,
        _final_candidate_logits(model_output),
        _final_cls_logits(model_output),
        config=step_config,
        solutions=batch.solutions,
    )
    pool_stats = pool.apply_step_result(batch, step_result, targets.next_last_nonempty_target)
    return TrainingStepOutput(
        loss=loss,
        targets=targets,
        pool_stats=pool_stats,
        model_output=model_output,
    )


def _logits_for_loss(model_output: LDTOutput | Any) -> tuple[Tensor, Tensor]:
    if hasattr(model_output, "candidate_logits") and hasattr(model_output, "cls_logits"):
        return model_output.candidate_logits, model_output.cls_logits
    candidate_logits, cls_logits = model_output
    if candidate_logits.ndim == 3:
        candidate_logits = candidate_logits.unsqueeze(0)
    if cls_logits.ndim == 1:
        cls_logits = cls_logits.unsqueeze(0)
    return candidate_logits, cls_logits


def _final_candidate_logits(model_output: LDTOutput | Any) -> Any:
    if hasattr(model_output, "final_candidate_logits"):
        return model_output.final_candidate_logits
    candidate_logits, _ = model_output
    return candidate_logits[-1] if getattr(candidate_logits, "ndim", 0) == 4 else candidate_logits


def _final_cls_logits(model_output: LDTOutput | Any) -> Any:
    if hasattr(model_output, "final_cls_logits"):
        return model_output.final_cls_logits
    _, cls_logits = model_output
    return cls_logits[-1] if getattr(cls_logits, "ndim", 0) == 2 else cls_logits


def _stack_previous_targets(entries: tuple[Any, ...]) -> LatticeState:
    candidates = np.stack([entry.last_nonempty_target.candidates for entry in entries], axis=0)
    active = np.stack([entry.last_nonempty_target.active for entry in entries], axis=0)
    return LatticeState(candidates, active)


def _pack_extra_features(
    entries: tuple[Any, ...],
    state: LatticeState,
    extra_features_fn: ExtraFeaturesFn | None,
    device: str | torch.device,
) -> Tensor | None:
    if extra_features_fn is None:
        return None

    flat_candidates = state.candidates.reshape((-1, state.num_positions, state.vocab_size))
    flat_active = state.active.reshape((-1, state.num_positions))
    features: list[NDArray[np.float32] | Tensor] = []
    for entry, candidates, active in zip(entries, flat_candidates, flat_active, strict=True):
        feature = extra_features_fn(entry.raw, LatticeState(candidates, active))
        if feature is None:
            raise ValueError("extra_features_fn returned None")
        features.append(feature)

    if isinstance(features[0], Tensor):
        tensor_features = [feature for feature in features if isinstance(feature, Tensor)]
        return torch.stack(tensor_features).to(device)
    return torch.as_tensor(np.stack(features, axis=0), dtype=torch.float32, device=device)
