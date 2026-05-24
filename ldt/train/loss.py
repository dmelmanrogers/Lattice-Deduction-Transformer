from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional

from ldt.lattice import LatticeState, alpha, consistent_solutions, meet


@dataclass(frozen=True)
class LDTLossConfig:
    positive_weight: float = 4.0
    negative_weight: float = 0.5
    lambda_cls: float = 0.1
    lambda_ce: float = 0.2

    def __post_init__(self) -> None:
        if self.positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        if self.negative_weight <= 0:
            raise ValueError("negative_weight must be positive")
        if self.lambda_cls < 0:
            raise ValueError("lambda_cls must be non-negative")
        if self.lambda_ce < 0:
            raise ValueError("lambda_ce must be non-negative")


@dataclass(frozen=True)
class LDTTargets:
    candidate_target: Tensor
    cls_target: Tensor
    active_mask: Tensor
    ce_mask: Tensor
    ce_target: Tensor
    no_survivors: NDArray[np.bool_]
    next_last_nonempty_target: LatticeState


@dataclass(frozen=True)
class LDTLossOutput:
    total: Tensor
    candidate_bce: Tensor
    cls_bce: Tensor
    singleton_ce: Tensor


def compute_targets(
    state: LatticeState,
    solutions: NDArray[np.integer[Any]] | np.ndarray,
    *,
    last_nonempty_target: LatticeState | None = None,
    device: torch.device | str | None = None,
    inactive_value: int = -1,
) -> LDTTargets:
    """Compute state-conditional paper targets for a batch of lattice states."""

    flat_state, active = _flatten_state(state)
    solution_array = _normalize_solutions(solutions, flat_state.shape[0], state.num_positions)
    previous = _normalize_previous_target(last_nonempty_target, state)

    targets = np.zeros_like(flat_state, dtype=np.bool_)
    next_previous = previous.copy()
    no_survivors = np.zeros(flat_state.shape[0], dtype=np.bool_)

    for row_idx in range(flat_state.shape[0]):
        row_state = LatticeState(flat_state[row_idx], active[row_idx])
        row_solutions = solution_array[row_idx]
        valid_rows = ~np.all(row_solutions == inactive_value, axis=1)
        row_solutions = row_solutions[valid_rows]
        consistent = consistent_solutions(
            row_state,
            row_solutions,
            inactive_value=inactive_value,
        )

        if consistent.any():
            abstract = alpha(
                row_solutions[consistent],
                vocab_size=state.vocab_size,
                active=active[row_idx],
                inactive_value=inactive_value,
            )
            target_state = meet(row_state, abstract)
            next_previous[row_idx] = target_state.candidates
        else:
            target_state = meet(row_state, LatticeState(previous[row_idx], active[row_idx]))
            no_survivors[row_idx] = True

        targets[row_idx] = target_state.candidates

    target_state = LatticeState(targets.reshape(state.candidates.shape), state.active)
    previous_state = LatticeState(next_previous.reshape(state.candidates.shape), state.active)
    target_counts = targets.sum(axis=-1)
    ce_mask_np = (target_counts == 1) & active
    ce_target_np = targets.argmax(axis=-1).astype(np.int64)

    torch_device = device or "cpu"
    return LDTTargets(
        candidate_target=torch.as_tensor(
            targets.reshape(state.candidates.shape),
            dtype=torch.float32,
            device=torch_device,
        ),
        cls_target=torch.as_tensor(
            no_survivors.reshape(state.batch_shape),
            dtype=torch.float32,
            device=torch_device,
        ),
        active_mask=torch.as_tensor(
            np.array(state.active, dtype=np.bool_, copy=True),
            dtype=torch.bool,
            device=torch_device,
        ),
        ce_mask=torch.as_tensor(
            ce_mask_np.reshape(state.active.shape),
            dtype=torch.bool,
            device=torch_device,
        ),
        ce_target=torch.as_tensor(
            ce_target_np.reshape(state.active.shape),
            dtype=torch.long,
            device=torch_device,
        ),
        no_survivors=no_survivors.reshape(state.batch_shape),
        next_last_nonempty_target=previous_state,
    )


def compute_loss(
    candidate_logits: Tensor,
    cls_logits: Tensor,
    targets: LDTTargets,
    config: LDTLossConfig | None = None,
) -> LDTLossOutput:
    """Compute the per-internal-iteration LDT objective."""

    loss_config = config or LDTLossConfig()
    if candidate_logits.ndim != 4:
        raise ValueError("candidate_logits must have shape (loops, batch, positions, vocab)")
    if cls_logits.ndim != 2:
        raise ValueError("cls_logits must have shape (loops, batch)")

    loop_count = candidate_logits.shape[0]
    candidate_target = targets.candidate_target.to(candidate_logits.device)
    cls_target = targets.cls_target.to(cls_logits.device)
    active_mask = targets.active_mask.to(candidate_logits.device)
    ce_mask = targets.ce_mask.to(candidate_logits.device)
    ce_target = targets.ce_target.to(candidate_logits.device)
    if candidate_target.ndim == 2 and candidate_logits.ndim == 4:
        candidate_target = candidate_target.unsqueeze(0)
    if active_mask.ndim == 1 and candidate_logits.ndim == 4:
        active_mask = active_mask.unsqueeze(0)
    if ce_mask.ndim == 1 and candidate_logits.ndim == 4:
        ce_mask = ce_mask.unsqueeze(0)
        ce_target = ce_target.unsqueeze(0)
    if cls_target.ndim == 0 and cls_logits.ndim == 2:
        cls_target = cls_target.unsqueeze(0)

    candidate_losses: list[Tensor] = []
    cls_losses: list[Tensor] = []
    ce_losses: list[Tensor] = []

    for loop_idx in range(loop_count):
        candidate_losses.append(
            _weighted_candidate_bce(
                candidate_logits[loop_idx],
                candidate_target,
                active_mask,
                loss_config,
            )
        )
        cls_losses.append(
            functional.binary_cross_entropy_with_logits(cls_logits[loop_idx], cls_target)
        )
        ce_losses.append(
            _singleton_ce(candidate_logits[loop_idx], ce_mask, ce_target)
        )

    candidate_bce = torch.stack(candidate_losses).mean()
    cls_bce = torch.stack(cls_losses).mean()
    singleton_ce = torch.stack(ce_losses).mean()
    total = (
        candidate_bce
        + loss_config.lambda_cls * cls_bce
        + loss_config.lambda_ce * singleton_ce
    )
    return LDTLossOutput(
        total=total,
        candidate_bce=candidate_bce.detach(),
        cls_bce=cls_bce.detach(),
        singleton_ce=singleton_ce.detach(),
    )


def _weighted_candidate_bce(
    logits: Tensor,
    target: Tensor,
    active_mask: Tensor,
    config: LDTLossConfig,
) -> Tensor:
    loss = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(
        target.bool(),
        torch.full_like(target, config.positive_weight),
        torch.full_like(target, config.negative_weight),
    )
    mask = active_mask.unsqueeze(-1).to(dtype=loss.dtype)
    denom = mask.sum() * logits.shape[-1]
    return (loss * weights * mask).sum() / denom.clamp_min(1.0)


def _singleton_ce(logits: Tensor, ce_mask: Tensor, ce_target: Tensor) -> Tensor:
    if not bool(ce_mask.any()):
        return logits.sum() * 0.0
    selected_logits = logits[ce_mask]
    selected_target = ce_target[ce_mask]
    return functional.cross_entropy(selected_logits, selected_target)


def _flatten_state(state: LatticeState) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    return (
        state.candidates.reshape((-1, state.num_positions, state.vocab_size)),
        state.active.reshape((-1, state.num_positions)),
    )


def _normalize_solutions(
    solutions: NDArray[np.integer[Any]] | np.ndarray,
    batch_size: int,
    num_positions: int,
) -> NDArray[np.int64]:
    solution_array = np.asarray(solutions, dtype=np.int64)
    if solution_array.ndim == 2:
        if solution_array.shape[1] != num_positions:
            raise ValueError("solutions have wrong num_positions")
        solution_array = np.broadcast_to(solution_array, (batch_size, *solution_array.shape)).copy()
    elif solution_array.ndim == 3:
        if solution_array.shape[0] != batch_size or solution_array.shape[2] != num_positions:
            raise ValueError("batched solutions must have shape (batch, K, positions)")
    else:
        raise ValueError("solutions must have shape (K, positions) or (batch, K, positions)")
    return solution_array


def _normalize_previous_target(
    previous: LatticeState | None,
    state: LatticeState,
) -> NDArray[np.bool_]:
    if previous is None:
        return state.candidates.reshape((-1, state.num_positions, state.vocab_size)).copy()
    previous.require_compatible(state)
    return previous.candidates.reshape((-1, state.num_positions, state.vocab_size)).copy()
