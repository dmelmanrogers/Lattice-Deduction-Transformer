from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ldt.domains.base import LatticeDomain, PuzzleInstance
from ldt.lattice import LatticeState
from ldt.solve.pool import _stack_states
from ldt.solve.step import LatticeStepConfig, lattice_projection_step


class LDTModelLike(Protocol):
    training: bool

    def train(self, mode: bool = True) -> Any: ...

    def __call__(self, candidates: Tensor, extra_features: Tensor | None = None) -> Any: ...


ExtraFeaturesFn = Callable[[PuzzleInstance, LatticeState], NDArray[np.float32] | Tensor | None]


@dataclass(frozen=True)
class InferenceConfig:
    num_slots: int = 8
    chains_per_slot: int = 64
    round_budget: int = 1000
    step: LatticeStepConfig = LatticeStepConfig(theta_cls=0.6)
    device: str | torch.device = "cpu"
    model_train_mode: bool = False

    def __post_init__(self) -> None:
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.chains_per_slot <= 0:
            raise ValueError("chains_per_slot must be positive")
        if self.round_budget <= 0:
            raise ValueError("round_budget must be positive")


@dataclass(frozen=True)
class PuzzleSolveResult:
    puzzle_id: str
    accepted: bool
    abstained: bool
    reason: str
    rounds: int
    conflicts: int
    invalid_solutions: int
    accepted_chain: int | None
    solution_state: LatticeState | None
    solution: Any | None


@dataclass(frozen=True)
class InferenceMetrics:
    total: int
    accepted: int
    abstained: int
    timeouts: int
    conflicts: int
    invalid_solutions: int
    model_forwards: int
    row_forwards: int
    total_rounds: int

    @property
    def solve_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.abstained / self.total if self.total else 0.0


@dataclass
class _Slot:
    instance: PuzzleInstance
    chains: list[LatticeState]
    rounds: int = 0
    conflicts: int = 0
    invalid_solutions: int = 0

    @classmethod
    def from_instance(cls, instance: PuzzleInstance, chains_per_slot: int) -> _Slot:
        return cls(
            instance=instance,
            chains=[instance.initial_state.copy() for _ in range(chains_per_slot)],
        )


class BatchedInferenceEngine:
    """Batched LDT inference with puzzle slots and parallel stochastic chains."""

    def __init__(
        self,
        model: LDTModelLike,
        domain: LatticeDomain,
        config: InferenceConfig | None = None,
        *,
        rng: np.random.Generator | None = None,
        extra_features_fn: ExtraFeaturesFn | None = None,
    ) -> None:
        self.model = model
        self.domain = domain
        self.config = config or InferenceConfig()
        self.rng = rng or np.random.default_rng()
        self.extra_features_fn = extra_features_fn

    def solve(self, puzzles: Iterable[Any]) -> tuple[list[PuzzleSolveResult], InferenceMetrics]:
        queue = list(puzzles)
        if not queue:
            return [], _metrics_from_results([], 0, 0)

        previous_mode = self.model.training
        self.model.train(self.config.model_train_mode)
        try:
            return self._solve_with_current_model_mode(queue)
        finally:
            self.model.train(previous_mode)

    def _solve_with_current_model_mode(
        self,
        queue: list[Any],
    ) -> tuple[list[PuzzleSolveResult], InferenceMetrics]:
        results: list[PuzzleSolveResult] = []
        active_slots: list[_Slot] = []
        next_puzzle = 0
        model_forwards = 0
        row_forwards = 0

        while next_puzzle < len(queue) and len(active_slots) < self.config.num_slots:
            active_slots.append(self._new_slot(queue[next_puzzle], next_puzzle))
            next_puzzle += 1

        while active_slots:
            batch_state, slot_rows = self._pack_slots(active_slots)
            candidates = torch.as_tensor(
                np.array(batch_state.candidates, dtype=np.float32, copy=True),
                dtype=torch.float32,
                device=self.config.device,
            )
            extra_features = self._pack_extra_features(active_slots, slot_rows)

            with torch.no_grad():
                model_output = self.model(candidates, extra_features)
            candidate_logits, cls_logits = _final_logits(model_output)
            step_result = lattice_projection_step(
                batch_state,
                candidate_logits,
                cls_logits,
                config=self.config.step,
                rng=self.rng,
            )
            model_forwards += 1
            row_forwards += candidates.shape[0]

            completed_indices: list[int] = []
            for slot_idx, slot in enumerate(active_slots):
                start = slot_idx * self.config.chains_per_slot
                stop = start + self.config.chains_per_slot
                chain_result = _handle_slot_step(
                    slot,
                    step_result,
                    start,
                    stop,
                    self.domain,
                )
                slot.rounds += 1
                if chain_result is not None:
                    results.append(chain_result)
                    completed_indices.append(slot_idx)
                elif slot.rounds >= self.config.round_budget:
                    results.append(
                        PuzzleSolveResult(
                            puzzle_id=slot.instance.puzzle_id,
                            accepted=False,
                            abstained=True,
                            reason="timeout",
                            rounds=slot.rounds,
                            conflicts=slot.conflicts,
                            invalid_solutions=slot.invalid_solutions,
                            accepted_chain=None,
                            solution_state=None,
                            solution=None,
                        )
                    )
                    completed_indices.append(slot_idx)

            for slot_idx in reversed(completed_indices):
                del active_slots[slot_idx]
            while next_puzzle < len(queue) and len(active_slots) < self.config.num_slots:
                active_slots.append(self._new_slot(queue[next_puzzle], next_puzzle))
                next_puzzle += 1

        return results, _metrics_from_results(results, model_forwards, row_forwards)

    def _new_slot(self, puzzle: Any, puzzle_index: int) -> _Slot:
        instance = self.domain.encode(puzzle, puzzle_id=f"puzzle-{puzzle_index}")
        return _Slot.from_instance(instance, self.config.chains_per_slot)

    def _pack_slots(self, slots: list[_Slot]) -> tuple[LatticeState, list[tuple[int, int]]]:
        states: list[LatticeState] = []
        slot_rows: list[tuple[int, int]] = []
        row = 0
        for slot in slots:
            start = row
            states.extend(slot.chains)
            row += len(slot.chains)
            slot_rows.append((start, row))
        return _stack_states(tuple(states)), slot_rows

    def _pack_extra_features(
        self,
        slots: list[_Slot],
        slot_rows: list[tuple[int, int]],
    ) -> Tensor | None:
        if self.extra_features_fn is None:
            return None

        features: list[NDArray[np.float32] | Tensor] = []
        for slot, (start, stop) in zip(slots, slot_rows, strict=True):
            for chain in slot.chains:
                feature = self.extra_features_fn(slot.instance, chain)
                if feature is None:
                    raise ValueError(
                        "extra_features_fn returned None for a model that expects features"
                    )
                features.append(feature)
            if stop - start != len(slot.chains):
                raise ValueError("slot row bookkeeping mismatch")

        if not features:
            return None
        if isinstance(features[0], Tensor):
            return torch.stack([feature for feature in features if isinstance(feature, Tensor)]).to(
                self.config.device
            )
        return torch.as_tensor(
            np.stack(features, axis=0),
            dtype=torch.float32,
            device=self.config.device,
        )


def _handle_slot_step(
    slot: _Slot,
    step_result: Any,
    start: int,
    stop: int,
    domain: LatticeDomain,
) -> PuzzleSolveResult | None:
    for row_idx in range(start, stop):
        chain_idx = row_idx - start
        row_state = LatticeState(
            step_result.state.candidates[row_idx],
            step_result.state.active[row_idx],
        )
        if bool(step_result.solved[row_idx]):
            validation = domain.validate_solution(slot.instance.raw, row_state)
            if validation.valid:
                return PuzzleSolveResult(
                    puzzle_id=slot.instance.puzzle_id,
                    accepted=True,
                    abstained=False,
                    reason="accepted",
                    rounds=slot.rounds + 1,
                    conflicts=slot.conflicts,
                    invalid_solutions=slot.invalid_solutions,
                    accepted_chain=chain_idx,
                    solution_state=row_state,
                    solution=domain.decode_solution(row_state),
                )
            slot.invalid_solutions += 1
            slot.chains[chain_idx] = slot.instance.initial_state.copy()
            continue

        if bool(step_result.conflict[row_idx]):
            slot.conflicts += 1
            slot.chains[chain_idx] = slot.instance.initial_state.copy()
        else:
            slot.chains[chain_idx] = row_state
    return None


def _final_logits(model_output: Any) -> tuple[Any, Any]:
    has_final_logits = hasattr(model_output, "final_candidate_logits") and hasattr(
        model_output,
        "final_cls_logits",
    )
    if has_final_logits:
        return model_output.final_candidate_logits, model_output.final_cls_logits
    if isinstance(model_output, tuple) and len(model_output) == 2:
        candidate_logits, cls_logits = model_output
        if hasattr(candidate_logits, "ndim") and candidate_logits.ndim == 4:
            candidate_logits = candidate_logits[-1]
        if hasattr(cls_logits, "ndim") and cls_logits.ndim == 2:
            cls_logits = cls_logits[-1]
        return candidate_logits, cls_logits
    raise TypeError("model output must be LDTOutput-like or a (candidate_logits, cls_logits) tuple")


def _metrics_from_results(
    results: list[PuzzleSolveResult],
    model_forwards: int,
    row_forwards: int,
) -> InferenceMetrics:
    accepted = sum(result.accepted for result in results)
    abstained = sum(result.abstained for result in results)
    return InferenceMetrics(
        total=len(results),
        accepted=int(accepted),
        abstained=int(abstained),
        timeouts=sum(result.reason == "timeout" for result in results),
        conflicts=sum(result.conflicts for result in results),
        invalid_solutions=sum(result.invalid_solutions for result in results),
        model_forwards=model_forwards,
        row_forwards=row_forwards,
        total_rounds=sum(result.rounds for result in results),
    )
