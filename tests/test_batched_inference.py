from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from ldt.domains.base import DomainSpec, PuzzleInstance, ValidationResult
from ldt.lattice import LatticeState
from ldt.solve import BatchedInferenceEngine, InferenceConfig, LatticeStepConfig


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


@dataclass(frozen=True)
class TinyPuzzle:
    solution: tuple[int, int] = (0, 1)


class TinyBinaryDomain:
    def __init__(self) -> None:
        self._spec = DomainSpec(
            name="tiny",
            num_positions=2,
            vocab_size=2,
            active=np.ones(2, dtype=np.bool_),
            position_shape=(1, 2),
        )

    @property
    def spec(self) -> DomainSpec:
        return self._spec

    def encode(self, puzzle: TinyPuzzle, *, puzzle_id: str = "") -> PuzzleInstance:
        return PuzzleInstance(
            puzzle_id=puzzle_id,
            initial_state=LatticeState(np.ones((2, 2), dtype=np.bool_)),
            solutions=np.asarray([puzzle.solution], dtype=np.int64),
            raw=puzzle,
        )

    def solution_to_state(self, solution: tuple[int, int]) -> LatticeState:
        candidates = np.zeros((2, 2), dtype=np.bool_)
        for position, value in enumerate(solution):
            candidates[position, value] = True
        return LatticeState(candidates)

    def decode_solution(self, state: LatticeState) -> tuple[int, int]:
        decoded = state.as_solution_indices()
        return int(decoded[0]), int(decoded[1])

    def validate_solution(self, puzzle: TinyPuzzle, state: LatticeState) -> ValidationResult:
        if bool(state.is_conflict()):
            return ValidationResult(False, "conflict")
        if not bool(state.is_complete()):
            return ValidationResult(False, "partial")
        return ValidationResult(self.decode_solution(state) == puzzle.solution)


class StaticModel(nn.Module):
    def __init__(
        self,
        candidate_logits: Tensor,
        cls_logits: Tensor,
        *,
        sequence: list[tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.base_candidate_logits = candidate_logits
        self.base_cls_logits = cls_logits
        self.sequence = sequence or []
        self.calls = 0

    def forward(
        self,
        candidates: Tensor,
        extra_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del extra_features
        if self.calls < len(self.sequence):
            candidate_logits, cls_logits = self.sequence[self.calls]
        else:
            candidate_logits, cls_logits = self.base_candidate_logits, self.base_cls_logits
        self.calls += 1
        batch = candidates.shape[0]
        return (
            candidate_logits.to(candidates.device).expand(batch, -1, -1),
            cls_logits.to(candidates.device).expand(batch),
        )


def _solving_logits(solution: tuple[int, int] = (0, 1)) -> Tensor:
    logits = torch.full((2, 2), _logit(0.01))
    for position, value in enumerate(solution):
        logits[position, value] = _logit(0.99)
    return logits


def test_inference_accepts_valid_solutions_across_slots_and_reports_metrics() -> None:
    model = StaticModel(_solving_logits(), torch.tensor(_logit(0.01)))
    engine = BatchedInferenceEngine(
        model,
        TinyBinaryDomain(),
        InferenceConfig(
            num_slots=2,
            chains_per_slot=3,
            round_budget=5,
            step=LatticeStepConfig(theta_elim=0.1, enable_branching=False),
        ),
    )

    results, metrics = engine.solve([TinyPuzzle(), TinyPuzzle(), TinyPuzzle()])

    assert [result.accepted for result in results] == [True, True, True]
    assert [result.solution for result in results] == [(0, 1), (0, 1), (0, 1)]
    assert metrics.total == 3
    assert metrics.accepted == 3
    assert metrics.abstained == 0
    assert metrics.model_forwards == 2
    assert metrics.row_forwards == 9
    assert metrics.solve_rate == 1.0


def test_inference_abstains_on_round_budget_timeout() -> None:
    model = StaticModel(torch.full((2, 2), _logit(0.99)), torch.tensor(_logit(0.01)))
    engine = BatchedInferenceEngine(
        model,
        TinyBinaryDomain(),
        InferenceConfig(
            num_slots=1,
            chains_per_slot=2,
            round_budget=2,
            step=LatticeStepConfig(theta_elim=0.1, enable_branching=False),
        ),
    )

    results, metrics = engine.solve([TinyPuzzle()])

    assert len(results) == 1
    assert results[0].abstained
    assert results[0].reason == "timeout"
    assert results[0].rounds == 2
    assert metrics.timeouts == 1
    assert metrics.model_forwards == 2
    assert metrics.row_forwards == 4
    assert metrics.abstention_rate == 1.0


def test_conflicted_chains_reset_and_can_accept_on_later_round() -> None:
    conflict_logits = (torch.full((2, 2), _logit(0.99)), torch.tensor(_logit(0.99)))
    solve_logits = (_solving_logits(), torch.tensor(_logit(0.01)))
    model = StaticModel(
        _solving_logits(),
        torch.tensor(_logit(0.01)),
        sequence=[conflict_logits, solve_logits],
    )
    engine = BatchedInferenceEngine(
        model,
        TinyBinaryDomain(),
        InferenceConfig(
            num_slots=1,
            chains_per_slot=2,
            round_budget=3,
            step=LatticeStepConfig(theta_elim=0.1, theta_cls=0.6, enable_branching=False),
        ),
    )

    results, metrics = engine.solve([TinyPuzzle()])

    assert len(results) == 1
    assert results[0].accepted
    assert results[0].rounds == 2
    assert results[0].conflicts == 2
    assert metrics.conflicts == 2
    assert metrics.model_forwards == 2


def test_invalid_complete_solutions_are_rejected_and_counted_as_abstentions() -> None:
    model = StaticModel(_solving_logits((1, 0)), torch.tensor(_logit(0.01)))
    engine = BatchedInferenceEngine(
        model,
        TinyBinaryDomain(),
        InferenceConfig(
            num_slots=1,
            chains_per_slot=2,
            round_budget=1,
            step=LatticeStepConfig(theta_elim=0.1, enable_branching=False),
        ),
    )

    results, metrics = engine.solve([TinyPuzzle((0, 1))])

    assert len(results) == 1
    assert results[0].abstained
    assert results[0].invalid_solutions == 2
    assert metrics.invalid_solutions == 2
    assert metrics.accepted == 0


def test_inference_restores_model_training_mode_after_solve() -> None:
    model = StaticModel(_solving_logits(), torch.tensor(_logit(0.01)))
    model.train(True)
    engine = BatchedInferenceEngine(
        model,
        TinyBinaryDomain(),
        InferenceConfig(
            num_slots=1,
            chains_per_slot=1,
            round_budget=1,
            step=LatticeStepConfig(theta_elim=0.1, enable_branching=False),
            model_train_mode=False,
        ),
    )

    engine.solve([TinyPuzzle()])

    assert model.training
