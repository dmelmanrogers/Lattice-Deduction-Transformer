from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ldt.domains.base import LatticeDomain
from ldt.solve import BatchedInferenceEngine, InferenceConfig, InferenceMetrics, PuzzleSolveResult


@dataclass(frozen=True)
class EvaluationSummary:
    results: list[PuzzleSolveResult]
    metrics: InferenceMetrics

    @property
    def solved_ids(self) -> list[str]:
        return [result.puzzle_id for result in self.results if result.accepted]


def evaluate_puzzles(
    model: Any,
    domain: LatticeDomain,
    puzzles: Iterable[Any],
    *,
    config: InferenceConfig | None = None,
) -> EvaluationSummary:
    engine = BatchedInferenceEngine(model, domain, config)
    results, metrics = engine.solve(puzzles)
    return EvaluationSummary(results=results, metrics=metrics)
