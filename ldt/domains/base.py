from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ldt.lattice import LatticeState


@dataclass(frozen=True)
class DomainSpec:
    """Shape and masking information needed by domain-agnostic LDT components."""

    name: str
    num_positions: int
    vocab_size: int
    active: NDArray[np.bool_]
    position_shape: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        active = np.asarray(self.active, dtype=np.bool_)
        if active.shape != (self.num_positions,):
            raise ValueError("active must have shape (num_positions,)")
        if self.num_positions <= 0:
            raise ValueError("num_positions must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        active_copy = active.copy()
        active_copy.flags.writeable = False
        object.__setattr__(self, "active", active_copy)


@dataclass(frozen=True)
class PuzzleInstance:
    """Encoded puzzle plus optional concrete solution samples for training."""

    puzzle_id: str
    initial_state: LatticeState
    solutions: NDArray[np.int64] | None = None
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LatticeDomain(Protocol):
    """Solver-facing interface implemented by every benchmark/domain adapter."""

    @property
    def spec(self) -> DomainSpec:
        """Return static lattice shape and active-position metadata."""

    def encode(self, puzzle: Any, *, puzzle_id: str = "") -> PuzzleInstance:
        """Encode a raw puzzle into an initial lattice state and optional solutions."""

    def solution_to_state(self, solution: Any) -> LatticeState:
        """Convert a concrete solution representation into a singleton lattice state."""

    def decode_solution(self, state: LatticeState) -> Any:
        """Convert a complete singleton lattice state into the domain's solution object."""

    def validate_solution(self, puzzle: Any, state: LatticeState) -> ValidationResult:
        """Validate a completed lattice state against the raw puzzle constraints."""
