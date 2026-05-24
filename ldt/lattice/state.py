from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True)
class LatticeState:
    """A powerset lattice state over fixed positions and finite candidate vocabularies.

    `candidates[..., p, v]` is true when candidate value `v` is still possible at
    position `p`. `active[..., p]` marks positions that belong to the current
    domain instance. Inactive positions are excluded from completion/conflict
    checks and are normalized to have no candidates.
    """

    candidates: BoolArray
    active: BoolArray

    def __init__(
        self,
        candidates: NDArray[np.bool_] | np.ndarray,
        active: NDArray[np.bool_] | np.ndarray | None = None,
    ) -> None:
        candidate_array = np.asarray(candidates, dtype=np.bool_)
        if candidate_array.ndim < 2:
            raise ValueError("candidates must have shape (..., positions, vocab_size)")

        positions = candidate_array.shape[-2]
        batch_shape = candidate_array.shape[:-2]
        if active is None:
            active_array = np.ones((*batch_shape, positions), dtype=np.bool_)
        else:
            active_array = np.asarray(active, dtype=np.bool_)
            expected_shapes = {batch_shape + (positions,), (positions,)}
            if active_array.shape not in expected_shapes:
                raise ValueError(
                    "active must have shape (..., positions) matching candidates or (positions,)"
                )
            if active_array.shape == (positions,) and batch_shape:
                active_array = np.broadcast_to(active_array, (*batch_shape, positions)).copy()

        normalized = candidate_array.copy()
        normalized[~active_array, :] = False

        active_copy = active_array.copy()
        normalized.flags.writeable = False
        active_copy.flags.writeable = False

        object.__setattr__(self, "candidates", normalized)
        object.__setattr__(self, "active", active_copy)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.candidates.shape[:-2]

    @property
    def num_positions(self) -> int:
        return self.candidates.shape[-2]

    @property
    def vocab_size(self) -> int:
        return self.candidates.shape[-1]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.candidates.shape

    def copy(self) -> LatticeState:
        return LatticeState(self.candidates.copy(), self.active.copy())

    def candidate_counts(self) -> NDArray[np.int64]:
        return self.candidates.sum(axis=-1, dtype=np.int64)

    def singleton_mask(self) -> BoolArray:
        return (self.candidate_counts() == 1) & self.active

    def conflict_mask(self) -> BoolArray:
        return (self.candidate_counts() == 0) & self.active

    def complete_mask(self) -> BoolArray:
        return self.singleton_mask() | ~self.active

    def is_conflict(self) -> NDArray[np.bool_] | np.bool_:
        return self.conflict_mask().any(axis=-1)

    def is_complete(self) -> NDArray[np.bool_] | np.bool_:
        return self.complete_mask().all(axis=-1) & ~self.is_conflict()

    def as_solution_indices(self, inactive_value: int = -1) -> NDArray[np.int64]:
        """Return singleton indices for complete rows, using `inactive_value` elsewhere.

        Raises if any active position is not a singleton. This makes accidental
        decoding of partial lattice states explicit.
        """

        if np.any(~self.complete_mask() & self.active):
            raise ValueError("cannot decode a partial lattice state as a concrete solution")
        if np.any(self.conflict_mask()):
            raise ValueError("cannot decode a conflicted lattice state as a concrete solution")

        indices = self.candidates.argmax(axis=-1).astype(np.int64, copy=False)
        return np.where(self.active, indices, inactive_value)

    def with_candidates(self, candidates: NDArray[np.bool_] | np.ndarray) -> LatticeState:
        return LatticeState(candidates, self.active)

    def require_compatible(self, other: LatticeState) -> None:
        if self.candidates.shape != other.candidates.shape:
            raise ValueError(
                f"candidate shapes differ: {self.candidates.shape} != {other.candidates.shape}"
            )
        if not np.array_equal(self.active, other.active):
            raise ValueError("active masks differ")
