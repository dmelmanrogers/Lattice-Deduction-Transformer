from __future__ import annotations

import pytest

from ldt.domains import LatticeDomain, ValidationResult
from ldt.domains.sudoku import SudokuDomain


def _accepts_domain_protocol(domain: LatticeDomain) -> ValidationResult:
    puzzle = (
        "530070000600195000098000060800060003400803001700020006"
        "060000280000419005000080079"
    )
    instance = domain.encode(puzzle, puzzle_id="fixture")
    return domain.validate_solution(puzzle, instance.initial_state)


def test_sudoku_domain_implements_solver_facing_protocol() -> None:
    result = _accepts_domain_protocol(SudokuDomain())

    assert not result.valid
    assert result.reason == "state is not complete"


def test_domain_spec_is_copied_on_construction() -> None:
    domain = SudokuDomain()
    active = domain.spec.active

    with pytest.raises(ValueError, match="read-only"):
        active[:] = False
