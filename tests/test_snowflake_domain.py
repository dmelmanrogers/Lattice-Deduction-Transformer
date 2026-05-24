from __future__ import annotations

import numpy as np

from ldt.domains.snowflake import SNOWFLAKE_POSITIONS, SnowflakeDomain, make_snowflake_puzzle


def test_snowflake_encodes_givens_and_in_puzzle_mask() -> None:
    active = np.zeros(SNOWFLAKE_POSITIONS, dtype=np.bool_)
    active[:6] = True
    givens = np.zeros(SNOWFLAKE_POSITIONS, dtype=np.int64)
    givens[0] = 1
    solution = np.zeros(SNOWFLAKE_POSITIONS, dtype=np.int64)
    solution[:6] = [1, 2, 3, 4, 5, 6]
    puzzle = make_snowflake_puzzle(givens, [range(6)], active=active, solution=solution)
    domain = SnowflakeDomain(active)

    instance = domain.encode(puzzle)

    assert instance.initial_state.candidates.shape == (150, 6)
    assert np.array_equal(
        instance.initial_state.candidates[0],
        [True, False, False, False, False, False],
    )
    assert not instance.initial_state.active[6]
    assert instance.solutions is not None
    assert instance.solutions[0, 0] == 0
    assert np.array_equal(domain.extra_features(puzzle, instance.initial_state)[:7, 0], active[:7])


def test_snowflake_validates_all_different_groups() -> None:
    active = np.zeros(SNOWFLAKE_POSITIONS, dtype=np.bool_)
    active[:6] = True
    givens = np.zeros(SNOWFLAKE_POSITIONS, dtype=np.int64)
    solution = np.zeros(SNOWFLAKE_POSITIONS, dtype=np.int64)
    solution[:6] = [1, 2, 3, 4, 5, 6]
    puzzle = make_snowflake_puzzle(givens, [range(6)], active=active, solution=solution)
    domain = SnowflakeDomain(active)

    valid = domain.validate_solution(puzzle, domain.solution_to_state(solution))
    assert valid.valid

    bad = solution.copy()
    bad[1] = 1
    invalid = domain.validate_solution(
        make_snowflake_puzzle(givens, [range(6)], active=active),
        domain.solution_to_state(bad),
    )
    assert not invalid.valid
    assert "duplicate" in invalid.reason
