from __future__ import annotations

import numpy as np

from ldt.domains.maze import (
    MazeDomain,
    MazePuzzle,
    enumerate_shortest_paths,
    shortest_path_distance,
)


def test_maze_encodes_static_features_and_samples_shortest_paths() -> None:
    walls = np.zeros((3, 3), dtype=np.bool_)
    puzzle = MazePuzzle(walls, (0, 0), (0, 2))
    domain = MazeDomain(3, 3)
    solutions = domain.sample_shortest_solutions(
        puzzle,
        max_solutions=2,
        rng=np.random.default_rng(0),
    )
    puzzle = MazePuzzle(walls, (0, 0), (0, 2), solutions)

    instance = domain.encode(puzzle)
    features = domain.extra_features(puzzle, instance.initial_state)

    assert instance.initial_state.candidates.shape == (9, 2)
    assert np.array_equal(instance.initial_state.candidates[0], [False, True])
    assert instance.solutions is not None
    assert instance.solutions.shape[1] == 9
    assert features.shape == (9, 3)
    assert features[0, 1] == 1.0
    assert features[2, 2] == 1.0


def test_maze_validates_shortest_simple_path() -> None:
    walls = np.zeros((3, 3), dtype=np.bool_)
    puzzle = MazePuzzle(walls, (0, 0), (0, 2))
    domain = MazeDomain(3, 3)
    path = [(0, 0), (0, 1), (0, 2)]

    result = domain.validate_solution(puzzle, domain.solution_to_state(path))

    assert result.valid


def test_maze_rejects_non_shortest_path_when_required() -> None:
    walls = np.zeros((3, 3), dtype=np.bool_)
    puzzle = MazePuzzle(walls, (0, 0), (0, 2))
    domain = MazeDomain(3, 3)
    long_path = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]

    result = domain.validate_solution(puzzle, domain.solution_to_state(long_path))

    assert not result.valid
    assert result.reason == "path is not shortest"


def test_maze_shortest_path_helpers() -> None:
    walls = np.zeros((2, 3), dtype=np.bool_)
    walls[0, 1] = True

    assert shortest_path_distance(walls, (0, 0), (0, 2)) == 4
    paths = enumerate_shortest_paths(
        walls,
        (0, 0),
        (0, 2),
        max_paths=4,
        rng=np.random.default_rng(0),
    )
    assert paths
    assert all(len(path) == 5 for path in paths)
