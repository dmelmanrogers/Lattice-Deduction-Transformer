from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ldt.domains.base import DomainSpec, PuzzleInstance, ValidationResult
from ldt.lattice import LatticeState

NOT_PATH = 0
PATH = 1


@dataclass(frozen=True)
class MazePuzzle:
    walls: NDArray[np.bool_]
    start: tuple[int, int]
    goal: tuple[int, int]
    solutions: tuple[NDArray[np.bool_], ...] = ()

    def __post_init__(self) -> None:
        walls = np.asarray(self.walls, dtype=np.bool_)
        if walls.ndim != 2:
            raise ValueError("walls must be a 2D grid")
        _check_coord(self.start, walls.shape, "start")
        _check_coord(self.goal, walls.shape, "goal")
        if walls[self.start] or walls[self.goal]:
            raise ValueError("start and goal must not be walls")
        normalized = tuple(
            np.asarray(solution, dtype=np.bool_).copy() for solution in self.solutions
        )
        for solution in normalized:
            if solution.shape != walls.shape:
                raise ValueError("solution masks must match walls shape")
        object.__setattr__(self, "walls", walls.copy())
        object.__setattr__(self, "solutions", normalized)


class MazeDomain:
    """Maze-Hard adapter using a binary path-membership lattice."""

    def __init__(self, height: int, width: int, *, require_shortest: bool = True) -> None:
        self.height = height
        self.width = width
        self.require_shortest = require_shortest
        self._spec = DomainSpec(
            name="maze",
            num_positions=height * width,
            vocab_size=2,
            active=np.ones(height * width, dtype=np.bool_),
            position_shape=(height, width),
            metadata={"extra_input_channels": 3},
        )

    @property
    def spec(self) -> DomainSpec:
        return self._spec

    def encode(self, puzzle: MazePuzzle, *, puzzle_id: str = "") -> PuzzleInstance:
        self._check_shape(puzzle)
        candidates = np.ones((self.spec.num_positions, 2), dtype=np.bool_)
        flat_walls = puzzle.walls.reshape(-1)
        candidates[flat_walls, :] = False
        candidates[flat_walls, NOT_PATH] = True
        for coord in (puzzle.start, puzzle.goal):
            position = self._position(coord)
            candidates[position, :] = False
            candidates[position, PATH] = True

        solutions = None
        if puzzle.solutions:
            solutions = np.stack(
                [solution.reshape(-1).astype(np.int64) for solution in puzzle.solutions],
                axis=0,
            )

        return PuzzleInstance(
            puzzle_id=puzzle_id,
            initial_state=LatticeState(candidates, self.spec.active),
            solutions=solutions,
            raw=puzzle,
        )

    def extra_features(self, puzzle: MazePuzzle, state: LatticeState) -> NDArray[np.float32]:
        del state
        features = np.zeros((self.spec.num_positions, 3), dtype=np.float32)
        features[:, 0] = puzzle.walls.reshape(-1).astype(np.float32)
        features[self._position(puzzle.start), 1] = 1.0
        features[self._position(puzzle.goal), 2] = 1.0
        return features

    def solution_to_state(
        self,
        solution: NDArray[np.bool_] | Iterable[tuple[int, int]],
    ) -> LatticeState:
        mask = self._solution_mask(solution)
        candidates = np.zeros((self.spec.num_positions, 2), dtype=np.bool_)
        candidates[:, NOT_PATH] = ~mask.reshape(-1)
        candidates[:, PATH] = mask.reshape(-1)
        return LatticeState(candidates, self.spec.active)

    def decode_solution(self, state: LatticeState) -> NDArray[np.bool_]:
        decoded = state.as_solution_indices()
        return decoded.reshape(self.height, self.width).astype(np.bool_)

    def validate_solution(self, puzzle: MazePuzzle, state: LatticeState) -> ValidationResult:
        if bool(state.is_conflict()):
            return ValidationResult(False, "state is conflicted")
        if not bool(state.is_complete()):
            return ValidationResult(False, "state is not complete")
        path = self.decode_solution(state)
        if np.any(path & puzzle.walls):
            return ValidationResult(False, "path crosses a wall")
        if not path[puzzle.start] or not path[puzzle.goal]:
            return ValidationResult(False, "path must include start and goal")
        if not _is_single_simple_path(path, puzzle.start, puzzle.goal):
            return ValidationResult(False, "path is not a single simple start-goal path")
        if self.require_shortest:
            distance = shortest_path_distance(puzzle.walls, puzzle.start, puzzle.goal)
            if distance is None:
                return ValidationResult(False, "maze has no start-goal path")
            if int(path.sum()) != distance + 1:
                return ValidationResult(False, "path is not shortest")
        return ValidationResult(True)

    def sample_shortest_solutions(
        self,
        puzzle: MazePuzzle,
        *,
        max_solutions: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray[np.bool_], ...]:
        paths = enumerate_shortest_paths(
            puzzle.walls,
            puzzle.start,
            puzzle.goal,
            max_paths=max_solutions,
            rng=rng,
        )
        return tuple(_path_to_mask(path, puzzle.walls.shape) for path in paths)

    def _check_shape(self, puzzle: MazePuzzle) -> None:
        if puzzle.walls.shape != (self.height, self.width):
            raise ValueError("puzzle shape does not match MazeDomain")

    def _position(self, coord: tuple[int, int]) -> int:
        return coord[0] * self.width + coord[1]

    def _solution_mask(
        self,
        solution: NDArray[np.bool_] | Iterable[tuple[int, int]],
    ) -> NDArray[np.bool_]:
        if isinstance(solution, np.ndarray):
            mask = np.asarray(solution, dtype=np.bool_)
            if mask.shape != (self.height, self.width):
                raise ValueError("solution mask has wrong shape")
            return mask.copy()
        mask = np.zeros((self.height, self.width), dtype=np.bool_)
        for coord in solution:
            _check_coord(coord, (self.height, self.width), "path coordinate")
            mask[coord] = True
        return mask


def shortest_path_distance(
    walls: NDArray[np.bool_],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> int | None:
    distances = _bfs_distances(walls, start)
    distance = distances[goal]
    return None if distance < 0 else int(distance)


def enumerate_shortest_paths(
    walls: NDArray[np.bool_],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    max_paths: int,
    rng: np.random.Generator | None = None,
) -> list[list[tuple[int, int]]]:
    if max_paths <= 0:
        raise ValueError("max_paths must be positive")
    start_dist = _bfs_distances(walls, start)
    goal_dist = _bfs_distances(walls, goal)
    shortest = start_dist[goal]
    if shortest < 0:
        return []

    generator = rng or np.random.default_rng()
    paths: list[list[tuple[int, int]]] = []

    def visit(coord: tuple[int, int], path: list[tuple[int, int]]) -> None:
        if len(paths) >= max_paths:
            return
        if coord == goal:
            paths.append(path.copy())
            return
        next_coords = [
            neighbor
            for neighbor in _neighbors(coord, walls.shape)
            if not walls[neighbor]
            and start_dist[neighbor] == start_dist[coord] + 1
            and start_dist[neighbor] + goal_dist[neighbor] == shortest
        ]
        generator.shuffle(next_coords)
        for neighbor in next_coords:
            path.append(neighbor)
            visit(neighbor, path)
            path.pop()

    visit(start, [start])
    return paths


def _bfs_distances(walls: NDArray[np.bool_], start: tuple[int, int]) -> NDArray[np.int64]:
    distances = np.full(walls.shape, -1, dtype=np.int64)
    queue: deque[tuple[int, int]] = deque([start])
    distances[start] = 0
    while queue:
        coord = queue.popleft()
        for neighbor in _neighbors(coord, walls.shape):
            if walls[neighbor] or distances[neighbor] >= 0:
                continue
            distances[neighbor] = distances[coord] + 1
            queue.append(neighbor)
    return distances


def _neighbors(coord: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = coord
    height, width = shape
    candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
    return [
        (next_row, next_col)
        for next_row, next_col in candidates
        if 0 <= next_row < height and 0 <= next_col < width
    ]


def _is_single_simple_path(
    path: NDArray[np.bool_],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    path_coords = [tuple(coord) for coord in np.argwhere(path)]
    if not path_coords:
        return False
    path_set = set(path_coords)
    degrees = {
        coord: sum(neighbor in path_set for neighbor in _neighbors(coord, path.shape))
        for coord in path_set
    }
    if start == goal:
        return path.sum() == 1
    if degrees.get(start) != 1 or degrees.get(goal) != 1:
        return False
    for coord, degree in degrees.items():
        if coord not in {start, goal} and degree != 2:
            return False

    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        coord = queue.popleft()
        for neighbor in _neighbors(coord, path.shape):
            if neighbor in path_set and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == path_set


def _path_to_mask(path: list[tuple[int, int]], shape: tuple[int, int]) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=np.bool_)
    for coord in path:
        mask[coord] = True
    return mask


def _check_coord(coord: tuple[int, int], shape: tuple[int, int], name: str) -> None:
    row, col = coord
    if not 0 <= row < shape[0] or not 0 <= col < shape[1]:
        raise ValueError(f"{name} coordinate out of range")
