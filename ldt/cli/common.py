from __future__ import annotations

from pathlib import Path
from typing import Any

from ldt.config import ExperimentConfig
from ldt.data import load_sudoku_rows
from ldt.domains import LatticeDomain, MazeDomain, SnowflakeDomain, SudokuDomain
from ldt.domains.base import PuzzleInstance
from ldt.domains.sudoku import SudokuPuzzle
from ldt.model import RecurrentLDT


def build_domain(config: ExperimentConfig) -> LatticeDomain:
    if config.domain == "sudoku":
        return SudokuDomain()
    if config.domain == "snowflake":
        return SnowflakeDomain()
    if config.domain == "maze":
        if len(config.model.position_shape) != 2:
            raise ValueError("maze config requires 2D model.position_shape")
        height, width = config.model.position_shape
        return MazeDomain(height, width)
    raise ValueError(f"unsupported domain: {config.domain}")


def build_model(config: ExperimentConfig) -> RecurrentLDT:
    return RecurrentLDT(config.model)


def load_puzzles(config: ExperimentConfig, path: str | Path) -> list[Any]:
    if config.domain == "sudoku":
        return load_sudoku_rows(path)
    raise ValueError(f"dataset loader for domain {config.domain!r} is not implemented")


def encode_training_instances(
    domain: LatticeDomain,
    puzzles: list[Any],
) -> list[PuzzleInstance]:
    instances = [
        domain.encode(puzzle, puzzle_id=f"train-{idx}")
        for idx, puzzle in enumerate(puzzles)
    ]
    missing = [instance.puzzle_id for instance in instances if instance.solutions is None]
    if missing:
        raise ValueError(f"training instances require sampled/reference solutions: {missing[:5]}")
    return instances


def sudoku_fixture() -> SudokuPuzzle:
    return SudokuPuzzle(
        givens=_grid(
            "530070000600195000098000060800060003400803001700020006"
            "060000280000419005000080079"
        ),
        solution=_grid(
            "534678912672195348198342567859761423426853791713924856"
            "961537284287419635345286179"
        ),
    )


def _grid(text: str):
    import numpy as np

    values = [int(char) for char in text]
    return np.asarray(values, dtype=np.int64).reshape(9, 9)
