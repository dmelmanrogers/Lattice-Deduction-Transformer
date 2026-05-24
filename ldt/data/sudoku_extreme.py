from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from ldt.domains.sudoku import SudokuPuzzle, parse_puzzle


@dataclass(frozen=True)
class SudokuExtremePrepConfig:
    input_path: Path
    output_path: Path
    limit: int | None = None
    min_rating: int | None = None
    max_rating: int | None = None
    shuffle: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if (
            self.min_rating is not None
            and self.max_rating is not None
            and self.min_rating > self.max_rating
        ):
            raise ValueError("min_rating cannot exceed max_rating")


@dataclass(frozen=True)
class SudokuExtremePrepStats:
    read: int
    selected: int
    written: int
    filtered_by_rating: int


@dataclass(frozen=True)
class _PreparedRow:
    givens: str
    solution: str
    rating: int | None


def prepare_sudoku_extreme_csv(config: SudokuExtremePrepConfig) -> SudokuExtremePrepStats:
    """Convert Sapient Sudoku-Extreme CSV rows into trainer-ready text records."""

    prepared: list[_PreparedRow] = []
    read = 0
    filtered_by_rating = 0

    with config.input_path.open(newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(reader.fieldnames)
        for row in reader:
            read += 1
            rating = _parse_rating(row.get("rating", ""))
            if not _rating_allowed(rating, config.min_rating, config.max_rating):
                filtered_by_rating += 1
                continue
            prepared.append(_prepare_row(row, rating))
            if not config.shuffle and config.limit is not None and len(prepared) >= config.limit:
                break

    if config.shuffle:
        rng = random.Random(config.seed)
        rng.shuffle(prepared)

    selected = len(prepared) if config.limit is None else min(len(prepared), config.limit)
    if config.limit is not None and selected < config.limit:
        raise ValueError(
            f"requested {config.limit} rows, but only {selected} matched the filters"
        )

    output_rows = prepared[:selected]
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        "".join(f"{row.givens} {row.solution}\n" for row in output_rows)
    )

    return SudokuExtremePrepStats(
        read=read,
        selected=selected,
        written=len(output_rows),
        filtered_by_rating=filtered_by_rating,
    )


def _require_columns(fieldnames: list[str] | None) -> None:
    required = {"question", "answer"}
    if fieldnames is None:
        raise ValueError("input CSV is missing a header row")
    missing = required.difference(fieldnames)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"input CSV is missing required columns: {missing_text}")


def _parse_rating(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _rating_allowed(
    rating: int | None,
    min_rating: int | None,
    max_rating: int | None,
) -> bool:
    if min_rating is not None and (rating is None or rating < min_rating):
        return False
    return not (max_rating is not None and (rating is None or rating > max_rating))


def _prepare_row(row: dict[str, str], rating: int | None) -> _PreparedRow:
    puzzle = parse_puzzle(f"{row['question']}\n{row['answer']}")
    return _PreparedRow(
        givens=_givens_text(puzzle),
        solution=_solution_text(puzzle),
        rating=rating,
    )


def _givens_text(puzzle: SudokuPuzzle) -> str:
    return "".join(str(int(value)) for value in puzzle.givens.reshape(-1))


def _solution_text(puzzle: SudokuPuzzle) -> str:
    if puzzle.solution is None:
        raise ValueError("Sudoku-Extreme rows must include reference solutions")
    return "".join(str(int(value)) for value in puzzle.solution.reshape(-1))
