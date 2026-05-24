from __future__ import annotations

from pathlib import Path

from ldt.domains.sudoku import SudokuPuzzle, parse_puzzle


def load_sudoku_rows(path: str | Path) -> list[SudokuPuzzle]:
    """Load Sudoku rows from text, CSV-like, or two-line records.

    Each non-comment record may contain either one 81-cell givens string or a
    givens/solution pair separated by comma, whitespace, or tab.
    """

    puzzles: list[SudokuPuzzle] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = stripped.replace(",", " ").replace("\t", " ")
        parts = [part for part in normalized.split(" ") if part]
        if len(parts) == 1:
            puzzles.append(parse_puzzle(parts[0]))
        elif len(parts) >= 2:
            puzzles.append(parse_puzzle(f"{parts[0]}\n{parts[1]}"))
        else:
            raise ValueError(f"could not parse Sudoku row: {line}")
    return puzzles
