"""Problem-domain adapters for LDT benchmarks."""

from ldt.domains.base import DomainSpec, LatticeDomain, PuzzleInstance, ValidationResult
from ldt.domains.maze import MazeDomain, MazePuzzle
from ldt.domains.snowflake import SnowflakeDomain, SnowflakePuzzle
from ldt.domains.sudoku import SudokuDomain, SudokuPuzzle

__all__ = [
    "DomainSpec",
    "LatticeDomain",
    "MazeDomain",
    "MazePuzzle",
    "PuzzleInstance",
    "SnowflakeDomain",
    "SnowflakePuzzle",
    "SudokuDomain",
    "SudokuPuzzle",
    "ValidationResult",
]
