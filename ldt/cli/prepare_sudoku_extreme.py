from __future__ import annotations

import argparse
import json
from pathlib import Path

from ldt.data import SudokuExtremePrepConfig, prepare_sudoku_extreme_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Sapient Sudoku-Extreme CSV data for LDT training."
    )
    parser.add_argument("--input", required=True, help="Path to train.csv or test.csv.")
    parser.add_argument("--output", required=True, help="Output text file path.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to emit.")
    parser.add_argument("--min-rating", type=int, default=None, help="Minimum tdoku rating.")
    parser.add_argument("--max-rating", type=int, default=None, help="Maximum tdoku rating.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Deterministically shuffle matching rows before applying --limit.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    args = parser.parse_args(argv)

    stats = prepare_sudoku_extreme_csv(
        SudokuExtremePrepConfig(
            input_path=Path(args.input),
            output_path=Path(args.output),
            limit=args.limit,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    )
    print(json.dumps(stats.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
