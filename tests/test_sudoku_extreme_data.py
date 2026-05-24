from __future__ import annotations

from ldt.data import (
    SudokuExtremePrepConfig,
    load_sudoku_rows,
    prepare_sudoku_extreme_csv,
)

GIVENS_1 = (
    "530070000600195000098000060800060003400803001700020006"
    "060000280000419005000080079"
)
GIVENS_2 = (
    "000260701680070090190004500820100040004602900050003028"
    "009300074040050036703018000"
)
SOLUTION_1 = (
    "534678912672195348198342567859761423426853791713924856"
    "961537284287419635345286179"
)
SOLUTION_2 = (
    "435269781682571493197834562826195347374682915951743628"
    "519326874248957136763418259"
)


def test_prepare_sudoku_extreme_filters_limits_and_loads_output(tmp_path) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "train-1.txt"
    input_path.write_text(
        "\n".join(
            [
                "source,question,answer,rating",
                f"low,{GIVENS_1},{SOLUTION_1},3",
                f"high,{GIVENS_2},{SOLUTION_2},17",
                f"higher,{GIVENS_1.replace('0', '.')},{SOLUTION_1},41",
                "",
            ]
        )
    )

    stats = prepare_sudoku_extreme_csv(
        SudokuExtremePrepConfig(
            input_path=input_path,
            output_path=output_path,
            limit=1,
            min_rating=10,
        )
    )

    assert stats.read == 2
    assert stats.filtered_by_rating == 1
    assert stats.selected == 1
    assert stats.written == 1
    assert output_path.read_text() == f"{GIVENS_2} {SOLUTION_2}\n"
    assert len(load_sudoku_rows(output_path)) == 1


def test_prepare_sudoku_extreme_shuffle_is_seeded(tmp_path) -> None:
    input_path = tmp_path / "train.csv"
    out_a = tmp_path / "a.txt"
    out_b = tmp_path / "b.txt"
    input_path.write_text(
        "\n".join(
            [
                "source,question,answer,rating",
                f"one,{GIVENS_1},{SOLUTION_1},11",
                f"two,{GIVENS_2},{SOLUTION_2},12",
                f"three,{GIVENS_1},{SOLUTION_1},13",
                "",
            ]
        )
    )

    config_kwargs = {
        "input_path": input_path,
        "limit": 2,
        "shuffle": True,
        "seed": 7,
    }
    prepare_sudoku_extreme_csv(SudokuExtremePrepConfig(output_path=out_a, **config_kwargs))
    prepare_sudoku_extreme_csv(SudokuExtremePrepConfig(output_path=out_b, **config_kwargs))

    assert out_a.read_text() == out_b.read_text()
