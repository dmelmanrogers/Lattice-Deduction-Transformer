from __future__ import annotations

import json

from ldt.cli.smoke_overfit import main as smoke_main
from ldt.cli.train import main as train_main

GIVENS = (
    "530070000600195000098000060800060003400803001700020006"
    "060000280000419005000080079"
)
SOLUTION = (
    "534678912672195348198342567859761423426853791713924856"
    "961537284287419635345286179"
)


def _small_config() -> dict:
    return {
        "domain": "sudoku",
        "metadata": {"train_steps": 1},
        "model": {
            "candidate_channels": 9,
            "d_model": 32,
            "dropout": 0.0,
            "ffn_multiplier": 2.0,
            "input_channels": 9,
            "num_heads": 4,
            "num_layers": 1,
            "num_loops": 2,
            "num_positions": 81,
            "position_shape": [9, 9],
            "use_rope": False,
        },
        "name": "small-sudoku-cli",
        "optimizer": {
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "lr": 0.001,
            "warmup_fraction": 0.1,
            "weight_decay": 0.0,
        },
        "pool": {"max_age": 100, "pool_size": 1},
        "step": {"enable_branching": False, "tau_decide": 1.5, "theta_cls": 0.6, "theta_elim": 0.0},
    }


def test_smoke_overfit_cli_writes_checkpoint(tmp_path) -> None:
    output = tmp_path / "smoke.pt"

    exit_code = smoke_main(["--steps", "8", "--output", str(output)])

    assert exit_code == 0
    assert output.exists()


def test_train_cli_runs_one_bounded_step_and_writes_checkpoint(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    data_path = tmp_path / "sudoku.txt"
    output_path = tmp_path / "model.pt"
    config_path.write_text(json.dumps(_small_config()))
    data_path.write_text(f"{GIVENS},{SOLUTION}\n")

    exit_code = train_main(
        [
            "--config",
            str(config_path),
            "--train-data",
            str(data_path),
            "--output",
            str(output_path),
            "--steps",
            "1",
            "--batch-size",
            "1",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
