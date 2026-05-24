from __future__ import annotations

import numpy as np
import torch

from ldt.config import load_experiment_config, save_experiment_config
from ldt.data.sudoku import load_sudoku_rows
from ldt.domains.sudoku import SudokuDomain
from ldt.eval import evaluate_puzzles
from ldt.io import CheckpointMetadata, load_checkpoint, save_checkpoint
from ldt.model import LDTConfig, RecurrentLDT
from ldt.solve import InferenceConfig, LatticeStepConfig


def test_config_round_trip(tmp_path) -> None:
    config = load_experiment_config("configs/sudoku_extreme.json")
    path = tmp_path / "config.json"

    save_experiment_config(config, path)
    loaded = load_experiment_config(path)

    assert loaded.name == config.name
    assert loaded.model.num_loops == 16


def test_checkpoint_round_trip(tmp_path) -> None:
    model = RecurrentLDT(
        LDTConfig(
            num_positions=4,
            input_channels=2,
            candidate_channels=2,
            position_shape=(2, 2),
            d_model=16,
            num_layers=1,
            num_heads=4,
            num_loops=1,
        )
    )
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model=model, metadata=CheckpointMetadata(step=7, metrics={"loss": 1.25}))
    metadata = load_checkpoint(path, model=model)

    assert metadata.step == 7
    assert metadata.metrics["loss"] == 1.25


def test_sudoku_loader_accepts_givens_and_solution_rows(tmp_path) -> None:
    givens = (
        "530070000600195000098000060800060003400803001700020006"
        "060000280000419005000080079"
    )
    solution = (
        "534678912672195348198342567859761423426853791713924856"
        "961537284287419635345286179"
    )
    path = tmp_path / "sudoku.txt"
    path.write_text(f"# comment\n{givens},{solution}\n")

    puzzles = load_sudoku_rows(path)

    assert len(puzzles) == 1
    assert puzzles[0].givens.shape == (9, 9)
    assert puzzles[0].solution is not None


class AlwaysSolveSudoku(torch.nn.Module):
    def __init__(self, solution: np.ndarray) -> None:
        super().__init__()
        logits = torch.full((81, 9), -10.0)
        for position, digit in enumerate(solution.reshape(-1)):
            logits[position, int(digit) - 1] = 10.0
        self.register_buffer("logits", logits)

    def forward(self, candidates, extra_features=None):
        del extra_features
        batch = candidates.shape[0]
        return self.logits.expand(batch, -1, -1), torch.full(
            (batch,),
            -10.0,
            device=candidates.device,
        )


def test_evaluate_puzzles_wraps_batched_inference() -> None:
    domain = SudokuDomain()
    puzzle = (
        "530070000600195000098000060800060003400803001700020006"
        "060000280000419005000080079\n"
        "534678912672195348198342567859761423426853791713924856"
        "961537284287419635345286179"
    )
    instance = domain.encode(puzzle)
    assert instance.solutions is not None
    solution = instance.solutions[0].reshape(9, 9) + 1

    summary = evaluate_puzzles(
        AlwaysSolveSudoku(solution),
        domain,
        [instance.raw],
        config=InferenceConfig(
            num_slots=1,
            chains_per_slot=1,
            round_budget=1,
            step=LatticeStepConfig(theta_elim=0.1, enable_branching=False),
        ),
    )

    assert summary.metrics.accepted == 1
    assert summary.solved_ids == ["puzzle-0"]
