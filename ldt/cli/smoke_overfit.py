from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ldt.cli.common import sudoku_fixture
from ldt.domains import SudokuDomain
from ldt.io import CheckpointMetadata, save_checkpoint
from ldt.model import LDTConfig, RecurrentLDT
from ldt.solve import LatticeStepConfig, OnPolicyTrainingPool, PoolConfig
from ldt.train import (
    AdamWConfig,
    TrainingLoopConfig,
    create_adamw,
    create_warmup_cosine_scheduler,
    run_training_step,
    run_training_steps,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny Sudoku smoke-overfit check.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Optional checkpoint path.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    domain = SudokuDomain()
    instance = domain.encode(sudoku_fixture(), puzzle_id="smoke-sudoku")
    model = RecurrentLDT(
        LDTConfig(
            num_positions=81,
            input_channels=9,
            candidate_channels=9,
            position_shape=(9, 9),
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_loops=2,
            dropout=0.0,
        )
    ).to(args.device)
    step_config = LatticeStepConfig(theta_elim=0.0, enable_branching=False)

    pool = OnPolicyTrainingPool([instance], PoolConfig(pool_size=1, max_age=10_000))
    optimizer = create_adamw(model, AdamWConfig(lr=3e-3, weight_decay=0.0))
    initial = run_training_step(
        model,
        pool,
        batch_size=1,
        step_config=step_config,
        device=args.device,
    ).loss.total.item()
    scheduler = create_warmup_cosine_scheduler(optimizer, total_steps=max(1, args.steps))
    stats = run_training_steps(
        model,
        pool,
        optimizer,
        config=TrainingLoopConfig(steps=args.steps, batch_size=1),
        step_config=step_config,
        scheduler=scheduler,
        device=args.device,
    )
    final = stats.last_loss
    result = {
        "initial_loss": initial,
        "final_loss": final,
        "improved": final is not None and final < initial,
        "steps": args.steps,
    }
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            output,
            model=model,
            optimizer=optimizer,
            metadata=CheckpointMetadata(step=args.steps, metrics={"final_loss": final or -1.0}),
        )
    print(json.dumps(result, sort_keys=True))
    if not result["improved"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
