from __future__ import annotations

import argparse
import json
from pathlib import Path

from ldt.cli.common import build_domain, build_model, encode_training_instances, load_puzzles
from ldt.config import load_experiment_config
from ldt.io import CheckpointMetadata, save_checkpoint
from ldt.solve import OnPolicyTrainingPool
from ldt.train import (
    AdamWConfig,
    TrainingLoopConfig,
    create_adamw,
    create_warmup_cosine_scheduler,
    run_training_steps,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded LDT training.")
    parser.add_argument("--config", required=True, help="Experiment config JSON path.")
    parser.add_argument("--train-data", required=True, help="Training dataset path.")
    parser.add_argument("--output", required=True, help="Checkpoint output path.")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    domain = build_domain(config)
    puzzles = load_puzzles(config, args.train_data)
    instances = encode_training_instances(domain, puzzles)

    model = build_model(config).to(args.device)
    pool = OnPolicyTrainingPool(instances, config.pool)
    optimizer = create_adamw(
        model,
        AdamWConfig(
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
        ),
    )
    steps = args.steps if args.steps is not None else int(config.metadata.get("train_steps", 0))
    if steps <= 0:
        raise ValueError(
            "training steps must be provided by --steps or config.metadata.train_steps"
        )
    batch_size = args.batch_size if args.batch_size is not None else min(config.pool.pool_size, 512)
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        total_steps=steps,
        warmup_fraction=config.optimizer.warmup_fraction,
    )
    stats = run_training_steps(
        model,
        pool,
        optimizer,
        config=TrainingLoopConfig(
            steps=steps,
            batch_size=batch_size,
            grad_clip=config.optimizer.grad_clip,
        ),
        loss_config=config.loss,
        step_config=config.step,
        scheduler=scheduler,
        device=args.device,
        extra_features_fn=getattr(domain, "extra_features", None),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output,
        model=model,
        optimizer=optimizer,
        metadata=CheckpointMetadata(
            step=steps,
            config={"name": config.name, "domain": config.domain},
            metrics={"last_loss": -1.0 if stats.last_loss is None else stats.last_loss},
        ),
    )
    print(json.dumps(stats.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
