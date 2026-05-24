from __future__ import annotations

import argparse
import json

from ldt.cli.common import build_domain, build_model, load_puzzles
from ldt.config import load_experiment_config
from ldt.eval import evaluate_puzzles
from ldt.io import load_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an LDT checkpoint.")
    parser.add_argument("--config", required=True, help="Experiment config JSON path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--eval-data", required=True, help="Evaluation dataset path.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    domain = build_domain(config)
    model = build_model(config).to(args.device)
    load_checkpoint(args.checkpoint, model=model, map_location=args.device)
    puzzles = load_puzzles(config, args.eval_data)
    summary = evaluate_puzzles(model, domain, puzzles, config=config.inference)
    print(json.dumps(summary.metrics.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
