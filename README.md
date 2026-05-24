# Lattice Deduction Transformer

This repository implements the Lattice Deduction Transformer described by
Davis, Haller, Alfarano, and Santolucito in "Lattice Deduction Transformers"
(arXiv:2605.08605).

The implementation is organized around the structure outlined in the paper:
lattice states, recurrent LDT model passes, on-policy lattice projection, and
validator-backed inference. Benchmark domains such as Sudoku-Extreme,
Snowflake Sudoku, and Maze-Hard are included.

## Current Slice

- Powerset lattice state representation.
- Meet, join, alpha, consistency filtering, candidate pinning, and elimination.
- Solver-facing domain protocol.
- Sudoku-Extreme benchmark adapter for encoding and validation only.
- Recurrent LDT model core with learned 2D position embeddings, optional 2D
  RoPE, per-loop candidate logits, and per-loop CLS conflict logits.
- Shared lattice projection step with threshold elimination, conflict handling,
  train-time solution verification, and stochastic singleton branching.
- On-policy training pool for recent partially-deduced lattice states.
- Batched inference engine with slots, parallel chains, validator-backed
  acceptance, timeout abstention, conflict resets, and aggregate metrics.
- Paper loss targets/objective, bounded training-step orchestration, optimizer
  factories, configs, checkpoints, dataset loaders, and evaluation wrappers.
- Snowflake Sudoku and Maze-Hard domain adapters with solver-facing feature
  channels and validators.
- Deterministic tests for lattice laws, alpha semantics, masks, and Sudoku validation.

## Development

Create a local environment and run tests:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install numpy pytest
.venv/bin/python -m pytest
```

Run the tiny smoke-overfit check:

```bash
.venv/bin/python -m ldt.cli.smoke_overfit --steps 25 --output .context/smoke-overfit.pt
```

Train/evaluate entrypoints are available as Python modules or installed console
commands:

```bash
.venv/bin/python -m ldt.cli.train --config configs/sudoku_extreme.json --train-data data/sudoku.txt --steps 100 --output runs/sudoku.pt
.venv/bin/python -m ldt.cli.eval --config configs/sudoku_extreme.json --checkpoint runs/sudoku.pt --eval-data data/sudoku-test.txt
```
