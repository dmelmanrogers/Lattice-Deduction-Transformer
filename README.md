# Lattice Deduction Transformer

This repository implements the Lattice Deduction Transformer described by
Davis, Haller, Alfarano, and Santolucito in "Lattice Deduction Transformers"
(arXiv:2605.08605).

The implementation is organized around the paper's domain-agnostic structure:
lattice states, recurrent LDT model passes, on-policy lattice projection, and
validator-backed inference. Benchmark domains such as Sudoku-Extreme,
Snowflake Sudoku, and Maze-Hard are adapters over that core rather than
problem-specific solver implementations.

## Current Slice

- Powerset lattice state representation.
- Meet, join, alpha, consistency filtering, candidate pinning, and elimination.
- Solver-facing domain protocol.
- Sudoku-Extreme benchmark adapter for encoding and validation only.
- Deterministic tests for lattice laws, alpha semantics, masks, and Sudoku validation.

## Development

Create a local environment and run tests:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install numpy pytest
.venv/bin/python -m pytest
```
