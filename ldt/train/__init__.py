"""Training-ready utilities for LDT.

These modules define objectives and step orchestration. Importing them does not
start a training run.
"""

from ldt.train.loss import LDTLossConfig, LDTLossOutput, LDTTargets, compute_loss, compute_targets
from ldt.train.optim import AdamWConfig, create_adamw, create_warmup_cosine_scheduler
from ldt.train.runner import TrainingLoopConfig, TrainingRunStats, run_training_steps
from ldt.train.step import TrainingStepOutput, run_training_step

__all__ = [
    "AdamWConfig",
    "LDTLossConfig",
    "LDTLossOutput",
    "LDTTargets",
    "TrainingLoopConfig",
    "TrainingRunStats",
    "TrainingStepOutput",
    "compute_loss",
    "compute_targets",
    "create_adamw",
    "create_warmup_cosine_scheduler",
    "run_training_steps",
    "run_training_step",
]
