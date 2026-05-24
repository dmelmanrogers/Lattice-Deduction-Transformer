from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ldt.domains.base import DomainSpec


@dataclass(frozen=True)
class LDTConfig:
    """Configuration for the recurrent Lattice Deduction Transformer."""

    num_positions: int
    input_channels: int
    candidate_channels: int
    position_shape: tuple[int, ...] = ()
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_loops: int = 16
    ffn_multiplier: float = 4.0
    dropout: float = 0.1
    use_rope: bool = False
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.num_positions <= 0:
            raise ValueError("num_positions must be positive")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.candidate_channels <= 0:
            raise ValueError("candidate_channels must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_loops <= 0:
            raise ValueError("num_loops must be positive")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.position_shape and len(self.position_shape) != 2:
            raise ValueError("position_shape must be empty or 2D")
        if self.position_shape and _product(self.position_shape) != self.num_positions:
            raise ValueError("position_shape product must equal num_positions")
        if self.use_rope:
            if len(self.position_shape) != 2:
                raise ValueError("use_rope requires a 2D position_shape")
            if (self.d_model // self.num_heads) % 4 != 0:
                raise ValueError("2D RoPE requires per-head dimension divisible by 4")

    @classmethod
    def from_domain_spec(
        cls,
        spec: DomainSpec,
        *,
        extra_input_channels: int = 0,
        **overrides: Any,
    ) -> LDTConfig:
        values: dict[str, Any] = {
            "num_positions": spec.num_positions,
            "input_channels": spec.vocab_size + extra_input_channels,
            "candidate_channels": spec.vocab_size,
            "position_shape": spec.position_shape,
        }
        values.update(overrides)
        return cls(**values)


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
