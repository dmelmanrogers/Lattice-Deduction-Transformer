from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ldt.model.attention import LearnedPositionEmbedding, TransformerBlock
from ldt.model.config import LDTConfig


@dataclass(frozen=True)
class LDTOutput:
    """Per-loop candidate and conflict logits from a recurrent LDT forward pass."""

    candidate_logits: Tensor
    cls_logits: Tensor
    final_hidden: Tensor
    hidden_states: Tensor | None = None

    @property
    def final_candidate_logits(self) -> Tensor:
        return self.candidate_logits[-1]

    @property
    def final_cls_logits(self) -> Tensor:
        return self.cls_logits[-1]


class RecurrentLDT(nn.Module):
    """Recurrent Lattice Deduction Transformer core.

    The model consumes domain-encoded lattice candidate tensors plus optional
    read-only feature channels. It emits candidate logits for lattice projection
    and a CLS conflict logit at every internal recurrent iteration.
    """

    def __init__(self, config: LDTConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_channels, config.d_model)
        self.position_embedding = LearnedPositionEmbedding(
            config.num_positions,
            config.d_model,
            config.position_shape,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_multiplier,
                    config.dropout,
                    num_positions=config.num_positions,
                    position_shape=config.position_shape,
                    use_rope=config.use_rope,
                    rope_base=config.rope_base,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.candidate_head = nn.Linear(config.d_model, config.candidate_channels)
        self.cls_head = nn.Linear(config.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        candidates: Tensor,
        extra_features: Tensor | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> LDTOutput:
        features = self._combine_features(candidates, extra_features)
        batch = features.shape[0]

        lattice_signal = self.input_projection(features)
        position_signal = self.position_embedding().to(
            device=lattice_signal.device,
            dtype=lattice_signal.dtype,
        )
        cell_tokens = lattice_signal + position_signal
        cls_token = self.cls_token.expand(batch, -1, -1)

        hidden = torch.cat([cls_token, cell_tokens], dim=1)
        hidden = self.input_dropout(hidden)

        zero_cls_signal = torch.zeros_like(cls_token)
        reinjection = torch.cat([zero_cls_signal, lattice_signal], dim=1)

        candidate_logits: list[Tensor] = []
        cls_logits: list[Tensor] = []
        hidden_states: list[Tensor] = []

        for loop_index in range(self.config.num_loops):
            if loop_index > 0:
                hidden = hidden + reinjection
            for block in self.blocks:
                hidden = block(hidden)

            candidate_logits.append(self.candidate_head(hidden[:, 1:, :]))
            cls_logits.append(self.cls_head(hidden[:, 0, :]).squeeze(-1))
            if return_hidden_states:
                hidden_states.append(hidden)

        stacked_hidden = torch.stack(hidden_states, dim=0) if return_hidden_states else None
        return LDTOutput(
            candidate_logits=torch.stack(candidate_logits, dim=0),
            cls_logits=torch.stack(cls_logits, dim=0),
            final_hidden=hidden,
            hidden_states=stacked_hidden,
        )

    def _combine_features(self, candidates: Tensor, extra_features: Tensor | None) -> Tensor:
        if candidates.ndim != 3:
            raise ValueError("candidates must have shape (batch, positions, candidate_channels)")
        if candidates.shape[1] != self.config.num_positions:
            raise ValueError("candidate positions do not match config.num_positions")
        if candidates.shape[2] != self.config.candidate_channels:
            raise ValueError("candidate channels do not match config.candidate_channels")

        if extra_features is None:
            if self.config.input_channels != self.config.candidate_channels:
                raise ValueError("extra_features required by config.input_channels")
            features = candidates
        else:
            if extra_features.ndim != 3:
                raise ValueError("extra_features must have shape (batch, positions, channels)")
            if extra_features.shape[:2] != candidates.shape[:2]:
                raise ValueError("extra_features batch/position dimensions must match candidates")
            features = torch.cat([candidates, extra_features], dim=-1)
            if features.shape[-1] != self.config.input_channels:
                raise ValueError("combined feature channels do not match config.input_channels")

        return features.to(dtype=self.input_projection.weight.dtype)


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    parameters = module.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
