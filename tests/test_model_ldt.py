from __future__ import annotations

import pytest
import torch

from ldt.model import LDTConfig, RecurrentLDT, count_parameters


def test_recurrent_ldt_emits_per_loop_candidate_and_cls_logits() -> None:
    torch.manual_seed(0)
    config = LDTConfig(
        num_positions=81,
        input_channels=9,
        candidate_channels=9,
        position_shape=(9, 9),
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_loops=3,
        dropout=0.0,
    )
    model = RecurrentLDT(config)
    candidates = torch.ones(2, 81, 9)

    output = model(candidates, return_hidden_states=True)

    assert output.candidate_logits.shape == (3, 2, 81, 9)
    assert output.cls_logits.shape == (3, 2)
    assert output.final_hidden.shape == (2, 82, 32)
    assert output.hidden_states is not None
    assert output.hidden_states.shape == (3, 2, 82, 32)
    assert output.final_candidate_logits.shape == (2, 81, 9)
    assert output.final_cls_logits.shape == (2,)
    assert torch.isfinite(output.candidate_logits).all()
    assert torch.isfinite(output.cls_logits).all()


def test_recurrent_loop_reuses_blocks_but_produces_distinct_iteration_outputs() -> None:
    torch.manual_seed(1)
    config = LDTConfig(
        num_positions=9,
        input_channels=4,
        candidate_channels=4,
        position_shape=(3, 3),
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_loops=3,
        dropout=0.0,
    )
    model = RecurrentLDT(config)
    candidates = torch.rand(1, 9, 4)

    output = model(candidates)

    assert len(model.blocks) == 1
    assert not torch.allclose(output.candidate_logits[0], output.candidate_logits[-1])


def test_model_accepts_read_only_extra_feature_channels() -> None:
    config = LDTConfig(
        num_positions=150,
        input_channels=7,
        candidate_channels=6,
        position_shape=(15, 10),
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_loops=2,
        dropout=0.0,
    )
    model = RecurrentLDT(config)
    candidates = torch.ones(4, 150, 6, dtype=torch.bool)
    in_puzzle_mask = torch.ones(4, 150, 1)

    output = model(candidates, in_puzzle_mask)

    assert output.candidate_logits.shape == (2, 4, 150, 6)
    assert output.cls_logits.shape == (2, 4)


def test_model_requires_extra_features_when_input_channels_exceed_candidates() -> None:
    config = LDTConfig(
        num_positions=9,
        input_channels=5,
        candidate_channels=4,
        position_shape=(3, 3),
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_loops=1,
        dropout=0.0,
    )
    model = RecurrentLDT(config)

    with pytest.raises(ValueError, match="extra_features required"):
        model(torch.ones(1, 9, 4))


def test_2d_rope_forward_path_is_shape_stable() -> None:
    torch.manual_seed(2)
    config = LDTConfig(
        num_positions=16,
        input_channels=3,
        candidate_channels=3,
        position_shape=(4, 4),
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_loops=2,
        dropout=0.0,
        use_rope=True,
    )
    model = RecurrentLDT(config)
    output = model(torch.rand(2, 16, 3))

    assert output.candidate_logits.shape == (2, 2, 16, 3)
    assert output.cls_logits.shape == (2, 2)


def test_base_sudoku_parameter_count_matches_paper_scale() -> None:
    config = LDTConfig(
        num_positions=81,
        input_channels=9,
        candidate_channels=9,
        position_shape=(9, 9),
    )
    model = RecurrentLDT(config)

    assert 780_000 <= count_parameters(model) <= 840_000
