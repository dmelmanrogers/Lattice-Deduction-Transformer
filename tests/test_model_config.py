from __future__ import annotations

import pytest

from ldt.domains.sudoku import SudokuDomain
from ldt.model import LDTConfig


def test_config_from_domain_spec_uses_candidate_channels_as_default_input() -> None:
    config = LDTConfig.from_domain_spec(
        SudokuDomain().spec,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_loops=2,
    )

    assert config.num_positions == 81
    assert config.input_channels == 9
    assert config.candidate_channels == 9
    assert config.position_shape == (9, 9)


def test_config_supports_read_only_extra_input_channels() -> None:
    config = LDTConfig.from_domain_spec(
        SudokuDomain().spec,
        extra_input_channels=1,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_loops=2,
    )

    assert config.input_channels == 10
    assert config.candidate_channels == 9


def test_config_rejects_invalid_rope_shape_and_head_width() -> None:
    with pytest.raises(ValueError, match="requires a 2D position_shape"):
        LDTConfig(
            num_positions=8,
            input_channels=3,
            candidate_channels=3,
            d_model=32,
            num_heads=4,
            use_rope=True,
        )

    with pytest.raises(ValueError, match="per-head dimension divisible by 4"):
        LDTConfig(
            num_positions=16,
            input_channels=3,
            candidate_channels=3,
            position_shape=(4, 4),
            d_model=40,
            num_heads=4,
            use_rope=True,
        )
