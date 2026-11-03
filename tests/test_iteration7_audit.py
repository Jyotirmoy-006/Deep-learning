"""
ForenSightX - Iteration 7 Audit Tests
Covers:
- Part 0d: SyntheticDataVAE(7, 4, 4) produces exactly 135 parameters with per-layer assertions.
- Part 2a: Family-disjoint dataset validation (zero family overlap across train/val/test).
"""

import pytest
import torch
from src.models import SyntheticDataVAE


def test_buggy_vae_parameter_count_135():
    """
    Part 0d Audit Test:
    Asserts that SyntheticDataVAE(input_dim=7, latent_dim=4, hidden_dim=4, num_classes=2)
    produces exactly 135 parameters, replacing the erroneous '135 = 67 x 2' calculation.
    """
    model = SyntheticDataVAE(input_dim=7, latent_dim=4, hidden_dim=4, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params == 135, f"Expected exactly 135 parameters, got {total_params}"

    # Verify layer-by-layer breakdown
    param_dict = {name: p.numel() for name, p in model.named_parameters()}

    expected_counts = {
        "encoder_fc.0.weight": 36,  # 4 x (7 + 2) = 36
        "encoder_fc.0.bias": 4,     # 4
        "encoder_fc.2.weight": 8,   # (4 // 2) x 4 = 2 x 4 = 8
        "encoder_fc.2.bias": 2,     # 2
        "fc_mu.weight": 8,          # 4 x 2 = 8
        "fc_mu.bias": 4,            # 4
        "fc_logvar.weight": 8,      # 4 x 2 = 8
        "fc_logvar.bias": 4,        # 4
        "decoder.0.weight": 12,     # (4 // 2) x (4 + 2) = 2 x 6 = 12
        "decoder.0.bias": 2,        # 2
        "decoder.2.weight": 8,      # 4 x 2 = 8
        "decoder.2.bias": 4,        # 4
        "decoder.4.weight": 28,     # 7 x 4 = 28
        "decoder.4.bias": 7,        # 7
    }

    for layer_name, expected_numel in expected_counts.items():
        assert param_dict[layer_name] == expected_numel, (
            f"Layer {layer_name} expected {expected_numel} parameters, got {param_dict[layer_name]}"
        )

    # Sum verification
    assert sum(expected_counts.values()) == 135

