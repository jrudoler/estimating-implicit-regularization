from __future__ import annotations

import torch

from core.bias import SmoothedPowerBias


def test_smoothed_power_bias_gradient_matches_autograd() -> None:
    flattened_params = torch.tensor([1.2, -0.7, 0.3], dtype=torch.float32, requires_grad=True)
    bias = SmoothedPowerBias(
        scale_init=0.7,
        exponent_init=1.4,
        epsilon=1e-4,
        trainable_scale=False,
        trainable_exponent=False,
    )

    penalty = bias(flattened_params, structured_params=None)
    autograd_gradient = torch.autograd.grad(penalty, flattened_params)[0]
    analytic_gradient = bias.penalty_gradient(flattened_params)

    assert torch.allclose(analytic_gradient, autograd_gradient, atol=1e-6, rtol=1e-5)


def test_smoothed_power_bias_transformed_quadratic_identity() -> None:
    flattened_params = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)
    bias = SmoothedPowerBias(
        scale_init=1.0,
        exponent_init=1.6,
        epsilon=1e-3,
        trainable_scale=False,
        trainable_exponent=False,
    )

    transformed = bias.transformed_parameters(flattened_params)
    penalty_terms = bias.elementwise_penalty(flattened_params)

    assert torch.allclose(transformed.pow(2), penalty_terms)
