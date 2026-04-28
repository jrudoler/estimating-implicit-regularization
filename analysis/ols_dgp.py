"""Shared data-generating process for synthetic OLS experiments."""

from __future__ import annotations

import torch


DEFAULT_OLS_SEED = 56
DEFAULT_OLS_N = 1000
DEFAULT_OLS_P = 10
DEFAULT_OLS_BETA_SCALE = 3.0
DEFAULT_OLS_NOISE_STD = 1.0


def sample_ols_design(
    *,
    n: int = DEFAULT_OLS_N,
    p: int = DEFAULT_OLS_P,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw the Gaussian design matrix used by the OLS experiments."""
    return torch.randn(n, p, generator=generator, dtype=dtype)


def sample_ols_beta(
    *,
    p: int = DEFAULT_OLS_P,
    beta_scale: float = DEFAULT_OLS_BETA_SCALE,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw the ground-truth regression coefficients."""
    return beta_scale * torch.randn(p, generator=generator, dtype=dtype)


def make_noisy_linear_response(
    X: torch.Tensor,
    beta: torch.Tensor,
    *,
    noise_std: float = DEFAULT_OLS_NOISE_STD,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate y = X beta + epsilon with Gaussian observation noise."""
    y = X @ beta
    if noise_std == 0.0:
        return y
    noise = noise_std * torch.randn(
        y.shape,
        generator=generator,
        dtype=y.dtype,
        device=y.device,
    )
    return y + noise


def sample_ols_problem(
    *,
    seed: int = DEFAULT_OLS_SEED,
    n: int = DEFAULT_OLS_N,
    p: int = DEFAULT_OLS_P,
    beta_scale: float = DEFAULT_OLS_BETA_SCALE,
    noise_std: float = DEFAULT_OLS_NOISE_STD,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a complete synthetic OLS problem from the shared DGP."""
    generator = torch.Generator().manual_seed(seed)
    X = sample_ols_design(n=n, p=p, generator=generator, dtype=dtype)
    beta = sample_ols_beta(
        p=p,
        beta_scale=beta_scale,
        generator=generator,
        dtype=dtype,
    )
    y = make_noisy_linear_response(X, beta, noise_std=noise_std, generator=generator)
    return X, y, beta
