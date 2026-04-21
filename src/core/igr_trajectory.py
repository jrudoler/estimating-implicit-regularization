"""Trajectory-based estimator for Barrett & Dherin (2022) IGR.

Fits the single scalar lambda in the gradient-penalty regularizer
    R(theta, lambda) = (lambda / p) * ||grad L(theta)||^2
against empirical trajectory targets collected during training.

At each checkpoint t along a training trajectory we have:
  - Hg_t: Hessian-vector product H(theta_t) @ grad L(theta_t)
  - target_t: the observed residual that grad R must match.
    In "flow-ref" mode: (Delta_theta_flow - Delta_theta_GD) / eta
    In "sgd" mode:      -Delta_theta_SGD / eta - grad L_full(theta_t)

The penalty gradient is (2 lambda / p) * Hg_t, so matching is a simple
1D least-squares problem in lambda.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class IGRFitResult:
    lambda_hat: float
    lambda_hat_per_param: float
    residual_ratio: float  # ||target - predicted|| / ||target||
    hg_norm_mean: float
    target_norm_mean: float
    num_checkpoints: int


def fit_igr_lambda_closed_form(
    hg_list: Sequence[Tensor],
    target_list: Sequence[Tensor],
    num_params: int,
) -> IGRFitResult:
    """Closed-form least-squares fit of lambda in (2 lambda / p) * Hg ~ target.

    Stacks per-checkpoint (Hg_t, target_t) vectors and solves
        lambda_hat = (p / 2) * <Hg_stack, target_stack> / <Hg_stack, Hg_stack>.
    """
    if len(hg_list) != len(target_list):
        raise ValueError(
            f"hg_list and target_list must have same length; got "
            f"{len(hg_list)} vs {len(target_list)}."
        )
    if len(hg_list) == 0:
        raise ValueError("Need at least one checkpoint.")

    hg = torch.cat([h.reshape(-1) for h in hg_list])
    target = torch.cat([t.reshape(-1) for t in target_list])
    if hg.shape != target.shape:
        raise ValueError(
            f"Stacked Hg and target must have matching shapes; "
            f"got {tuple(hg.shape)} vs {tuple(target.shape)}."
        )

    hg_sq = torch.dot(hg, hg)
    lambda_per_param = 0.5 * torch.dot(hg, target) / hg_sq
    lambda_hat = lambda_per_param * num_params

    predicted = (2.0 * lambda_per_param) * hg
    target_norm = target.norm().clamp_min(1e-30)
    residual_ratio = (target - predicted).norm() / target_norm

    hg_norms = torch.stack([h.reshape(-1).norm() for h in hg_list])
    target_norms = torch.stack([t.reshape(-1).norm() for t in target_list])

    return IGRFitResult(
        lambda_hat=float(lambda_hat),
        lambda_hat_per_param=float(lambda_per_param),
        residual_ratio=float(residual_ratio),
        hg_norm_mean=float(hg_norms.mean()),
        target_norm_mean=float(target_norms.mean()),
        num_checkpoints=len(hg_list),
    )


def compute_full_batch_grad_and_hvp(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    features: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute flat-vector g = grad L(theta) and Hg = H @ g on the full batch.

    Uses retain_graph to avoid double backward issues. Detaches outputs.
    """
    from collections import OrderedDict
    from torch.func import functional_call

    param_clones = OrderedDict(
        (name, p.detach().clone().requires_grad_(True))
        for name, p in model.named_parameters()
    )
    predictions = functional_call(model, param_clones, (features,))
    loss = loss_fn(predictions, targets)

    params_tuple = tuple(param_clones.values())
    grads = torch.autograd.grad(loss, params_tuple, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    hvp_tensors = torch.autograd.grad(
        grads,
        params_tuple,
        grad_outputs=grads,
        retain_graph=False,
    )
    flat_hvp = torch.cat([h.reshape(-1) for h in hvp_tensors])
    return flat_grad.detach(), flat_hvp.detach()
