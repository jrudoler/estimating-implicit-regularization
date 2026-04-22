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


def _compute_flat_grad_at(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    features: Tensor,
    targets: Tensor,
    flat_theta: Tensor,
) -> Tensor:
    """Evaluate flat grad L at a specified flat parameter vector without
    mutating the caller's model. Uses functional_call so the model's own
    parameters are untouched."""
    from collections import OrderedDict
    from torch.func import functional_call

    # Rebuild the structured param dict from the flat vector.
    param_dict: OrderedDict[str, Tensor] = OrderedDict()
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        chunk = flat_theta[offset : offset + n].view(p.shape)
        param_dict[name] = chunk.detach().clone().requires_grad_(True)
        offset += n

    predictions = functional_call(model, param_dict, (features,))
    loss = loss_fn(predictions, targets)
    grads = torch.autograd.grad(loss, tuple(param_dict.values()))
    return torch.cat([g.reshape(-1) for g in grads]).detach()


def _compute_flat_hg_at(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    features: Tensor,
    targets: Tensor,
    flat_theta: Tensor,
) -> tuple[Tensor, Tensor]:
    """Flat (grad, H@grad) at a specified flat parameter vector, without
    mutating the caller's model."""
    from collections import OrderedDict
    from torch.func import functional_call

    param_dict: OrderedDict[str, Tensor] = OrderedDict()
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        chunk = flat_theta[offset : offset + n].view(p.shape)
        param_dict[name] = chunk.detach().clone().requires_grad_(True)
        offset += n

    predictions = functional_call(model, param_dict, (features,))
    loss = loss_fn(predictions, targets)
    params_tuple = tuple(param_dict.values())
    grads = torch.autograd.grad(loss, params_tuple, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    hvp_tensors = torch.autograd.grad(
        grads, params_tuple, grad_outputs=grads, retain_graph=False
    )
    flat_hvp = torch.cat([h.reshape(-1) for h in hvp_tensors])
    return flat_grad.detach(), flat_hvp.detach()


def integrate_gradient_flow_rk4(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    features: Tensor,
    targets: Tensor,
    theta0: Tensor,
    t_end: float,
    n_steps: int,
) -> Tensor:
    """Fixed-step RK4 integration of gradient flow dtheta/dt = -grad L(theta).

    Returns theta(t_end). Does not mutate the model's own parameters.
    Local error O(h^5), global O(h^4) where h = t_end/n_steps, so even modest
    n_steps gives a reference trajectory essentially indistinguishable from
    the true ODE solution at the scales we care about (h << 1).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for RK4 integration.")
    h = t_end / n_steps
    theta = theta0.detach().clone()
    for _ in range(n_steps):
        k1 = -_compute_flat_grad_at(model, loss_fn, features, targets, theta)
        k2 = -_compute_flat_grad_at(model, loss_fn, features, targets, theta + 0.5 * h * k1)
        k3 = -_compute_flat_grad_at(model, loss_fn, features, targets, theta + 0.5 * h * k2)
        k4 = -_compute_flat_grad_at(model, loss_fn, features, targets, theta + h * k3)
        theta = theta + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return theta


def integrate_modified_flow_rk4(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    features: Tensor,
    targets: Tensor,
    theta0: Tensor,
    t_end: float,
    n_steps: int,
    eta: float,
) -> Tensor:
    """Fixed-step RK4 integration of the Barrett-modified flow
        dtheta/dt = -grad L(theta) - (eta/2) * H(theta) @ grad L(theta).

    This is the ODE whose trajectory Barrett & Dherin (2021) predict GD with
    step size eta follows to O(eta^2) per step. Returns theta(t_end).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for RK4 integration.")
    h = t_end / n_steps

    def drift(th: Tensor) -> Tensor:
        g, hg = _compute_flat_hg_at(model, loss_fn, features, targets, th)
        return -g - 0.5 * eta * hg

    theta = theta0.detach().clone()
    for _ in range(n_steps):
        k1 = drift(theta)
        k2 = drift(theta + 0.5 * h * k1)
        k3 = drift(theta + 0.5 * h * k2)
        k4 = drift(theta + h * k3)
        theta = theta + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return theta
