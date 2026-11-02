"""Training continuation and explicit practical stationarity tolerances."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def assess_convergence(control, *, baseline_norm, dropout, norm_absolute=1e-4,
                       norm_relative=.1, norm_reduction=.1,
                       coefficient_absolute=1e-7, coefficient_relative=.05):
    """Require small upper uncertainty bounds, not failure to reject zero.

    The norm interval is the existing approximate jackknife interval. This is a
    declared practical tolerance, not a proof of convergence or global optimality.
    The zero-dropout control uses an absolute norm tolerance because train=eval.
    """
    signed = control["endpoint"]["scale_star"]
    ci = control["direct_signed_lambda_mc_ci95"]
    if signed is None or ci is None:
        raise ValueError("A nonzero ridge basis is required")
    bias_interval = [value - signed for value in ci]
    coefficient_bound = max(abs(value) for value in bias_interval)
    coefficient_limit = max(coefficient_absolute, coefficient_relative * abs(signed))
    norm_limit = min(norm_reduction * baseline_norm,
                     max(norm_absolute, norm_relative * control["clean_gradient_norm"])
                     if dropout else norm_absolute)
    norm_upper_sq = control["train_gradient_norm_sq_ci95"][1]
    # An entirely negative norm interval is an MC anomaly, not stationarity.
    valid = math.isfinite(norm_upper_sq) and norm_upper_sq >= 0
    norm_pass = valid and norm_upper_sq <= norm_limit**2
    coefficient_pass = coefficient_bound <= coefficient_limit
    return {
        "passes": bool(norm_pass and coefficient_pass),
        "norm_pass": bool(norm_pass), "coefficient_pass": bool(coefficient_pass),
        "norm_upper_95": math.sqrt(norm_upper_sq) if valid else None,
        "norm_limit": norm_limit,
        "coefficient_bias_ci95": bias_interval,
        "coefficient_bias_upper_abs": coefficient_bound,
        "coefficient_bias_limit": coefficient_limit,
        "point_estimates_near_tolerance": bool(
            control["train_gradient_norm_sq_unbiased"] <= norm_limit**2
            and abs(control["signed_coefficient_contamination"]) <= coefficient_limit),
    }


def continuation_lr(epoch, *, initial=.01):
    """New schedule, in additional completed epochs; no validation early stop."""
    return initial * .2 ** sum(epoch >= milestone for milestone in (1000, 2500, 3500))


def train_epoch(model, batches, optimizer):
    """One shuffled SGD epoch on the unchanged dropout CE objective."""
    model.train()
    order = torch.randperm(len(batches.y), device=batches.y.device)
    loss_sum = torch.zeros((), device=batches.y.device)
    for index in order.split(batches.batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batches.x[index]), batches.y[index])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite continuation loss")
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * len(index)
    return float(loss_sum / len(batches.y))
