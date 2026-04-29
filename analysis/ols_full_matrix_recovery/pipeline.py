"""Helpers for the OLS full-matrix endpoint-recovery experiment.

Each "endpoint" is a (X, beta_k) -> y_k regression that is trained by full-batch
gradient descent until a callback decides we have early-stopped, yielding a
weight vector theta_k and the negative loss-gradient -g_k = -X.T (X theta_k - y_k) / n.
Stacking M of these endpoints gives a system

    sum_{ij} Q_{ij} * theta_k[j] = (-g_k)[i]      for k = 1..M, i = 1..p

with p*(p+1)/2 unknowns under the symmetry constraint Q_{ij} = Q_{ji}, and Mp
equations. With enough endpoints this least-squares problem recovers the full
theoretical regularizer Q_t implied by early-stopped GD.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def loss_grad(X: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    return X.T @ (X @ theta - y) / n


def gd_trajectory(
    X: torch.Tensor, y: torch.Tensor, steps: int, eps: float
) -> dict[str, torch.Tensor]:
    theta = torch.zeros(X.shape[1], dtype=X.dtype)
    thetas = [theta.clone()]
    for _ in range(steps):
        theta = theta - eps * loss_grad(X, y, theta)
        thetas.append(theta.clone())
    return {"theta": torch.stack(thetas)}


@dataclass
class Endpoint:
    beta: torch.Tensor          # ground-truth regression weights, [p]
    y: torch.Tensor             # response, [n]
    stop_step: int              # callback-selected GD step
    model_theta: torch.Tensor   # weights at the stop step, [p]
    neg_grad_stop: torch.Tensor # -loss_grad at the stop step, [p]
    Q_theory_stop: torch.Tensor # theoretical Q_{k_r} at stop_step (n=N), [p, p]


def fit_symmetric_matrix_from_points(
    theta_points: torch.Tensor, target_points: torch.Tensor
) -> torch.Tensor:
    """Solve for the symmetric Q in stacked Q theta_k = -g_k via least squares.

    theta_points: [M, p], target_points: [M, p]. Returns Q in [p, p].
    """
    if theta_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("theta_points and target_points must both be rank-2 tensors")
    if theta_points.shape != target_points.shape:
        raise ValueError("theta_points and target_points must have matching shapes")

    num_points, p = theta_points.shape
    pairs = [(i, j) for i in range(p) for j in range(i, p)]
    design = theta_points.new_zeros((num_points * p, len(pairs)))
    rows = torch.arange(num_points, device=theta_points.device) * p

    for col, (i, j) in enumerate(pairs):
        design[rows + i, col] += theta_points[:, j]
        if i != j:
            design[rows + j, col] += theta_points[:, i]

    coeffs = torch.linalg.lstsq(design, target_points.reshape(-1)).solution
    Q = theta_points.new_zeros((p, p))
    for coeff, (i, j) in zip(coeffs, pairs):
        Q[i, j] = coeff
        Q[j, i] = coeff
    return Q


def matrix_relative_distance(est: torch.Tensor, theory: torch.Tensor) -> float:
    return float(torch.linalg.norm(est - theory))
