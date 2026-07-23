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


def loss_grad_trajectory(
    X: torch.Tensor, y: torch.Tensor, thetas: torch.Tensor
) -> torch.Tensor:
    """Vectorised loss gradients for a whole trajectory.

    thetas: [T+1, p] — one row per GD iterate. Returns [T+1, p] where
    row k is loss_grad(X, y, thetas[k]).  Useful for trajectory visualisation;
    for a single point prefer calling loss_grad directly.
    """
    n = X.shape[0]
    c = (X.T @ y) / n
    A = (X.T @ X) / n
    return thetas @ A.T - c.unsqueeze(0)


def mse_loss(X: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return (X @ theta - y).square().mean()


def gd_trajectory(
    X: torch.Tensor, y: torch.Tensor, steps: int, eps: float
) -> dict[str, torch.Tensor]:
    theta = torch.zeros(X.shape[1], dtype=X.dtype)
    thetas = [theta.clone()]
    losses = [mse_loss(X, y, theta)]
    for _ in range(steps):
        theta = theta - eps * loss_grad(X, y, theta)
        thetas.append(theta.clone())
        losses.append(mse_loss(X, y, theta))
    return {"theta": torch.stack(thetas), "loss": torch.stack(losses)}


def callback_stop_step(
    monitored_losses: torch.Tensor, patience: int, min_delta: float
) -> int:
    """Lightning-style early-stopping over a precomputed loss trace.

    Returns the index k such that monitored_losses[k] is the iterate used as
    the final model — i.e. the last step before patience consecutive steps with
    no improvement >= min_delta.  Mirrors the behaviour of
    lightning.pytorch.callbacks.EarlyStopping(mode="min", restore_best_weights=False).
    """
    best = float("inf")
    wait = 0
    for epoch in range(len(monitored_losses) - 1):
        current = float(monitored_losses[epoch])
        if current < best - min_delta:
            best = current
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return epoch + 1
    return len(monitored_losses) - 1


def freeze_after_stop(theta_traj: torch.Tensor, stop_step: int) -> torch.Tensor:
    """Return a copy of theta_traj with all iterates after stop_step clamped.

    Useful for visualising what a GD trajectory looks like under early stopping:
    the trajectory is frozen at stop_step rather than continuing to converge.
    Not needed for scalar endpoint extraction — index theta_traj[stop_step] directly.
    """
    frozen = theta_traj.clone()
    if stop_step + 1 < frozen.shape[0]:
        frozen[stop_step + 1 :] = frozen[stop_step]
    return frozen


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
