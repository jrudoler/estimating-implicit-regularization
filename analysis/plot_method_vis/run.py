#!/usr/bin/env python3
"""Method-visualization figures: tradeoff surface + SGD-vs-full-batch landscape.

Ported from the corresponding cells of notebooks/method-vis.ipynb so that
Snakemake doesn't have to exec notebook cells. The notebook is still useful as
an exploration surface; it just isn't the source of truth for these figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cmap import Colormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


REPO_ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"

Vec2 = Tuple[float, float]


# ----------------------------------------------------------------------
# Figure 1: tradeoff-vis.pdf
# ----------------------------------------------------------------------
def quad_mse(
    center: Vec2 = (1.0, 1.0), scale: float = 0.5
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    cx, cy = center

    def f(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
        return scale * ((W1 - cx) ** 2 + (W2 - cy) ** 2)

    return f


def l2_penalty(lam: float = 0.3) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def f(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
        return lam * (W1**2 + W2**2)

    return f


def ridge_minimizer_2d(
    lam: float, mse_scale: float = 0.5, mse_center: Vec2 = (1.0, 1.0)
) -> Vec2:
    cx, cy = mse_center
    alpha = mse_scale / (mse_scale + 2.0 * lam)
    return alpha * cx, alpha * cy


def make_tradeoff_vis(out_path: Path) -> None:
    """3D surface showing MSE + L2 = total objective; wireframe for data-fit,
    floor contours for the L2 penalty."""
    plot_floor = -30.0
    mse_center = (2.0, 3.0)
    mse_scale = 1.0
    lam = 1.5

    components: List[Callable[[np.ndarray, np.ndarray], np.ndarray]] = [
        quad_mse(center=mse_center, scale=mse_scale),
        l2_penalty(lam=lam),
    ]

    w = np.linspace(-3.5, 3.5, 200)
    W1, W2 = np.meshgrid(w, w)
    Zs = [comp(W1, W2) for comp in components]
    Z_total = np.sum(Zs, axis=0)

    w_mse = mse_center
    w_l2 = (0.0, 0.0)
    w_ridge = ridge_minimizer_2d(lam=lam, mse_scale=mse_scale, mse_center=mse_center)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        W1, W2, Z_total, linewidth=0, antialiased=False, alpha=0.8, cmap="Greys"
    )
    ax.plot_wireframe(
        W1[::8, ::8],
        W2[::8, ::8],
        Zs[0][::8, ::8],
        linewidth=0.5,
        alpha=0.6,
        color="red",
    )
    ax.contour(
        W1,
        W2,
        Zs[1],
        zdir="z",
        offset=plot_floor,
        levels=12,
        alpha=0.8,
        zorder=-1,
        cmap="Blues",
    )

    def scatter_point(x: float, y: float, z: float, label: str, **kwargs) -> None:
        ax.scatter([x], [y], [z], s=60, **kwargs)
        ax.text(x, y, z, "  " + label)

    scatter_point(*w_mse, plot_floor, "MSE min", marker="o", color="red", zorder=5)
    scatter_point(*w_l2, plot_floor, "L2 min", marker="^", color="blue", zorder=5)
    scatter_point(
        *w_ridge,
        plot_floor,
        f"Reg min ({w_ridge[0]:.2f},{w_ridge[1]:.2f})",
        marker="x",
        color="gray",
        zorder=5,
    )

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_zlabel("Objective")
    ax.set_title("Composing objectives: data-fit + L2 → regularized surface")
    ax.set_zlim(plot_floor, float(np.max(Z_total)))

    proxies = [
        Line2D([0], [0], color="red", lw=4, label="Data-fit (MSE)", alpha=0.6),
        Line2D([0], [0], color="blue", lw=4, label="Penalty (L2)", alpha=0.6),
        Line2D([0], [0], color="grey", lw=4, label="Total objective", alpha=0.8),
    ]
    ax.legend(handles=proxies, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: sgd-vs-full-batch.pdf
# ----------------------------------------------------------------------
def two_param_net(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    weight, bias = theta
    return torch.tanh(weight * x + bias)


def mse_loss(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(two_param_net(theta, x), y)


def autograd_gradient(
    theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta_var = theta.clone().detach().requires_grad_(True)
    loss = mse_loss(theta_var, x, y)
    loss.backward()
    return loss.detach(), theta_var.grad.detach()


def fit_minimum_of_interest(
    x: torch.Tensor,
    y: torch.Tensor,
    theta_init: torch.Tensor,
    steps: int = 900,
    lr: float = 0.04,
) -> torch.Tensor:
    theta = theta_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        mse_loss(theta, x, y).backward()
        optimizer.step()
    return theta.detach()


def make_sgd_vs_full_batch(out_path: Path) -> None:
    """Loss landscape over (weight, bias) for tanh(wx+b); overlay gradient-flow
    trajectory, SGD iterates, and per-step quivers contrasting the full-batch
    gradient direction (red) with the actual mini-batch update (lightgray)."""
    torch.manual_seed(4)

    x_data = torch.linspace(-2.5, 2.5, 96)
    y_data = 0.75 * torch.tanh(1.4 * x_data - 0.35) + 0.12 * torch.sin(2.7 * x_data)

    theta_seed = torch.tensor([0.8, 0.1])
    minimum_of_interest = fit_minimum_of_interest(x_data, y_data, theta_seed)

    rng = torch.Generator().manual_seed(21)
    direction = torch.randn(2, generator=rng)
    direction = direction / direction.norm()
    distance_to_minimum = 0.40
    theta0 = minimum_of_interest + distance_to_minimum * direction

    # Gradient-flow proxy: tiny LR + many steps with full-batch gradients.
    flow_lr = 0.004
    flow_steps = 3000
    flow_trajectory: List[torch.Tensor] = []
    current_theta = theta0.clone().detach()
    for _ in range(flow_steps):
        _, full_grad = autograd_gradient(current_theta, x_data, y_data)
        flow_trajectory.append(current_theta.detach())
        current_theta = current_theta - flow_lr * full_grad.detach()
    flow_trajectory_t = torch.stack(flow_trajectory)

    # Discrete SGD steps with a fixed (large-ish) learning rate.
    batch_size = 8
    learning_rate = 0.40
    step_count = 6
    trajectory: List[dict] = []
    current_theta = theta0.clone().detach()
    for step_idx in range(step_count):
        _, full_grad = autograd_gradient(current_theta, x_data, y_data)
        batch_indices = torch.randperm(x_data.numel(), generator=rng)[:batch_size]
        x_batch = x_data[batch_indices]
        y_batch = y_data[batch_indices]
        _, batch_grad = autograd_gradient(current_theta, x_batch, y_batch)
        theta_batch_step = current_theta - learning_rate * batch_grad
        trajectory.append(
            {
                "step": step_idx,
                "theta": current_theta.detach(),
                "full_grad": full_grad.detach(),
                "batch_grad": batch_grad.detach(),
                "theta_batch_step": theta_batch_step.detach(),
            }
        )
        current_theta = theta_batch_step.detach()

    trajectory_path = torch.stack(
        [item["theta"] for item in trajectory] + [trajectory[-1]["theta_batch_step"]]
    )

    # Pad the (weight, bias) grid so all visited points sit comfortably inside
    # the heatmap.
    focus_points = torch.stack(
        [minimum_of_interest]
        + [item["theta"] for item in trajectory]
        + [item["theta_batch_step"] for item in trajectory]
    )
    weight_range = (focus_points[:, 0].max() - focus_points[:, 0].min()).item()
    bias_range = (focus_points[:, 1].max() - focus_points[:, 1].min()).item()
    weight_padding = max(0.1, 0.25 * weight_range)
    bias_padding = max(0.1, 0.25 * bias_range)

    weight_grid = torch.linspace(
        focus_points[:, 0].min().item() - weight_padding,
        focus_points[:, 0].max().item() + weight_padding,
        320,
    )
    bias_grid = torch.linspace(
        focus_points[:, 1].min().item() - bias_padding,
        focus_points[:, 1].max().item() + bias_padding,
        320,
    )
    Weight, Bias = torch.meshgrid(weight_grid, bias_grid, indexing="xy")
    Loss = torch.mean(
        (torch.tanh(Weight[..., None] * x_data + Bias[..., None]) - y_data) ** 2,
        dim=-1,
    )

    cmap = Colormap("crameri:imola").to_mpl()
    fig, ax = plt.subplots(figsize=(8, 6))

    level_min = Loss.min().item()
    level_max = max(torch.quantile(Loss, 0.985).item(), level_min + 1e-4)
    levels = np.linspace(level_min, level_max, 28)

    ax.contourf(
        Weight.numpy(),
        Bias.numpy(),
        Loss.numpy(),
        levels=levels,
        cmap=cmap,
        extend="max",
    )
    ax.contour(
        Weight.numpy(),
        Bias.numpy(),
        Loss.numpy(),
        levels=levels[::2],
        colors="white",
        linewidths=0.45,
        alpha=0.6,
    )

    ax.plot(
        flow_trajectory_t[:, 0].numpy(),
        flow_trajectory_t[:, 1].numpy(),
        color="red",
        linewidth=1.8,
        linestyle="--",
        alpha=0.95,
        label="Gradient flow trajectory",
        zorder=1,
    )
    ax.scatter(
        trajectory_path[1:, 0].numpy(),
        trajectory_path[1:, 1].numpy(),
        s=36,
        color="lightgray",
        edgecolors="black",
        linewidths=0.6,
        zorder=7,
        label="SGD iterates",
    )
    ax.scatter(
        minimum_of_interest[0].item(),
        minimum_of_interest[1].item(),
        marker="*",
        s=150,
        color="gold",
        edgecolors="black",
        linewidths=0.9,
        zorder=6,
        label=r"$\theta^*$",
    )
    ax.scatter(
        theta0[0].item(),
        theta0[1].item(),
        s=60,
        color="k",
        edgecolors="black",
        linewidths=0.8,
        zorder=7,
        label=r"$\theta_0$",
    )

    for item in trajectory:
        theta_np = item["theta"].numpy()
        full_delta = (-learning_rate * item["full_grad"]).numpy()
        batch_delta = (-learning_rate * item["batch_grad"]).numpy()
        ax.quiver(
            theta_np[0], theta_np[1], full_delta[0], full_delta[1],
            angles="xy", scale_units="xy", scale=1,
            color="red", width=0.006, alpha=0.85, zorder=1,
        )
        ax.quiver(
            theta_np[0], theta_np[1], batch_delta[0], batch_delta[1],
            angles="xy", scale_units="xy", scale=1,
            color="lightgray", edgecolor="black",
            width=0.006, alpha=0.85, zorder=1,
        )

    ax.set_xlim(weight_grid.min().item(), weight_grid.max().item())
    ax.set_ylim(bias_grid.min().item(), bias_grid.max().item())
    ax.set_xlabel(r"$\theta_1$", fontsize=12)
    ax.set_ylabel(r"$\theta_2$", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-tradeoff", required=True, type=Path)
    parser.add_argument("--out-sgd-vs-fb", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    make_tradeoff_vis(args.out_tradeoff)
    print(args.out_tradeoff)

    make_sgd_vs_full_batch(args.out_sgd_vs_fb)
    print(args.out_sgd_vs_fb)


if __name__ == "__main__":
    main()
