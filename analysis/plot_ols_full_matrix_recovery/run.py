#!/usr/bin/env python3
"""Two figures from the OLS full-matrix endpoint-recovery experiment.

Reads data/generated/ols_full_matrix_recovery/results.pt and produces:

  * ols_full_matrix_recovery.pdf
        Two heatmaps (estimated Q_hat from all endpoints vs. median full
        theoretical Q_t over the same endpoints) plus a 3-bar weight panel
        comparing the closed-form ridge weights under Q_hat, the GD-stopped
        weights, and the true weights for the first endpoint.

  * ols_full_matrix_distance_to_theory.pdf
        Mean (+/- 95% SE across pools) of ||Q_hat_m - Q_theory|| as a function
        of the number m of endpoints used in the symmetric least-squares fit,
        with a vertical marker at the visual-comparison count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from cmap import Colormap


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.utils import compute_beta_closed_form  # noqa: E402

from analysis.ols_full_matrix_recovery.pipeline import (  # noqa: E402
    fit_symmetric_matrix_from_points,
)


STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DIVERGING_CMAP = Colormap("crameri:vik").to_mpl()
DEFAULT_VIS_MARKER = 10  # vertical marker on the distance plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out-recovery",
        type=Path,
        required=True,
        help="Output path for the Q-heatmap + weights figure.",
    )
    parser.add_argument(
        "--out-distance",
        type=Path,
        required=True,
        help="Output path for the distance-to-theory curve.",
    )
    parser.add_argument(
        "--vis-marker",
        type=int,
        default=DEFAULT_VIS_MARKER,
        help="Endpoint count at which to drop a vertical marker on the curve.",
    )
    return parser.parse_args()


def plot_recovery(
    Q_est: torch.Tensor,
    Q_theory: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    theta_hat: torch.Tensor,
    theta_true: torch.Tensor,
    num_endpoints: int,
    out_path: Path,
) -> None:
    p = Q_est.shape[0]
    theta_q = compute_beta_closed_form(X, y, Q_est).detach().cpu().numpy()
    Q_est_np = Q_est.detach().cpu().numpy()
    Q_theory_np = Q_theory.detach().cpu().numpy()
    theta_hat_np = theta_hat.detach().cpu().numpy()
    theta_true_np = theta_true.detach().cpu().numpy()

    vmax_q = float(max(np.abs(Q_est_np).max(), np.abs(Q_theory_np).max()))
    vmax_w = float(
        max(np.abs(theta_q).max(), np.abs(theta_hat_np).max(), np.abs(theta_true_np).max())
    )
    norm_w = plt.Normalize(vmin=-vmax_w, vmax=vmax_w)

    fig = plt.figure(figsize=(13.8, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.38])

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        Q_est_np, cmap=DIVERGING_CMAP, vmin=-vmax_q, vmax=vmax_q, aspect="equal"
    )
    ax0.set_title(r"Estimated $\hat{Q}$")
    ax0.set_xlabel(r"$p$")
    ax0.set_ylabel(r"$p$")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(
        Q_theory_np, cmap=DIVERGING_CMAP, vmin=-vmax_q, vmax=vmax_q, aspect="equal"
    )
    ax1.set_title("Symmetric full-matrix theory target")
    ax1.set_xlabel(r"$p$")

    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])

    cbar0 = fig.colorbar(im0, ax=[ax0, ax1], shrink=0.9, pad=0.02)
    cbar0.ax.set_ylabel("value")

    weights_spec = gs[0, 2].subgridspec(2, 3, height_ratios=[1.0, 0.08], hspace=0.08, wspace=0.02)
    weight_axes = [fig.add_subplot(weights_spec[0, i]) for i in range(3)]
    cax = fig.add_subplot(weights_spec[1, :])

    for ax, theta_vec, theta_title in zip(
        weight_axes,
        [theta_q, theta_hat_np, theta_true_np],
        [r"$\hat{\theta}_{Q}$", r"$\hat{\theta}_{k}$", r"$\theta$"],
    ):
        sns.heatmap(
            theta_vec.reshape(-1, 1),
            cmap=DIVERGING_CMAP,
            norm=norm_w,
            square=True,
            cbar=False,
            yticklabels=False,
            xticklabels=False,
            ax=ax,
            linewidths=0.5,
            linecolor="k",
        )
        ax.set_title(theta_title, fontsize=20)

    weight_axes[0].set_ylabel(r"$p$", fontsize=16)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=norm_w)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="horizontal")

    fig.suptitle(
        f"{num_endpoints}-endpoint symmetric full-matrix fit "
        f"({p * (p + 1) // 2} parameters)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_distance_curve(
    distances_pool: torch.Tensor,
    vis_marker: int,
    out_path: Path,
) -> None:
    num_avg_seeds, num_endpoints = distances_pool.shape
    counts = np.arange(1, num_endpoints + 1)
    mean = distances_pool.mean(dim=0).numpy()
    sem95 = (
        1.96
        * distances_pool.std(dim=0, unbiased=True).numpy()
        / np.sqrt(num_avg_seeds)
    )
    lower = np.clip(mean - sem95, 1e-12, None)
    upper = mean + sem95

    devon = Colormap("crameri:devon").to_mpl()
    line_color = devon(0.45)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    ax.semilogy(
        counts,
        mean,
        marker="o",
        markersize=3,
        color=line_color,
        label="Mean distance to theory",
    )
    ax.fill_between(
        counts, lower, upper, color=line_color, alpha=0.2, label="95% SE"
    )
    ax.axvline(
        vis_marker,
        color="#222222",
        linestyle=":",
        linewidth=1.2,
        label=f"{vis_marker} endpoints",
    )
    ax.set_xlabel("Number of Endpoints Used")
    ax.set_ylabel("Distance to Theory")
    ax.set_xlim(1, num_endpoints)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    payload = torch.load(args.input, weights_only=False)
    config = payload["config"]
    theta_pool = payload["theta_pool"]
    target_pool = payload["target_pool"]
    Q_theory_pool = payload["Q_theory_pool"]
    distances_pool = payload["distances_pool"]
    X = payload["X"]
    vis_beta_first = payload["vis_beta_first"]
    vis_y_first = payload["vis_y_first"]
    vis_theta_first = payload["vis_theta_first"]

    # Visualise pool 0 with all endpoints; the experiment already produced a
    # median-over-endpoints theory matrix per pool that we use as the side-by-
    # side comparison.
    Q_est = fit_symmetric_matrix_from_points(theta_pool[0], target_pool[0])
    plot_recovery(
        Q_est=Q_est,
        Q_theory=Q_theory_pool[0],
        X=X,
        y=vis_y_first,
        theta_hat=vis_theta_first,
        theta_true=vis_beta_first,
        num_endpoints=config["num_endpoints"],
        out_path=args.out_recovery,
    )
    plot_distance_curve(
        distances_pool=distances_pool,
        vis_marker=args.vis_marker,
        out_path=args.out_distance,
    )

    print(f"Saved {args.out_recovery}")
    print(f"Saved {args.out_distance}")


if __name__ == "__main__":
    main()
