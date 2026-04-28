#!/usr/bin/env python3
"""Composite OLS implicit-regularization figure (replaces standalone Figs 4 & 5).

Two rows:

  Row 1 -- three matrix heatmaps that share a colorbar, plus a 3-bar weight panel.
    (A) theoretical Lambda
    (B) multi-endpoint estimated Q-hat (full symmetric matrix from many
        independent training endpoints)
    (C) single-endpoint estimated Lambda-hat (diagonal-only, from one
        early-stopped run)
    To the right of (C): a 3-bar weight comparison showing
        (theta_hat_prime, theta_hat, theta) -- closed-form ridge under
        Lambda-hat, single-endpoint trained weights, and ground truth.

  Row 2 -- two side-by-side panels.
    Left:  distance-to-theory curve as a function of how many endpoints
           are used in the symmetric LSQ fit (with a marker at the count
           used in panel B).
    Right: heuristic single-endpoint estimator -- fits a 1-parameter
           regularizer at each GD checkpoint and plots the recovered
           value over training time, alongside the closed-form analytic
           solution and the theoretical tr(Q_t)/p target.

Inputs
------
    --linear-data         data/generated/linear_regression_ols/results.pt
    --full-matrix-data    data/generated/ols_full_matrix_recovery/results.pt
    --lambda-epochs-data  data/generated/lambda_vs_epochs/results.pt
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
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"

HEATMAP_CMAP = Colormap("colorcet:CET-CBD1").to_mpl()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-data", type=Path, required=True)
    parser.add_argument("--full-matrix-data", type=Path, required=True)
    parser.add_argument("--lambda-epochs-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _se95(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return 1.96 * x.std(dim=0, unbiased=True) / float(np.sqrt(n))


def _add_panel_label(fig, ax, letter: str, fontsize: int = 18) -> None:
    """Bold panel label anchored to the top-left of the full subplot tight bbox.

    Uses get_tightbbox so the label aligns with the outermost left edge of the
    subplot region (including y-tick and y-axis labels), not just the y-axis spine.
    """
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        tight = ax.get_tightbbox(r)
        ax_bb = ax.get_window_extent(r)
        x_frac = (tight.x0 - ax_bb.x0) / ax_bb.width
    except Exception:
        x_frac = 0.0
    ax.text(
        x_frac, 1.02, letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=fontsize,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _add_retraining_arrow(ax: Axes) -> None:
    """Draw the bracketed source-to-recovered-weights annotation for panel C."""
    arrow_color = "#2d3740"
    arrow = FancyArrowPatch(
        (0.02, 0.58),
        (0.98, 0.58),
        transform=ax.transAxes,
        arrowstyle="fancy",
        connectionstyle="arc3,rad=-0.4",
        mutation_scale=18,
        linewidth=2.4,
        edgecolor=arrow_color,
        facecolor=arrow_color,
        zorder=30,
        clip_on=False,
    )
    ax.add_patch(arrow)
    ax.text(
        0.50,
        0.76,
        "Retrain with \nexplicit regularization",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color=arrow_color,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
        clip_on=False,
        zorder=31,
    )


def _centered_colorbar_axis(
    fig: plt.Figure, slot: SubplotSpec, *, width_fraction: float = 0.76
) -> Axes:
    side_fraction = (1.0 - width_fraction) / 2.0
    cbar_grid = slot.subgridspec(
        1,
        3,
        width_ratios=[side_fraction, width_fraction, side_fraction],
        wspace=0.0,
    )
    return fig.add_subplot(cbar_grid[0, 1])


def _format_cbar_tick(value: float, _pos: int | None) -> str:
    if np.isclose(value, 0.0, atol=1e-12):
        return "0"
    return f"{value:.3g}"


def _style_horizontal_colorbar(cbar: Colorbar) -> None:
    cbar.formatter = FuncFormatter(_format_cbar_tick)
    cbar.update_ticks()
    cbar.ax.tick_params(axis="x", labelsize=8, pad=1, length=2.5)


def _match_vertical_span(ax: Axes, anchor_ax: Axes) -> None:
    anchor_box = anchor_ax.get_position()
    ax_box = ax.get_position()
    ax.set_position([ax_box.x0, anchor_box.y0, ax_box.width, anchor_box.height])


def _add_aligned_weight_axes(
    fig: plt.Figure,
    anchor_ax: Axes,
    slot_ax: Axes,
    *,
    n_vectors: int,
    n_rows: int,
) -> list[Axes]:
    fig.canvas.draw()
    anchor_box = anchor_ax.get_position()
    slot_box = slot_ax.get_position()
    fig_width, fig_height = fig.get_size_inches()

    vector_width = anchor_box.height * fig_height / (n_rows * fig_width)
    gap = 0.18 * vector_width
    total_width = n_vectors * vector_width + (n_vectors - 1) * gap

    if total_width > slot_box.width:
        gap = 0.06 * slot_box.width
        vector_width = (slot_box.width - (n_vectors - 1) * gap) / n_vectors
        total_width = n_vectors * vector_width + (n_vectors - 1) * gap

    left = slot_box.x0 + (slot_box.width - total_width) / 2.0
    return [
        fig.add_axes(
            [
                left + idx * (vector_width + gap),
                anchor_box.y0,
                vector_width,
                anchor_box.height,
            ]
        )
        for idx in range(n_vectors)
    ]


def main() -> None:
    args = parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    # --- Load artifacts --------------------------------------------------
    lin = torch.load(args.linear_data, weights_only=False)
    fm = torch.load(args.full_matrix_data, weights_only=False)
    lve = torch.load(args.lambda_epochs_data, weights_only=False)

    # Theory (single canonical Q from the linear-regression early-stopped run).
    Q_theory = lin["Q"].numpy()
    # Single-endpoint estimated diag (full Lambda matrix is diagonal-only).
    Q_single = lin["q_hat_diag"].numpy()
    # Multi-endpoint full-matrix estimate from pool 0, all 100 endpoints.
    from analysis.ols_full_matrix_recovery.pipeline import (
        fit_symmetric_matrix_from_points,
    )

    Q_multi = fit_symmetric_matrix_from_points(
        fm["theta_pool"][0], fm["target_pool"][0]
    ).numpy()

    # Weight bars -- single-endpoint experiment.
    theta_real = lin["theta_real"].numpy()
    theta_hat = lin["theta_hat"].numpy()
    theta_hat_prime = lin["theta_hat_prime"].numpy()

    # Distance-to-theory (full-matrix): mean +/- 95% SE across pools.
    distances_pool = fm["distances_pool"]  # [pools, num_endpoints]
    num_pools, num_endpoints = distances_pool.shape
    counts = np.arange(1, num_endpoints + 1)
    dist_mean = distances_pool.mean(dim=0).numpy()
    dist_sem = _se95(distances_pool).numpy()
    dist_lo = np.clip(dist_mean - dist_sem, 1e-12, None)
    dist_hi = dist_mean + dist_sem

    # Lambda vs epochs.
    epoch_grid = np.asarray(lve["epoch_grid"])
    iter_lambdas = np.asarray(lve["iter_lambdas"])
    closed_lambdas = np.asarray(lve["closed_lambdas"])
    theory_lambdas = np.asarray(lve["theoretical_lambdas"])

    # --- Composite figure -------------------------------------------------
    vmax_w = float(
        max(np.abs(theta_real).max(), np.abs(theta_hat).max(), np.abs(theta_hat_prime).max())
    )

    fig = plt.figure(figsize=(15.5, 9.0), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.30)

    # Row 1: keep the three Lambda heatmaps as equal sibling panels, then
    # place the retraining annotation and recovered weights to the right of C.
    row1 = outer[0].subgridspec(
        2, 5,
        width_ratios=[1.0, 1.0, 1.0, 0.62, 0.55],
        height_ratios=[1.0, 0.045],
        wspace=0.16,
        hspace=-0.06,
    )
    ax_A = fig.add_subplot(row1[0, 0])
    ax_B = fig.add_subplot(row1[0, 1])
    ax_C = fig.add_subplot(row1[0, 2])
    ax_cbar_A = _centered_colorbar_axis(fig, row1[1, 0])
    ax_cbar_B = _centered_colorbar_axis(fig, row1[1, 1])
    ax_cbar_C = _centered_colorbar_axis(fig, row1[1, 2])
    ax_retrain = fig.add_subplot(row1[0, 3], frameon=False)
    ax_retrain.set_axis_off()
    ax_retrain.patch.set_alpha(0.0)
    ax_retrain.set_zorder(25)
    _ax_retrain_spacer = fig.add_subplot(row1[1, 3], frameon=False)
    _ax_retrain_spacer.set_axis_off()
    ax_weights_slot = fig.add_subplot(row1[0, 4], frameon=False)
    ax_weights_slot.set_axis_off()
    ax_cbar_w = fig.add_subplot(row1[1, 4])

    for ax, mat, title, ax_cbar in [
        (ax_A, Q_theory, r"Theoretical $\Lambda$", ax_cbar_A),
        (ax_B, Q_multi, r"Multi-endpoint $\hat{\Lambda}$", ax_cbar_B),
        (ax_C, Q_single, r"Single-endpoint $\hat{\Lambda}$", ax_cbar_C),
    ]:
        vmax = float(np.abs(mat).max())
        im = ax.imshow(mat, cmap=HEATMAP_CMAP, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        _style_horizontal_colorbar(cbar)

    _match_vertical_span(ax_retrain, ax_C)
    ax_w0, ax_w1, ax_w2 = _add_aligned_weight_axes(
        fig,
        ax_C,
        ax_weights_slot,
        n_vectors=3,
        n_rows=len(theta_real),
    )

    norm_w = plt.Normalize(vmin=-vmax_w, vmax=vmax_w)
    for ax, vec, title in [
        (ax_w0, theta_hat_prime, r"$\hat{\theta}_{\Lambda}$"),
        (ax_w1, theta_hat, r"$\hat{\theta}$"),
        (ax_w2, theta_real, r"$\theta$"),
    ]:
        sns.heatmap(
            np.asarray(vec).reshape(-1, 1),
            cmap=HEATMAP_CMAP,
            norm=norm_w,
            square=False,
            cbar=False,
            yticklabels=False,
            xticklabels=False,
            ax=ax,
            linewidths=0.5,
            linecolor="k",
        )
        ax.set_title(title, fontsize=14)

    sm_w = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=norm_w)
    sm_w.set_array([])
    cbar_w = fig.colorbar(sm_w, cax=ax_cbar_w, orientation="horizontal")
    _style_horizontal_colorbar(cbar_w)
    _add_retraining_arrow(ax_retrain)

    # Row 2: distance curve | lambda vs epochs.
    row2 = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax_dist = fig.add_subplot(row2[0, 0])
    ax_lvse = fig.add_subplot(row2[0, 1])

    devon = Colormap("crameri:devon").to_mpl()
    line_color = devon(0.45)
    ax_dist.semilogy(
        counts,
        dist_mean,
        marker="o",
        markersize=3,
        color=line_color,
        label="Mean distance to theory",
    )
    ax_dist.fill_between(
        counts, dist_lo, dist_hi, color=line_color, alpha=0.2, label="95% SE"
    )
    ax_dist.set_xlabel("Number of endpoints used")
    ax_dist.set_ylabel(r"$\| \hat{\Lambda}_m - \Lambda \|$")
    ax_dist.set_xlim(1, num_endpoints)
    ax_dist.grid(alpha=0.3, which="both")
    ax_dist.legend(frameon=False)
    ax_dist.set_title("Multi-endpoint estimation error")

    batlow = Colormap("crameri:batlow").to_mpl()
    c_iter, c_closed, c_theory = batlow(0.2), batlow(0.55), batlow(0.85)
    ax_lvse.loglog(
        epoch_grid,
        iter_lambdas,
        "o-",
        color=c_iter,
        markersize=6,
        label=r"Iterative $\hat{\lambda}_t$ (gradient matching)",
    )
    ax_lvse.loglog(
        epoch_grid,
        closed_lambdas,
        "x",
        color=c_closed,
        markersize=8,
        markeredgewidth=1.8,
        label=r"Closed-form $\hat{\lambda}_t$",
    )
    ax_lvse.loglog(
        epoch_grid,
        theory_lambdas,
        "--",
        color=c_theory,
        linewidth=1.5,
        label=r"Theoretical $\mathrm{tr}(\Lambda_t)/p$",
    )
    ax_lvse.set_xlabel("Gradient descent epochs $t$")
    ax_lvse.set_ylabel(r"Scalar ridge penalty $\hat{\lambda}_t$")
    ax_lvse.legend(frameon=False, loc="lower left", fontsize=10)
    ax_lvse.grid(True, which="both", alpha=0.3)
    ax_lvse.set_title("Heuristic single-endpoint estimator over training")

    ax_C.text(
        -0.14,
        1.02,
        "C",
        transform=ax_C.transAxes,
        fontweight="bold",
        fontsize=18,
        va="bottom",
        ha="left",
        clip_on=False,
    )
    for ax, letter in [(ax_A, "A"), (ax_B, "B"), (ax_dist, "D"), (ax_lvse, "E")]:
        _add_panel_label(fig, ax, letter)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
