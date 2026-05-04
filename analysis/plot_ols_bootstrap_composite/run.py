#!/usr/bin/env python3
"""Bootstrap OLS full-matrix recovery figure.

Six-panel appendix figure parallel to ols_composite.pdf, but replacing
independent re-draws of ground-truth weights with bootstrap resampling of the
original (X, y) dataset.  We use a noisier DGP (sigma=10) so that the OLS
sampling variance is large enough for bootstrap-induced theta variability to
span the 55-dim symmetric-matrix space.

  (A) Theoretical regularisation matrix Lambda^{(t)} from original X.
  (B) m=1: minimum-norm fit from the single canonical training endpoint
      (10 equations in 55 unknowns, so most of the matrix is left at zero).
  (C) m=50: 49 bootstrap resamples added to the canonical endpoint.
  (D) m=100: 99 bootstrap resamples added to the canonical endpoint.
  (E) Distance-to-theory ||Lambda_hat_m - Lambda^{(t)}|| as the number of
      endpoints m grows from 1 to 100, with markers at m=50, m=100.
  (F) Distance-to-theory at m=100 swept over observation noise sigma, with
      sigma=10 (the value used in panels A-E) marked.

Inputs
------
    --linear-data            data/generated/linear_regression_ols_noisy/results.pt
    --bootstrap-panel-b-data data/generated/ols_bootstrap_recovery_panel_b/results.pt
    --bootstrap-full-data    data/generated/ols_bootstrap_recovery/results.pt
    --sigma-sweep-data       data/generated/ols_bootstrap_sigma_sweep/results.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmap import Colormap
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
HEATMAP_CMAP = Colormap("colorcet:CET-CBD1").to_mpl()

PANEL_MS = [1, 50, 100]
HIGHLIGHT_SIGMA = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-data", type=Path, required=True)
    parser.add_argument("--bootstrap-panel-b-data", type=Path, required=True)
    parser.add_argument("--bootstrap-full-data", type=Path, required=True)
    parser.add_argument("--sigma-sweep-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _format_cbar_tick(value: float, _pos: int | None) -> str:
    if np.isclose(value, 0.0, atol=1e-12):
        return "0"
    return f"{value:.3g}"


def _style_horizontal_colorbar(cbar) -> None:
    cbar.formatter = FuncFormatter(_format_cbar_tick)
    cbar.update_ticks()
    cbar.ax.tick_params(axis="x", labelsize=8, pad=1, length=2.5)


def _add_panel_label(ax, letter: str, fontsize: int = 18, x: float = -0.05) -> None:
    ax.text(
        x, 1.02, letter, transform=ax.transAxes,
        fontweight="bold", fontsize=fontsize,
        va="bottom", ha="left", clip_on=False,
    )


def main() -> None:
    args = parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    lin = torch.load(args.linear_data, weights_only=False)
    pb = torch.load(args.bootstrap_panel_b_data, weights_only=False)
    sweep = torch.load(args.sigma_sweep_data, weights_only=False)

    Q_theory = lin["Q"].numpy()
    Qt_norm = float(np.linalg.norm(Q_theory))

    from analysis.ols_full_matrix_recovery.pipeline import (
        fit_symmetric_matrix_from_points,
    )

    # Build a single 100-endpoint pool with the canonical theta as endpoint 0
    # (bootstrap pool 0 contributes the remaining 99 resamples).  This makes
    # the m=1 column literally "the original dataset".
    theta_canonical = lin["theta_hat"].to(torch.float32)
    X_orig = lin["X"]
    y_orig = lin["y"]
    target_canonical = -(X_orig.T @ (X_orig @ theta_canonical - y_orig) / X_orig.shape[0])

    thetas = torch.cat([theta_canonical.unsqueeze(0), pb["theta_pool"][0, :99]], dim=0)
    targets = torch.cat([target_canonical.unsqueeze(0), pb["target_pool"][0, :99]], dim=0)

    matrices = {
        m: fit_symmetric_matrix_from_points(thetas[:m], targets[:m]).numpy()
        for m in PANEL_MS
    }
    counts = np.arange(1, thetas.shape[0] + 1)
    dist_curve = np.array([
        np.linalg.norm(
            fit_symmetric_matrix_from_points(thetas[:m], targets[:m]).numpy() - Q_theory
        )
        for m in counts
    ])

    sweep_sigmas = sweep["sigmas"].numpy()
    sweep_dist_pool = sweep["distances"].numpy()
    sweep_dist_mean = sweep_dist_pool.mean(axis=1)
    sweep_dist_se = sweep_dist_pool.std(axis=1, ddof=1) / np.sqrt(sweep_dist_pool.shape[1])
    sweep_dist_lo = sweep_dist_mean - 1.96 * sweep_dist_se
    sweep_dist_hi = sweep_dist_mean + 1.96 * sweep_dist_se

    fig = plt.figure(figsize=(15.5, 8.5), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.35)

    row1 = outer[0].subgridspec(
        2, 4, width_ratios=[1.0] * 4, height_ratios=[1.0, 0.05],
        wspace=0.18, hspace=-0.04,
    )
    panels = [
        (Q_theory, r"$\Lambda^{(t)}$ (theory)"),
        (matrices[1], r"$\hat{\Lambda}_{m=1}^{(t)}$ (original dataset only)"),
        (matrices[50], r"$\hat{\Lambda}_{m=50}^{(t)}$"),
        (matrices[100], r"$\hat{\Lambda}_{m=100}^{(t)}$"),
    ]
    panel_axes = []
    for k, (mat, title) in enumerate(panels):
        ax = fig.add_subplot(row1[0, k])
        cgrid = row1[1, k].subgridspec(1, 3, width_ratios=[0.12, 0.76, 0.12], wspace=0.0)
        ax_cbar = fig.add_subplot(cgrid[0, 1])
        vmax = max(np.abs(mat).max(), 1e-9)
        im = ax.imshow(mat, cmap=HEATMAP_CMAP, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        _style_horizontal_colorbar(cbar)
        panel_axes.append(ax)

    for ax, letter in zip(panel_axes, ["A", "B", "C", "D"]):
        _add_panel_label(ax, letter)

    devon = Colormap("crameri:devon").to_mpl()
    line_color = devon(0.45)
    marker_color = devon(0.15)

    row2 = outer[1].subgridspec(1, 2, wspace=0.28)
    ax_m = fig.add_subplot(row2[0, 0])
    ax_s = fig.add_subplot(row2[0, 1])

    ax_m.semilogy(counts, dist_curve, "o-", color=line_color, markersize=3)
    ax_m.axhline(Qt_norm, ls="--", color="gray",
                 label=r"$\|\Lambda^{(t)}\|_F$ (zero-baseline)")
    for m, lab in zip(PANEL_MS, ["B", "C", "D"]):
        ax_m.scatter([m], [dist_curve[m - 1]], s=85, color=marker_color, zorder=5,
                     edgecolor="white", linewidth=1.4)
        ax_m.annotate(lab, xy=(m, dist_curve[m - 1]),
                      xytext=(8, 8), textcoords="offset points",
                      fontsize=12, fontweight="bold", color=marker_color)
    ax_m.set_xlabel(r"number of endpoints $m$ (1 canonical + $m{-}1$ bootstrap)")
    ax_m.set_ylabel(r"$\|\hat{\Lambda}_m^{(t)} - \Lambda^{(t)}\|_F$")
    ax_m.set_title(rf"Distance vs $m$ (at $\sigma={int(HIGHLIGHT_SIGMA)}$)", fontsize=11)
    ax_m.set_xlim(1, counts[-1])
    ax_m.grid(True, which="both", alpha=0.3)
    ax_m.legend(frameon=False, fontsize=10, loc="upper right")
    _add_panel_label(ax_m, "E", x=-0.08)

    ax_s.fill_between(sweep_sigmas, np.clip(sweep_dist_lo, 1e-6, None), sweep_dist_hi,
                      color=line_color, alpha=0.2)
    ax_s.plot(sweep_sigmas, sweep_dist_mean, "o-", color=line_color, markersize=5)
    ax_s.axhline(Qt_norm, ls="--", color="gray",
                 label=r"$\|\Lambda^{(t)}\|_F$ (zero-baseline)")
    sw_idx = int(np.argmin(np.abs(sweep_sigmas - HIGHLIGHT_SIGMA)))
    ax_s.scatter([sweep_sigmas[sw_idx]], [sweep_dist_mean[sw_idx]],
                 s=120, color=marker_color, zorder=5,
                 edgecolor="white", linewidth=1.6,
                 label=rf"$\sigma={int(HIGHLIGHT_SIGMA)}$ (panels A--E)")
    ax_s.set_xlabel(r"observation noise $\sigma$")
    ax_s.set_ylabel(r"$\|\hat{\Lambda}_{m=100}^{(t)} - \Lambda^{(t)}\|_F$")
    ax_s.set_title(r"Distance vs $\sigma$ (at $m=100$)", fontsize=11)
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(frameon=False, fontsize=10, loc="upper right")
    _add_panel_label(ax_s, "F", x=-0.08)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
