#!/usr/bin/env python3
"""Produces Barrett-style Figure 2: R_IG vs estimated lambda, and test accuracy
vs estimated lambda.

Reads results/barrett_igr_figure2.pt (from experiments/barrett_igr_figure2.py)
and writes figures/barrett_igr_figure2.pdf with two panels, colored by number
of parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmap import Colormap


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FIGURES_DIR = REPO_ROOT / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "results" / "barrett_igr_figure2.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_FIGURES_DIR / "barrett_igr_figure2.pdf",
    )
    parser.add_argument(
        "--x-quantity",
        choices=("lambda_hat", "lambda_theoretical"),
        default="lambda_hat",
        help="Which lambda to put on the x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.style.use(str(REPO_ROOT / "clean_fig.mplstyle"))

    payload = torch.load(args.results, weights_only=False)
    results = payload["results"]
    if not results:
        raise SystemExit("No results in file.")

    xs = np.array([r[args.x_quantity] for r in results], dtype=float)
    r_ig = np.array([r["r_ig"] for r in results], dtype=float)
    test_acc = np.array([r["best_test_acc"] for r in results], dtype=float)
    num_params = np.array([r["num_params"] for r in results], dtype=int)
    widths = np.array([r["width"] for r in results], dtype=int)
    etas = np.array([r["eta"] for r in results], dtype=float)
    residuals = np.array([r["residual_ratio"] for r in results], dtype=float)

    unique_params = sorted(np.unique(num_params).tolist())
    cmap = Colormap("crameri:batlow").to_mpl()
    param_to_color = {
        m: cmap(i / max(1, len(unique_params) - 1))
        for i, m in enumerate(unique_params)
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): R_IG vs lambda
    for m in unique_params:
        mask = num_params == m
        order = np.argsort(xs[mask])
        axes[0].plot(
            xs[mask][order],
            r_ig[mask][order],
            marker="o",
            linestyle="--",
            color=param_to_color[m],
            label=f"{m}",
            markersize=6,
            alpha=0.85,
            linewidth=1.0,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    xlabel = r"$\hat{\lambda}$" if args.x_quantity == "lambda_hat" else r"Theoretical $\lambda = \eta m/4$"
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(r"$R_{IG} = \frac{1}{p}\|\nabla E(\theta)\|^2$")
    # axes[0].set_title("(a) Regularization vs $\\hat{\\lambda}$")
    axes[0].legend(title="# params", loc="lower left", fontsize=8, frameon=False)
    axes[0].grid(True, which="both", alpha=0.25)

    # Panel (b): test accuracy vs lambda
    for m in unique_params:
        mask = num_params == m
        order = np.argsort(xs[mask])
        axes[1].plot(
            xs[mask][order],
            test_acc[mask][order] * 100,
            marker="o",
            linestyle="--",
            color=param_to_color[m],
            label=f"{m}",
            markersize=6,
            alpha=0.85,
            linewidth=1.0,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Test Accuracy (%)")
    # axes[1].set_title("Test accuracy vs $\\hat{\\lambda}$")
    axes[1].legend(title="# params", loc="lower right", fontsize=8, frameon=False)
    axes[1].grid(True, which="both", alpha=0.25)

    config = payload.get("config", {})
    activation = config.get("activation", "?")
    train_samples = config.get("train_samples", "?")
    # fig.suptitle(
    #     f"Barrett & Dherin (2022) Figure 2 reproduction · {activation} MLP · "
    #     f"MNIST (n_train={train_samples})",
    #     fontsize=12,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved {args.out}")
    print(
        f"Data: n_runs={len(results)}, lambda_hat range=[{xs.min():.3g}, {xs.max():.3g}], "
        f"R_IG range=[{r_ig.min():.3g}, {r_ig.max():.3g}], "
        f"test_acc range=[{test_acc.min():.3f}, {test_acc.max():.3f}], "
        f"median residual_ratio={np.median(residuals):.3f}"
    )


if __name__ == "__main__":
    main()
