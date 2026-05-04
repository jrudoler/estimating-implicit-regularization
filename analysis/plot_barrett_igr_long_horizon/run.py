#!/usr/bin/env python3
"""Plots long-horizon 3-trajectory drift from experiments/barrett_igr_long_horizon.py.

Two panels:
  (a) L2 drift vs time: ||theta_GD - theta_orig_flow|| and ||theta_GD - theta_mod_flow||.
      Barrett predicts the modified-flow drift stays small while the original-flow
      drift grows ~linearly (per-step O(eta^2) accumulates over ~t/eta steps).
  (b) Drift RATIO: ||GD - mod|| / ||GD - orig||, should stay well below 1 throughout.

Accepts multiple input files to overlay different settings (eta, dataset, etc.).
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
        nargs="+",
        required=True,
        help="One or more .pt files produced by experiments/barrett_igr_long_horizon.py.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels per input file (must match length).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_FIGURES_DIR / "barrett_igr_long_horizon.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.style.use(str(REPO_ROOT / "clean_fig.mplstyle"))

    if args.labels is not None and len(args.labels) != len(args.results):
        raise SystemExit("--labels must match the number of --results files")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    devon = Colormap("crameri:devon").to_mpl()
    n_runs = len(args.results)
    # Sample devon at evenly-spaced positions, avoiding the near-white extreme.
    sample_positions = (
        [0.5] if n_runs == 1
        else [0.15 + 0.6 * i / (n_runs - 1) for i in range(n_runs)]
    )

    for i, path in enumerate(args.results):
        payload = torch.load(path, weights_only=False)
        drifts = payload["drifts"]
        config = payload.get("config", {})
        eta = config.get("eta", payload.get("eta", "?"))
        ds = config.get("dataset", "?")
        act = config.get("activation", "")
        label = (
            args.labels[i]
            if args.labels
            else f"{ds} {'('+act+')' if act and ds == 'mnist' else ''} eta={eta}"
        )

        t = np.array([d["time"] for d in drifts])
        d_orig = np.array([d["dist_gd_to_orig_flow"] for d in drifts])
        d_mod = np.array([d["dist_gd_to_mod_flow"] for d in drifts])

        color = devon(sample_positions[i])
        axes[0].plot(t, d_orig, "-", color=color, label=f"{label}: GD vs. original flow", linewidth=1.5, alpha=0.9)
        axes[0].plot(t, d_mod, "--", color=color, label=f"{label}: GD vs. modified flow", linewidth=1.5, alpha=0.9)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d_orig > 0, d_mod / d_orig, np.nan)
        axes[1].plot(t, ratio, "-", color=color, label=label, linewidth=1.5, alpha=0.9)

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"time $t = k \cdot \eta$")
    axes[0].set_ylabel(r"$\| \theta_{GD}(t) - \theta_{\cdot}(t) \|_2$")
    axes[0].set_title("(a) Drift between GD and ODE flows\nsolid: original loss; dashed: modified loss")
    axes[0].legend(loc="best", fontsize=7, frameon=False)
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"time $t = k \cdot \eta$")
    axes[1].set_ylabel(r"$\| GD-mod \|\;/\;\| GD-orig \|$")
    axes[1].set_title("(b) Drift ratio: modified-flow tracks GD\n better than original flow by this factor")
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    axes[1].grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Non-tautological test of Barrett & Dherin (2022) Thm 3.1:\n"
        "GD iterates track the modified-loss flow, not the original-loss flow",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {args.out}")
    for i, path in enumerate(args.results):
        payload = torch.load(path, weights_only=False)
        d = payload["drifts"][-1]
        print(
            f"  {path.name}: final drift_orig={d['dist_gd_to_orig_flow']:.3e}, "
            f"drift_mod={d['dist_gd_to_mod_flow']:.3e}, "
            f"ratio={d['dist_gd_to_mod_flow']/max(d['dist_gd_to_orig_flow'], 1e-30):.3g}"
        )


if __name__ == "__main__":
    main()
