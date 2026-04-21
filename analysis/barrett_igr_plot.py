#!/usr/bin/env python3
"""Figure generator for the Barrett et al. 2022 IGR empirical reproduction.

Reads sweep output produced by scripts/run_barrett_igr_sweep.py and emits
figures/barrett_igr_reproduction.pdf.

Panels:
  (a) Synthetic flow-ref: lambda_hat vs eta, points colored by p, theory
      line lambda = eta * p / 4 overlaid.
  (b) Synthetic flow-ref: recovery ratio lambda_hat / (eta * p / 4) vs
      trajectory step, curves per eta (pooled over p and seeds).
  (c) MNIST flow-ref: recovery ratio vs eta, colored by architecture.
      Shows clean recovery for linear/tanh, breakdown for ReLU.
  (d) Synthetic SGD: lambda_hat/p vs eta, by batch size. Contrast with flow_ref.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=REPO_ROOT / "results" / "barrett_igr_sweep.pt",
    )
    parser.add_argument(
        "--mnist",
        type=Path,
        default=REPO_ROOT / "results" / "barrett_igr_mnist_sweep.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "figures" / "barrett_igr_reproduction.pdf",
    )
    return parser.parse_args()


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = torch.load(path, weights_only=False)
    return data.get("runs", [])


def extract_pooled(runs: list[dict], mode: str, dataset: str | None = None) -> list[dict]:
    rows = []
    for r in runs:
        ax = r["summary"]["sweep_axes"]
        if ax.get("mode") != mode:
            continue
        if dataset is not None and ax.get("dataset", "synthetic") != dataset:
            continue
        rows.append(
            {
                **ax,
                "lambda_hat": r["summary"]["pooled"]["lambda_hat"],
                "lambda_hat_per_param": r["summary"]["pooled"]["lambda_hat_per_param"],
                "theory": r["summary"]["theoretical_lambda"],
                "residual_ratio": r["summary"]["pooled"]["residual_ratio"],
                "num_params": r["summary"]["num_params"],
                "per_step": r["summary"]["per_step"],
            }
        )
    return rows


def plot_panel_a(ax, rows_flow_synth):
    """lambda_hat vs eta, colored by p, theory line overlaid."""
    p_values = sorted({r["p"] for r in rows_flow_synth})
    cmap = plt.get_cmap("viridis")
    colors = {p: cmap(i / max(1, len(p_values) - 1)) for i, p in enumerate(p_values)}

    for p in p_values:
        subset = [r for r in rows_flow_synth if r["p"] == p]
        etas = np.array([r["eta"] for r in subset])
        lams = np.array([r["lambda_hat"] for r in subset])
        ax.scatter(etas, lams, color=colors[p], label=f"$p={p}$", s=40, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Theory line: lambda = eta * p / 4, for each p.
    eta_grid = np.geomspace(min(r["eta"] for r in rows_flow_synth), max(r["eta"] for r in rows_flow_synth), 50)
    for p in p_values:
        ax.plot(eta_grid, eta_grid * p / 4.0, color=colors[p], linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\hat{\lambda}$")
    ax.set_title(r"(a) Synthetic OLS: $\hat{\lambda}$ vs $\eta$" + "\n" + r"dashed: theory $\lambda=\eta p/4$")
    ax.legend(loc="lower right", fontsize=9, frameon=False)


def plot_panel_b(ax, runs):
    """Recovery ratio vs trajectory step, curves per eta (synth flow-ref)."""
    # Aggregate per-step lambda_hat across (p, seed) for each eta.
    by_eta: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    theory_by_eta_p: dict[tuple[float, int], float] = {}
    for r in runs:
        ax_ = r["summary"]["sweep_axes"]
        if ax_.get("mode") != "flow_ref" or ax_.get("dataset", "synthetic") != "synthetic":
            continue
        eta = ax_["eta"]
        p = ax_["p"]
        theory = eta * p / 4.0
        theory_by_eta_p[(eta, p)] = theory
        for row in r["summary"]["per_step"]:
            ratio = row["lambda_hat"] / theory if theory else float("nan")
            by_eta[eta][row["step"]].append(ratio)

    etas = sorted(by_eta.keys())
    cmap = plt.get_cmap("plasma")
    for i, eta in enumerate(etas):
        steps = sorted(by_eta[eta].keys())
        means = [np.mean(by_eta[eta][s]) for s in steps]
        color = cmap(i / max(1, len(etas) - 1))
        ax.plot(steps, means, "o-", color=color, label=rf"$\eta={eta:.0e}$", markersize=4, linewidth=1.5)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("trajectory step $t$")
    ax.set_ylabel(r"$\hat{\lambda}_t / (\eta p / 4)$")
    ax.set_title(r"(b) Recovery ratio along trajectory" + "\n" + "averaged over $p$, seeds")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_ylim(0.0, 1.1)


def plot_panel_c(ax, rows_mnist):
    """Recovery ratio vs eta for MNIST flow-ref, by architecture."""
    if not rows_mnist:
        ax.text(0.5, 0.5, "(MNIST sweep not available)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) MNIST flow-ref: recovery ratio by architecture")
        return

    archs = sorted({r["arch"] for r in rows_mnist})
    cmap = plt.get_cmap("tab10")
    for i, arch in enumerate(archs):
        subset = [r for r in rows_mnist if r["arch"] == arch]
        etas = np.array([r["eta"] for r in subset])
        ratios = np.array([r["lambda_hat"] / r["theory"] for r in subset])
        order = np.argsort(etas)
        ax.plot(etas[order], ratios[order], "o-", color=cmap(i), label=arch, markersize=6, linewidth=1.5, alpha=0.9)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\hat{\lambda} / (\eta p / 4)$")
    ax.set_title("(c) MNIST flow-ref:\nrecovery ratio by architecture")
    ax.legend(loc="best", fontsize=9, frameon=False)


def plot_panel_d(ax, rows_sgd):
    """SGD: lambda_hat / p vs eta, per batch size, compared to eta/4 line."""
    if not rows_sgd:
        ax.text(0.5, 0.5, "(SGD runs not available)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(d) SGD: $\\hat{\\lambda}/p$ vs $\\eta$")
        return

    bs_values = sorted({r["batch_size"] for r in rows_sgd})
    cmap = plt.get_cmap("cividis")
    for i, bs in enumerate(bs_values):
        subset = [r for r in rows_sgd if r["batch_size"] == bs]
        by_eta: dict[float, list[float]] = defaultdict(list)
        for r in subset:
            by_eta[r["eta"]].append(r["lambda_hat_per_param"])
        etas = sorted(by_eta.keys())
        means = [np.mean(by_eta[e]) for e in etas]
        sds = [np.std(by_eta[e]) for e in etas]
        color = cmap(i / max(1, len(bs_values) - 1))
        ax.errorbar(etas, means, yerr=sds, fmt="o-", color=color, label=f"B={bs}", capsize=3, markersize=5)

    # Theory reference: lambda_hat_per_param = eta / 4.
    eta_grid = np.geomspace(
        min(r["eta"] for r in rows_sgd), max(r["eta"] for r in rows_sgd), 50
    )
    ax.plot(eta_grid, eta_grid / 4.0, "--", color="black", linewidth=1.0, alpha=0.5, label=r"theory $\eta/4$")

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\hat{\lambda} / p$")
    ax.set_title("(d) Synthetic SGD:\ntarget dominated by batch noise")
    ax.legend(loc="best", fontsize=8, frameon=False)


def main() -> None:
    args = parse_args()
    plt.style.use(str(REPO_ROOT / "clean_fig.mplstyle"))

    synthetic_runs = load_runs(args.synthetic)
    mnist_runs = load_runs(args.mnist)

    rows_flow_synth = extract_pooled(synthetic_runs, "flow_ref", dataset="synthetic")
    rows_sgd = extract_pooled(synthetic_runs, "sgd", dataset="synthetic")
    rows_mnist = extract_pooled(mnist_runs, "flow_ref", dataset="mnist")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_panel_a(axes[0, 0], rows_flow_synth)
    plot_panel_b(axes[0, 1], synthetic_runs)
    plot_panel_c(axes[1, 0], rows_mnist)
    plot_panel_d(axes[1, 1], rows_sgd)

    fig.suptitle("Empirical reproduction of Barrett & Dherin (2022) IGR: $\\lambda = \\eta p / 4$", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
