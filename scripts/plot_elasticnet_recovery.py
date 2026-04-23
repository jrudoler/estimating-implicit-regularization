#!/usr/bin/env python3
"""
Build 2x2 heatmaps: mean and standard error of log(λ̂/λ) for L1 and L2 over replicates.

Reads a CSV (from W&B export or scripts/run_elasticnet_figure_batch.py) with columns including
true_l1, true_l2, smooth, seed, recovery/log_mult_l1, recovery/log_mult_l2.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_LN10 = math.log(10.0)
REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_from_wandb(sweep_id: str, entity_project: str) -> list[dict[str, str]]:
    import wandb

    api = wandb.Api()
    sweep = api.sweep(f"{entity_project}/{sweep_id}")
    rows: list[dict[str, str]] = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        row: dict[str, str] = {}
        cfg = dict(run.config)
        if "l1" in cfg:
            row["true_l1"] = str(cfg["l1"])
        if "l2" in cfg:
            row["true_l2"] = str(cfg["l2"])
        if "smooth" in cfg:
            row["smooth"] = str(cfg["smooth"])
        if "seed" in cfg:
            row["seed"] = str(cfg["seed"])
        s = run.summary._json_dict
        for k in (
            "recovery/log_mult_l1",
            "recovery/log_mult_l2",
            "recovery/lambda_1_hat",
            "recovery/lambda_2_hat",
            "bias/theta_1",
            "bias/theta_2",
        ):
            if k in s:
                row[k] = str(s[k])
        # Backward compatibility for older sweeps that only logged theta_1/theta_2.
        if "recovery/log_mult_l1" not in row and "bias/theta_1" in row and "true_l1" in row:
            l1_hat = float(np.exp(float(row["bias/theta_1"])))
            row["recovery/log_mult_l1"] = str(np.log(l1_hat / float(row["true_l1"])))
        if "recovery/log_mult_l2" not in row and "bias/theta_2" in row and "true_l2" in row:
            l2_hat = float(np.exp(float(row["bias/theta_2"])))
            row["recovery/log_mult_l2"] = str(np.log(l2_hat / float(row["true_l2"])))
        rows.append(row)
    return rows


def _aggregate(
    rows: list[dict[str, str]], smooth: float
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[float],
    list[float],
]:
    sub = [
        r
        for r in rows
        if r.get("smooth") is not None and np.isclose(float(r["smooth"]), smooth, rtol=0, atol=0)
    ]
    if not sub:
        raise ValueError(f"No rows with smooth={smooth}")

    keys = set()
    for r in sub:
        keys.add((float(r["true_l1"]), float(r["true_l2"])))

    idx = sorted({a for a, _ in keys})
    cols = sorted({b for _, b in keys})
    li = {v: i for i, v in enumerate(idx)}
    lj = {v: j for j, v in enumerate(cols)}

    buckets: dict[tuple[float, float], list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )
    for r in sub:
        a, b = float(r["true_l1"]), float(r["true_l2"])
        ln1 = float(r["recovery/log_mult_l1"])
        ln2 = float(r["recovery/log_mult_l2"])
        h1 = a * math.exp(ln1)
        h2 = b * math.exp(ln2)
        buckets[(a, b)].append((ln1, ln2, h1, h2))

    mean_l1_mat = np.full((len(idx), len(cols)), np.nan)
    se_l1_mat = np.full((len(idx), len(cols)), np.nan)
    mean_l2_mat = np.full((len(idx), len(cols)), np.nan)
    se_l2_mat = np.full((len(idx), len(cols)), np.nan)
    mean_hat_l1 = np.full((len(idx), len(cols)), np.nan)
    mean_hat_l2 = np.full((len(idx), len(cols)), np.nan)

    for (a, b), pairs in buckets.items():
        i, j = li[a], lj[b]
        v1_ln = np.array([p[0] for p in pairs], dtype=float)
        v2_ln = np.array([p[1] for p in pairs], dtype=float)
        v1_l10 = v1_ln / _LN10
        v2_l10 = v2_ln / _LN10
        mean_l1_mat[i, j] = float(np.mean(v1_l10))
        mean_l2_mat[i, j] = float(np.mean(v2_l10))
        mean_hat_l1[i, j] = float(np.mean([p[2] for p in pairs]))
        mean_hat_l2[i, j] = float(np.mean([p[3] for p in pairs]))
        if len(v1_ln) > 1:
            se_l1_mat[i, j] = float(np.std(v1_l10, ddof=1) / np.sqrt(len(v1_l10)))
            se_l2_mat[i, j] = float(np.std(v2_l10, ddof=1) / np.sqrt(len(v2_l10)))
        else:
            se_l1_mat[i, j] = 0.0
            se_l2_mat[i, j] = 0.0

    return (
        mean_l1_mat,
        se_l1_mat,
        mean_l2_mat,
        se_l2_mat,
        mean_hat_l1,
        mean_hat_l2,
        idx,
        cols,
    )


def _annotate_mean_panel(
    ax: plt.Axes, log10_mat: np.ndarray, mean_hat: np.ndarray
) -> None:
    n_rows, n_cols = log10_mat.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = log10_mat[i, j]
            est = mean_hat[i, j]
            if not np.isfinite(val) or not np.isfinite(est):
                continue
            ax.text(
                j + 0.5,
                i + 0.35,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",
            )
            ax.text(
                j + 0.5,
                i + 0.70,
                f"({est:.2g})",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )


def _annotate_se_panel(ax: plt.Axes, mat: np.ndarray) -> None:
    n_rows, n_cols = mat.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )


def plot_four_panels(
    mean_l1: np.ndarray,
    se_l1: np.ndarray,
    mean_l2: np.ndarray,
    se_l2: np.ndarray,
    mean_hat_l1: np.ndarray,
    mean_hat_l2: np.ndarray,
    idx: list[float],
    cols: list[float],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    style_path = Path(__file__).resolve().parents[1] / "clean_fig.mplstyle"
    plt.style.use(str(style_path))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=False)

    # Rows: smallest true λ₁ at the bottom, largest at the top, so moving up and right
    # increases both λ₁ and λ₂ (columns are already smallest → largest left to right).
    mean_l1_d = np.flipud(mean_l1)
    mean_l2_d = np.flipud(mean_l2)
    se_l1_d = np.flipud(se_l1)
    se_l2_d = np.flipud(se_l2)
    mean_hat_l1_d = np.flipud(mean_hat_l1)
    mean_hat_l2_d = np.flipud(mean_hat_l2)

    abs_max = float(
        np.nanmax(
            np.abs(np.r_[mean_l1_d.ravel(), mean_l2_d.ravel()].astype(float))
        )
    )
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 1.0
    mmin, mmax = -abs_max, abs_max

    smin = max(float(np.nanmin([se_l1_d, se_l2_d])), 0.0)
    smax = max(float(np.nanmax([se_l1_d, se_l2_d])), 1e-9)

    xlabels = [f"{c:g}" for c in cols]
    ylabels = [f"{r:g}" for r in reversed(idx)]

    x_lab = r"True $\lambda_2$"
    y_lab = r"True $\lambda_1$"

    # Top row: mean log10 error + dual annotations (notebook style).
    for ax, data, hat, ttl in [
        (
            axes[0, 0],
            mean_l1_d,
            mean_hat_l1_d,
            r"Mean $\log_{10}(\hat\lambda_1/\lambda_1)$",
        ),
        (
            axes[0, 1],
            mean_l2_d,
            mean_hat_l2_d,
            r"Mean $\log_{10}(\hat\lambda_2/\lambda_2)$",
        ),
    ]:
        hm = sns.heatmap(
            data,
            ax=ax,
            cmap="bwr",
            vmin=mmin,
            vmax=mmax,
            center=0.0,
            cbar=True,
            cbar_kws={"shrink": 0.82, "label": r"$\log_{10}(\hat\lambda/\lambda)$"},
            xticklabels=xlabels,
            yticklabels=ylabels,
            linewidths=0.5,
            linecolor="white",
            annot=False,
        )
        _annotate_mean_panel(ax, data, hat)
        hm.collections[0].colorbar.ax.tick_params(labelsize=10)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_title(ttl + title_suffix)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    # Bottom row: SE of log10(mean is per-replicate log10 error).
    for ax, data, ttl in [
        (
            axes[1, 0],
            se_l1_d,
            r"SE of $\log_{10}(\hat\lambda_1/\lambda_1)$",
        ),
        (
            axes[1, 1],
            se_l2_d,
            r"SE of $\log_{10}(\hat\lambda_2/\lambda_2)$",
        ),
    ]:
        hm = sns.heatmap(
            data,
            ax=ax,
            cmap="viridis",
            vmin=smin,
            vmax=smax,
            cbar=True,
            cbar_kws={"shrink": 0.82},
            xticklabels=xlabels,
            yticklabels=ylabels,
            linewidths=0.5,
            linecolor="white",
            annot=False,
        )
        _annotate_se_panel(ax, data)
        hm.collections[0].colorbar.ax.tick_params(labelsize=10)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_title(ttl + title_suffix)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.12,
        top=0.88,
        wspace=0.38,
        hspace=0.45,
    )
    # Column headers: λ₁ recovery (left), λ₂ recovery (right), above both rows.
    for j, col_title in enumerate(
        (r"$\lambda_1$ recovery", r"$\lambda_2$ recovery")
    ):
        ax_top = axes[0, j]
        pos = ax_top.get_position()
        x_c = 0.5 * (pos.x0 + pos.x1)
        y = min(pos.y1 + 0.05, 0.99)
        fig.text(
            x_c,
            y,
            col_title,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="semibold",
            transform=fig.transFigure,
        )
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, help="CSV from W&B export or local batch script")
    p.add_argument("--sweep-id", type=str, help="W&B sweep id (use with --entity-project)")
    p.add_argument(
        "--entity-project",
        type=str,
        default="jhrudoler-penn/inductive-bias",
        help="entity/project for W&B API",
    )
    p.add_argument("--smooth", type=float, default=1e-3, help="Smoothing beta (paper default 1e-3)")
    p.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURES_DIR / "elasticnet_recovery_mean_se.pdf",
    )
    args = p.parse_args()

    if args.csv:
        rows = _load_csv(args.csv)
    elif args.sweep_id:
        rows = _load_from_wandb(args.sweep_id, args.entity_project)
    else:
        raise SystemExit("Provide --csv or --sweep-id")

    mean_l1, se_l1, mean_l2, se_l2, mean_hat_l1, mean_hat_l2, idx, cols = _aggregate(
        rows, args.smooth
    )
    plot_four_panels(
        mean_l1,
        se_l1,
        mean_l2,
        se_l2,
        mean_hat_l1,
        mean_hat_l2,
        idx,
        cols,
        args.output,
    )


if __name__ == "__main__":
    main()
