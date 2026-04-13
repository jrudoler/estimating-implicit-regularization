#!/usr/bin/env python3
"""
Build 2x2 heatmaps: mean and standard error of log(λ̂/λ) for L1 and L2 over replicates.

Reads a CSV (from W&B export or scripts/run_elasticnet_figure_batch.py) with columns including
true_l1, true_l2, smooth, seed, recovery/log_mult_l1, recovery/log_mult_l2.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        ):
            if k in s:
                row[k] = str(s[k])
        rows.append(row)
    return rows


def _aggregate(
    rows: list[dict[str, str]], smooth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
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

    buckets: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
    for r in sub:
        a, b = float(r["true_l1"]), float(r["true_l2"])
        buckets[(a, b)].append(
            (float(r["recovery/log_mult_l1"]), float(r["recovery/log_mult_l2"]))
        )

    mean_l1_mat = np.full((len(idx), len(cols)), np.nan)
    se_l1_mat = np.full((len(idx), len(cols)), np.nan)
    mean_l2_mat = np.full((len(idx), len(cols)), np.nan)
    se_l2_mat = np.full((len(idx), len(cols)), np.nan)

    for (a, b), pairs in buckets.items():
        i, j = li[a], lj[b]
        v1 = np.array([p[0] for p in pairs], dtype=float)
        v2 = np.array([p[1] for p in pairs], dtype=float)
        mean_l1_mat[i, j] = float(np.mean(v1))
        mean_l2_mat[i, j] = float(np.mean(v2))
        if len(v1) > 1:
            se_l1_mat[i, j] = float(np.std(v1, ddof=1) / np.sqrt(len(v1)))
            se_l2_mat[i, j] = float(np.std(v2, ddof=1) / np.sqrt(len(v2)))
        else:
            se_l1_mat[i, j] = 0.0
            se_l2_mat[i, j] = 0.0

    return mean_l1_mat, se_l1_mat, mean_l2_mat, se_l2_mat, idx, cols


def plot_four_panels(
    mean_l1: np.ndarray,
    se_l1: np.ndarray,
    mean_l2: np.ndarray,
    se_l2: np.ndarray,
    idx: list[float],
    cols: list[float],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    vmin = float(np.nanmin([mean_l1, mean_l2]))
    vmax = float(np.nanmax([mean_l1, mean_l2]))
    smin = max(float(np.nanmin([se_l1, se_l2])), 0.0)
    smax = max(float(np.nanmax([se_l1, se_l2])), 1e-9)

    xlabels = [f"{c:g}" for c in cols]
    ylabels = [f"{r:g}" for r in idx]

    for ax, data, ttl, cmap, v0, v1 in [
        (axes[0, 0], mean_l1, r"Mean $\log(\hat\lambda_1/\lambda_1)$", "viridis", vmin, vmax),
        (axes[0, 1], mean_l2, r"Mean $\log(\hat\lambda_2/\lambda_2)$", "viridis", vmin, vmax),
        (axes[1, 0], se_l1, r"SE of $\log(\hat\lambda_1/\lambda_1)$", "magma", smin, smax),
        (axes[1, 1], se_l2, r"SE of $\log(\hat\lambda_2/\lambda_2)$", "magma", smin, smax),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=v0, vmax=v1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel(r"true $\lambda_2$")
        ax.set_ylabel(r"true $\lambda_1$")
        ax.set_title(ttl + title_suffix)
        plt.colorbar(im, ax=ax, fraction=0.046)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
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
        default=Path("figures/elasticnet_recovery_mean_se.pdf"),
    )
    args = p.parse_args()

    if args.csv:
        rows = _load_csv(args.csv)
    elif args.sweep_id:
        rows = _load_from_wandb(args.sweep_id, args.entity_project)
    else:
        raise SystemExit("Provide --csv or --sweep-id")

    mean_l1, se_l1, mean_l2, se_l2, idx, cols = _aggregate(rows, args.smooth)
    plot_four_panels(mean_l1, se_l1, mean_l2, se_l2, idx, cols, args.output)


if __name__ == "__main__":
    main()
