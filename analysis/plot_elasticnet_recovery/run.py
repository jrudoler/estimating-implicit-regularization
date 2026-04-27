#!/usr/bin/env python3
"""
Two-panel line plot for elastic-net lambda recovery.

Left panel:  true λ₁ (x) vs estimated λ₁ (y), one line per true λ₂.
Right panel: true λ₂ (x) vs estimated λ₂ (y), one line per true λ₁.
Error bars: ±1 SE across seeds. Dashed diagonal = perfect recovery.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.wandb_utils import get_sweep_runs, wandb_summary_df

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FIGURES_DIR = REPO_ROOT / "results" / "figures"


def _fmt_lambda(v: float) -> str:
    """Format a lambda value as a LaTeX power-of-ten string."""
    exp = int(round(math.log10(v)))
    return rf"$10^{{{exp}}}$"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_parquet(path: Path) -> list[dict[str, object]]:
    df = pd.read_parquet(path)
    return _dataframe_rows(df)


def _load_from_wandb(
    sweep_id: str,
    entity_project: str | None,
) -> list[dict[str, object]]:
    runs = get_sweep_runs(
        sweep_id=sweep_id,
        entity_project=entity_project,
        state="finished",
    )
    return _dataframe_rows(wandb_summary_df(runs, include_system_metrics=False))


def _dataframe_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in df.to_dict("records"):
        rows.append({k: v for k, v in row.items() if not _is_missing(v)})
    return rows


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _parse_records(rows: list[dict[str, object]], smooth: float) -> list[dict[str, float]]:
    """Return one tidy record per run with true and estimated lambda values."""
    sub = [
        r
        for r in rows
        if r.get("smooth") is not None
        and np.isclose(float(r["smooth"]), smooth, rtol=0, atol=0)
    ]
    if not sub:
        raise ValueError(f"No rows with smooth={smooth}")

    records: list[dict[str, float]] = []
    for r in sub:
        true_l1_raw = r.get("true_l1", r.get("l1"))
        true_l2_raw = r.get("true_l2", r.get("l2"))
        if true_l1_raw is None or true_l2_raw is None:
            continue
        true_l1 = float(true_l1_raw)
        true_l2 = float(true_l2_raw)

        if "recovery/lambda_1_hat" in r:
            hat_l1 = float(r["recovery/lambda_1_hat"])
        elif "recovery/log_mult_l1" in r:
            hat_l1 = true_l1 * math.exp(float(r["recovery/log_mult_l1"]))
        else:
            continue

        if "recovery/lambda_2_hat" in r:
            hat_l2 = float(r["recovery/lambda_2_hat"])
        elif "recovery/log_mult_l2" in r:
            hat_l2 = true_l2 * math.exp(float(r["recovery/log_mult_l2"]))
        else:
            continue

        records.append(
            dict(true_l1=true_l1, true_l2=true_l2, hat_l1=hat_l1, hat_l2=hat_l2)
        )
    return records


def _plot_panel(
    ax: plt.Axes,
    records: list[dict[str, float]],
    x_key: str,
    hat_key: str,
    other_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    other_label: str,
) -> None:
    x_vals = sorted(set(r[x_key] for r in records))
    other_vals = sorted(set(r[other_key] for r in records))
    n = len(other_vals)
    from cmap import Colormap
    cmap = Colormap("crameri:imola").to_mpl()
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # Perfect-recovery diagonal
    lo, hi = min(x_vals), max(x_vals)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, zorder=0)

    for ci, ov in enumerate(other_vals):
        xs, means, ses = [], [], []
        for xv in x_vals:
            vals = [
                r[hat_key]
                for r in records
                if np.isclose(r[x_key], xv) and np.isclose(r[other_key], ov)
            ]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            xs.append(xv)
            means.append(float(np.mean(arr)))
            ses.append(
                float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            )

        ax.errorbar(
            xs,
            means,
            yerr=ses,
            marker="o",
            markersize=4,
            capsize=3,
            capthick=0.8,
            lw=1.5,
            color=colors[ci],
            label=_fmt_lambda(ov),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        title=other_label,
        fontsize=8,
        title_fontsize=9,
        loc="lower right",
        framealpha=0.7,
    )


def plot_lineplot(
    records: list[dict[str, float]],
    out_path: Path,
) -> None:
    style_path = REPO_ROOT / "clean_fig.mplstyle"
    plt.style.use(str(style_path))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    _plot_panel(
        axes[0],
        records,
        x_key="true_l1",
        hat_key="hat_l1",
        other_key="true_l2",
        xlabel=r"True $\lambda_1$",
        ylabel=r"Estimated $\hat\lambda_1$",
        title=r"$\lambda_1$ recovery",
        other_label=r"True $\lambda_2$",
    )
    _plot_panel(
        axes[1],
        records,
        x_key="true_l2",
        hat_key="hat_l2",
        other_key="true_l1",
        xlabel=r"True $\lambda_2$",
        ylabel=r"Estimated $\hat\lambda_2$",
        title=r"$\lambda_2$ recovery",
        other_label=r"True $\lambda_1$",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-parquet",
        type=Path,
        help="Local W&B sweep snapshot produced by pull_wandb_sweep.",
    )
    p.add_argument("--csv", type=Path, help="CSV from W&B export or local batch script")
    p.add_argument(
        "--sweep-id",
        type=str,
        help="Manual fallback: live W&B sweep id (use with --entity-project).",
    )
    p.add_argument(
        "--entity-project",
        type=str,
        help=(
            "Manual fallback: W&B '<entity>/<project>'. If omitted, uses "
            "WANDB_ENTITY_PROJECT or WANDB_ENTITY plus WANDB_PROJECT."
        ),
    )
    p.add_argument("--smooth", type=float, default=1e-3, help="Smoothing beta (paper default 1e-3)")
    p.add_argument(
        "--output",
        type=Path,
        default=RESULTS_FIGURES_DIR / "elasticnet_recovery_mean_se.pdf",
    )
    args = p.parse_args()

    if args.runs_parquet:
        rows = _load_parquet(args.runs_parquet)
    elif args.csv:
        rows = _load_csv(args.csv)
    elif args.sweep_id:
        rows = _load_from_wandb(args.sweep_id, args.entity_project)
    else:
        raise SystemExit("Provide --runs-parquet, --csv, or --sweep-id")

    records = _parse_records(rows, args.smooth)
    plot_lineplot(records, args.output)


if __name__ == "__main__":
    main()
