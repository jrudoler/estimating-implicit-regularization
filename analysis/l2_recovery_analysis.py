#!/usr/bin/env python3
"""Analyze L2 recovery sweep results and generate diagnostic plots."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from math import ceil
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "results" / "data"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
LOG_DIR = REPO_ROOT / "logs" / "analysis"
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"

LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO") -> Path:
    """Configure root logging with console and file handlers."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"l2_recovery_analysis_{timestamp}.log"

    root = logging.getLogger()
    root.handlers.clear()
    level = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    LOGGER.debug("Logging configured. Log file: %s", log_path)
    return log_path


def apply_plot_style() -> None:
    """Apply the repository-wide matplotlib style, if available."""
    if STYLE_PATH.exists():
        plt.style.use(STYLE_PATH)
        LOGGER.debug("Applied matplotlib style from %s", STYLE_PATH)
    else:
        LOGGER.warning("Matplotlib style file not found at %s", STYLE_PATH)


def sanitize_value(value: Any) -> Any:
    """Convert complex config values to scalars or strings for DataFrame use."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return str(value)


def flatten_dict(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionaries using '/' as a separator."""
    items: dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        new_key = f"{prefix}{key}" if not prefix else f"{prefix}{key}"
        if isinstance(value, Mapping):
            items.update(flatten_dict(value, prefix=f"{new_key}/"))
        else:
            items[new_key] = sanitize_value(value)
    return items


def coalesce_columns(df: pd.DataFrame, target: str, candidates: Iterable[str]) -> pd.Series:
    """Pick the first non-null, numeric column among candidates and assign to target."""
    for column in candidates:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.notna().any():
                df[target] = numeric
                return df[target]
    df[target] = np.nan
    return df[target]


def fetch_sweep_runs(
    entity: str,
    project: str,
    sweep_id: str,
    max_runs: Optional[int] = None,
) -> tuple[pd.DataFrame, str]:
    """Download sweep runs from Weights & Biases and return a DataFrame."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    records: list[dict[str, Any]] = []
    for idx, run in enumerate(sweep.runs):
        if max_runs is not None and idx >= max_runs:
            break
        record: dict[str, Any] = {
            "run_id": run.id,
            "run_name": run.name,
            "run_state": run.state,
            "sweep_id": sweep_id,
            "sweep_name": sweep.name,
            "created_at": getattr(run, "created_at", None),
        }
        record.update(flatten_dict(run.config))
        record.update(flatten_dict(getattr(run.summary, "_json_dict", {})))
        records.append(record)
    if not records:
        return pd.DataFrame(), sweep.name or sweep_id
    df = pd.DataFrame.from_records(records)
    return df, sweep.name or sweep_id


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and derive common error metrics."""
    prepared = df.copy()

    coalesce_columns(
        prepared,
        target="true_l2",
        candidates=(
            "l2_true",
            "l2/true",
            "l2_lambda",
            "bias/lambda",
        ),
    )
    coalesce_columns(
        prepared,
        target="estimated_l2",
        candidates=(
            "l2_estimated",
            "l2/estimated",
            "bias/beta",
        ),
    )
    coalesce_columns(
        prepared,
        target="abs_error",
        candidates=("l2_abs_error", "l2/abs_error"),
    )
    coalesce_columns(
        prepared,
        target="rel_error",
        candidates=("l2_rel_error", "l2/rel_error"),
    )
    
    # Try to extract performance metrics
    for col in prepared.columns:
        if "test/acc" in col or "val/acc" in col:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    numeric_cols = [
        "true_l2",
        "estimated_l2",
        "abs_error",
        "rel_error",
        "depth",
        "width",
        "noise_std",
        "batch_size",
        "max_epochs_model",
        "max_epochs_bias",
        "lr_model",
        "lr_bias",
        "l2_lambda",
    ]
    for column in numeric_cols:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if "function_class" in prepared.columns:
        prepared["function_class"] = prepared["function_class"].astype("string")
    if "dataset" in prepared.columns:
        prepared["dataset"] = prepared["dataset"].astype("string")

    missing_true = prepared["true_l2"].isna()
    missing_est = prepared["estimated_l2"].isna()
    if missing_true.any() or missing_est.any():
        LOGGER.warning(
            "Dropping %d rows without true/estimated L2 values.",
            int((missing_true | missing_est).sum()),
        )
        prepared = prepared[~(missing_true | missing_est)].copy()

    if "abs_error" not in prepared.columns or prepared["abs_error"].isna().any():
        prepared["abs_error"] = (prepared["estimated_l2"] - prepared["true_l2"].astype(float)).abs()
    if "rel_error" not in prepared.columns or prepared["rel_error"].isna().any():
        prepared["rel_error"] = prepared["abs_error"] / prepared["true_l2"].clip(lower=1e-12)

    with np.errstate(divide="ignore"):
        prepared["log10_ratio"] = np.log10(
            prepared["estimated_l2"] / prepared["true_l2"].clip(lower=1e-12)
        )

    prepared.sort_values("abs_error", inplace=True)
    prepared.reset_index(drop=True, inplace=True)
    return prepared


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate errors by key hyperparameters."""
    group_cols = [col for col in ("dataset", "function_class", "depth", "width", "noise_std", "l2_lambda") if col in df.columns]
    if not group_cols:
        LOGGER.debug("No grouping columns available for summary table.")
        return pd.DataFrame()

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("run_id", "nunique"),
            mean_abs_error=("abs_error", "mean"),
            std_abs_error=("abs_error", "std"),
            median_rel_error=("rel_error", "median"),
            mean_rel_error=("rel_error", "mean"),
        )
        .reset_index()
    )
    return summary


def plot_estimated_vs_true(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot comparing estimated and true L2 values."""
    if df.empty:
        LOGGER.warning("Skipping scatter plot: no data available.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    # Prefer dataset over function_class for hue
    hue_col = "dataset" if "dataset" in df.columns else ("function_class" if "function_class" in df.columns else None)
    hue_order = sorted(df[hue_col].dropna().unique()) if hue_col else None
    style_order = sorted(df["depth"].dropna().unique()) if "depth" in df.columns else None

    sns.scatterplot(
        data=df,
        x="true_l2",
        y="estimated_l2",
        hue=hue_col,
        style="depth" if style_order else None,
        palette="tab10",
        ax=ax,
        s=70,
    )

    finite_true = df["true_l2"].replace(0.0, np.nan).dropna()
    finite_est = df["estimated_l2"].replace(0.0, np.nan).dropna()
    if not finite_true.empty and not finite_est.empty:
        min_val = float(min(finite_true.min(), finite_est.min()))
        max_val = float(max(finite_true.max(), finite_est.max()))
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Ideal")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(r"True $\lambda_2$")
    ax.set_ylabel(r"Estimated $\lambda_2$")
    ax.set_title("L2 Recovery: Estimated vs True")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved scatter plot to %s", output_path)


def plot_architecture_heatmaps(df: pd.DataFrame, output_path: Path, metric: str = "abs_error") -> None:
    """Heatmaps of the chosen metric across architecture settings per function class."""
    required_cols = {"depth", "width", metric}
    if not required_cols.issubset(df.columns):
        LOGGER.warning("Skipping heatmaps: missing columns %s", required_cols - set(df.columns))
        return

    subset = df.dropna(subset=list(required_cols)).copy()
    if subset.empty:
        LOGGER.warning("Skipping heatmaps: no rows left after dropping NaNs.")
        return

    # Prefer dataset over function_class for grouping
    group_col = "dataset" if "dataset" in subset.columns else ("function_class" if "function_class" in subset.columns else None)
    if group_col:
        classes = sorted(subset[group_col].dropna().unique())
    else:
        classes = ["All"]
        subset = subset.assign(group="All")
        group_col = "group"

    n_classes = len(classes)
    ncols = min(3, n_classes)
    nrows = ceil(n_classes / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5 * ncols, 4.8 * nrows), squeeze=False)
    for ax in axes.flat[n_classes:]:
        ax.remove()

    for idx, class_val in enumerate(classes):
        ax = axes.flat[idx]
        data = subset[subset[group_col] == class_val]
        pivot = data.pivot_table(index="depth", columns="width", values=metric, aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="magma_r", ax=ax, cbar=idx == n_classes - 1)
        ax.set_title(f"{class_val}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Depth")

    fig.suptitle(f"Mean {metric.replace('_', ' ').title()} by Architecture", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved architecture heatmaps to %s", output_path)


def plot_error_vs_noise(df: pd.DataFrame, output_path: Path) -> None:
    """Line plot of relative error against noise level."""
    if "noise_std" not in df.columns or "rel_error" not in df.columns:
        LOGGER.warning("Skipping error-vs-noise plot: required columns missing.")
        return

    subset = df.dropna(subset=["noise_std", "rel_error"]).copy()
    if subset.empty:
        LOGGER.warning("Skipping error-vs-noise plot: no data available after dropna.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.lineplot(
        data=subset,
        x="noise_std",
        y="rel_error",
        hue="function_class" if "function_class" in subset.columns else None,
        style="depth" if "depth" in subset.columns else None,
        marker="o",
        ax=ax,
        estimator="median",
        errorbar="sd",
    )
    ax.set_xlabel("Noise standard deviation")
    ax.set_ylabel("Relative error")
    ax.set_title("Relative L2 Error vs Noise")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved error-vs-noise plot to %s", output_path)


def plot_performance_vs_l2(df: pd.DataFrame, output_path: Path) -> None:
    """Plot validation/test performance as a function of L2 penalty."""
    # Try to find performance metrics
    perf_cols = [col for col in df.columns if any(x in col.lower() for x in ["val/acc", "test/acc", "val_acc", "test_acc", "accuracy"])]
    if not perf_cols or "true_l2" not in df.columns:
        LOGGER.warning("Skipping performance-vs-l2 plot: required columns missing.")
        return
    
    perf_col = perf_cols[0]  # Use first available performance column
    subset = df.dropna(subset=["true_l2", perf_col]).copy()
    if subset.empty:
        LOGGER.warning("Skipping performance-vs-l2 plot: no data available after dropna.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Plot 1: Performance vs L2 (log scale)
    ax = axes[0]
    hue_order = sorted(subset["dataset"].dropna().unique()) if "dataset" in subset.columns else None
    sns.lineplot(
        data=subset,
        x="true_l2",
        y=perf_col,
        hue="dataset" if hue_order else None,
        style="depth" if "depth" in subset.columns else None,
        marker="o",
        ax=ax,
        estimator="mean",
        errorbar="sd",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"L2 Penalty ($\lambda$)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance vs L2 Penalty")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    if hue_order:
        ax.legend(title="Dataset")
    
    # Plot 2: Recovery error vs L2 scale
    ax = axes[1]
    sns.scatterplot(
        data=subset,
        x="true_l2",
        y="rel_error",
        hue="dataset" if hue_order else None,
        style="depth" if "depth" in subset.columns else None,
        ax=ax,
        s=70,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"L2 Penalty ($\lambda$)")
    ax.set_ylabel("Relative Recovery Error")
    ax.set_title("L2 Recovery Error vs Penalty Scale")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    if hue_order:
        ax.legend(title="Dataset")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved performance-vs-l2 plot to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze L2 recovery sweep results")
    parser.add_argument("--sweep-id", type=str, help="W&B sweep ID (e.g., abc1234)")
    parser.add_argument("--entity", type=str, default="jhrudoler-penn", help="W&B entity/username")
    parser.add_argument("--project", type=str, default="inductive-bias-experiments", help="W&B project name")
    parser.add_argument("--input-csv", type=Path, help="Existing CSV of sweep results to load")
    parser.add_argument("--max-runs", type=int, help="Optional cap on the number of runs to fetch")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, ...)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_level)
    LOGGER.info("Log file: %s", log_path)

    apply_plot_style()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df: pd.DataFrame
    sweep_label: str

    if args.input_csv:
        if not args.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
        raw_df = pd.read_csv(args.input_csv)
        sweep_label = args.input_csv.stem
        LOGGER.info("Loaded %d rows from %s", len(raw_df), args.input_csv)
    else:
        if not args.sweep_id:
            raise ValueError("Either --input-csv or --sweep-id must be provided.")
        raw_df, sweep_label = fetch_sweep_runs(
            entity=args.entity,
            project=args.project,
            sweep_id=args.sweep_id,
            max_runs=args.max_runs,
        )
        if raw_df.empty:
            LOGGER.error("No runs retrieved for sweep %s", args.sweep_id)
            return
        csv_path = DATA_DIR / f"l2_sweep_{sweep_label}_results.csv"
        raw_df.to_csv(csv_path, index=False)
        LOGGER.info("Saved raw sweep data to %s", csv_path)

    prepared_df = prepare_dataframe(raw_df)
    LOGGER.info("Prepared dataframe with %d rows.", len(prepared_df))

    summary_df = build_summary_table(prepared_df)
    if not summary_df.empty:
        summary_path = DATA_DIR / f"l2_sweep_{sweep_label}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        LOGGER.info("Saved summary statistics to %s", summary_path)
        best_rows = summary_df.nsmallest(10, "mean_abs_error")
        LOGGER.info("Top configurations by mean absolute error:\n%s", best_rows)
    else:
        LOGGER.warning("Summary table could not be created (insufficient columns).")

    scatter_path = FIGURES_DIR / f"l2_sweep_{sweep_label}_estimated_vs_true.pdf"
    heatmap_path = FIGURES_DIR / f"l2_sweep_{sweep_label}_architecture_heatmap.pdf"
    noise_path = FIGURES_DIR / f"l2_sweep_{sweep_label}_error_vs_noise.pdf"
    perf_path = FIGURES_DIR / f"l2_sweep_{sweep_label}_performance_vs_l2.pdf"

    plot_estimated_vs_true(prepared_df, scatter_path)
    plot_architecture_heatmaps(prepared_df, heatmap_path)
    plot_error_vs_noise(prepared_df, noise_path)
    plot_performance_vs_l2(prepared_df, perf_path)

    LOGGER.info("Analysis complete.")


if __name__ == "__main__":
    main()
