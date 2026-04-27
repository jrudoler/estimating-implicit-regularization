#!/usr/bin/env python3
"""
Analysis script for Mixed Bias Recovery experiments.

Loads results from W&B sweeps and generates figures showing:
1. True vs estimated regularization coefficients (scatter plot)
2. Recovery accuracy by ground truth condition
3. Identifiability: inactive regularizers should be near zero
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.wandb_utils import get_sweep_runs, wandb_summary_df

# Use clean style if available
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))

# Output directory
FIGURES_DIR = REPO_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze mixed bias recovery results")
    parser.add_argument(
        "--sweep-id",
        type=str,
        required=True,
        help="W&B sweep ID for mixed_bias_recovery experiment",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="jhrudoler-penn",
        help="W&B entity (organization/user)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="inductive-bias-experiments",
        help="W&B project name",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mixed_bias_recovery",
        help="Prefix for output figure files",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["png", "pdf", "svg"],
        help="Output figure format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output formats",
    )
    return parser.parse_args()


def load_sweep_data(sweep_id: str, entity: str, project: str) -> pd.DataFrame:
    """Load and preprocess sweep data from W&B."""
    print(f"Loading runs from sweep {entity}/{project}/{sweep_id}...")
    runs = get_sweep_runs(sweep_id, entity=entity, project=project, state="finished")
    print(f"Found {len(runs)} finished runs")

    df = wandb_summary_df(runs, include_system_metrics=False)

    # Ensure required columns exist
    required_cols = [
        "gt_ridge_lambda",
        "gt_coherence_lambda",
        "estimated_ridge",
        "estimated_coherence",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sweep data: {missing}")

    # Add derived columns
    df["ridge_active"] = df["gt_ridge_lambda"] > 0
    df["coherence_active"] = df["gt_coherence_lambda"] > 0
    df["experiment_type"] = df.apply(
        lambda r: _get_experiment_type(r["ridge_active"], r["coherence_active"]),
        axis=1,
    )

    return df


def _get_experiment_type(ridge_active: bool, coherence_active: bool) -> str:
    """Classify experiment by which regularizers are active."""
    if ridge_active and coherence_active:
        return "Both Active"
    elif ridge_active:
        return "Ridge Only"
    elif coherence_active:
        return "Coherence Only"
    else:
        return "Neither Active"


def plot_true_vs_estimated(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Scatter plot of true vs estimated values for both regularizers.
    Points should lie on the diagonal for perfect recovery.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Ridge (L2) ---
    ax = axes[0]
    scatter = ax.scatter(
        df["gt_ridge_lambda"],
        df["estimated_ridge"],
        c=df["train_bias/loss"] if "train_bias/loss" in df.columns else "steelblue",
        cmap="viridis_r",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        s=50,
    )

    # Reference line (y=x)
    lims = [
        min(df["gt_ridge_lambda"].min(), df["estimated_ridge"].min()) * 0.9,
        max(df["gt_ridge_lambda"].max(), df["estimated_ridge"].max()) * 1.1,
    ]
    # Handle zero values
    lims[0] = max(lims[0], -0.01)
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True Ridge λ")
    ax.set_ylabel("Estimated Ridge λ")
    ax.set_title("Ridge (L2) Recovery")
    ax.legend(loc="lower right")

    # --- Coherence ---
    ax = axes[1]
    scatter = ax.scatter(
        df["gt_coherence_lambda"],
        df["estimated_coherence"],
        c=df["train_bias/loss"] if "train_bias/loss" in df.columns else "steelblue",
        cmap="viridis_r",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        s=50,
    )

    lims = [
        min(df["gt_coherence_lambda"].min(), df["estimated_coherence"].min()) * 0.9,
        max(df["gt_coherence_lambda"].max(), df["estimated_coherence"].max()) * 1.1,
    ]
    lims[0] = max(lims[0], -0.01)
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True Coherence λ")
    ax.set_ylabel("Estimated Coherence λ")
    ax.set_title("WeightCoherence Recovery")
    ax.legend(loc="lower right")

    # Colorbar for loss
    if "train_bias/loss" in df.columns:
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label("Bias Estimation Loss")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_identifiability_bars(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Bar plot showing estimated coefficients grouped by experiment type.
    Key insight: inactive regularizers should be estimated near zero.
    """
    # Melt to long format for easier plotting
    df_long = df.melt(
        id_vars=["run_id", "experiment_type", "depth", "seed"],
        value_vars=["estimated_ridge", "estimated_coherence"],
        var_name="regularizer",
        value_name="estimated_value",
    )
    df_long["regularizer"] = df_long["regularizer"].map({
        "estimated_ridge": "Ridge",
        "estimated_coherence": "Coherence",
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=df_long,
        x="experiment_type",
        y="estimated_value",
        hue="regularizer",
        ax=ax,
        palette=["#4C72B0", "#DD8452"],
        edgecolor="white",
        errorbar="sd",
    )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Ground Truth Condition")
    ax.set_ylabel("Estimated Coefficient")
    ax.set_title("Identifiability: Inactive Regularizers Should Be ≈ 0")
    ax.legend(title="Regularizer")

    # Rotate x labels for readability
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_error_by_condition(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Box/strip plot of relative errors by ground truth condition.
    """
    # Compute relative errors (only for active regularizers)
    df = df.copy()
    df["ridge_rel_error"] = np.where(
        df["gt_ridge_lambda"] > 0,
        np.abs(df["estimated_ridge"] - df["gt_ridge_lambda"]) / df["gt_ridge_lambda"],
        np.nan,
    )
    df["coherence_rel_error"] = np.where(
        df["gt_coherence_lambda"] > 0,
        np.abs(df["estimated_coherence"] - df["gt_coherence_lambda"])
        / df["gt_coherence_lambda"],
        np.nan,
    )

    # Melt for plotting
    df_errors = df.melt(
        id_vars=["run_id", "experiment_type", "depth"],
        value_vars=["ridge_rel_error", "coherence_rel_error"],
        var_name="regularizer",
        value_name="relative_error",
    ).dropna()

    df_errors["regularizer"] = df_errors["regularizer"].map({
        "ridge_rel_error": "Ridge",
        "coherence_rel_error": "Coherence",
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    # Violin + strip combination
    sns.violinplot(
        data=df_errors,
        x="regularizer",
        y="relative_error",
        hue="experiment_type",
        ax=ax,
        inner=None,
        alpha=0.3,
        cut=0,
    )
    sns.stripplot(
        data=df_errors,
        x="regularizer",
        y="relative_error",
        hue="experiment_type",
        ax=ax,
        dodge=True,
        alpha=0.7,
        size=4,
        linewidth=0.5,
        edgecolor="white",
    )

    # Success threshold line
    ax.axhline(0.3, color="green", linestyle="--", alpha=0.7, label="30% threshold")

    ax.set_xlabel("Regularizer")
    ax.set_ylabel("Relative Error")
    ax.set_title("Recovery Error for Active Regularizers")

    # Fix legend (strip + violin creates duplicates)
    handles, labels = ax.get_legend_handles_labels()
    n_unique = len(df_errors["experiment_type"].unique())
    ax.legend(
        handles[: n_unique + 1],
        labels[: n_unique + 1],
        title="Condition",
        loc="upper right",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_faceted_recovery(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    FacetGrid: true vs estimated, faceted by depth and ground truth condition.
    """
    # Long format with both regularizers
    df_ridge = df[["run_id", "depth", "gt_ridge_lambda", "estimated_ridge"]].copy()
    df_ridge.columns = ["run_id", "depth", "true_value", "estimated_value"]
    df_ridge["regularizer"] = "Ridge"
    df_ridge["is_active"] = df_ridge["true_value"] > 0

    df_coh = df[
        ["run_id", "depth", "gt_coherence_lambda", "estimated_coherence"]
    ].copy()
    df_coh.columns = ["run_id", "depth", "true_value", "estimated_value"]
    df_coh["regularizer"] = "Coherence"
    df_coh["is_active"] = df_coh["true_value"] > 0

    df_long = pd.concat([df_ridge, df_coh], ignore_index=True)

    g = sns.FacetGrid(
        df_long,
        col="regularizer",
        row="depth",
        margin_titles=True,
        height=3,
        aspect=1.1,
    )

    def scatter_with_refline(data, **kwargs):
        ax = plt.gca()
        ax.scatter(
            data["true_value"],
            data["estimated_value"],
            c=data["is_active"].map({True: "#4C72B0", False: "#DD8452"}),
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            s=40,
        )
        # Reference line
        lims = [
            min(data["true_value"].min(), data["estimated_value"].min()),
            max(data["true_value"].max(), data["estimated_value"].max()),
        ]
        lims = [max(lims[0] * 0.9, -0.01), lims[1] * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    g.map_dataframe(scatter_with_refline)
    g.set_axis_labels("True λ", "Estimated λ")
    g.set_titles(col_template="{col_name}", row_template="Depth {row_name}")

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#4C72B0", label="Active (λ > 0)"),
        Patch(facecolor="#DD8452", label="Inactive (λ = 0)"),
    ]
    g.figure.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
    )

    plt.tight_layout()
    g.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a summary table of recovery results."""
    print("\n" + "=" * 80)
    print("RECOVERY SUMMARY")
    print("=" * 80)

    # Group by experiment type
    summary = (
        df.groupby("experiment_type")
        .agg(
            {
                "estimated_ridge": ["mean", "std"],
                "estimated_coherence": ["mean", "std"],
                "train_bias/loss": "mean",
            }
        )
        .round(4)
    )

    # Add ground truth for reference
    gt_summary = (
        df.groupby("experiment_type")
        .agg({"gt_ridge_lambda": "first", "gt_coherence_lambda": "first"})
        .round(4)
    )

    print("\nEstimated Values by Condition:")
    print(summary)
    print("\nGround Truth by Condition:")
    print(gt_summary)

    # Success rates
    print("\n" + "-" * 80)
    print("SUCCESS RATES (relative error < 30% for active, estimate < 0.01 for inactive)")
    print("-" * 80)

    for exp_type in df["experiment_type"].unique():
        subset = df[df["experiment_type"] == exp_type]
        n = len(subset)

        # Ridge success
        if subset["ridge_active"].iloc[0]:
            ridge_success = (
                np.abs(subset["estimated_ridge"] - subset["gt_ridge_lambda"])
                / subset["gt_ridge_lambda"]
                < 0.3
            ).mean()
        else:
            ridge_success = (subset["estimated_ridge"].abs() < 0.01).mean()

        # Coherence success
        if subset["coherence_active"].iloc[0]:
            coh_success = (
                np.abs(subset["estimated_coherence"] - subset["gt_coherence_lambda"])
                / subset["gt_coherence_lambda"]
                < 0.3
            ).mean()
        else:
            coh_success = (subset["estimated_coherence"].abs() < 0.01).mean()

        print(
            f"{exp_type:20s}: Ridge {ridge_success*100:5.1f}% | Coherence {coh_success*100:5.1f}% (n={n})"
        )


def main():
    args = parse_args()

    # Load data
    df = load_sweep_data(args.sweep_id, args.entity, args.project)
    print(f"Loaded {len(df)} runs")
    print(f"Columns: {list(df.columns)}")

    # Print summary
    print_summary_table(df)

    # Generate figures
    prefix = args.output_prefix
    fmt = args.format
    dpi = args.dpi

    plot_true_vs_estimated(df, FIGURES_DIR / f"{prefix}_true_vs_estimated.{fmt}", dpi)
    plot_identifiability_bars(df, FIGURES_DIR / f"{prefix}_identifiability.{fmt}", dpi)
    plot_error_by_condition(df, FIGURES_DIR / f"{prefix}_error_by_condition.{fmt}", dpi)
    plot_faceted_recovery(df, FIGURES_DIR / f"{prefix}_faceted.{fmt}", dpi)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
