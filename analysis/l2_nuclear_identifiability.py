#!/usr/bin/env python3
"""Analyze L2 + Nuclear Norm identifiability from sweep results."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.wandb_utils import wandb_summary_df


def fetch_sweep_data(sweep_id: str, entity: str = "jhrudoler-penn", project: str = "inductive-bias"):
    """Fetch all runs from a W&B sweep."""
    api = wandb.Api(timeout=60)
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs
    
    df = wandb_summary_df(runs)
    return df


def compute_identifiability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute identifiability metrics across parameter combinations."""
    # Group by true parameter values
    grouped = df.groupby(['l2_lambda', 'nuclear_lambda']).agg({
        'l2_rel_error': ['mean', 'std', 'min', 'max'],
        'nuclear_rel_error': ['mean', 'std', 'min', 'max'],
        'mean_rel_error': ['mean', 'std'],
        'max_rel_error': ['mean', 'std'],
        'seed': 'count',
    }).reset_index()
    
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    return grouped


def plot_recovery_scatter(df: pd.DataFrame, save_path: Path = None):
    """Plot true vs estimated parameters for both L2 and nuclear norm."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 recovery
    axes[0].scatter(df['l2_lambda'], df['l2_estimated'], alpha=0.6, s=30)
    min_val = min(df['l2_lambda'].min(), df['l2_estimated'].min())
    max_val = max(df['l2_lambda'].max(), df['l2_estimated'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Recovery')
    axes[0].set_xlabel('True L2 λ')
    axes[0].set_ylabel('Estimated L2 λ')
    axes[0].set_title('L2 Parameter Recovery')
    axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # Nuclear norm recovery
    axes[1].scatter(df['nuclear_lambda'], df['nuclear_estimated'], alpha=0.6, s=30)
    min_val = min(df['nuclear_lambda'].min(), df['nuclear_estimated'].min())
    max_val = max(df['nuclear_lambda'].max(), df['nuclear_estimated'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Recovery')
    axes[1].set_xlabel('True Nuclear Norm λ')
    axes[1].set_ylabel('Estimated Nuclear Norm λ')
    axes[1].set_title('Nuclear Norm Parameter Recovery')
    axes[1].legend()
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved recovery scatter plot to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_error_heatmap(df: pd.DataFrame, save_path: Path = None):
    """Plot heatmap of recovery errors across parameter combinations."""
    # Aggregate by true parameter combinations
    pivot_data = df.groupby(['l2_lambda', 'nuclear_lambda'])['mean_rel_error'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='nuclear_lambda', columns='l2_lambda', values='mean_rel_error')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Mean Relative Error'})
    ax.set_xlabel('True L2 λ')
    ax.set_ylabel('True Nuclear Norm λ')
    ax.set_title('Identifiability: Mean Relative Error Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error heatmap to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_correlation_analysis(df: pd.DataFrame, save_path: Path = None):
    """Analyze correlation between L2 and nuclear norm estimation errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error correlation
    axes[0].scatter(df['l2_rel_error'], df['nuclear_rel_error'], alpha=0.6, s=30)
    axes[0].set_xlabel('L2 Relative Error')
    axes[0].set_ylabel('Nuclear Norm Relative Error')
    axes[0].set_title('Estimation Error Correlation')
    
    # Compute correlation
    corr = np.corrcoef(df['l2_rel_error'], df['nuclear_rel_error'])[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=axes[0].transAxes, verticalalignment='top')
    
    # Ratio of true parameters vs mean relative error
    df['param_ratio'] = df['l2_lambda'] / df['nuclear_lambda']
    axes[1].scatter(df['param_ratio'], df['mean_rel_error'], alpha=0.6, s=30)
    axes[1].set_xlabel('L2 / Nuclear Norm Ratio')
    axes[1].set_ylabel('Mean Relative Error')
    axes[1].set_title('Error vs Parameter Ratio')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation analysis to {save_path}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze L2 + Nuclear Norm identifiability results")
    parser.add_argument("sweep_id", type=str, help="W&B sweep ID")
    parser.add_argument("--entity", type=str, default="jhrudoler-penn", help="W&B entity")
    parser.add_argument("--project", type=str, default="inductive-bias", help="W&B project")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"), help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching sweep data for {args.sweep_id}...")
    df = fetch_sweep_data(args.sweep_id, args.entity, args.project)
    print(f"Loaded {len(df)} runs")
    
    # Filter out failed runs
    df = df.dropna(subset=['l2_estimated', 'nuclear_estimated'])
    print(f"After filtering: {len(df)} successful runs")
    
    # Compute summary statistics
    print("\n=== Overall Statistics ===")
    print(f"L2 Mean Rel Error: {df['l2_rel_error'].mean():.4f} ± {df['l2_rel_error'].std():.4f}")
    print(f"Nuclear Mean Rel Error: {df['nuclear_rel_error'].mean():.4f} ± {df['nuclear_rel_error'].std():.4f}")
    print(f"Combined Mean Rel Error: {df['mean_rel_error'].mean():.4f} ± {df['mean_rel_error'].std():.4f}")
    
    # Compute identifiability metrics
    print("\n=== Identifiability by Parameter Combination ===")
    id_metrics = compute_identifiability_metrics(df)
    print(id_metrics)
    
    # Save metrics
    metrics_path = args.output_dir / f"identifiability_metrics_{args.sweep_id}.csv"
    id_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_recovery_scatter(df, args.output_dir / f"recovery_scatter_{args.sweep_id}.png")
    plot_error_heatmap(df, args.output_dir / f"error_heatmap_{args.sweep_id}.png")
    plot_correlation_analysis(df, args.output_dir / f"correlation_analysis_{args.sweep_id}.png")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

