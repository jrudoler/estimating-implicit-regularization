#!/usr/bin/env python3
"""Plot attention/MLP ridge estimates with bootstrap or token-chunk intervals."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "data" / "generated" / "olmo_ridge_estimation" / "bootstrap_grid"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "figures" / "llm_ridge_bootstrap_grid.pdf"
LOGGER = logging.getLogger(__name__)

SCOPE_SPECS: tuple[tuple[str, str], ...] = (
    ("attention_plus_mlp", "Attention + MLP"),
    ("attention", "Attention"),
    ("mlp", "MLP"),
)
FAMILY_ORDER = ("Gemma 3", "Qwen2.5", "Qwen3", "Pythia", "OLMo 2")
FAMILY_COLORS = {
    "Gemma 3": "#0072B2",
    "Qwen2.5": "#D55E00",
    "Qwen3": "#CC79A7",
    "Pythia": "#6C6C6C",
    "OLMo 2": "#009E73",
}
FAMILY_MARKERS = {
    "Gemma 3": "o",
    "Qwen2.5": "s",
    "Qwen3": "D",
    "Pythia": "v",
    "OLMo 2": "^",
}
FAMILY_LABEL_Y_OFFSETS = {
    "Gemma 3": 0,
    "Qwen2.5": -5,
    "Qwen3": 5,
    "Pythia": 0,
    "OLMo 2": 5,
}


@dataclass(frozen=True)
class BootstrapPoint:
    run_id: str
    family: str
    label: str
    model_params: int
    attention_mlp_params: int
    lambdas: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]
    error_sources: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pattern",
        default="*_wikitext103_full_bootstrap.json",
        help="Glob pattern under --input-dir.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def family_name(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen2.5" in lowered:
        return "Qwen2.5"
    if "gemma-3" in lowered:
        return "Gemma 3"
    if "qwen3" in lowered:
        return "Qwen3"
    if "pythia" in lowered:
        return "Pythia"
    if "olmo-2" in lowered:
        return "OLMo 2"
    return "Other"


def model_label(run_id: str, model_id: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?[BM])", model_id, flags=re.IGNORECASE)
    label = match.group(1).upper() if match else run_id
    return label.replace("0.5B", "0.5B")


def nominal_model_params(model_id: str) -> int:
    match = re.search(r"(\d+(?:\.\d+)?)([BM])", model_id, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse nominal model size from {model_id!r}")
    value = float(match.group(1))
    multiplier = 1_000_000_000 if match.group(2).upper() == "B" else 1_000_000
    return int(value * multiplier)


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


def scope_payload(estimates: Mapping[str, Any], scope: str) -> Mapping[str, Any]:
    if scope == "attention_plus_mlp":
        return estimates[scope]
    return estimates["groups"][scope]


def estimate_with_interval(scope_data: Mapping[str, Any]) -> tuple[float, float, float, str]:
    chunk_estimates = scope_data.get("chunk_token_estimates", {})
    if chunk_estimates:
        mean = float(chunk_estimates["mean"])
        sem = float(chunk_estimates["sem"])
        return mean, mean - sem, mean + sem, "chunk_sem"

    bootstrap = scope_data.get("bootstrap_batches", {})
    if bootstrap:
        return (
            float(scope_data["lambda_hat"]),
            float(bootstrap.get("ci95_lower", float("nan"))),
            float(bootstrap.get("ci95_upper", float("nan"))),
            "bootstrap_ci95",
        )

    lambda_hat = float(scope_data["lambda_hat"])
    return lambda_hat, float("nan"), float("nan"), "none"


def point_from_payload(path: Path, payload: Mapping[str, Any]) -> BootstrapPoint:
    config = payload.get("config", {})
    model_id = str(config.get("model_id", ""))
    estimates = payload["ridge_estimates"]
    family = family_name(model_id)
    attention_mlp_params = int(estimates["attention_plus_mlp"]["param_count"])
    lambdas: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    error_sources: dict[str, str] = {}
    for scope, _ in SCOPE_SPECS:
        scope_data = scope_payload(estimates, scope)
        lambdas[scope], lower[scope], upper[scope], error_sources[scope] = (
            estimate_with_interval(scope_data)
        )
    return BootstrapPoint(
        run_id=path.stem,
        family=family,
        label=model_label(path.stem, model_id),
        model_params=nominal_model_params(model_id),
        attention_mlp_params=attention_mlp_params,
        lambdas=lambdas,
        lower=lower,
        upper=upper,
        error_sources=error_sources,
    )


def load_points(input_dir: Path, pattern: str) -> list[BootstrapPoint]:
    points = [
        point_from_payload(path, load_payload(path))
        for path in sorted(input_dir.glob(pattern))
    ]
    return sorted(points, key=lambda point: (point.family, point.model_params))


def ordered_families(points: Sequence[BootstrapPoint]) -> list[str]:
    available = {point.family for point in points}
    ordered = [family for family in FAMILY_ORDER if family in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def format_model_params(value: float, _position: int) -> str:
    if value < 1_000_000_000:
        return f"{value / 1_000_000:g}M"
    return f"{value / 1_000_000_000:g}B"


def configure_model_param_axis(
    ax: plt.Axes,
    points: Sequence[BootstrapPoint],
) -> None:
    model_params = sorted({point.model_params for point in points})
    ax.set_xlim(model_params[0] / 1.25, model_params[-1] * 1.25)
    ax.xaxis.set_major_locator(FixedLocator(model_params))
    ax.xaxis.set_major_formatter(FuncFormatter(format_model_params))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def plot_panel(
    ax: plt.Axes,
    points: Sequence[BootstrapPoint],
    *,
    scope: str,
    title: str,
    show_legend: bool,
) -> None:
    for family in ordered_families(points):
        family_points = sorted(
            (point for point in points if point.family == family),
            key=lambda point: point.model_params,
        )
        x_values = [point.model_params for point in family_points]
        y_values = [point.lambdas[scope] for point in family_points]
        yerr_lower = [
            max(point.lambdas[scope] - point.lower[scope], 0.0)
            if math.isfinite(point.lower[scope])
            else 0.0
            for point in family_points
        ]
        yerr_upper = [
            max(point.upper[scope] - point.lambdas[scope], 0.0)
            if math.isfinite(point.upper[scope])
            else 0.0
            for point in family_points
        ]
        ax.errorbar(
            x_values,
            y_values,
            yerr=[yerr_lower, yerr_upper],
            linewidth=1.8,
            marker=None,
            elinewidth=1.2,
            capsize=2.5,
            label=family,
            color=FAMILY_COLORS.get(family),
        )
        for point in family_points:
            ax.annotate(
                point.label,
                xy=(point.model_params, point.lambdas[scope]),
                xytext=(4, FAMILY_LABEL_Y_OFFSETS.get(family, 0)),
                textcoords="offset points",
                fontsize=7,
                va="center",
            )
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-9)
    configure_model_param_axis(ax, points)
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.12)
    if show_legend:
        ax.legend(frameon=False, loc="best", fontsize=8)


def make_plot(points: Sequence[BootstrapPoint], output: Path) -> None:
    if not points:
        raise ValueError("No bootstrap-grid JSON outputs found.")
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    source_names = {
        source
        for point in points
        for source in point.error_sources.values()
        if source != "none"
    }
    if source_names == {"chunk_sem"}:
        figure_title = (
            "Attention/MLP ridge estimates across contiguous 10M-token chunks "
            "(mean +/- 1 SEM)"
        )
    elif source_names == {"bootstrap_ci95"}:
        figure_title = "Attention/MLP ridge estimates with 95% batch-bootstrap intervals"
    else:
        figure_title = "Attention/MLP ridge estimates"

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharex=True, constrained_layout=True)
    for index, (scope, panel_title) in enumerate(SCOPE_SPECS):
        plot_panel(
            axes[index],
            points,
            scope=scope,
            title=panel_title,
            show_legend=index == 0,
        )
    axes[0].set_ylabel("lambda_hat")
    fig.supxlabel("Model parameters (nominal)")
    fig.suptitle(figure_title, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", output)


def main() -> None:
    configure_logging()
    args = parse_args()
    points = load_points(args.input_dir, args.pattern)
    make_plot(points, args.output)


if __name__ == "__main__":
    main()
