#!/usr/bin/env python3
"""Plot ridge estimates across the LLM model grid."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "generated" / "olmo_ridge_estimation" / "model_grid"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "figures" / "llm_ridge_model_grid.pdf"
SUPERSEDED_RUN_IDS = {"olmo2_7b_stage1_wiki0001_50m"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelGridPoint:
    run_id: str
    family: str
    label: str
    total_params: int
    values: dict[str, float]


SCOPE_SPECS: tuple[tuple[str, str], ...] = (
    ("all", "All parameters"),
    ("decay_eligible", "Decay-eligible"),
    ("attention_plus_mlp", "Attention + MLP"),
    ("embedding_or_lm_head", "Embedding / LM head"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_wiki0001_50m.json model-grid outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination PDF path.",
    )
    parser.add_argument(
        "--include-superseded",
        action="store_true",
        help="Include superseded checkpoints such as the earlier OLMo 2 7B stage1 run.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _scope_from_sufficient_stats(stats: Mapping[str, float | int]) -> dict[str, float]:
    theta_dot_grad = float(stats["theta_dot_grad"])
    theta_sq_norm = float(stats["theta_sq_norm"])
    grad_sq_norm = float(stats["grad_sq_norm"])
    lambda_hat = (
        -theta_dot_grad / (2.0 * theta_sq_norm) if theta_sq_norm > 0.0 else float("nan")
    )
    denom = math.sqrt(theta_sq_norm * grad_sq_norm)
    cosine_alignment = -theta_dot_grad / denom if denom > 0.0 else float("nan")
    return {
        "lambda_hat": lambda_hat,
        "cosine_alignment": cosine_alignment,
        "param_count": float(stats.get("param_count", 0)),
    }


def attention_plus_mlp_scope(ridge_estimates: Mapping[str, Any]) -> Mapping[str, Any]:
    if "attention_plus_mlp" in ridge_estimates:
        return ridge_estimates["attention_plus_mlp"]

    groups = ridge_estimates.get("groups", {})
    summed = {
        "param_count": 0,
        "theta_dot_grad": 0.0,
        "theta_sq_norm": 0.0,
        "grad_sq_norm": 0.0,
    }
    for group_name in ("attention", "mlp"):
        group = groups.get(group_name)
        if group is None:
            continue
        summed["param_count"] += int(group.get("param_count", 0))
        summed["theta_dot_grad"] += float(group.get("theta_dot_grad", 0.0))
        summed["theta_sq_norm"] += float(group.get("theta_sq_norm", 0.0))
        summed["grad_sq_norm"] += float(group.get("grad_sq_norm", 0.0))
    return _scope_from_sufficient_stats(summed)


def family_name(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen2.5" in lowered:
        return "Qwen2.5"
    if "qwen3" in lowered:
        return "Qwen3"
    if "gemma-3" in lowered:
        return "Gemma 3"
    if "pythia" in lowered:
        return "Pythia"
    if "olmo-2" in lowered:
        return "OLMo 2"
    return "Other"


def model_label(run_id: str, model_id: str, revision: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?[BM])", model_id, flags=re.IGNORECASE)
    label = match.group(1).upper() if match else run_id.removesuffix("_wiki0001_50m")
    if run_id == "olmo2_7b_stage1_final_wiki0001_50m":
        return f"{label} final"
    if revision and revision != "main" and "olmo" in model_id.lower():
        return f"{label} stage1"
    return label


def scope_lambda(ridge_estimates: Mapping[str, Any], scope_name: str) -> float:
    if scope_name == "attention_plus_mlp":
        scope = attention_plus_mlp_scope(ridge_estimates)
    elif scope_name in {"embedding_or_lm_head", "attention", "mlp"}:
        scope = ridge_estimates.get("groups", {}).get(scope_name, {})
    else:
        scope = ridge_estimates.get(scope_name, {})
    return float(scope.get("lambda_hat", float("nan")))


def point_from_payload(path: Path, payload: Mapping[str, Any]) -> ModelGridPoint:
    config = payload.get("config", {})
    model_id = str(config.get("model_id", ""))
    revision = str(config.get("revision", ""))
    ridge_estimates = payload.get("ridge_estimates", {})
    total_params = int(ridge_estimates.get("all", {}).get("param_count", 0))
    values = {
        scope_name: scope_lambda(ridge_estimates, scope_name)
        for scope_name, _ in SCOPE_SPECS
    }
    return ModelGridPoint(
        run_id=path.stem,
        family=family_name(model_id),
        label=model_label(path.stem, model_id, revision),
        total_params=total_params,
        values=values,
    )


def load_points(input_dir: Path, *, include_superseded: bool) -> list[ModelGridPoint]:
    paths = sorted(input_dir.glob("*_wiki0001_50m.json"))
    points: list[ModelGridPoint] = []
    for path in paths:
        if not include_superseded and path.stem in SUPERSEDED_RUN_IDS:
            LOGGER.info("Skipping superseded run %s", path.stem)
            continue
        points.append(point_from_payload(path, load_payload(path)))
    return sorted(points, key=lambda point: (point.family, point.total_params, point.run_id))


def ordered_families(points: Sequence[ModelGridPoint]) -> list[str]:
    available = {point.family for point in points}
    ordered = [family for family in FAMILY_ORDER if family in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def format_billions(value: float, _position: int) -> str:
    return f"{value / 1e9:g}"


def annotate_last_points(
    ax: plt.Axes,
    family_points: Sequence[ModelGridPoint],
    scope_name: str,
) -> None:
    for point in family_points:
        y_value = point.values[scope_name]
        if not math.isfinite(y_value):
            continue
        ax.annotate(
            point.label,
            xy=(point.total_params, y_value),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
            va="center",
        )


def plot_scope_panel(
    ax: plt.Axes,
    *,
    points: Sequence[ModelGridPoint],
    scope_name: str,
    title: str,
    show_legend: bool,
) -> None:
    for family in ordered_families(points):
        family_points = sorted(
            (point for point in points if point.family == family),
            key=lambda point: point.total_params,
        )
        x_values = [point.total_params for point in family_points]
        y_values = [point.values[scope_name] for point in family_points]
        ax.plot(
            x_values,
            y_values,
            marker=FAMILY_MARKERS.get(family, "o"),
            linewidth=1.8,
            markersize=4.5,
            label=family,
            color=FAMILY_COLORS.get(family),
        )
        annotate_last_points(ax, family_points, scope_name)

    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-9)
    ax.xaxis.set_major_formatter(format_billions)
    ax.set_title(title)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.12)
    if show_legend:
        ax.legend(frameon=False, loc="best", fontsize=8)


def make_plot(points: Sequence[ModelGridPoint], output: Path) -> None:
    if not points:
        raise ValueError("No model-grid points found to plot.")
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), sharex=True, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for index, (scope_name, title) in enumerate(SCOPE_SPECS):
        plot_scope_panel(
            flat_axes[index],
            points=points,
            scope_name=scope_name,
            title=title,
            show_legend=index == 0,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("lambda_hat")
    for ax in axes[-1, :]:
        ax.set_xlabel("Total unique trainable parameters (B)")

    fig.suptitle("LLM ridge estimates by model size and parameter group", fontsize=13)
    fig.text(
        0.5,
        0.005,
        "Y axis uses signed symlog scaling with a zero reference line; points are 50M-token wiki shard estimates.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", output)


def main() -> None:
    configure_logging()
    args = parse_args()
    points = load_points(args.input_dir, include_superseded=args.include_superseded)
    make_plot(points, args.output)


if __name__ == "__main__":
    main()
