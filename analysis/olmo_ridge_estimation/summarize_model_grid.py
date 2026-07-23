#!/usr/bin/env python3
"""Summarize LLM ridge-estimation model-grid JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_GLOB = (
    REPO_ROOT / "data" / "generated" / "olmo_ridge_estimation" / "model_grid" / "*_wiki0001_50m.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Model-grid JSON outputs. Defaults to all *_wiki0001_50m.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown table output path. Defaults to stdout only.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


def _scope_from_sufficient_stats(stats: Mapping[str, float | int]) -> dict[str, float | int]:
    theta_dot_grad = float(stats["theta_dot_grad"])
    theta_sq_norm = float(stats["theta_sq_norm"])
    grad_sq_norm = float(stats["grad_sq_norm"])
    lambda_hat = (
        -theta_dot_grad / (2.0 * theta_sq_norm) if theta_sq_norm > 0.0 else float("nan")
    )
    denom = math.sqrt(theta_sq_norm * grad_sq_norm)
    cosine_alignment = -theta_dot_grad / denom if denom > 0.0 else float("nan")
    if grad_sq_norm > 0.0 and math.isfinite(lambda_hat):
        residual_sq = (
            grad_sq_norm
            + 4.0 * lambda_hat * theta_dot_grad
            + 4.0 * lambda_hat * lambda_hat * theta_sq_norm
        )
        residual_ratio = math.sqrt(max(residual_sq, 0.0) / grad_sq_norm)
    else:
        residual_ratio = float("nan")
    return {
        **dict(stats),
        "lambda_hat": lambda_hat,
        "lambda_hat_nonnegative": max(lambda_hat, 0.0) if math.isfinite(lambda_hat) else lambda_hat,
        "lambda_mean_loss": lambda_hat,
        "cosine_alignment": cosine_alignment,
        "residual_ratio": residual_ratio,
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
    if "gemma-3" in lowered:
        return "Gemma 3"
    if "olmo-2" in lowered:
        return "OLMo 2"
    if "qwen3" in lowered:
        return "Qwen3"
    if "pythia" in lowered:
        return "Pythia"
    return "Other"


def fmt_float(value: Any, precision: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{precision}g}"


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return ""


def row_from_payload(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    estimates = payload.get("ridge_estimates", {})
    groups = estimates.get("groups", {})
    attention_mlp = attention_plus_mlp_scope(estimates)
    embedding = groups.get("embedding_or_lm_head", {})
    decay = estimates.get("decay_eligible", {})
    loss = payload.get("loss", {})
    gradient_artifact = payload.get("gradient_artifact", {})
    return {
        "run_id": path.stem,
        "family": family_name(str(config.get("model_id", ""))),
        "model_id": config.get("model_id", ""),
        "revision": config.get("revision", ""),
        "block_params": int(attention_mlp.get("param_count", 0)),
        "lambda_attention_mlp": attention_mlp.get("lambda_hat"),
        "cosine_attention_mlp": attention_mlp.get("cosine_alignment"),
        "lambda_embedding_lm_head": embedding.get("lambda_hat"),
        "lambda_decay_eligible": decay.get("lambda_hat"),
        "loss": loss.get("cross_entropy"),
        "perplexity": loss.get("perplexity"),
        "gradient_manifest": gradient_artifact.get("manifest", ""),
    }


def resolve_inputs(inputs: Sequence[Path]) -> list[Path]:
    if inputs:
        return sorted(path.expanduser().resolve() for path in inputs)
    return sorted(DEFAULT_INPUT_GLOB.parent.glob(DEFAULT_INPUT_GLOB.name))


def render_markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    headers = [
        "run_id",
        "family",
        "block_params",
        "lambda_attention_mlp",
        "cosine_attention_mlp",
        "lambda_embedding_lm_head",
        "lambda_decay_eligible",
        "loss",
        "perplexity",
        "gradient_manifest",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            str(row["run_id"]),
            str(row["family"]),
            fmt_int(row["block_params"]),
            fmt_float(row["lambda_attention_mlp"], precision=5),
            fmt_float(row["cosine_attention_mlp"], precision=4),
            fmt_float(row["lambda_embedding_lm_head"], precision=5),
            fmt_float(row["lambda_decay_eligible"], precision=5),
            fmt_float(row["loss"], precision=5),
            fmt_float(row["perplexity"], precision=5),
            str(row["gradient_manifest"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args.inputs)
    rows = [row_from_payload(path, load_payload(path)) for path in input_paths]
    rows.sort(key=lambda row: int(row["block_params"]))
    table = render_markdown(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
