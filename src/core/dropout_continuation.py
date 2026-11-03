"""Verified imports and schedules for separately named continuation studies."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from core.scalar_estimation import file_sha256, model_sha256


def scheduled_lr(epoch: int, *, initial: float, milestones: list[int], factor: float) -> float:
    """LR for an epoch indexed from zero within this continuation stage."""
    if epoch < 0 or not math.isfinite(initial) or initial <= 0:
        raise ValueError("Epoch and initial learning rate are invalid")
    if not 0 < factor < 1 or milestones != sorted(set(milestones)) or any(m <= 0 for m in milestones):
        raise ValueError("Require increasing positive milestones and a decay factor in (0, 1)")
    return initial * factor ** sum(epoch >= milestone for milestone in milestones)


def validate_import(source: dict[str, Any], payload: dict[str, Any], checkpoint: Path,
                    model: torch.nn.Module) -> dict[str, Any]:
    """Import only the final, hash-verified state of a completed prior stage."""
    if source["status"] not in ("budget_exhausted", "converged_to_declared_tolerance"):
        raise ValueError("The source study must have a completed result")
    endpoint = source["trajectory"][-1]
    expected = endpoint["provenance"]
    if file_sha256(checkpoint) != expected["checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash mismatch")
    if model_sha256(model) != expected["model_sha256"]:
        raise ValueError("Source model hash mismatch")
    if payload["epoch"] != expected["epoch"] or payload["additional_epochs"] != endpoint["additional_epochs"]:
        raise ValueError("Source checkpoint is not the declared final endpoint")
    for key, value in payload["signature"].items():
        if source.get(key) != value:
            raise ValueError(f"Source signature mismatch: {key}")
    if payload["baseline_norm"] != source["baseline_norm"]:
        raise ValueError("Original convergence baseline changed")
    return expected


def restore_rng(payload: dict[str, Any]) -> None:
    torch.set_rng_state(payload["cpu_rng"].cpu())
    if torch.cuda.is_available():
        if len(payload["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved random stream")
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)
