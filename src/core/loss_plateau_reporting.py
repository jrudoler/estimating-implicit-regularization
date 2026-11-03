"""Validate published endpoint projections without reading training scratch."""
from __future__ import annotations

from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.scalar_estimation import file_sha256

DEPTHS, WIDTHS = (3, 5), (128, 256, 512)
RATES = (0., .05, .1, .15, .2, .3, .4, .5)
SEEDS = (123, 42, 666, 314, 17, 111, 325, 643, 432, 51)
GRID = tuple(product(DEPTHS, WIDTHS, RATES, SEEDS))


def collect_paths(paths: Iterable[Path], *, study: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a fixed set of published runs; explicitly record missing cells."""
    rows, sources, protocols, code_hashes = [], [], [], []
    seen: set[tuple[int, int, float, int]] = set()
    for path in paths:
        data = json.loads(path.read_text())
        cfg, endpoint = data["config"], data["endpoint"]
        cell = tuple(cfg[k] for k in ("depth", "width", "dropout", "seed"))
        depth, width, rate, seed = cell
        if cell not in GRID or path.parent.name != f"d{depth}_w{width}_do{rate}_s{seed}":
            raise ValueError(f"Wrong configuration: {path}")
        if cell in seen:
            raise ValueError(f"Duplicate configuration: {path}")
        seen.add(cell)
        if data["signature"]["study"] != study:
            raise ValueError("Cannot mix different studies")
        if data["status"] not in ("loss_plateau", "budget_exhausted"):
            raise ValueError(f"Incomplete run: {path}")
        checkpoint = path.parent / Path(endpoint["provenance"]["checkpoint"]).name
        checkpoint_hash = file_sha256(checkpoint)
        if checkpoint_hash != endpoint["provenance"]["checkpoint_sha256"]:
            raise ValueError(f"Changed checkpoint: {checkpoint}")
        ridge = endpoint["ridge"]
        signed, positive = ridge["scale_star"], ridge["scale_star_clamped"]
        if signed is None or not math.isfinite(signed) or positive != max(0., signed):
            raise ValueError(f"Invalid ridge coefficient: {path}")
        cosine, norm, target_norm = ridge["grad_cosine"], ridge["grad_p_norm"], ridge["target_norm"]
        if cosine is not None and not math.isclose(signed, cosine * target_norm / norm, rel_tol=1e-9, abs_tol=1e-16):
            raise ValueError(f"Inconsistent scalar projection: {path}")
        r2 = ridge["projection_r2_clamped"]
        if cosine is not None and not math.isclose(r2, max(cosine, 0.) ** 2, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError(f"Inconsistent explained fraction: {path}")
        protocols.append(data["protocol"])
        code_hashes.append(data["signature"]["code_sha256"])
        rows.append({**cfg, "status": data["status"], "epochs": endpoint["epoch"],
                     "lambda_signed": signed, "lambda_nonnegative": positive,
                     "projection_r2": r2, "grad_cosine": cosine,
                     "train_loss_mean": data["history"][-1]["mean_train_loss"],
                     "final_lr": endpoint["lr"], "result": str(path)})
        sources.append({"cell": list(cell), "result": str(path),
                        "sha256": file_sha256(path), "checkpoint_sha256": checkpoint_hash})
    if not rows:
        raise ValueError("No completed runs to plot")
    if any(p != protocols[0] for p in protocols) or any(c != code_hashes[0] for c in code_hashes):
        raise ValueError("Grid mixes training protocols or executed source versions")
    frame = pd.DataFrame(rows).sort_values(["depth", "width", "dropout", "seed"])
    receipt = {"n": len(frame), "expected_n": len(GRID), "protocol": protocols[0],
               "code_sha256": code_hashes[0], "statuses": frame.status.value_counts().to_dict(),
               "sources": sources, "missing_cells": [list(c) for c in GRID if c not in seen],
               "interpretation": "Endpoint ridge projection under loss-only stopping. Not a direct measurement of dropout's causal penalty. No budget-limited cells excluded."}
    return frame, receipt


def collect(root: Path, *, study: str = "dropout_loss_plateau") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require every one of the 480 predeclared configurations for the final figure."""
    paths = [root / f"d{d}_w{w}_do{p}_s{s}" / "result.json" for d, w, p, s in GRID]
    return collect_paths(paths, study=study)
