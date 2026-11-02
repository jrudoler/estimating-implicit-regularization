"""Aggregate all paired seeds without silently dropping missing control cells."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from core.dropout_grid import SEEDS
from core.scalar_estimation import file_sha256


def validate_record(record):
    c = record["control"]
    if c["mask_draws"] != 32 or len(c["direct_signed_lambda_draws"]) != 32:
        raise ValueError("The scientific design requires 32 independent mask draws")
    for name in ("endpoint", "direct_dropout"):
        fit = c[name]
        if fit["scale_star_clamped"] != max(fit["scale_star"], 0.):
            raise ValueError("Incorrect nonnegative coefficient")
        if fit["grad_cosine"] is not None:
            expected = max(fit["grad_cosine"], 0.)**2
            if not math.isclose(fit["projection_r2_clamped"], expected, abs_tol=1e-12):
                raise ValueError("Inconsistent float64 fit diagnostics")
    difference = c["direct_dropout"]["scale_star"] - c["endpoint"]["scale_star"]
    if not math.isclose(difference, c["signed_coefficient_contamination"], abs_tol=1e-12):
        raise ValueError("Endpoint/direct decomposition failed")
    audit = record.get("optimizer_audit")
    if audit and not math.isclose(audit["closed_form"]["estimated_ridge"],
                                  audit["reference_nonnegative_lstsq"], abs_tol=1e-12):
        raise ValueError("Closed form disagrees with float64 least squares")
    evidence = record["provenance"]
    if not evidence["checkpoint_sha256"] or not evidence["dataset"]["split_indices_sha256"]:
        raise ValueError("Missing fixed-checkpoint or dataset-split provenance")


def verify_checkpoint(record):
    """Verify original or locally recovered generated checkpoint bytes."""
    evidence = record["provenance"]
    original = Path(evidence["checkpoint"])
    checkpoint = original
    if not checkpoint.exists() and "/data/generated/" in str(original):
        checkpoint = Path("data/generated") / str(original).split("/data/generated/", 1)[1]
    if not checkpoint.is_file() or file_sha256(checkpoint) != evidence["checkpoint_sha256"]:
        raise ValueError(f"Missing or changed checkpoint: {original}")
    return {"original": str(original), "verified_path": str(checkpoint.resolve()),
            "sha256": evidence["checkpoint_sha256"]}


def interval(values, seed=123):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (10000, len(values)), replace=True).mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "n": len(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    args = p.parse_args()
    rows = []
    sources = []
    verified_checkpoints = []
    seen = set()
    for path in sorted(args.results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        sources.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
        cfg = data["config"]
        key = (cfg["dropout"], cfg["seed"])
        if key in seen or cfg["depth"] != 3 or cfg["width"] != 256:
            raise ValueError(f"Duplicate or wrong architecture: {path}")
        seen.add(key)
        if sorted(r["fraction"] for r in data["trajectory"]) != [.5, .8, 1.]:
            raise ValueError(f"Missing/duplicate fractions: {path}")
        for record in data["trajectory"]:
            validate_record(record)
            verified_checkpoints.append(verify_checkpoint(record))
            c = record["control"]
            # The archived first run averaged before subtracting g_eval. For
            # zero dropout this produces only roundoff (<7e-17 gradient norm).
            # Preserve raw JSON; use the exact analytic zero in summaries so
            # bootstrap CIs cannot manufacture significance from that artifact.
            zero_control = cfg["dropout"] == 0.
            if zero_control and c["direct_dropout"]["target_norm"] > 1e-12:
                raise ValueError("Zero-dropout control failed")
            rows.append(dict(dropout=cfg["dropout"], seed=cfg["seed"], fraction=record["fraction"],
                             endpoint=c["endpoint"]["scale_star_clamped"],
                             endpoint_signed=c["endpoint"]["scale_star"],
                             direct=0. if zero_control else c["direct_dropout"]["scale_star_clamped"],
                             direct_signed=0. if zero_control else c["direct_dropout"]["scale_star"],
                             direct_r2=None if zero_control else c["direct_dropout"]["projection_r2_clamped"],
                             direct_signed_r2=None if zero_control else c["direct_dropout"]["projection_r2"],
                             direct_mc_low=c["direct_signed_lambda_mc_ci95"][0],
                             direct_mc_high=c["direct_signed_lambda_mc_ci95"][1],
                             train_norm_sq=c["train_gradient_norm_sq_unbiased"],
                             train_norm_sq_low=c["train_gradient_norm_sq_ci95"][0],
                             train_norm_sq_high=c["train_gradient_norm_sq_ci95"][1]))
    if seen != {(p, s) for p in (0., .3, .5) for s in SEEDS}:
        raise ValueError("Expected all 30 control runs; no incomplete-seed summary")
    df = pd.DataFrame(rows)
    summaries = []
    for (rate, fraction), group in df.groupby(["dropout", "fraction"]):
        summaries.append({"dropout": rate, "fraction": fraction,
                          "endpoint": interval(group.endpoint), "direct": interval(group.direct),
                          "signed_contamination": interval(group.direct_signed - group.endpoint_signed),
                          "direct_signed": interval(group.direct_signed),
                          # Exact zero-dropout contributions have no defined
                          # relative fit. Never summarize roundoff-only angles.
                          "direct_r2": interval(group.direct_r2) if rate else None,
                          "direct_signed_r2": interval(group.direct_signed_r2) if rate else None,
                          "direct_mc_intervals_include_zero": int(((group.direct_mc_low <= 0) & (group.direct_mc_high >= 0)).sum()),
                          "stationarity_inconclusive": int(((group.train_norm_sq_low <= 0) & (group.train_norm_sq_high >= 0)).sum())})
    contrasts = []
    for fraction in (.5, .8, 1.):
        for field in ("endpoint", "direct", "direct_signed"):
            wide = df[df.fraction == fraction].pivot(index="seed", columns="dropout", values=field)
            for rate in (.3, .5):
                contrasts.append({"fraction": fraction, "metric": field, "dropout": rate,
                                  "paired_difference_from_zero": interval(wide[rate] - wide[0.])})
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"n_runs": len(seen), "by_condition": summaries,
                                       "sources": sources,
                                       "verified_checkpoints": verified_checkpoints,
                                       "zero_control": "Raw roundoff checked below 1e-12 gradient norm; analytic zero used in summary coefficients, relative fit undefined.",
                                       "paired_contrasts": contrasts,
                                       "uncertainty": "Seed CIs bootstrap paired runs; per-checkpoint MC intervals are separate. Norm intervals crossing zero do not prove stationarity."}, indent=2, allow_nan=False))
    df.to_csv(args.table, index=False)


if __name__ == "__main__":
    main()
