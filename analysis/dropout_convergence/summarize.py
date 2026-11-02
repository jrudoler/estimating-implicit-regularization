"""Summarize the complete continuation design without hiding unconverged runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from core.dropout_grid import SEEDS


def summarize(paths):
    records, hashes = [], {}
    for directory in paths:
        path = directory / "result.json"
        raw = path.read_bytes()
        result = json.loads(raw)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        final = result["trajectory"][-1]
        evaluation = final.get("confirmation", final)
        control = evaluation["control"]
        assessment = evaluation["assessment"]
        if result["status"] == "converged_to_declared_tolerance" and (
            "confirmation" not in final or not assessment["passes"]
        ):
            raise ValueError(f"Convergence claim lacks passing independent confirmation: {path}")
        records.append({**result["config"], "status": result["status"],
                        "additional_epochs": final["additional_epochs"],
                        "initial_endpoint_signed": result["trajectory"][0]["control"]["endpoint"]["scale_star"],
                        "endpoint_signed": control["endpoint"]["scale_star"],
                        "endpoint_nonnegative": control["endpoint"]["estimated_ridge"],
                        "direct_signed": control["direct_dropout"]["scale_star"],
                        "direct_nonnegative": control["direct_dropout"]["estimated_ridge"],
                        "direct_ridge_r2_nonnegative": control["direct_dropout"]["projection_r2_clamped"],
                        "coefficient_contamination": control["signed_coefficient_contamination"],
                        "control": control, "assessment": assessment,
                        "checkpoint_provenance": final["provenance"]})
    expected = {(3, 256, p, s) for p in (0., .3, .5) for s in SEEDS}
    keys = [(r["depth"], r["width"], r["dropout"], r["seed"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Require exactly all 30 expected endpoints, with no duplicates")
    rng = np.random.default_rng(123)
    conditions = []
    for rate in (0., .3, .5):
        rows = sorted((r for r in records if r["dropout"] == rate), key=lambda r: r["seed"])
        indexes = rng.integers(0, 10, size=(10000, 10))
        summary = {"dropout": rate,
                   "converged_count": sum(r["status"] == "converged_to_declared_tolerance" for r in rows),
                   "run_count": len(rows)}
        for field in ("endpoint_signed", "endpoint_nonnegative", "direct_signed",
                      "direct_nonnegative", "coefficient_contamination"):
            values = np.array([r[field] for r in rows])
            summary[field] = {"mean": float(values.mean()),
                              "seed_bootstrap_ci95": np.quantile(values[indexes].mean(1), [.025, .975]).tolist()}
        conditions.append(summary)
    return {"source_sha256": hashes, "conditions": conditions, "records": records,
            "interpretation": "All seeds retained, including budget-exhausted runs. Seed intervals are separate from checkpoint mask uncertainty."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
