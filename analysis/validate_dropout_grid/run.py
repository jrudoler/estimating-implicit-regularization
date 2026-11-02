"""Validate corrected grid coverage and retain explicit historical provenance."""
import argparse
import json
from pathlib import Path

from core.dropout_grid import load_grid
from core.scalar_estimation import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--historical-source", help="Explicit source label for the old checkpoint-free reruns")
    args = p.parse_args()
    # Historical float32 vector norms over up to 1.5M parameters exhibit
    # discrepancies up to 5.04e-4 in this identity. Keep that limitation explicit;
    # newly computed float64 statistics must satisfy the strict tolerance.
    tolerance = 1e-3 if args.historical_source else 1e-5
    records = load_grid(args.results_dir, diagnostic_tolerance=tolerance)
    sources = []
    for key, (path, record) in sorted(records.items()):
        evidence = record.get("provenance")
        if not evidence and not args.historical_source:
            raise ValueError(f"Missing checkpoint provenance in {path}; supply an explicit historical source label")
        sources.append({"cell": list(key), "file": str(path.resolve()), "sha256": file_sha256(path),
                        "provenance": evidence or {
                            "endpoint_kind": "historical_retrained_endpoint",
                            "source": args.historical_source,
                            "checkpoint_sha256": None, "versions": None,
                            "accumulation_dtype": "float32",
                            "limitation": "Original corrected sweep disabled checkpointing; not same-weight comparison to legacy sweep."}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    discrepancies = [abs(r["closed_form"]["ridge"]["residual_ratio_clamped"]**2 -
                         (1 - max(r["closed_form"]["ridge"]["grad_cosine"], 0)**2))
                     for _, r in records.values() if r["closed_form"]["ridge"]["grad_cosine"] is not None]
    args.output.write_text(json.dumps({"schema_version": 2, "n_runs": len(sources),
                                      "diagnostic_tolerance": tolerance,
                                      "max_diagnostic_identity_error": max(discrepancies, default=0.),
                                      "historical_source": args.historical_source,
                                      "constraint": "nonnegative", "sources": sources}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
