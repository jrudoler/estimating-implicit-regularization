"""Report fixed pilot extension checks without changing stopping decisions."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core.dropout_continuation import atomic_json
from core.loss_plateau import coefficient_stability
from core.scalar_estimation import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = []
    expected = {(3, 256, p, 123) for p in (0., .3, .5)}
    if args.expanded:
        expected |= {(d, w, .5, 123) for d, w in ((3, 128), (3, 512), (5, 128), (5, 512))}
    seen = set()
    for path in args.runs:
        result = json.loads(path.read_text())
        cfg = result["config"]
        key = tuple(cfg[k] for k in ("depth", "width", "dropout", "seed"))
        if key not in expected or key in seen:
            raise ValueError("Expected the declared pilot configurations")
        seen.add(key)
        first, last = result["endpoint"], result["extension"]
        if last is None or last["epoch"] < 2 * first["epoch"]:
            raise ValueError("Missing extension of at least one training duration")
        for record in (first, last):
            checkpoint = path.parent / Path(record["provenance"]["checkpoint"]).name
            if file_sha256(checkpoint) != record["provenance"]["checkpoint_sha256"]:
                raise ValueError("Pilot checkpoint hash mismatch")
        stability = coefficient_stability(first["ridge"]["scale_star"], last["ridge"]["scale_star"])
        if stability != last["stability"]:
            raise ValueError("Stored pilot stability does not reproduce")
        rows.append({"config": cfg, "status": result["status"], "stop_epoch": first["epoch"],
                     "extended_epoch": last["epoch"], "stability": stability,
                     "endpoint_direct_dropout": first["control"]["direct_dropout"]["scale_star"],
                     "endpoint_direct_ci95": first["control"]["direct_signed_lambda_mc_ci95"],
                     "extended_direct_dropout": last["control"]["direct_dropout"]["scale_star"],
                     "extended_direct_ci95": last["control"]["direct_signed_lambda_mc_ci95"],
                     "result": str(path), "result_sha256": file_sha256(path)})
    if seen != expected:
        raise ValueError("Pilot coverage mismatch")
    ready = all(row["status"] == "loss_plateau" and row["stability"]["passes"] for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {"ready_for_grid_sensitivity_study": ready, "pilots": rows,
                "interpretation": "Practical coefficient robustness only, not a proof of stationarity. Direct-dropout comparisons remain separate and must not be hidden by a stable endpoint fit."})
    logging.info("Pilot ready for grid sensitivity study: %s", ready)
    for row in rows:
        logging.info("dropout=%s status=%s stop=%s extended=%s stability=%s", row["config"]["dropout"],
                     row["status"], row["stop_epoch"], row["extended_epoch"], row["stability"])


if __name__ == "__main__":
    main()
