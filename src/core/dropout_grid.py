"""Coverage and provenance checks for the fixed Figure-5 design."""
from itertools import product
import json
import math
from pathlib import Path

WIDTHS = (128, 256, 512)
DEPTHS = (3, 5)
DROPOUTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
SEEDS = (123, 42, 666, 314, 17, 111, 325, 643, 432, 51)
EXPECTED = set(product(DEPTHS, WIDTHS, DROPOUTS, SEEDS))


def load_grid(directory, *, diagnostic_tolerance=1e-5):
    records = {}
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text())
        cfg = record["config"]
        key = tuple(cfg[k] for k in ("depth", "width", "dropout", "seed"))
        if key in records:
            raise ValueError(f"Duplicate grid cell {key}: {path}")
        if key not in EXPECTED:
            raise ValueError(f"Unexpected grid cell {key}: {path}")
        fit = record["closed_form"]["ridge"]
        signed, nonnegative = fit["scale_star"], fit["scale_star_clamped"]
        if signed is None or not math.isfinite(signed) or nonnegative != max(signed, 0):
            raise ValueError(f"Invalid ridge estimate: {path}")
        for name in ("residual_ratio", "residual_ratio_clamped", "grad_cosine"):
            value = fit.get(name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Nonfinite {name}: {path}")
        resid, cosine = fit.get("residual_ratio_clamped"), fit.get("grad_cosine")
        if resid is not None and not 0 <= resid <= 1 + 1e-6:
            raise ValueError(f"Nonoptimal nonnegative fit: {path}")
        if cosine is not None and (abs(cosine) > 1 + 1e-6 or resid is None or
                abs(resid**2 - (1 - max(cosine, 0)**2)) > diagnostic_tolerance):
            raise ValueError(f"Inconsistent fit diagnostics: {path}")
        records[key] = (path, record)
    missing = EXPECTED - records.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} of 480 corrected cells under {directory}; no legacy fallback")
    return records
