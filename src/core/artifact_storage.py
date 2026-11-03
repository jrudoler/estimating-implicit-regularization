"""Quota-guarded publication of final artifacts from node-local scratch."""
from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any

from core.dropout_continuation import atomic_json
from core.scalar_estimation import file_sha256

LOG = logging.getLogger(__name__)


def home_quota(text: str) -> tuple[int, int]:
    """Parse quota -w's 1-KiB block counts; fail closed if home is absent."""
    for line in text.splitlines():
        columns = line.split()
        if columns and columns[0].endswith(":/home") and len(columns) >= 4:
            used, soft, hard = (int(value.rstrip("*")) * 1024 for value in columns[1:4])
            return used, soft or hard
    raise ValueError("Cannot verify /home quota; leaving artifacts on scratch")


def publish_endpoint(source: Path, destination: Path, *, node: str,
                     lock_path: Path, max_home_gib: float = 48.) -> dict[str, Any]:
    """Keep full trajectories on scratch, publish only final endpoint(s)/JSON.

    A shared lock serializes our quota checks and writes across array tasks.
    Existing mismatching artifacts are never overwritten. The result JSON is
    copied last, so it remains a completion marker for downstream workflow rules.
    """
    if source.resolve() == destination.resolve():
        return {"mode": "already_at_destination"}
    result = json.loads((source / "result.json").read_text())
    records = [result["endpoint"]]
    if result.get("extension") is not None:
        records.append(result["extension"])
    checkpoints: list[Path] = []
    for record in records:
        path = source / Path(record["provenance"]["checkpoint"]).name
        if file_sha256(path) != record["provenance"]["checkpoint_sha256"]:
            raise ValueError(f"Scratch checkpoint hash mismatch: {path}")
        checkpoints.append(path)
    files = checkpoints + [source / "result.json"]
    destination.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        response = subprocess.run(["quota", "-w"], text=True, capture_output=True, timeout=30)
        used, limit = home_quota(response.stdout)
        # Keep two GiB below the quota even if a future account limit is lower.
        ceiling = min(int(max_home_gib * 1024**3), limit - 2 * 1024**3)
        additional = sum(path.stat().st_size for path in files if not (destination / path.name).exists())
        # Include another 256 MiB allowance for concurrent non-experiment writes.
        if used + additional + 256 * 1024**2 > ceiling:
            raise RuntimeError(f"Home quota guard: used={used / 1024**3:.3f} GiB, "
                               f"new={additional / 1024**3:.3f} GiB, ceiling={ceiling / 1024**3:.3f} GiB. "
                               f"Complete results retained at {node}:{source}.")
        receipt = {"node": node, "scratch_directory": str(source.resolve()),
                   "home_used_bytes_before_copy": used, "home_quota_bytes": limit,
                   "home_ceiling_bytes": ceiling, "additional_bytes": additional,
                   "policy": "all intermediate/recovery checkpoints remain on scratch; only endpoint(s) and result JSON copied home",
                   "retained_home_files": [path.name for path in files]}
        for path in files:
            target = destination / path.name
            expected_hash = file_sha256(path)
            if target.exists():
                if file_sha256(target) != expected_hash:
                    raise ValueError(f"Refusing to overwrite different published artifact: {target}")
                continue
            if path.name == "result.json":
                atomic_json(destination / "storage.json", receipt)
            pending = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(path, pending)
            if file_sha256(pending) != expected_hash:
                raise RuntimeError(f"Copied artifact hash mismatch: {pending}")
            pending.replace(target)
        LOG.info("Published %s; %.3f GiB home before copy, %.3f MiB added", destination,
                 used / 1024**3, additional / 1024**2)
        return receipt
