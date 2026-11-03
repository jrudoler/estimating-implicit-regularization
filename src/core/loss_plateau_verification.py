"""Independent replay and retained-checkpoint checks for the absolute-loss study."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import socket
from typing import Any

from core.loss_plateau import coefficient_stability
from core.loss_plateau_reporting import GRID, collect
from core.scalar_estimation import file_sha256

LOG = logging.getLogger(__name__)
STUDY = "dropout_loss_plateau_absolute"
ARCHIVE_SHA256 = "15fb6023d4d207bf5eafb51e68998958528767a4c74790a57ae7c107ac18e278"
SPLIT_SHA256 = "c485001e12d02300cb3dc8ed5f639f82684102ed59103f4160365219dc89fa8f"
PILOTS = {(3, 256, p, 123) for p in (0., .3, .5)} | {
    (d, w, .5, 123) for d, w in ((3, 128), (3, 512), (5, 128), (5, 512))}


def require(condition: bool, message: str) -> None:
    """Use explicit failures: checks must remain enabled under python -O."""
    if not condition:
        raise ValueError(message)


def verify_history(result: dict[str, Any]) -> int:
    """Replay saved loss blocks without calling the training controller."""
    protocol, history = result["protocol"], result["history"]
    settings = protocol["stopping"]
    require(bool(history), "Missing training-loss history")
    best, bad, lr = None, 0, protocol["initial_lr"]
    action = "continue"
    for index, record in enumerate(history, start=1):
        loss = record["mean_train_loss"]
        require(math.isfinite(loss) and loss >= 0, "Invalid training loss")
        require(record["epoch"] == index * settings["window"], "Missing or reordered loss blocks")
        require(math.isclose(record["lr"], lr, rel_tol=1e-14), "Recorded LR does not replay")
        if best is None or loss < min(best * (1 - settings["relative_improvement"]),
                                      best - settings["absolute_improvement"]):
            best, bad = loss, 0
        else:
            bad += settings["window"]
        require(record["best_mean_loss"] == best and record["bad_epochs"] == bad,
                "Recorded best loss or patience does not replay")
        action, next_lr = "continue", lr
        if bad >= settings["lr_patience"] and lr > settings["min_lr"] * (1 + 1e-12):
            action, next_lr = "reduce_lr", max(settings["min_lr"], lr * settings["factor"])
            best, bad = None, 0
        elif bad >= settings["patience"]:
            action = "stop"
            require(index == len(history), "Training history continues after its stopping decision")
        require(record["action"] == action, "Recorded action does not replay")
        require(math.isclose(record["next_lr"], next_lr, rel_tol=1e-14), "Recorded next LR does not replay")
        lr = next_lr
    endpoint = result["endpoint"]
    require(endpoint["epoch"] == history[-1]["epoch"], "Endpoint/history epoch mismatch")
    require(math.isclose(endpoint["lr"], lr, rel_tol=1e-14), "Endpoint/history LR mismatch")
    if result["status"] == "loss_plateau":
        require(action == "stop", "Plateau status without a stopping decision")
    else:
        require(result["status"] == "budget_exhausted" and action != "stop"
                and endpoint["epoch"] == protocol["max_epochs"], "Invalid budget-exhaustion status")
    return len(history)


def verify_trajectory(result: dict[str, Any], published: Path, *, scratch_root: Path,
                      expected_node: str) -> dict[str, int]:
    """Verify all retained trajectory files; recovery is checked for presence only."""
    storage = json.loads((published / "storage.json").read_text())
    require(storage["node"].split(".")[0] == expected_node
            and socket.gethostname().split(".")[0] == expected_node, "Wrong audit/scratch node")
    directory = Path(storage["scratch_directory"])
    require(directory == scratch_root / published.name, "Unexpected scratch directory")
    step, stop = result["protocol"]["checkpoint_every"], result["endpoint"]["epoch"]
    expected = [0, *range(step, stop, step), stop]
    extension = result.get("extension")
    if extension is not None:
        last = extension["epoch"]
        require(last >= 2 * stop, "Pilot extension is shorter than one additional training duration")
        expected.extend([*range((stop // step + 1) * step, last, step), last])
    require([r["epoch"] for r in result["trajectory"]] == expected, "Missing/reordered trajectory epochs")
    size = 0
    for record in result["trajectory"]:
        provenance = record["provenance"]
        checkpoint = Path(provenance["checkpoint"])
        require(checkpoint.parent == directory, "Trajectory checkpoint outside its recorded directory")
        require(provenance["epoch"] == record["epoch"], "Trajectory provenance epoch mismatch")
        require(file_sha256(checkpoint) == provenance["checkpoint_sha256"], f"Changed checkpoint: {checkpoint}")
        size += checkpoint.stat().st_size
    for record in [result["endpoint"]] + ([extension] if extension is not None else []):
        require(record in result["trajectory"], "Endpoint missing from trajectory")
        checkpoint = published / Path(record["provenance"]["checkpoint"]).name
        require(file_sha256(checkpoint) == record["provenance"]["checkpoint_sha256"],
                f"Changed published endpoint: {checkpoint}")
    require((directory / "latest.ckpt").is_file(), "Missing latest recovery checkpoint")
    return {"files": len(expected), "bytes": size}


def verify_grid(root: Path, *, scratch_root: Path, expected_node: str,
                pilot_review: Path, task_file: Path) -> dict[str, Any]:
    """Require the full fixed 480-model grid, including any budget-limited runs."""
    _, receipt = collect(root, study=STUDY)  # Missing cells fail; no partial mode.
    require(receipt["n"] == 480 and not receipt["missing_cells"], "Incomplete grid")
    archive = root / "source-v1.tar.gz"
    require(file_sha256(archive) == ARCHIVE_SHA256, "Changed frozen source archive")
    for name, digest in receipt["code_sha256"].items():
        require(file_sha256(root / "source-v1" / name) == digest, f"Changed executed source: {name}")
    protocol = receipt["protocol"]
    for key, value in {"max_epochs": 100000, "initial_lr": .05, "momentum": .9,
                       "weight_decay": 0., "checkpoint_every": 1000}.items():
        require(protocol[key] == value, f"Changed declared protocol: {key}")
    for key, value in {"window": 100, "relative_improvement": 0., "absolute_improvement": 1e-7,
                       "patience": 1000, "lr_patience": 500, "factor": .1, "min_lr": .0005,
                       "losses": [], "best": None, "bad_epochs": 0, "reductions": 0, "epochs": 0}.items():
        require(protocol["stopping"][key] == value, f"Changed declared stopping rule: {key}")
    tasks = task_file.read_text().splitlines()
    expected_tasks = {f"d{d}_w{w}_do{p}_s{s}" for d, w, p, s in GRID if (d, w, p, s) not in PILOTS}
    require(len(tasks) == 473 and set(tasks) == expected_tasks, "Task-list coverage mismatch")
    review = json.loads(pilot_review.read_text())
    require(review["ready_for_grid_sensitivity_study"] is True, "Pilot gate did not pass")
    review_rows = {tuple(r["config"][k] for k in ("depth", "width", "dropout", "seed")): r
                   for r in review["pilots"]}
    require(len(review["pilots"]) == 7 and set(review_rows) == PILOTS, "Pilot-review coverage mismatch")
    blocks = files = size = 0
    for index, source in enumerate(receipt["sources"], start=1):
        path = Path(source["result"])
        result = json.loads(path.read_text())
        cell = tuple(source["cell"])
        signature = result["signature"]
        require(signature["config"] == result["config"] and signature["protocol"] == protocol,
                f"Signature mismatch: {path}")
        require(result["config"]["batch_size"] == 2048 and result["config"]["batchnorm"] is False,
                f"Changed model/batch configuration: {path}")
        require(signature["split_indices_sha256"] == SPLIT_SHA256, f"Changed training split: {path}")
        require(signature["audit"]["enabled"] == (cell in PILOTS), f"Incorrect pilot assignment: {path}")
        if cell in PILOTS:
            extension = result["extension"]
            require(result["status"] == "loss_plateau" and extension is not None, f"Incomplete pilot: {path}")
            stability = coefficient_stability(result["endpoint"]["ridge"]["scale_star"],
                                              extension["ridge"]["scale_star"])
            require(stability["passes"] and stability == extension["stability"]
                    and stability == review_rows[cell]["stability"], f"Pilot stability mismatch: {path}")
            require(review_rows[cell]["result_sha256"] == source["sha256"], f"Changed reviewed pilot: {path}")
            require(signature["audit"]["draws"] == 512, f"Changed pilot uncertainty budget: {path}")
        else:
            require(result.get("extension") is None, f"Undeclared extension: {path}")
        for record in result["trajectory"]:
            provenance = record["provenance"]
            require(provenance["source_bundle_sha256"] == ARCHIVE_SHA256, f"Wrong source bundle: {path}")
            require(provenance["dataset"]["n"] == 54000
                    and provenance["dataset"]["split_indices_sha256"] == SPLIT_SHA256,
                    f"Wrong diagnostic dataset: {path}")
        blocks += verify_history(result)
        retained = verify_trajectory(result, path.parent, scratch_root=scratch_root, expected_node=expected_node)
        files += retained["files"]
        size += retained["bytes"]
        if index % 40 == 0:
            LOG.info("Verified %d/480 runs; %d retained checkpoints", index, files)
    receipt.update({"verified_utc": datetime.now(timezone.utc).isoformat(), "node": socket.gethostname(),
                    "verification_source_sha256": file_sha256(Path(__file__)),
                    "verification_dependencies_sha256": {
                        name: file_sha256(Path(__file__).with_name(name)) for name in
                        ("loss_plateau.py", "loss_plateau_reporting.py", "scalar_estimation.py", "dropout_continuation.py")},
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "pilot_review_sha256": file_sha256(pilot_review), "task_file_sha256": file_sha256(task_file),
                    "saved_loss_blocks_replayed": blocks, "trajectory_checkpoint_files": files,
                    "trajectory_checkpoint_bytes": size, "full_grid_verification_passed": True,
                    "qualification": "Verifies declared protocol, stopping replay, pilot robustness and file retention. Recovery files checked for presence; not a proof of exact stationarity or a new causal dropout estimate."})
    return receipt
