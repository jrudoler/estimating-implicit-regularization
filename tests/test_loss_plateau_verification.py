from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
from typing import Any

import pytest

from core.loss_plateau import LossPlateau, coefficient_stability
from core.loss_plateau_reporting import GRID
import core.loss_plateau_verification as verification
from core.loss_plateau_verification import verify_grid, verify_history, verify_trajectory
from core.scalar_estimation import file_sha256


def stopped_result() -> dict[str, Any]:
    controller = LossPlateau(relative_improvement=0., absolute_improvement=1e-7)
    protocol = {"stopping": controller.state_dict(), "initial_lr": .05, "max_epochs": 100000,
                "checkpoint_every": 1000}
    lr, history = .05, []
    for _ in range(10000):
        record = controller.update(.1, lr)
        if record is not None:
            history.append(record)
            lr = record["next_lr"]
            if record["action"] == "stop":
                break
    return {"protocol": protocol, "history": history, "status": "loss_plateau",
            "endpoint": {"epoch": controller.epochs, "lr": lr}}


def test_history_replay_and_false_plateau_rejection() -> None:
    result = stopped_result()
    assert verify_history(result) == len(result["history"])
    bad = deepcopy(result)
    bad["history"][3]["bad_epochs"] += 100
    with pytest.raises(ValueError, match="patience"):
        verify_history(bad)
    bad = deepcopy(result)
    bad["history"] = bad["history"][:-1]
    bad["endpoint"]["epoch"] -= 100
    with pytest.raises(ValueError, match="without a stopping"):
        verify_history(bad)
    bad["status"] = "budget_exhausted"
    bad["protocol"]["max_epochs"] = bad["endpoint"]["epoch"]
    assert verify_history(bad) == len(bad["history"])
    bad["protocol"]["max_epochs"] += 100
    with pytest.raises(ValueError, match="budget-exhaustion"):
        verify_history(bad)


def test_history_rejects_missing_blocks_and_wrong_lr() -> None:
    result = stopped_result()
    result["history"][2]["epoch"] += 100
    with pytest.raises(ValueError, match="reordered loss blocks"):
        verify_history(result)
    result = stopped_result()
    result["history"][2]["next_lr"] *= .1
    with pytest.raises(ValueError, match="next LR"):
        verify_history(result)


def trajectory_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, str]:
    result = stopped_result()
    published = tmp_path / "published" / "d3_w256_do0.5_s123"
    scratch_root = tmp_path / "scratch"
    scratch = scratch_root / published.name
    published.mkdir(parents=True)
    scratch.mkdir(parents=True)
    node = socket.gethostname().split(".")[0]
    (published / "storage.json").write_text(json.dumps({"node": node, "scratch_directory": str(scratch)}))
    result["trajectory"] = []
    stop = result["endpoint"]["epoch"]
    for epoch in [0, *range(1000, stop, 1000), stop]:
        checkpoint = scratch / f"epoch_{epoch}.ckpt"
        checkpoint.write_bytes(f"synthetic-{epoch}".encode())
        record = {"epoch": epoch, "lr": result["endpoint"]["lr"],
                  "provenance": {"epoch": epoch, "checkpoint": str(checkpoint),
                                 "checkpoint_sha256": file_sha256(checkpoint)}}
        result["trajectory"].append(record)
    result["endpoint"] = result["trajectory"][-1]
    name = Path(result["endpoint"]["provenance"]["checkpoint"]).name
    (published / name).write_bytes((scratch / name).read_bytes())
    (scratch / "latest.ckpt").write_bytes(b"synthetic-recovery-presence-only")
    return result, published, scratch_root, node


def test_trajectory_hashes_coverage_and_recovery(tmp_path: Path) -> None:
    result, published, scratch, node = trajectory_fixture(tmp_path)
    receipt = verify_trajectory(result, published, scratch_root=scratch, expected_node=node)
    assert receipt["files"] == 4 and receipt["bytes"] > 0
    incomplete = deepcopy(result)
    incomplete["trajectory"].pop(1)
    with pytest.raises(ValueError, match="trajectory epochs"):
        verify_trajectory(incomplete, published, scratch_root=scratch, expected_node=node)
    checkpoint = Path(result["trajectory"][1]["provenance"]["checkpoint"])
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Changed checkpoint"):
        verify_trajectory(result, published, scratch_root=scratch, expected_node=node)
    checkpoint.write_bytes(original)
    (scratch / published.name / "latest.ckpt").unlink()
    with pytest.raises(ValueError, match="recovery checkpoint"):
        verify_trajectory(result, published, scratch_root=scratch, expected_node=node)


def test_trajectory_refuses_wrong_node_and_short_extension(tmp_path: Path) -> None:
    result, published, scratch, node = trajectory_fixture(tmp_path)
    with pytest.raises(ValueError, match="Wrong audit/scratch node"):
        verify_trajectory(result, published, scratch_root=scratch, expected_node="not-this-node")
    result["extension"] = {"epoch": result["endpoint"]["epoch"] + 10}
    with pytest.raises(ValueError, match="shorter"):
        verify_trajectory(result, published, scratch_root=scratch, expected_node=node)


def test_full_grid_verifier_refuses_partial_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_grid(tmp_path, scratch_root=tmp_path / "scratch", expected_node="node9",
                    pilot_review=tmp_path / "review.json", task_file=tmp_path / "tasks.txt")


def test_complete_synthetic_grid_and_provenance_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the full verifier on explicitly synthetic, tiny pytest artifacts."""
    root, scratch = tmp_path / "published", tmp_path / "scratch"
    source = root / "source-v1"
    source.mkdir(parents=True)
    (source / "train.py").write_bytes(b"synthetic frozen source")
    archive = root / "source-v1.tar.gz"
    archive.write_bytes(b"synthetic archive, never used for training")
    archive_hash = file_sha256(archive)
    monkeypatch.setattr(verification, "ARCHIVE_SHA256", archive_hash)
    node = socket.gethostname().split(".")[0]
    tasks, reviews = [], []
    for depth, width, rate, seed in GRID:
        cell = depth, width, rate, seed
        tag = f"d{depth}_w{width}_do{rate}_s{seed}"
        published, directory = root / tag, scratch / tag
        published.mkdir()
        directory.mkdir(parents=True)
        result = stopped_result()
        result["protocol"].update(momentum=.9, weight_decay=0.)
        result["config"] = {"depth": depth, "width": width, "dropout": rate, "seed": seed,
                            "batchnorm": False, "batch_size": 2048}
        pilot = cell in verification.PILOTS
        result["signature"] = {"study": verification.STUDY, "config": result["config"],
                               "protocol": result["protocol"], "audit": {"enabled": pilot, "draws": 512},
                               "split_indices_sha256": verification.SPLIT_SHA256,
                               "code_sha256": {"train.py": file_sha256(source / "train.py")}}
        stop = result["endpoint"]["epoch"]
        epochs = [0, *range(1000, stop, 1000), stop]
        if pilot:
            epochs.extend([*range((stop // 1000 + 1) * 1000, 2 * stop, 1000), 2 * stop])
        result["trajectory"] = []
        result["extension"] = None
        for epoch in epochs:
            checkpoint = directory / f"epoch_{epoch}.ckpt"
            checkpoint.write_bytes(f"synthetic {tag} epoch {epoch}".encode())
            signed, norm, target_norm = 1e-8, 2., 1e-6
            cosine = signed * norm / target_norm
            record = {"epoch": epoch, "lr": result["endpoint"]["lr"],
                      "ridge": {"scale_star": signed, "scale_star_clamped": signed,
                                "grad_cosine": cosine, "grad_p_norm": norm, "target_norm": target_norm,
                                "projection_r2_clamped": cosine ** 2},
                      "provenance": {"epoch": epoch, "checkpoint": str(checkpoint),
                                     "checkpoint_sha256": file_sha256(checkpoint),
                                     "source_bundle_sha256": archive_hash,
                                     "dataset": {"n": 54000, "split_indices_sha256": verification.SPLIT_SHA256}}}
            result["trajectory"].append(record)
            if epoch == stop:
                result["endpoint"] = record
                (published / checkpoint.name).write_bytes(checkpoint.read_bytes())
            elif epoch == 2 * stop:
                result["extension"] = record
                record["stability"] = coefficient_stability(signed, signed)
                (published / checkpoint.name).write_bytes(checkpoint.read_bytes())
        (directory / "latest.ckpt").write_bytes(b"synthetic recovery")
        (published / "storage.json").write_text(json.dumps({"node": node, "scratch_directory": str(directory)}))
        result_path = published / "result.json"
        result_path.write_text(json.dumps(result))
        if pilot:
            reviews.append({"config": result["config"], "stability": result["extension"]["stability"],
                            "result_sha256": file_sha256(result_path)})
        else:
            tasks.append(tag)
    task_file = tmp_path / "tasks.txt"
    task_file.write_text("\n".join(tasks) + "\n")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"ready_for_grid_sensitivity_study": True, "pilots": reviews}))
    receipt = verify_grid(root, scratch_root=scratch, expected_node=node, pilot_review=review, task_file=task_file)
    assert receipt["full_grid_verification_passed"] is True
    assert receipt["n"] == 480 and not receipt["missing_cells"]
    assert receipt["trajectory_checkpoint_files"] == 4 * 480 + 3 * 7
    review.write_text(json.dumps({"ready_for_grid_sensitivity_study": False, "pilots": reviews}))
    with pytest.raises(ValueError, match="Pilot gate"):
        verify_grid(root, scratch_root=scratch, expected_node=node, pilot_review=review, task_file=task_file)
    (source / "train.py").write_bytes(b"changed synthetic source")
    with pytest.raises(ValueError, match="Changed executed source"):
        verify_grid(root, scratch_root=scratch, expected_node=node, pilot_review=review, task_file=task_file)
