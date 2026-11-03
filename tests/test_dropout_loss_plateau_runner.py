"""Small CPU integration checks, including interrupted extension recovery."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


class TinySplit:
    def __init__(self) -> None:
        self.indices = list(range(9))
        self.dataset = SimpleNamespace(data=torch.arange(9 * 28 * 28).reshape(9, 28, 28) % 256,
                                       targets=torch.arange(9) % 3)

    def __len__(self) -> int:
        return len(self.indices)


class TinyData:
    def __init__(self, **kwargs: object) -> None:
        self._train_dataset = TinySplit()

    def setup(self, stage: str) -> None:
        pass


def runner():
    path = Path(__file__).resolve().parents[1] / "analysis/dropout_loss_plateau/run.py"
    spec = importlib.util.spec_from_file_location("plateau_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(path: Path, *, audit: bool = False, max_epochs: int = 8) -> list[str]:
    result = ["run.py", "--dropout", ".3", "--seed", "123", "--width", "4", "--depth", "1",
              "--output-dir", str(path), "--device", "cpu", "--max-epochs", str(max_epochs),
              "--window", "1", "--patience", "3", "--lr-patience", "2", "--min-lr", ".05",
              "--checkpoint-every", "2", "--audit-min-epochs", "2", "--audit-draws", "4"]
    return result + (["--audit"] if audit else [])


def test_runner_extension_and_recovery_are_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = runner()
    monkeypatch.setattr(module, "MNISTLightningDataModule", TinyData)
    actual_train = module.train_epoch

    def constant_loss(*values: object) -> float:
        actual_train(*values)
        return 1.

    monkeypatch.setattr(module, "train_epoch", constant_loss)
    uninterrupted = tmp_path / "uninterrupted"
    monkeypatch.setattr(sys, "argv", args(uninterrupted, audit=True))
    module.main()
    first = json.loads((uninterrupted / "result.json").read_text())
    assert first["status"] == "loss_plateau"
    assert first["endpoint"]["epoch"] == 4
    assert first["extension"]["epoch"] == 8
    assert len(first["history"]) == 4
    assert first["endpoint"]["provenance"]["model_sha256"] != first["extension"]["provenance"]["model_sha256"]

    interrupted = tmp_path / "interrupted"
    calls = 0

    def fail_after_saved_epoch(*values: object) -> float:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("simulated preemption")
        return constant_loss(*values)

    monkeypatch.setattr(module, "train_epoch", fail_after_saved_epoch)
    monkeypatch.setattr(sys, "argv", args(interrupted, audit=True))
    with pytest.raises(RuntimeError, match="preemption"):
        module.main()
    assert not (interrupted / "result.json").exists()
    monkeypatch.setattr(module, "train_epoch", constant_loss)
    module.main()
    second = json.loads((interrupted / "result.json").read_text())
    assert first["endpoint"]["ridge"] == second["endpoint"]["ridge"]
    assert first["extension"]["control"] == second["extension"]["control"]
    assert first["extension"]["provenance"]["model_sha256"] == second["extension"]["provenance"]["model_sha256"]
    # Completed output is preserved, not retrained.
    before = (interrupted / "result.json").read_bytes()
    module.main()
    assert (interrupted / "result.json").read_bytes() == before


def test_budget_exhaustion_is_not_relabelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = runner()
    monkeypatch.setattr(module, "MNISTLightningDataModule", TinyData)
    monkeypatch.setattr(sys, "argv", args(tmp_path, max_epochs=2))
    module.main()
    data = json.loads((tmp_path / "result.json").read_text())
    assert data["status"] == "budget_exhausted"
    assert data["extension"] is None
    monkeypatch.setattr(sys, "argv", args(tmp_path, max_epochs=3))
    with pytest.raises(ValueError, match="overwrite"):
        module.main()
