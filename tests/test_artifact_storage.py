import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.artifact_storage import home_quota, publish_endpoint
from core.scalar_estimation import file_sha256


def test_home_quota() -> None:
    assert home_quota("head:/home 45000000 52428800 53477376 123 0 0") == (45000000 * 1024, 52428800 * 1024)
    with pytest.raises(ValueError):
        home_quota("quota RPC unavailable")


def test_only_final_files_are_published_and_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, target = tmp_path / "scratch", tmp_path / "home"
    source.mkdir()
    checkpoint = source / "endpoint.ckpt"
    checkpoint.write_bytes(b"test weights")
    (source / "latest.ckpt").write_bytes(b"not published")
    (source / "trajectory_epoch.ckpt").write_bytes(b"not published")
    result = {"endpoint": {"provenance": {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}},
              "extension": None}
    (source / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr("core.artifact_storage.subprocess.run", lambda *a, **kw: SimpleNamespace(stdout="head:/home 45000000 52428800 53477376 1 0 0"))
    receipt = publish_endpoint(source, target, node="test-node", lock_path=tmp_path / ".lock")
    assert set(p.name for p in target.iterdir()) == {"endpoint.ckpt", "result.json", "storage.json"}
    assert receipt["node"] == "test-node"
    assert file_sha256(target / "endpoint.ckpt") == file_sha256(checkpoint)
    monkeypatch.setattr("core.artifact_storage.subprocess.run", lambda *a, **kw: SimpleNamespace(stdout="head:/home 51000000 52428800 53477376 1 0 0"))
    with pytest.raises(RuntimeError, match="quota guard"):
        publish_endpoint(source, tmp_path / "full-home", node="test-node", lock_path=tmp_path / ".lock")
    assert not (tmp_path / "full-home/result.json").exists()
    assert checkpoint.exists()
