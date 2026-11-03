from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core.loss_plateau_reporting import collect, collect_paths
from core.scalar_estimation import file_sha256


def make_run(root: Path, *, rate: float = .5, seed: int = 123, signed: float = -1e-9) -> Path:
    directory = root / f"d3_w256_do{rate}_s{seed}"
    directory.mkdir()
    checkpoint = directory / "endpoint.ckpt"
    checkpoint.write_bytes(b"synthetic-test-checkpoint")
    cosine = signed * 2. / 1e-6
    record = {"config": {"depth": 3, "width": 256, "dropout": rate, "seed": seed},
              "signature": {"study": "dropout_loss_plateau_absolute", "code_sha256": {"code": "same"}},
              "status": "loss_plateau", "protocol": {"test_only": True},
              "history": [{"mean_train_loss": .1}],
              "endpoint": {"epoch": 2000, "lr": .0005,
                  "ridge": {"scale_star": signed, "scale_star_clamped": max(0., signed),
                            "grad_cosine": cosine, "grad_p_norm": 2., "target_norm": 1e-6,
                            "projection_r2_clamped": max(cosine, 0.) ** 2},
                  "provenance": {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}}}
    result = directory / "result.json"
    result.write_text(json.dumps(record))
    return result


def test_partial_receipt_preserves_missing_and_zero(tmp_path: Path) -> None:
    path = make_run(tmp_path)
    frame, receipt = collect_paths([path], study="dropout_loss_plateau_absolute")
    assert receipt["n"] == 1
    assert len(receipt["missing_cells"]) == 479
    assert frame.iloc[0].lambda_nonnegative == 0.
    assert frame.iloc[0].lambda_signed < 0.
    with pytest.raises(FileNotFoundError):
        collect(tmp_path, study="dropout_loss_plateau_absolute")
    with pytest.raises(ValueError, match="Duplicate"):
        collect_paths([path, path], study="dropout_loss_plateau_absolute")
    with pytest.raises(ValueError, match="No completed"):
        collect_paths([], study="dropout_loss_plateau_absolute")
    with pytest.raises(ValueError, match="different studies"):
        collect_paths([path], study="other")
    (path.parent / "endpoint.ckpt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed checkpoint"):
        collect_paths([path], study="dropout_loss_plateau_absolute")


def test_partial_plot_does_not_connect_incomplete_cells(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "analysis/plot_dropout_loss_plateau_interim/run.py"
    spec = importlib.util.spec_from_file_location("interim_figure_test", script)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    frame, _ = collect_paths([make_run(tmp_path)], study="dropout_loss_plateau_absolute")
    figure = plot.build_figure(frame, snapshot="synthetic test")
    assert len(figure.axes) == 7
    assert all(line.get_ydata().size == 8 for ax in figure.axes[:6] for line in ax.lines)
    assert all(all(value != value for value in line.get_ydata()) for ax in figure.axes[:6] for line in ax.lines)
    assert any(text.get_text() == "1/10" for text in figure.axes[1].texts)
    assert any("No completed" in text.get_text() for text in figure.axes[0].texts)
    figure.savefig(tmp_path / "synthetic-interim.pdf")
    plot.plt.close(figure)
    assert (tmp_path / "synthetic-interim.pdf").stat().st_size > 1000
