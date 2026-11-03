from __future__ import annotations

import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import pytest

from core.scalar_estimation import file_sha256


def module():
    path = Path(__file__).resolve().parents[1] / "analysis/plot_dropout_loss_plateau/run.py"
    spec = importlib.util.spec_from_file_location("plateau_figure_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_full_coverage_receipt_and_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plot = module()
    keys = list(product(plot.DEPTHS, plot.WIDTHS, plot.RATES, plot.SEEDS))
    for depth, width, rate, seed in keys:
        directory = tmp_path / f"d{depth}_w{width}_do{rate}_s{seed}"
        directory.mkdir()
        checkpoint = directory / "endpoint.ckpt"
        checkpoint.write_bytes(b"synthetic-test-checkpoint")
        # Algebraically consistent synthetic projections, only in pytest temp files.
        cosine, target_norm, basis_norm = .2, (1 + rate + seed / 1000) * 1e-5, 2.
        signed = cosine * target_norm / basis_norm
        record = {"config": {"depth": depth, "width": width, "dropout": rate, "seed": seed},
                  "signature": {"study": "dropout_loss_plateau", "code_sha256": {"code": "same"}},
                  "status": "loss_plateau", "protocol": {"test_only": True},
                  "history": [{"mean_train_loss": .1}],
                  "endpoint": {"epoch": 2000, "lr": .0005,
                    "ridge": {"scale_star": signed, "scale_star_clamped": signed,
                              "grad_cosine": cosine, "grad_p_norm": basis_norm,
                              "target_norm": target_norm, "projection_r2_clamped": cosine ** 2},
                    "provenance": {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}}}
        (directory / "result.json").write_text(json.dumps(record))
    frame, receipt = plot.collect(tmp_path)
    assert len(frame) == receipt["n"] == 480
    figure = plot.build_figure(frame)
    figure.savefig(tmp_path / "synthetic-test.pdf")
    plot.plt.close(figure)
    assert (tmp_path / "synthetic-test.pdf").stat().st_size > 1000
    # A separate log-scale presentation must preserve zeros and capped runs.
    frame.loc[frame.index[0], "lambda_nonnegative"] = 0.
    frame.loc[frame.index[0], "status"] = "budget_exhausted"
    figure = plot.build_figure(frame, yscale="symlog")
    assert all(ax.get_yscale() == "symlog" for ax in figure.axes[:6])
    points = [collection.get_offsets() for ax in figure.axes[:6] for collection in ax.collections]
    assert sum(len(offsets) for offsets in points) == 480
    assert any(float(offsets[0, 1]) == 0. for offsets in points)
    assert all(ax.get_ylim()[0] < 0 for ax in figure.axes[:6])
    figure.savefig(tmp_path / "synthetic-test-symlog.pdf")
    plot.plt.close(figure)
    assert (tmp_path / "synthetic-test-symlog.pdf").stat().st_size > 1000
    with pytest.raises(ValueError, match="yscale"):
        plot.build_figure(frame, yscale="log")
    # Exercise the real CLI, including exact figure/table/source provenance.
    output_figure = tmp_path / "cli-symlog.pdf"
    output_table = tmp_path / "cli-symlog.csv"
    output_receipt = tmp_path / "cli-symlog.json"
    style = Path(__file__).resolve().parents[1] / "clean_fig.mplstyle"
    monkeypatch.setattr(sys, "argv", ["run.py", "--results-dir", str(tmp_path),
        "--figure", str(output_figure), "--table", str(output_table),
        "--receipt", str(output_receipt), "--style", str(style), "--yscale", "symlog"])
    plot.main()
    saved = json.loads(output_receipt.read_text())
    assert saved["n"] == 480 and saved["missing_cells"] == []
    assert saved["yscale"] == "symlog" and saved["symlog_linthresh"] == 1e-10
    assert saved["figure_sha256"] == file_sha256(output_figure)
    assert saved["table_sha256"] == file_sha256(output_table)
    assert saved["renderer_sha256"] == file_sha256(Path(plot.__file__))
    assert saved["reporting_sha256"] == file_sha256(Path(plot.collect.__code__.co_filename))
    assert saved["style_sha256"] == file_sha256(style)
    broken = tmp_path / "d3_w128_do0.0_s123/result.json"
    data = json.loads(broken.read_text())
    data["signature"]["code_sha256"]["code"] = "different"
    broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mixes training"):
        plot.collect(tmp_path)
    broken.unlink()
    with pytest.raises(FileNotFoundError):
        plot.collect(tmp_path)
