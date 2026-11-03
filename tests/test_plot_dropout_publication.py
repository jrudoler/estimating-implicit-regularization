from __future__ import annotations

from copy import deepcopy
import importlib.util
from itertools import product
from pathlib import Path

from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "analysis/plot_dropout_bias_ridge_panel/run.py"
    spec = importlib.util.spec_from_file_location("publication_figure_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_publication_style_preserves_all_points_and_diagnostic(tmp_path: Path) -> None:
    plot = module()
    rows = [{"depth": d, "width": w, "dropout": p, "seed": s,
             "lambda_nonnegative": (1 + p + s / 1000) * 1e-8,
             "projection_r2": p, "status": "loss_plateau"}
            for d, w, p, s in product(plot.DEPTHS, plot.WIDTHS, plot.RATES, plot.SEEDS)]
    frame = pd.DataFrame(rows)
    frame.loc[0, "lambda_nonnegative"] = 0.
    frame.loc[0, "projection_r2"] = np.nan
    frame.loc[1, "status"] = "budget_exhausted"
    fig = plot.build_figure(frame)
    assert len(fig.axes) == 7
    assert fig._suptitle is None
    assert len(fig.texts) == 2  # shared axis labels only, no commentary
    assert all(ax.get_yscale() == "symlog" for ax in fig.axes[:6])
    points = [c for ax in fig.axes[:6] for c in ax.collections if isinstance(c, PathCollection)]
    offsets = np.concatenate([c.get_offsets() for c in points])
    assert len(offsets) == 480
    assert sum(offsets[:, 1] == 0.) == 1
    assert all(ax.get_ylim()[0] < 0 for ax in fig.axes[:6])
    assert fig.axes[-1].get_ylim() == (0., 1.)
    assert fig.axes[-1].get_ylabel() == r"$R_{\mathrm{grad}}^2$"
    fig.savefig(tmp_path / "publication.pdf")
    plot.plt.close(fig)
    assert (tmp_path / "publication.pdf").stat().st_size > 1000


def test_publication_requires_matching_completed_audit() -> None:
    plot = module()
    receipt = {"n": 480, "missing_cells": [], "protocol": {"test": True},
               "code_sha256": {}, "statuses": {"loss_plateau": 480},
               "sources": [{"cell": [i], "sha256": str(i), "checkpoint_sha256": str(i)}
                           for i in range(480)]}
    audit = {**deepcopy(receipt), "full_grid_verification_passed": True}
    plot.validate_audit(receipt, audit)
    audit["sources"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="Source hashes"):
        plot.validate_audit(receipt, audit)
    audit["full_grid_verification_passed"] = False
    with pytest.raises(ValueError, match="has not passed"):
        plot.validate_audit(receipt, audit)
