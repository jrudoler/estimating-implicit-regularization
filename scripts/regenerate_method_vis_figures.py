#!/usr/bin/env python3
"""Regenerate the paper figures exported from notebooks/method-vis.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "method-vis.ipynb"
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def save_paper_figure(
    fig: plt.Figure,
    filename: str,
    *,
    dpi: int = 300,
    bbox_inches: str | None = "tight",
) -> Path:
    out_path = PAPER_FIGURES_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    print(out_path)
    return out_path


def load_notebook_code_cells(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def main() -> None:
    cells = load_notebook_code_cells(NOTEBOOK_PATH)
    execution_globals = {
        "__name__": "__main__",
        "ROOT": REPO_ROOT,
        "PAPER_FIGURES_DIR": PAPER_FIGURES_DIR,
        "save_paper_figure": save_paper_figure,
    }

    # `method-vis.ipynb` currently exports the paper figures from these code cells:
    # - code cell 2: tradeoff-vis.png
    # - code cell 11: sgd-vs-full-batch.png
    for cell_index in (2, 11):
        exec(cells[cell_index], execution_globals)
        plt.close("all")


if __name__ == "__main__":
    main()
