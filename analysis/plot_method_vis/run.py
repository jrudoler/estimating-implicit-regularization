#!/usr/bin/env python3
"""Regenerate the method-visualization figures exported from notebooks/method-vis.ipynb.

The notebook's code cells write figures via a `save_paper_figure(fig, filename, ...)`
call. This wrapper re-executes just the two relevant cells and routes each named
output to the explicit CLI path provided by the Snakemake rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "method-vis.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-tradeoff", required=True, type=Path)
    parser.add_argument("--out-sgd-vs-fb", required=True, type=Path)
    return parser.parse_args()


def load_notebook_code_cells(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def main() -> None:
    args = parse_args()
    name_to_path: dict[str, Path] = {
        "tradeoff-vis.png": args.out_tradeoff,
        "sgd-vs-full-batch.png": args.out_sgd_vs_fb,
    }

    def save_paper_figure(
        fig: plt.Figure,
        filename: str,
        *,
        dpi: int = 300,
        bbox_inches: str | None = "tight",
    ) -> Path:
        if filename not in name_to_path:
            raise KeyError(
                f"notebook asked to save '{filename}' but this rule only routes "
                f"{sorted(name_to_path)}"
            )
        out_path = name_to_path[filename]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
        print(out_path)
        return out_path

    cells = load_notebook_code_cells(NOTEBOOK_PATH)
    execution_globals = {
        "__name__": "__main__",
        "ROOT": REPO_ROOT,
        "save_paper_figure": save_paper_figure,
    }

    # method-vis.ipynb emits the paper figures from these code cells:
    # - code cell 2: tradeoff-vis.png
    # - code cell 4: sgd-vs-full-batch.png
    for cell_index in (2, 4):
        exec(cells[cell_index], execution_globals)
        plt.close("all")


if __name__ == "__main__":
    main()
