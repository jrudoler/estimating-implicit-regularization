#!/usr/bin/env python3
"""Export notebook-derived or preserved manuscript figures to a target path."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"


@dataclass(frozen=True)
class FigureAsset:
    figure_id: str
    source: Path
    default_out: Path
    provenance: str


FIGURE_ASSETS: dict[str, FigureAsset] = {
    "sgd-vs-full-batch": FigureAsset(
        figure_id="sgd-vs-full-batch",
        source=REPO_ROOT / "notebooks" / "sgd-vs-full-batch.png",
        default_out=PAPER_FIGURES_DIR / "sgd-vs-full-batch.png",
        provenance="Notebook-generated asset from notebooks/method-vis.ipynb.",
    ),
    "tradeoff-vis": FigureAsset(
        figure_id="tradeoff-vis",
        source=PAPER_FIGURES_DIR / "tradeoff-vis.png",
        default_out=PAPER_FIGURES_DIR / "tradeoff-vis.png",
        provenance="Preserved manuscript asset derived from notebooks/method-vis.ipynb.",
    ),
    "OLS_early_stopping_figure": FigureAsset(
        figure_id="OLS_early_stopping_figure",
        source=PAPER_FIGURES_DIR / "OLS_early_stopping_figure.pdf",
        default_out=PAPER_FIGURES_DIR / "OLS_early_stopping_figure.pdf",
        provenance="Preserved manuscript asset assembled externally from automated OLS component figures.",
    ),
    "radial_sine_function_data_2d": FigureAsset(
        figure_id="radial_sine_function_data_2d",
        source=PAPER_FIGURES_DIR / "radial_sine_function_data_2d.pdf",
        default_out=PAPER_FIGURES_DIR / "radial_sine_function_data_2d.pdf",
        provenance="Notebook-generated lineage from notebooks/kernel-regression.ipynb, preserved here as an asset.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=sorted(FIGURE_ASSETS))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override the default destination in paper/figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset = FIGURE_ASSETS[args.figure]
    out_path = args.out or asset.default_out

    if not asset.source.exists():
        raise SystemExit(
            f"Missing source asset for {asset.figure_id}: {asset.source}\n{asset.provenance}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if asset.source.resolve() != out_path.resolve():
        shutil.copy2(asset.source, out_path)
    print(f"Prepared {out_path}")
    print(asset.provenance)


if __name__ == "__main__":
    main()
