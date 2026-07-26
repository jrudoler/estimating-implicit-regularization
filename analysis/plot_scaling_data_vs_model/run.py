"""Data-vs-model-size scaling figure for the rebuttal (theme R1 / GHL1 Q1).

Turns the PAC bound (app_pac.tex) into the plot reviewers asked for. Key reframe:
each of a model's p parameters is one stationarity equation, so a *single* trained
model of size p identifies any regularizer with r <= p coefficients, and estimation
error scales O(sqrt(r / p_eff)) with p_eff = p / rho -- i.e. estimation gets *easier*
as models grow. Extra "data" (independent endpoints/datasets) is needed only when
r > p or to beat gradient collinearity (rho > 1).

Panel A: relative estimation error factor  ~ sqrt(r * rho / p)   (constants folded into y-scale)
Panel B: independent endpoints needed       m_min = max(1, ceil(r / p))

Outputs a vector PDF and logs the inline m_min table used in the rebuttal text.
Pure CPU, no data dependency.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = Path("results/figures/scaling_data_vs_model.pdf")
DEFAULT_STYLE = Path("clean_fig.mplstyle")
ANCHORS = {
    "MLP ~1e7": 1.1e7,
    "OLMo-2-1B": 1.48e9,
    "OLMo-2-7B": 7.3e9,
    "8B": 8.0e9,
    "100B": 1.0e11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    return parser.parse_args()


def minimum_endpoints(
    parameter_count: np.ndarray, coefficient_count: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Return the minimum number of independent endpoints for identification."""
    return np.maximum(1.0, np.ceil(coefficient_count(parameter_count) / parameter_count))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use(args.style)
    p = np.logspace(6, 11, 400)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Panel A: relative estimation error factor ~ sqrt(r * rho / p) ----
    def relative_error(r: int, rho: int) -> np.ndarray:
        return np.sqrt(r * rho / p)

    curvesA = [
        (1, 1, "r=1 (scalar ridge)", "-"),
        (10, 1, "r=10", "-"),
        (100, 1, "r=100", "-"),
        (1, 10, "r=1, rho=10 (collinear)", "--"),
        (100, 10, "r=100, rho=10", "--"),
    ]
    for r, rho, lbl, ls in curvesA:
        axA.loglog(p, relative_error(r, rho), ls, label=lbl)
    axA.set_xlabel("Model parameter count  p")
    axA.set_ylabel(r"Relative estimation error  $\propto \sqrt{r/p_{\mathrm{eff}}}$"
                   "\n(up to constants $\\sigma,\\kappa,\\delta$)")
    axA.set_title("Error decreases as models grow")
    axA.legend(fontsize=7, loc="upper right")

    # ---- Panel B: independent endpoints needed  m_min = max(1, ceil(r/p)) ----
    curvesB = [
        (lambda pp: np.full_like(pp, 1.0), "r=1 (scalar)", "-"),
        (lambda pp: np.full_like(pp, 100.0), "r=100", "-"),
        (lambda pp: np.full_like(pp, 1e4), "r=1e4", "-"),
        (lambda pp: pp, "r=p (diagonal)", "-."),
        (lambda pp: pp * (pp + 1) / 2, "r=p(p+1)/2 (full matrix)", ":"),
    ]
    for r_of_p, lbl, ls in curvesB:
        axB.loglog(p, minimum_endpoints(p, r_of_p), ls, label=lbl)
    axB.set_xlabel("Model parameter count  p")
    axB.set_ylabel("Independent endpoints / datasets needed  m")
    axB.set_title("A single model suffices except for a\nfamily that grows faster than p")
    axB.legend(fontsize=7, loc="upper left")

    for ax in (axA, axB):
        for name, pv in ANCHORS.items():
            ax.axvline(pv, color="0.8", lw=0.8, zorder=0)
        ax.set_xlim(p.min(), p.max())
    # annotate anchors on Panel A top
    ymax = axA.get_ylim()[1]
    for name, pv in ANCHORS.items():
        axA.text(pv, ymax, name, rotation=90, va="top", ha="right",
                 fontsize=6, color="0.4")

    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    LOGGER.info("Wrote %s", args.output)

    # ---- inline m_min table for the rebuttal text ----
    lines = ["Inline table (endpoints m_min needed to identify an r-parameter regularizer):"]
    header = f"{'p':>10} | {'r=1':>6} | {'r=p':>6} | {'r=p(p+1)/2':>12}"
    lines.extend((header, "-" * len(header)))
    for pv in (1e6, 1e9, 1e11):
        m1 = max(1, int(np.ceil(1 / pv)))
        mp = max(1, int(np.ceil(pv / pv)))
        mfull = max(1, int(np.ceil((pv * (pv + 1) / 2) / pv)))
        lines.append(f"{pv:>10.0e} | {m1:>6d} | {mp:>6d} | {mfull:>12,d}")
    LOGGER.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
