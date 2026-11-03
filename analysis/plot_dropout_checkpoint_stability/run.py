"""Plot saved loss-plateau ridge estimates at actual training epochs."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def read_rows(paths: list[Path]) -> pd.DataFrame:
    rows = []
    seeds = set()
    protocols = set()
    sources = set()
    for path in paths:
        raw = path.read_bytes()
        result = json.loads(raw)
        config = result["config"]
        seed = config["seed"]
        if seed in seeds:
            raise ValueError(f"Duplicate seed {seed}")
        seeds.add(seed)
        if (config["depth"], config["width"], config["dropout"]) != (3, 256, 0.3):
            raise ValueError(f"Unexpected configuration: {path}")
        protocols.add(json.dumps(result["protocol"], sort_keys=True))
        endpoint = result["endpoint"]
        if endpoint["status"] != "loss_plateau":
            raise ValueError(f"Run did not stop at a loss plateau: {path}")
        points = [p for p in result["trajectory"] if p["kind"] == "trajectory"]
        interval = result["protocol"]["checkpoint_every"]
        if sorted(p["epoch"] for p in points) != list(range(interval, endpoint["epoch"], interval)):
            raise ValueError(f"Incomplete checkpoint sequence: {path}")
        # The stopping endpoint is shown only as x, even on a regular interval.
        points.append(endpoint)
        for point in points:
            sources.add(point["provenance"]["source_bundle_sha256"])
            ridge = point["ridge"]
            rows.append({
                "seed": seed, "epoch": point["epoch"], "kind": point["kind"],
                "lambda_signed": ridge["scale_star"],
                "lambda_nonnegative": ridge["scale_star_clamped"],
                "projection_r2": ridge["projection_r2"],
                "learning_rate": point["lr"],
                "checkpoint_sha256": point["provenance"]["checkpoint_sha256"],
                "source_result": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "source_result_sha256": hashlib.sha256(raw).hexdigest(),
            })
    if len(seeds) != 10 or len(protocols) != 1 or len(sources) != 1:
        raise ValueError("Require ten distinct seeds with a common protocol and source")
    frame = pd.DataFrame(rows).sort_values(["seed", "epoch"])
    if frame["lambda_signed"].isna().any() or (frame["lambda_signed"] <= 0).any():
        raise ValueError("Logarithmic plot requires finite positive signed coefficients")
    return frame


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    color = "#6B9AC4"
    for _, group in frame.groupby("seed"):
        ax.plot(group["epoch"], group["lambda_signed"], color=color,
                linewidth=1.2, alpha=0.45)
        regular = group[group["kind"] == "trajectory"]
        final = group[group["kind"] == "endpoint"]
        ax.scatter(regular["epoch"], regular["lambda_signed"], color=color,
                   marker="o", s=16, alpha=0.45, zorder=3)
        ax.scatter(final["epoch"], final["lambda_signed"], color=color,
                   marker="x", s=65, linewidths=2, alpha=0.85, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"Estimated ridge penalty $\hat\lambda$")
    ax.set_title("Depth 3, width 256, dropout 0.3", fontsize=16)
    ax.grid(True, which="major", alpha=0.45)
    ax.legend(handles=[
        Line2D([], [], color=color, marker="o", linestyle="none", markersize=4,
               label="1,000-epoch checkpoint"),
        Line2D([], [], color=color, marker="x", linestyle="none", markersize=7,
               markeredgewidth=2, label="Training endpoint"),
    ], loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    frame = read_rows([p.resolve() for p in args.inputs])
    plt.style.use(ROOT / "clean_fig.mplstyle")
    fig = build_figure(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    frame.to_csv(args.table, index=False)
    LOG.info("Saved %s and %s: %d measurements, 10 stopping endpoints", args.output, args.table, len(frame))


if __name__ == "__main__":
    main()
