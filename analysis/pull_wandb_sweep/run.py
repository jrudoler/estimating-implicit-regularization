#!/usr/bin/env python3
"""Pull summary rows for a W&B sweep into a local parquet snapshot.

Thin CLI over src/core/wandb_utils.py. Run via the Snakemake
`pull_wandb_sweep` rule; downstream figure rules depend on the parquet, not
on the live W&B API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.wandb_utils import get_sweep_runs, wandb_summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID (e.g. 9a7ll8aa).")
    parser.add_argument(
        "--entity-project",
        required=True,
        help="'<entity>/<project>', e.g. jhrudoler-penn/inductive-bias.",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Filter runs by state; pass empty string for all.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entity, project = args.entity_project.split("/", 1)
    state = args.state or None
    runs = get_sweep_runs(args.sweep_id, entity=entity, project=project, state=state)
    df = wandb_summary_df(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
