#!/usr/bin/env python3
"""
list_wandb_run_ids.py

Usage:
    uv run python list_wandb_run_ids.py <entity>/<project>/<sweep_id> [state]

Example:
    uv run python list_wandb_run_ids.py dobriban-wharton-dobriban/fancy-sweep-300 failed
"""

import sys
import wandb

if len(sys.argv) < 2:
    print(
        "Usage: python list_wandb_run_ids.py <entity>/<project>/<sweep_id> [state]",
        file=sys.stderr,
    )
    sys.exit(1)

sweep_path = sys.argv[1]
state = sys.argv[2] if len(sys.argv) > 2 else "failed"

api = wandb.Api()
sweep = api.sweep(sweep_path)

for run in sweep.runs:
    if run.state.lower() == state.lower():
        print(run.id)
