#!/usr/bin/env python3
"""Create a fresh W&B sweep, submit N SLURM agents, and block until all finish.

Writes the new sweep ID back to a local sweep id file under the provided
analysis key. Intended to be run via the Snakemake `launch_wandb_sweep`
rule as an opt-in per-analysis target.

After this rule finishes, the normal `pull_wandb_sweep` rule materializes
the parquet snapshot consumed by plot rules.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name=wandb_agent_{analysis}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
#SBATCH --chdir={chdir}
set -euo pipefail
uv run wandb agent --count 1 "{sweep_id}"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="Analysis key in sweep-ids file.")
    parser.add_argument("--sweep-config", required=True, type=Path, help="Sweep YAML to register.")
    parser.add_argument(
        "--entity-project",
        required=True,
        help="W&B '<entity>/<project>' where the sweep should be created.",
    )
    parser.add_argument("--n-agents", type=int, default=8, help="Number of sbatch agents to submit.")
    parser.add_argument("--sweep-ids-file", required=True, type=Path, help="YAML file to update with the new sweep ID.")
    parser.add_argument("--partition", default="gpu", help="SLURM partition.")
    parser.add_argument("--repo-root", default=str(Path.cwd()), help="--chdir value for sbatch.")
    parser.add_argument("--no-wait", action="store_true", help="Fire-and-forget; do not sbatch --wait.")
    return parser.parse_args()


def create_sweep(sweep_config: Path, entity_project: str) -> str:
    entity, project = entity_project.split("/", 1)
    cmd = ["uv", "run", "wandb", "sweep", "--entity", entity, "--project", project, str(sweep_config)]
    print("$", " ".join(cmd))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    combined = out.stdout + "\n" + out.stderr
    print(combined)
    m = re.search(r"wandb agent\s+([\w\-]+/[\w\-]+/[\w]+)", combined)
    if not m:
        sys.exit("Could not parse sweep ID from `wandb sweep` output.")
    full_id = m.group(1)
    return full_id


def update_sweep_ids_file(
    path: Path,
    analysis: str,
    sweep_id: str,
    entity_project: str,
) -> None:
    if path.exists():
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    data.setdefault("wandb_entity_project", entity_project)
    data.setdefault(analysis, {})
    data[analysis]["id"] = sweep_id
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    print(f"Updated {path}: {analysis}.id = {sweep_id}")


def submit_agents(sweep_id: str, n_agents: int, analysis: str, partition: str, chdir: str, wait: bool) -> None:
    Path("logs/slurm").mkdir(parents=True, exist_ok=True)
    script_body = SBATCH_TEMPLATE.format(
        partition=partition, analysis=analysis, chdir=chdir, sweep_id=sweep_id
    )
    script_path = Path(f"logs/slurm/wandb_agent_{analysis}.sbatch")
    script_path.write_text(script_body)

    job_ids: list[str] = []
    for i in range(n_agents):
        cmd = ["sbatch", "--parsable", str(script_path)]
        print(f"$ ({i + 1}/{n_agents})", " ".join(cmd))
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = out.stdout.strip().split(";", 1)[0]
        job_ids.append(job_id)
        print(f"  submitted {job_id}")

    if not wait:
        print("Launched without waiting; pull_wandb_sweep will block on run states later.")
        return

    dep = ",".join(f"afterany:{jid}" for jid in job_ids)
    print(f"Waiting on {len(job_ids)} agent jobs via sbatch --wait...")
    subprocess.run(
        ["srun", "--partition", partition, "--dependency", dep, "--time", "00:01:00", "true"],
        check=True,
    )
    print("All agents finished.")


def main() -> None:
    args = parse_args()
    full_sweep_id = create_sweep(args.sweep_config, args.entity_project)
    sweep_id = full_sweep_id.rsplit("/", 1)[-1]
    update_sweep_ids_file(
        args.sweep_ids_file,
        args.analysis,
        sweep_id,
        args.entity_project,
    )
    submit_agents(
        sweep_id=full_sweep_id,
        n_agents=args.n_agents,
        analysis=args.analysis,
        partition=args.partition,
        chdir=args.repo_root,
        wait=not args.no_wait,
    )


if __name__ == "__main__":
    main()
