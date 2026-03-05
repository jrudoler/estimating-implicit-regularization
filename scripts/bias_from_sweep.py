#!/usr/bin/env python3
"""Download checkpoints for a wandb sweep and run bias estimation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import wandb


@dataclass
class SweepRunArtifacts:
    run_id: str
    run_name: str
    checkpoint_path: Path
    config_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download checkpoints for each run in a wandb sweep and optionally fit bias models.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "sweep_id", type=str, help="Target wandb sweep ID (e.g. 2stzqc65)."
    )
    parser.add_argument(
        "--entity", type=str, default=None, help="wandb entity/organization."
    )
    parser.add_argument(
        "--project", type=str, required=True, help="wandb project name."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "bias_checkpoints",
        help="Directory where checkpoints and configs will be stored.",
    )
    parser.add_argument(
        "--states",
        type=str,
        nargs="*",
        default=["finished"],
        help="Run states to include (default: finished).",
    )
    parser.add_argument(
        "--run-bias",
        action="store_true",
        help="Immediately launch bias estimation runs after downloading checkpoints.",
    )
    parser.add_argument(
        "--bias-script",
        type=Path,
        default=Path("experiments") / "dropout_bias_estimation.py",
        help="Path to the bias estimation script.",
    )
    parser.add_argument(
        "--bias-project",
        type=str,
        default=None,
        help="wandb project to log bias runs (defaults to training project if omitted).",
    )
    parser.add_argument(
        "--bias-entity",
        type=str,
        default=None,
        help="wandb entity for bias runs (defaults to training entity if omitted).",
    )
    parser.add_argument(
        "--bias-run-template",
        type=str,
        default="{run_name}-bias",
        help="Format string for bias run names. Available fields: run_id, run_name, sweep_id.",
    )
    parser.add_argument(
        "--bias-arg",
        action="append",
        default=[],
        help="Additional arguments to forward to the bias script (repeat per flag).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of bias estimation processes to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print bias commands instead of executing them.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to write a JSONL manifest of runs (defaults to <output-dir>/manifest.jsonl).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download checkpoints/configs (implied when --run-bias is not passed).",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_model_artifact(run) -> Optional[wandb.apis.public.Artifact]:  # type: ignore[name-defined]
    artifacts = [
        artifact for artifact in run.logged_artifacts() if artifact.type == "model"
    ]
    if not artifacts:
        return None
    artifacts.sort(key=lambda art: (art.updated_at or art.created_at or ""))
    return artifacts[-1]


def download_checkpoint(artifact, destination: Path) -> Path:
    artifact_dir = Path(artifact.download(root=str(destination)))
    checkpoint_candidates = sorted(artifact_dir.rglob("*.ckpt"))
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No checkpoint files found in artifact {artifact.name}"
        )
    preferred = [p for p in checkpoint_candidates if "best" in p.name]
    return preferred[0] if preferred else checkpoint_candidates[0]


def materialize_run(run, base_dir: Path) -> Optional[SweepRunArtifacts]:
    artifact = find_latest_model_artifact(run)
    if artifact is None:
        return None
    run_dir = ensure_directory(base_dir / run.id)
    checkpoint_path = download_checkpoint(artifact, run_dir / "artifacts")
    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(run.config, fp, indent=2)
    return SweepRunArtifacts(
        run_id=run.id,
        run_name=run.name or run.id,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


def iter_bias_commands(
    artifacts: Iterable[SweepRunArtifacts],
    args: argparse.Namespace,
    sweep_id: str,
    entity: str,
) -> Iterable[List[str]]:
    bias_script = args.bias_script.resolve()
    bias_project = args.bias_project or args.project
    bias_entity = args.bias_entity or entity

    for record in artifacts:
        run_name = args.bias_run_template.format(
            run_id=record.run_id,
            run_name=record.run_name,
            sweep_id=sweep_id,
        )
        command = [
            "uv",
            "run",
            "python",
            str(bias_script),
            "--checkpoint-path",
            str(record.checkpoint_path),
            "--config-path",
            str(record.config_path),
            "--wandb-project",
            bias_project,
        ]
        if bias_entity is not None:
            command.extend(["--wandb-entity", bias_entity])
        command.extend(["--wandb-run-name", run_name])
        command.extend(args.bias_arg)
        yield command


def run_bias_commands(
    commands: Iterable[List[str]], max_workers: int, dry_run: bool
) -> None:
    if dry_run:
        for cmd in commands:
            print("[DRY-RUN]", " ".join(shlex.quote(part) for part in cmd))
        return

    commands_list = list(commands)
    if not commands_list:
        print("No bias estimation commands to execute.")
        return

    def launch(cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, check=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(launch, cmd): cmd for cmd in commands_list}
        for future in as_completed(futures):
            cmd = futures[future]
            try:
                future.result()
            except subprocess.CalledProcessError as exc:
                print("Command failed:", " ".join(shlex.quote(part) for part in cmd))
                raise exc


def write_manifest(records: Iterable[SweepRunArtifacts], manifest_path: Path) -> None:
    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(
                json.dumps(
                    {
                        "run_id": record.run_id,
                        "run_name": record.run_name,
                        "checkpoint_path": str(record.checkpoint_path),
                        "config_path": str(record.config_path),
                    }
                )
            )
            fp.write("\n")


def main() -> None:
    args = parse_args()

    api = wandb.Api()
    sweep_path = (
        f"{args.entity}/{args.project}/{args.sweep_id}"
        if args.entity
        else f"{args.project}/{args.sweep_id}"
    )
    sweep = api.sweep(sweep_path)
    entity = args.entity or sweep.entity

    ensure_directory(args.output_dir)
    target_states = set(state.lower() for state in args.states)

    records: List[SweepRunArtifacts] = []
    for run in sweep.runs:
        if run.state.lower() not in target_states:
            continue
        artifact_record = materialize_run(run, args.output_dir)
        if artifact_record is not None:
            records.append(artifact_record)
            try:
                rel_path = artifact_record.checkpoint_path.relative_to(Path.cwd())
            except ValueError:
                rel_path = artifact_record.checkpoint_path
            print(f"Prepared run {run.id} ({run.name}) -> {rel_path}")
        else:
            print(
                f"Skipping run {run.id} ({run.name}) - no checkpoint artifact detected."
            )

    if not records:
        print("No runs matched the requested sweep/state criteria.")
        return

    manifest_path = args.manifest or (args.output_dir / "manifest.jsonl")
    write_manifest(records, manifest_path)
    print(f"Wrote manifest with {len(records)} entries to {manifest_path}")

    if args.run_bias and not args.download_only:
        commands = list(iter_bias_commands(records, args, args.sweep_id, entity))
        run_bias_commands(commands, args.max_workers, args.dry_run)
    else:
        print("Bias estimation commands (rerun with --run-bias to execute):")
        for cmd in iter_bias_commands(records, args, args.sweep_id, entity):
            print("  ", " ".join(shlex.quote(part) for part in cmd))


if __name__ == "__main__":  # pragma: no cover
    main()
