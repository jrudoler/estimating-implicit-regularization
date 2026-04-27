# Shared paths and config loading.

configfile: "config/sweeps.yaml"

import os
from pathlib import Path

import yaml


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


LOCAL_SWEEPS_CONFIG = Path("config/sweeps.local.yaml")
if LOCAL_SWEEPS_CONFIG.exists():
    with LOCAL_SWEEPS_CONFIG.open() as handle:
        _deep_update(config, yaml.safe_load(handle) or {})


def sweep_id(analysis: str) -> str:
    value = config.get(analysis, {}).get("id")
    if not value:
        raise ValueError(
            f"No W&B sweep id configured for analysis '{analysis}'. "
            "Run the sweep first or set the id in config/sweeps.local.yaml."
        )
    return value


def wandb_entity_project(analysis: str) -> str:
    value = (
        config.get(analysis, {}).get("entity_project")
        or config.get("wandb_entity_project")
        or os.environ.get("WANDB_ENTITY_PROJECT")
    )
    if value:
        return value

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if entity and project:
        return f"{entity}/{project}"

    raise ValueError(
        "Missing W&B entity/project. Set 'wandb_entity_project' in "
        "config/sweeps.local.yaml, pass --config wandb_entity_project=<entity/project>, "
        "or export WANDB_ENTITY_PROJECT."
    )


def sweep_ids_file() -> str:
    return config.get("sweep_ids_file", "config/sweeps.local.yaml")


# Device override for training rules. Defaults to autodetect: the training
# scripts pick CUDA > MPS > CPU on their own. Override at the CLI:
#   uv run snakemake --config device=cpu figures
#   uv run snakemake --config device=cuda --profile workflow/profiles/slurm paper
def device_arg() -> str:
    value = config.get("device")
    return f"--device {value}" if value else ""
