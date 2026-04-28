import collections.abc
import os
from typing import Optional
import re

import pandas as pd


def parse_entity_project(entity_project: str) -> tuple[str, str]:
    """Split a W&B '<entity>/<project>' value with a clear validation error."""
    if "/" not in entity_project:
        raise ValueError(
            "W&B entity/project must have the form '<entity>/<project>', "
            f"got {entity_project!r}."
        )
    entity, project = entity_project.split("/", 1)
    if not entity or not project:
        raise ValueError(
            "W&B entity/project must have nonempty entity and project parts, "
            f"got {entity_project!r}."
        )
    return entity, project


def resolve_entity_project(
    *,
    entity: str | None = None,
    project: str | None = None,
    entity_project: str | None = None,
) -> tuple[str, str]:
    """Resolve W&B location from explicit args or environment.

    Precedence:
      1. explicit entity_project
      2. explicit entity/project, with missing pieces filled from environment
      3. WANDB_ENTITY_PROJECT
      4. WANDB_ENTITY + WANDB_PROJECT
    """
    if entity_project:
        return parse_entity_project(entity_project)

    env_entity_project = os.environ.get("WANDB_ENTITY_PROJECT")
    if entity is None and project is None and env_entity_project:
        return parse_entity_project(env_entity_project)

    entity = entity or os.environ.get("WANDB_ENTITY")
    project = project or os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise ValueError(
            "Missing W&B location. Provide '<entity>/<project>' via "
            "--entity-project, Snakemake config 'wandb_entity_project', "
            "WANDB_ENTITY_PROJECT, or WANDB_ENTITY and WANDB_PROJECT."
        )
    return entity, project


def get_sweep_runs(
    sweep_id: str,
    entity: str | None = None,
    project: str | None = None,
    entity_project: str | None = None,
    state: Optional[str] = "finished",
    timeout: int = 60,
) -> list:
    """Get runs from a W&B sweep.

    Args:
        sweep_id: The sweep ID (not including entity/project).
        entity: W&B entity (organization/user). Defaults to environment.
        project: W&B project name. Defaults to environment.
        entity_project: Combined '<entity>/<project>' value.
        state: Filter runs by state ("finished", "running", etc). None for all.
        timeout: API timeout in seconds.

    Returns:
        A list of wandb.Run objects.

    Example:
        >>> runs = get_sweep_runs("abc123", entity_project="my-team/my-project")
        >>> df = wandb_summary_df(runs)
    """
    import wandb

    entity, project = resolve_entity_project(
        entity=entity, project=project, entity_project=entity_project
    )
    api = wandb.Api(timeout=timeout)
    full_sweep_id = f"{entity}/{project}/{sweep_id}"
    sweep = api.sweep(full_sweep_id)
    runs = sweep.runs
    if state is not None:
        runs = [r for r in runs if r.state == state]
    return runs


def get_project_runs(
    entity: str | None = None,
    project: str | None = None,
    entity_project: str | None = None,
    filters: Optional[dict] = None,
    state: Optional[str] = "finished",
    timeout: int = 60,
) -> list:
    """Get runs from a W&B project, optionally filtered.

    Args:
        entity: W&B entity (organization/user). Defaults to environment.
        project: W&B project name. Defaults to environment.
        entity_project: Combined '<entity>/<project>' value.
        filters: Optional dict of filters (e.g., {"config.depth": 3}).
        state: Filter runs by state ("finished", "running", etc). None for all.
        timeout: API timeout in seconds.

    Returns:
        A list of wandb.Run objects.

    Example:
        >>> runs = get_project_runs(
        ...     entity_project="my-team/my-project",
        ...     filters={"config.gt_ridge_lambda": 0.05},
        ... )
    """
    import wandb

    entity, project = resolve_entity_project(
        entity=entity, project=project, entity_project=entity_project
    )
    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}"
    runs = api.runs(path, filters=filters)
    if state is not None:
        runs = [r for r in runs if r.state == state]
    return runs


def wandb_summary_df(runs: list, include_system_metrics: bool = True) -> pd.DataFrame:
    """Make a DataFrame of run configs, metrics, and optionally GPU stats.

    Args:
        runs: A list of wandb.Run objects.
        include_system_metrics: Whether to include GPU memory/utilization.

    Returns:
        A pandas DataFrame of run configs, metrics, and GPU memory and utilization.
    """
    rows = []
    for run in runs:
        # get run config
        config = run.config
        # get run metrics
        metrics = dict(run.summary)
        # Drop nested mapping values; both plain dicts and wandb's
        # SummarySubDict (which is a bare object that quacks like a dict but
        # doesn't subclass Mapping) break parquet writes via pyarrow.
        metrics = {
            k: v
            for k, v in metrics.items()
            if not isinstance(v, collections.abc.Mapping)
            and not (hasattr(v, "keys") and callable(getattr(v, "keys", None)))
        }

        row = {"run_id": run.id, "run_name": run.name, **config, **metrics}

        if include_system_metrics:
            system_metrics = run.system_metrics
            system_metrics = {
                k: v
                for k, v in system_metrics.items()
                if re.match(r"^system\.gpu\.\d+\.(gpu|memory)$", k)
            }
            row.update(system_metrics)

        rows.append(row)
    return pd.DataFrame(rows)


def get_run_history(
    run,
    keys: Optional[list[str]] = None,
    samples: int = 10000,
) -> pd.DataFrame:
    """Get history DataFrame for a single run.

    Args:
        run: A wandb.Run object.
        keys: List of metric keys to retrieve. None for all.
        samples: Max number of samples to retrieve.

    Returns:
        A pandas DataFrame with the run history.
    """
    if keys is not None:
        return run.history(keys=keys, samples=samples, pandas=True)
    return run.history(samples=samples, pandas=True)
