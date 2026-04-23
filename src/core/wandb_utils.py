import wandb
import pandas as pd
import re
from typing import Optional


def get_sweep_runs(
    sweep_id: str,
    entity: str = "jhrudoler-penn",
    project: str = "inductive-bias",
    state: Optional[str] = "finished",
    timeout: int = 60,
) -> list:
    """Get runs from a W&B sweep.

    Args:
        sweep_id: The sweep ID (not including entity/project).
        entity: W&B entity (organization/user).
        project: W&B project name.
        state: Filter runs by state ("finished", "running", etc). None for all.
        timeout: API timeout in seconds.

    Returns:
        A list of wandb.Run objects.

    Example:
        >>> runs = get_sweep_runs("abc123")
        >>> df = wandb_summary_df(runs)
    """
    api = wandb.Api(timeout=timeout)
    full_sweep_id = f"{entity}/{project}/{sweep_id}"
    sweep = api.sweep(full_sweep_id)
    runs = sweep.runs
    if state is not None:
        runs = [r for r in runs if r.state == state]
    return runs


def get_project_runs(
    entity: str = "jhrudoler-penn",
    project: str = "inductive-bias",
    filters: Optional[dict] = None,
    state: Optional[str] = "finished",
    timeout: int = 60,
) -> list:
    """Get runs from a W&B project, optionally filtered.

    Args:
        entity: W&B entity (organization/user).
        project: W&B project name.
        filters: Optional dict of filters (e.g., {"config.depth": 3}).
        state: Filter runs by state ("finished", "running", etc). None for all.
        timeout: API timeout in seconds.

    Returns:
        A list of wandb.Run objects.

    Example:
        >>> runs = get_project_runs(filters={"config.gt_ridge_lambda": 0.05})
    """
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
        # remove nested dicts (can cause issues)
        metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}

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
