import wandb
import pandas as pd
import re


def wandb_summary_df(runs):
    """Make a DataFrame of run configs, metrics, and GPU memory and utilization.

    Args:
        runs: A list of wandb.Run objects.

    Returns:
        A pandas DataFrame of run configs, metrics, and GPU memory and utilization.
    """
    rows = []
    for run in runs:
        # get run config
        config = run.config
        # get run metrics
        metrics = run.summary
        # get gpu memory and utilization
        system_metrics = run.system_metrics
        # get gpu memory and utilization by regex
        system_metrics = {
            k: v
            for k, v in system_metrics.items()
            if re.match(r"^system\.gpu\.\d+\.(gpu|memory)$", k)
        }
        # append to rows
        rows.append({"run_id": run.id, **config, **metrics, **system_metrics})
    return pd.DataFrame(rows)

