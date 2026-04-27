# W&B integration rules.
#
# `pull_wandb_sweep` is the default pathway: given a sweep ID and W&B
# entity/project from config or environment, fetch run summaries and write a parquet
# snapshot to data/generated/<analysis>/runs.parquet. Figures depend on the
# parquet, not on W&B directly, so figure rebuilds never re-pull by default.
#
# `launch_wandb_sweep_<analysis>` is an opt-in per-analysis target that creates
# a fresh sweep, submits N SLURM agents via sbatch --wait, captures the ID
# into config/sweeps.local.yaml by default, and touches a marker file. After
# launch finishes, a normal `pull_wandb_sweep` pass materializes the parquet.


rule pull_wandb_sweep:
    input:
        script="analysis/pull_wandb_sweep/run.py",
    output:
        parquet="data/generated/{analysis}/runs.parquet",
    params:
        sweep_id=lambda wc: sweep_id(wc.analysis),
        entity_project=lambda wc: wandb_entity_project(wc.analysis),
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--sweep-id {params.sweep_id} "
        "--entity-project {params.entity_project} "
        "--output {output.parquet}"


rule launch_wandb_sweep:
    input:
        sweep_config=lambda wc: config[wc.analysis]["sweep_config"],
        launcher="analysis/launch_wandb_sweep/run.py",
    output:
        marker=touch("data/generated/{analysis}/.sweep_done"),
    params:
        n_agents=lambda wc: config[wc.analysis].get("n_agents", 8),
        analysis=lambda wc: wc.analysis,
        entity_project=lambda wc: wandb_entity_project(wc.analysis),
        sweep_ids_file=sweep_ids_file(),
    shell:
        "PYTHONPATH=src uv run python {input.launcher} "
        "--analysis {params.analysis} "
        "--sweep-config {input.sweep_config} "
        "--entity-project {params.entity_project} "
        "--n-agents {params.n_agents} "
        "--sweep-ids-file {params.sweep_ids_file}"
