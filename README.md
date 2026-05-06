# Estimating implicit regularization

This repository contains the code for **Estimating Implicit Regularization in Deep Learning**. The project studies how to empirically recover and compare hypothesized inductive biases by treating a trained model as if it were the solution to a regularized learning problem, then matching the gradient of a candidate regularizer to the residual gradient of the predictive loss.

The central idea is a gradient-matching formulation of implicit regularization. Given trained weights and data, the code estimates regularizer parameters that best explain the observed training residuals. This lets the project do several things in one framework:

- recover explicit regularizers such as L2 weight decay
- test theoretical predictions about implicit bias, including early stopping style effects
- compare candidate regularizers under identifiability and collinearity constraints
- study practical settings such as dropout and deep nonlinear MNIST models

## Research Focus

The manuscript in [`paper/`](paper) frames the project around a simple inverse problem:

- train a predictive model with some algorithm or architectural choice
- compute the predictive loss gradient at the learned solution
- specify a parameterized family of candidate regularizers
- fit the regularizer parameters so their gradient matches the negative loss gradient

## Repository layout

The repo follows a reproducible-science layout with an explicit [`Snakefile`](workflow/Snakefile) that captures every dependency from raw input to final PDF.

```text
data/
  raw/           external downloads (e.g. MNIST)
  provided/      stable project inputs (sweep YAMLs, paper assets)
  generated/     workflow-produced intermediates (W&B snapshots, .pt checkpoints)
analysis/
  <rule>/run.py  one folder per Snakemake rule
src/core/        shared Python package (imports as `from core.X import ...`)
results/
  data/          final tables
  figures/       final manuscript figures (source of truth)
paper/           submodule; `paper/figures/` staged by the workflow
workflow/
  Snakefile
  rules/         common, wandb, train, plot, paper
  profiles/      local/ (cores=4, dev defaults) and slurm/ (sbatch executor)
config/          sweeps.yaml (account-neutral W&B sweep manifest)
```

See [science-repo-skill.md](science-repo-skill.md) for the canonical description of the layout. The figure-by-figure provenance is also encoded in the Snakemake DAG itself: each `analysis/<rule>/run.py` declares its inputs and outputs, and `workflow/rules/{train,plot,paper}.smk` wires them together.

## Getting started

```bash
git clone --recurse-submodules <PARENT_REPO_URL>
cd inductive-bias
bash scripts/bootstrap.sh
```

The bootstrap script installs `uv` if missing, syncs the dev group (which includes Snakemake), creates the log and generated directories, and prints the installed snakemake version. It is idempotent.

If you already have `uv`:

```bash
uv sync --group dev
```

## Workflow

Every step runs through Snakemake. The top-level targets, defined in [workflow/Snakefile](workflow/Snakefile), are:

- `figures` — every paper-bound figure under `results/figures/`.
- `paper` — `paper/main.pdf` (depends on staged figures and runs `latexmk`).
- `all` — alias for `paper`.

```bash
# Rebuild every paper figure. Snakemake reuses existing data/generated/ snapshots
# and only re-runs upstream rules (training, W&B pulls) for outputs that are missing.
uv run snakemake -s workflow/Snakefile --cores 4 figures

# Full end-to-end via SLURM: train, pull W&B snapshots, plot, latexmk -> paper/main.pdf
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/slurm paper

# Rebuild one figure (only its upstream subgraph).
uv run snakemake -s workflow/Snakefile --cores 4 results/figures/barrett_igr_figure2.pdf

# Force re-pull one W&B snapshot without touching others.
uv run snakemake -s workflow/Snakefile \
    --forcerun pull_wandb_sweep \
    results/figures/elasticnet_recovery_mean_se.pdf
```

Analysis sub-DAGs are independent. Rebuilding the OLS figures never triggers the dropout pipeline, and vice versa.
Experiment-backed figures are split into `data/generated/<analysis>/` data rules and `results/figures/` plot rules so plot iteration can rebuild PDFs without rerunning training or data analysis. The methods visualizations are the exception because they have no meaningful intermediate data.

### Adding a new analysis

1. Decide whether the output is an intermediate (`data/generated/`) or a final result (`results/`).
2. For experiment-backed figures, create a data script that writes `data/generated/<analysis>/...` and a separate `analysis/plot_<analysis>/run.py` that reads those artifacts and writes `results/figures/...`.
3. Move reusable logic into `src/core/`.
4. Add the rule to [workflow/rules/train.smk](workflow/rules/train.smk) (training) or [workflow/rules/plot.smk](workflow/rules/plot.smk) (plotting). For sweep-driven analyses, the snapshot/launch rules in [workflow/rules/wandb.smk](workflow/rules/wandb.smk) are reused via the analysis name.
5. If the analysis consumes W&B runs, add the sweep config to [config/sweeps.yaml](config/sweeps.yaml) and store personal sweep IDs in `config/sweeps.local.yaml`.

### W&B sweeps

[config/sweeps.yaml](config/sweeps.yaml) is intentionally account-neutral. It
records which analyses have W&B sweep configs, but not a required W&B
entity/project. Put your local account and finished sweep IDs in
`config/sweeps.local.yaml`:

```bash
cp config/sweeps.local.example.yaml config/sweeps.local.yaml
# edit wandb_entity_project and any finished sweep ids
```

As alternatives to the local file, pass `--config
wandb_entity_project=<entity/project>` or export `WANDB_ENTITY_PROJECT`.

Figures never query W&B directly. The workflow first snapshots finished runs to
`data/generated/<analysis>/runs.parquet` via `pull_wandb_sweep`; plot rules read
that local parquet. This keeps figure iteration fast and reproducible.

Pull a finished sweep snapshot:

```bash
uv run snakemake -s workflow/Snakefile --cores 4 \
    data/generated/elasticnet_train_and_recover/runs.parquet
```

Refresh a snapshot and rebuild its downstream figure:

```bash
uv run snakemake -s workflow/Snakefile --cores 4 \
    --forcerun pull_wandb_sweep \
    results/figures/elasticnet_recovery_mean_se.pdf
```

To reproduce from scratch on your own W&B account, launch a fresh sweep and
agents. The launch rule writes the new sweep ID to ignored
`config/sweeps.local.yaml` by default:

```bash
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/slurm \
    --config wandb_entity_project=<entity/project> \
    data/generated/<analysis>/.sweep_done
```

Then pull the snapshot and build figures:

```bash
uv run snakemake -s workflow/Snakefile --cores 4 \
    data/generated/<analysis>/runs.parquet
uv run snakemake -s workflow/Snakefile --cores 4 figures
```

### Local vs SLURM

Two profiles ship under [workflow/profiles/](workflow/profiles):

- `local/` — `cores: 4` plus dev-friendly defaults (`keep-incomplete`, `printshellcmds`). Use for development on a workstation; training rules run wherever the shell sees a device.
- `slurm/` — submits one sbatch job per rule via `snakemake-executor-plugin-slurm`. Training rules declare cluster resources inline via `resources:` blocks in [workflow/rules/train.smk](workflow/rules/train.smk) (partition, runtime, mem_mb, cpus_per_task, `slurm_extra="--gres=gpu:1"`). Cheap rules (plotting, W&B pulls, staging, latexmk) are listed under `localrules:` in [workflow/Snakefile](workflow/Snakefile) and always run on the submitting host.

```bash
# Run everything locally; training rules autodetect CUDA -> MPS -> CPU.
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/local figures

# Submit training rules to SLURM; cheap rules stay local automatically.
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/slurm paper
```

Per-rule overrides: tweak the `resources:` block on the offending rule, or override at the CLI:

```bash
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/slurm \
    --set-resources barrett_igr_long_horizon_mnist_tanh:runtime=720 \
    results/figures/barrett_igr_long_horizon.pdf
```

### Device (CPU vs GPU) for training rules

The training scripts autodetect CUDA → MPS → CPU. To force one, pass `--config device=<value>`:

On Apple Silicon, MPS is supported for local iteration, but the Barrett
training entrypoints will fall back to float32 if you request double
precision. That keeps the runs working on MPS without silently pretending
float64 is available there.

```bash
# Force CPU (e.g., for reproducibility or when no GPU is present):
uv run snakemake -s workflow/Snakefile --config device=cpu --cores 4 figures

# Force CUDA (also a useful sanity check that the env sees the GPU):
uv run snakemake -s workflow/Snakefile --config device=cuda --cores 4 figures

# Defaults (autodetect):
uv run snakemake -s workflow/Snakefile --cores 4 figures
```

The flag threads through to every training rule via [workflow/rules/common.smk](workflow/rules/common.smk) → `device_arg()` → each rule's `{params.device}`.

### Paper assembly

The manuscript is the submodule at [`paper/`](paper). The workflow writes canonical final figures to `results/figures/` and stages manuscript copies into `paper/figures/` (what `\includegraphics{figures/<fig>}` in `paper/main.tex` resolves to). `paper_pdf` depends on every figure having been staged, so `uv run snakemake -s workflow/Snakefile paper` stages figures first and then runs `latexmk`.

Operationally treat the submodule as separate: the parent repo only writes `paper/figures/` and bumps the submodule pointer. Edits to `paper/main.tex` or any hand-authored LaTeX are submodule commits — make and push them from inside `paper/`.

## Core library

- [`src/core/bias.py`](src/core/bias.py): candidate regularizers and bias parameterizations
- [`src/core/estimators.py`](src/core/estimators.py): Lightning-based gradient-matching estimators
- [`src/core/igr_trajectory.py`](src/core/igr_trajectory.py): trajectory-based estimator for the Barrett--Dherin implicit gradient regularizer
- [`src/core/models.py`](src/core/models.py): predictive models (MNIST MLPs, small linear/nonlinear nets)
- [`src/core/data.py`](src/core/data.py): datamodules and dataset loaders
- [`src/core/wandb_utils.py`](src/core/wandb_utils.py): W&B API helpers used by `pull_wandb_sweep`
- [`src/core/plotting.py`](src/core/plotting.py): shared plotting defaults

## Tests

```bash
PYTHONPATH=src uv run pytest tests/
```
