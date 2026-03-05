# Inductive Bias Estimation

This repository contains the code for the inductive bias estimation project.

## Getting Started

To get started, you need to install the dependencies. You can do this by running:

```bash
uv sync
```

## Running the Code

### Experiment Workflows

- Canonical experiment inventory: `experiments/REGISTRY.yaml`
- Canonical sweep launcher: `scripts/wandb_sweep.slurm`
- Sweep definitions: `sweeps/*.yaml`

Typical workflow:

```bash
wandb sweep sweeps/<config>.yaml
sbatch scripts/wandb_sweep.slurm <SWEEP_ID>
```