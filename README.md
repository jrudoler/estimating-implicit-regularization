# Inductive Bias Estimation

This repository contains the code for **A Framework for Estimating the Implicit Regularization Effect of Learning Algorithms**. The project studies how to empirically recover and compare hypothesized inductive biases by treating a trained model as if it were the solution to a regularized learning problem, then matching the gradient of a candidate regularizer to the residual gradient of the predictive loss.

The central idea is a gradient-matching formulation of implicit regularization. Given trained weights and data, the code estimates regularizer parameters that best explain the observed training residuals. This lets the project do several things in one framework:

- recover explicit regularizers such as L2 weight decay
- test theoretical predictions about implicit bias, including early stopping style effects
- compare candidate regularizers under identifiability and collinearity constraints
- study practical settings such as dropout and deep nonlinear MNIST models

## Research Focus

The manuscript in [`paper/`](/home/jrudoler/inductive-bias/paper) frames the project around a simple inverse problem:

- train a predictive model with some algorithm or architectural choice
- compute the predictive loss gradient at the learned solution
- specify a parameterized family of candidate regularizers
- fit the regularizer parameters so their gradient matches the negative loss gradient

In the codebase, these fitted parameters are the estimated effective regularization or best effective regularizer for the chosen bias family. The repo includes both controlled synthetic studies and larger neural-network experiments to evaluate when this recovery is accurate, when it is misspecified, and when geometry makes the problem ill-conditioned.

## Getting Started

This repo uses `uv` for environment management.

```bash
git clone --recurse-submodules git@github.com:jrudoler/inductive-bias.git
cd inductive-bias
uv sync
git submodule update --init --recursive
```

If you already cloned the parent repo without submodules:

```bash
git submodule update --init --recursive
```

The manuscript repo is checked out at [`paper/`](/home/jrudoler/inductive-bias/paper) as a submodule. It is available as read-only context for ordinary coding work. By default, implementation and method updates should go to code and non-paper docs, not the manuscript.

## Typical Workflows

### Run a Single Experiment

Use `uv run` for Python entrypoints:

```bash
uv run python experiments/l2_train_and_recover.py
```

Other common entrypoints include:

- `experiments/function_class_identifiability.py`
- `experiments/mixed_bias_recovery.py`
- `experiments/dropout_bias_estimation.py`
- `experiments/bootstrap_bias_recovery.py`

### Run A W&B Sweep

The canonical experiment inventory lives in [`experiments/REGISTRY.yaml`](/home/jrudoler/inductive-bias/experiments/REGISTRY.yaml). The canonical Slurm launcher is [`scripts/wandb_sweep.slurm`](/home/jrudoler/inductive-bias/scripts/wandb_sweep.slurm).

Typical workflow:

```bash
uv run wandb sweep sweeps/<config>.yaml
sbatch scripts/wandb_sweep.slurm <SWEEP_ID>
```

### Work With The Manuscript As Context

Refresh the manuscript locally when needed:

```bash
git -C paper fetch origin
git -C paper pull --ff-only origin main
```

Read from `paper/` for research context, definitions, and framing, but do not edit it unless manuscript changes are explicitly requested.

## Codebase Overview

### Core Library

- [`src/core/bias.py`](/home/jrudoler/inductive-bias/src/core/bias.py): candidate regularizers and bias parameterizations, including L2-adjacent, spectral, orthogonality, and gradient-penalty style bias models
- [`src/core/estimators.py`](/home/jrudoler/inductive-bias/src/core/estimators.py): Lightning-based gradient-matching estimators that fit bias parameters against predictive loss gradients
- [`src/core/models.py`](/home/jrudoler/inductive-bias/src/core/models.py): predictive models used in experiments, including MNIST MLPs and smaller linear or nonlinear networks
- [`src/core/data.py`](/home/jrudoler/inductive-bias/src/core/data.py): datamodules and dataset loaders
- [`src/core/plotting.py`](/home/jrudoler/inductive-bias/src/core/plotting.py): plotting helpers for experiment summaries
- [`src/core/callbacks.py`](/home/jrudoler/inductive-bias/src/core/callbacks.py), [`src/core/utils.py`](/home/jrudoler/inductive-bias/src/core/utils.py), [`src/core/wandb_utils.py`](/home/jrudoler/inductive-bias/src/core/wandb_utils.py): training utilities, logging helpers, and support code

### Experiments

- [`experiments/REGISTRY.yaml`](/home/jrudoler/inductive-bias/experiments/REGISTRY.yaml): canonical inventory of active, appendix, legacy, and deprecated experiment assets
- recovery scripts such as [`experiments/l2_train_and_recover.py`](/home/jrudoler/inductive-bias/experiments/l2_train_and_recover.py), [`experiments/l2_nuclear_train_and_recover.py`](/home/jrudoler/inductive-bias/experiments/l2_nuclear_train_and_recover.py), and [`experiments/l2_orthogonal_train_and_recover.py`](/home/jrudoler/inductive-bias/experiments/l2_orthogonal_train_and_recover.py)
- mixed-bias and identifiability studies such as [`experiments/mixed_bias_recovery.py`](/home/jrudoler/inductive-bias/experiments/mixed_bias_recovery.py), [`experiments/function_class_identifiability.py`](/home/jrudoler/inductive-bias/experiments/function_class_identifiability.py), and [`experiments/nonlinear_multi_geometry_suite.py`](/home/jrudoler/inductive-bias/experiments/nonlinear_multi_geometry_suite.py)
- practical deep-learning settings such as [`experiments/dropout_bias_estimation.py`](/home/jrudoler/inductive-bias/experiments/dropout_bias_estimation.py), [`experiments/mnist_implicit_reg.py`](/home/jrudoler/inductive-bias/experiments/mnist_implicit_reg.py), and [`experiments/mnist_deep_relu_bias.py`](/home/jrudoler/inductive-bias/experiments/mnist_deep_relu_bias.py)
- stability and resampling work such as [`experiments/bootstrap_bias_recovery.py`](/home/jrudoler/inductive-bias/experiments/bootstrap_bias_recovery.py)

### Analysis And Documentation

- [`analysis/`](/home/jrudoler/inductive-bias/analysis): experiment summaries and analysis scripts, including bootstrap and identifiability writeups
- [`docs/`](/home/jrudoler/inductive-bias/docs): project notes, handoff context, manuscript-adjacent notes, and autonomous work logs
- [`docs/autonomous_notes.md`](/home/jrudoler/inductive-bias/docs/autonomous_notes.md): default non-paper log for autonomous method or implementation updates

### Sweep And Cluster Tooling

- [`scripts/wandb_sweep.slurm`](/home/jrudoler/inductive-bias/scripts/wandb_sweep.slurm): canonical GPU sweep launcher on the cluster
- [`scripts/bias_from_sweep.py`](/home/jrudoler/inductive-bias/scripts/bias_from_sweep.py): download checkpoints from a W&B sweep and optionally run downstream bias estimation
- [`scripts/build_paper_figures.py`](/home/jrudoler/inductive-bias/scripts/build_paper_figures.py): regenerate or verify the figures referenced by [`paper/main.tex`](/home/jrudoler/inductive-bias/paper/main.tex)
- [`scripts/`](/home/jrudoler/inductive-bias/scripts): sweep helpers, launch utilities, and reporting scripts

### Tests And Profiling

- [`tests/test_predictive_loss_grad.py`](/home/jrudoler/inductive-bias/tests/test_predictive_loss_grad.py): checks predictive-loss gradient calculations used by the estimators
- [`tests/test_function_class_identifiability_structured.py`](/home/jrudoler/inductive-bias/tests/test_function_class_identifiability_structured.py): structured regression and identifiability coverage
- [`tests/PROFILING_README.md`](/home/jrudoler/inductive-bias/tests/PROFILING_README.md): profiling notes and supporting assets

## Project Structure At A Glance

```text
src/core/        Core bias models, estimators, data modules, and utilities
experiments/     Main experimental entrypoints and registry
analysis/        Analysis code and result summaries
scripts/         Sweep launchers and experiment utilities
tests/           Tests, smoke checks, and profiling helpers
docs/            Project notes and autonomous documentation
paper/           Manuscript submodule for read-only context by default
archive/         Archived experiment assets retained for reproducibility
```

## Notes On Manuscript Workflow

The manuscript lives in the separate git submodule at [`paper/`](/home/jrudoler/inductive-bias/paper). Treat it as operationally separate from the parent repo:

- use it freely as context during coding tasks
- do not edit it unless manuscript changes are explicitly requested
- keep manuscript commits separate from parent repo commits
- if manuscript changes are made, decide separately whether to commit the updated submodule pointer in the parent repo

## Current State

The repository includes active work on:

- explicit regularizer recovery in deep networks
- multi-regularizer identifiability and collinearity analysis
- synthetic function-class experiments
- dropout and early-stopping style implicit bias studies
- bootstrap and subsampling diagnostics for recovery stability

For the most complete research framing, read the manuscript in [`paper/main.tex`](/home/jrudoler/inductive-bias/paper/main.tex). For the most accurate experiment inventory, use [`experiments/REGISTRY.yaml`](/home/jrudoler/inductive-bias/experiments/REGISTRY.yaml).
