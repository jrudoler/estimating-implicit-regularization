#!/usr/bin/env python3
"""Resampling study for shared-lambda recovery in the matrix-spectrum control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, Sequence

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from function_class_identifiability import (  # noqa: E402
    compute_target_and_bias_gradients,
    parse_bool,
    parse_csv_floats,
    parse_csv_list,
    set_seed,
)
from matrix_spectrum_identifiability import (  # noqa: E402
    LinearMatrixRegressor,
    evaluate_split_metrics,
    generate_matrix_dataset,
    mean_relative_error,
    solve_least_squares_lambdas,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResamplingSummary:
    full_pair_cosine: float
    full_design_condition: float
    full_ols_mean_rel_error: float
    full_ols_max_rel_error: float
    single_resample_mean_rel_error: float
    single_resample_max_rel_error: float
    aggregate_mean_rel_error: float
    aggregate_max_rel_error: float
    stacked_mean_rel_error: float
    stacked_max_rel_error: float
    stacked_relative_residual: float
    stacked_design_condition: float
    replicate_pair_cosine_mean: float
    replicate_pair_cosine_std: float
    replicate_design_condition_mean: float
    replicate_design_condition_std: float


def build_design_matrix(
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
) -> Tensor:
    return torch.stack([bias_gradients[name] for name in bias_names], dim=1)


def design_condition_number(design: Tensor) -> float:
    gram = design.T @ design
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return float(torch.linalg.cond(gram + 1e-8 * eye).item())


def pair_cosine_from_design(design: Tensor) -> float:
    if design.shape[1] != 2:
        return float("nan")
    left = design[:, 0]
    right = design[:, 1]
    denom = float(left.norm().item() * right.norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item() / denom)


def make_resample_indices(
    n_total: int,
    mode: str,
    sample_fraction: float,
    generator: torch.Generator,
) -> Tensor:
    sample_size = max(1, int(round(sample_fraction * n_total)))
    if mode == "subsample":
        return torch.randperm(n_total, generator=generator)[:sample_size]
    if mode == "bootstrap":
        return torch.randint(0, n_total, (sample_size,), generator=generator)
    raise ValueError(f"Unsupported resample mode: {mode}")


def dataset_from_indices(dataset: TensorDataset, indices: Tensor) -> tuple[Tensor, Tensor]:
    inputs, targets = dataset.tensors
    return inputs[indices], targets[indices]


def average_lambda_dicts(
    lambda_dicts: Sequence[Dict[str, float]],
    bias_names: Sequence[str],
) -> Dict[str, float]:
    if not lambda_dicts:
        return {name: 0.0 for name in bias_names}
    result: Dict[str, float] = {}
    for name in bias_names:
        result[name] = float(sum(entry.get(name, 0.0) for entry in lambda_dicts) / len(lambda_dicts))
    return result


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return float(tensor.std(unbiased=True).item())


def run_resampling_study(
    predictive_model: LinearMatrixRegressor,
    train_dataset: TensorDataset,
    bias_names: Sequence[str],
    ground_truth: Dict[str, float],
    resample_mode: str,
    n_replicates: int,
    sample_fraction: float,
    seed: int,
) -> ResamplingSummary:
    full_target_gradient, full_bias_gradients, _ = compute_target_and_bias_gradients(
        predictive_model.network.eval(),
        train_dataset.tensors,
        bias_names,
    )
    full_design = build_design_matrix(full_bias_gradients, bias_names)
    full_estimated, _ = solve_least_squares_lambdas(full_target_gradient, full_bias_gradients, bias_names)
    full_ols_mean_rel_error, full_ols_max_rel_error = mean_relative_error(
        full_estimated,
        ground_truth,
        bias_names,
    )

    per_replicate_estimated: list[Dict[str, float]] = []
    stacked_targets: list[Tensor] = []
    stacked_designs: list[Tensor] = []
    replicate_pair_cosines: list[float] = []
    replicate_design_conditions: list[float] = []

    n_total = len(train_dataset)
    base_generator = torch.Generator().manual_seed(seed + 100_000)
    for replicate_idx in range(n_replicates):
        replicate_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=base_generator).item())
        replicate_generator = torch.Generator().manual_seed(replicate_seed)
        indices = make_resample_indices(
            n_total=n_total,
            mode=resample_mode,
            sample_fraction=sample_fraction,
            generator=replicate_generator,
        )
        batch = dataset_from_indices(train_dataset, indices)
        target_gradient, bias_gradients, _ = compute_target_and_bias_gradients(
            predictive_model.network.eval(),
            batch,
            bias_names,
        )
        design = build_design_matrix(bias_gradients, bias_names)
        estimated, _ = solve_least_squares_lambdas(target_gradient, bias_gradients, bias_names)
        per_replicate_estimated.append(estimated)
        stacked_targets.append(target_gradient)
        stacked_designs.append(design)
        replicate_pair_cosines.append(abs(pair_cosine_from_design(design)))
        replicate_design_conditions.append(design_condition_number(design))

    single_errors = [
        mean_relative_error(estimated, ground_truth, bias_names)
        for estimated in per_replicate_estimated
    ]
    single_resample_mean_rel_error = float(sum(item[0] for item in single_errors) / len(single_errors))
    single_resample_max_rel_error = float(sum(item[1] for item in single_errors) / len(single_errors))

    aggregate_estimated = average_lambda_dicts(per_replicate_estimated, bias_names)
    aggregate_mean_rel_error, aggregate_max_rel_error = mean_relative_error(
        aggregate_estimated,
        ground_truth,
        bias_names,
    )

    stacked_target = torch.cat(stacked_targets, dim=0)
    stacked_design = torch.cat(stacked_designs, dim=0)
    stacked_solution = torch.linalg.lstsq(stacked_design, stacked_target).solution
    stacked_reconstructed = stacked_design @ stacked_solution
    stacked_relative_residual = float(
        (stacked_target - stacked_reconstructed).norm().item()
        / max(stacked_target.norm().item(), 1e-8)
    )
    stacked_estimated = {
        name: float(stacked_solution[idx].item())
        for idx, name in enumerate(bias_names)
    }
    stacked_mean_rel_error, stacked_max_rel_error = mean_relative_error(
        stacked_estimated,
        ground_truth,
        bias_names,
    )

    return ResamplingSummary(
        full_pair_cosine=abs(pair_cosine_from_design(full_design)),
        full_design_condition=design_condition_number(full_design),
        full_ols_mean_rel_error=full_ols_mean_rel_error,
        full_ols_max_rel_error=full_ols_max_rel_error,
        single_resample_mean_rel_error=single_resample_mean_rel_error,
        single_resample_max_rel_error=single_resample_max_rel_error,
        aggregate_mean_rel_error=aggregate_mean_rel_error,
        aggregate_max_rel_error=aggregate_max_rel_error,
        stacked_mean_rel_error=stacked_mean_rel_error,
        stacked_max_rel_error=stacked_max_rel_error,
        stacked_relative_residual=stacked_relative_residual,
        stacked_design_condition=design_condition_number(stacked_design),
        replicate_pair_cosine_mean=float(sum(replicate_pair_cosines) / len(replicate_pair_cosines)),
        replicate_pair_cosine_std=std(replicate_pair_cosines),
        replicate_design_condition_mean=float(
            sum(replicate_design_conditions) / len(replicate_design_conditions)
        ),
        replicate_design_condition_std=std(replicate_design_conditions),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["adam", "lbfgs"])
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-spectrum",
        type=str,
        default="identity",
        choices=["identity", "uniform", "spiked", "power_law"],
    )
    parser.add_argument("--input-rank", type=int, default=0)
    parser.add_argument("--input-spectrum-decay", type=float, default=1.5)
    parser.add_argument(
        "--teacher-spectrum",
        type=str,
        default="power_law",
        choices=["identity", "uniform", "spiked", "power_law"],
    )
    parser.add_argument("--teacher-rank", type=int, default=0)
    parser.add_argument("--teacher-spectrum-decay", type=float, default=0.0)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--gt-bias-types", type=str, default="ridge,nuclear_norm")
    parser.add_argument("--gt-lambdas", type=str, default="0.05,0.05")
    parser.add_argument("--auto-balance-gt-lambdas", action="store_true", default=False)
    parser.add_argument("--gt-lambda-scale", type=float, default=0.05)
    parser.add_argument("--resample-mode", type=str, default="subsample", choices=["subsample", "bootstrap"])
    parser.add_argument("--n-replicates", type=int, default=16)
    parser.add_argument("--sample-fraction", type=float, default=0.25)
    parser.add_argument("--wandb-project", type=str, default="inductive-bias-experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = wandb.init(project=args.wandb_project, job_type="matrix_spectrum_resampling")
    cfg = run.config

    n_samples = int(cfg.get("n_samples", args.n_samples))
    input_dim = int(cfg.get("input_dim", args.input_dim))
    output_dim = int(cfg.get("output_dim", args.output_dim))
    noise_std = float(cfg.get("noise_std", args.noise_std))
    batch_size = int(cfg.get("batch_size", args.batch_size))
    lr = float(cfg.get("lr", args.lr))
    optimizer_name = str(cfg.get("optimizer", args.optimizer))
    max_epochs = int(cfg.get("max_epochs", args.max_epochs))
    patience = int(cfg.get("patience", args.patience))
    val_fraction = float(cfg.get("val_fraction", args.val_fraction))
    test_fraction = float(cfg.get("test_fraction", args.test_fraction))
    seed = int(cfg.get("seed", args.seed))
    input_spectrum = str(cfg.get("input_spectrum", args.input_spectrum))
    input_rank_raw = int(cfg.get("input_rank", args.input_rank))
    input_spectrum_decay = float(cfg.get("input_spectrum_decay", args.input_spectrum_decay))
    teacher_spectrum = str(cfg.get("teacher_spectrum", args.teacher_spectrum))
    teacher_rank_raw = int(cfg.get("teacher_rank", args.teacher_rank))
    teacher_spectrum_decay = float(cfg.get("teacher_spectrum_decay", args.teacher_spectrum_decay))
    target_scale = float(cfg.get("target_scale", args.target_scale))
    gt_bias_types = parse_csv_list(str(cfg.get("gt_bias_types", args.gt_bias_types)))
    gt_lambdas = parse_csv_floats(str(cfg.get("gt_lambdas", args.gt_lambdas)))
    auto_balance_gt_lambdas = parse_bool(cfg.get("auto_balance_gt_lambdas", args.auto_balance_gt_lambdas))
    gt_lambda_scale = float(cfg.get("gt_lambda_scale", args.gt_lambda_scale))
    resample_mode = str(cfg.get("resample_mode", args.resample_mode))
    n_replicates = int(cfg.get("n_replicates", args.n_replicates))
    sample_fraction = float(cfg.get("sample_fraction", args.sample_fraction))

    if len(gt_bias_types) != len(gt_lambdas):
        raise ValueError("gt_bias_types and gt_lambdas must have matching lengths.")
    if len(gt_bias_types) != 2:
        raise ValueError("This resampling study expects exactly two bias types.")

    set_seed(seed)

    dataset, metadata = generate_matrix_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        noise_std=noise_std,
        seed=seed,
        input_rank=input_rank_raw if input_rank_raw > 0 else input_dim,
        input_spectrum=input_spectrum,
        input_spectrum_decay=input_spectrum_decay,
        teacher_rank=teacher_rank_raw if teacher_rank_raw > 0 else min(input_dim, output_dim),
        teacher_spectrum=teacher_spectrum,
        teacher_spectrum_decay=teacher_spectrum_decay,
        target_scale=target_scale,
    )
    test_size = int(len(dataset) * test_fraction)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size - test_size
    train_split, val_split, test_split = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LinearMatrixRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        lr=lr,
        optimizer_name=optimizer_name,
        explicit_bias_types=gt_bias_types,
        explicit_lambdas=gt_lambdas,
    )
    if auto_balance_gt_lambdas:
        from function_class_identifiability import calibrate_explicit_lambdas  # noqa: E402

        gt_lambdas = calibrate_explicit_lambdas(model.network, gt_bias_types, gt_lambda_scale)
        model.explicit_lambdas = list(gt_lambdas)
        model.hparams.explicit_lambdas = list(gt_lambdas)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    model_logger = WandbLogger(
        project=run.project,
        name=f"matrix-resample-train-{teacher_spectrum}-s{seed}",
        experiment=run,
        log_model=False,
    )
    model_trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=model_logger,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="val/loss", mode="min", patience=patience)],
        log_every_n_steps=25,
    )
    model_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    full_inputs, full_targets = dataset.tensors
    train_indices = torch.as_tensor(train_split.indices, dtype=torch.long)
    val_indices = torch.as_tensor(val_split.indices, dtype=torch.long)
    test_indices = torch.as_tensor(test_split.indices, dtype=torch.long)
    train_dataset = TensorDataset(full_inputs[train_indices], full_targets[train_indices])
    val_dataset = TensorDataset(full_inputs[val_indices], full_targets[val_indices])
    test_dataset = TensorDataset(full_inputs[test_indices], full_targets[test_indices])

    train_mse, train_loss = evaluate_split_metrics(model, train_dataset)
    val_mse, val_loss = evaluate_split_metrics(model, val_dataset)
    test_mse, test_loss = evaluate_split_metrics(model, test_dataset)

    gt_map = {name: value for name, value in zip(gt_bias_types, gt_lambdas)}
    summary = run_resampling_study(
        predictive_model=model,
        train_dataset=train_dataset,
        bias_names=gt_bias_types,
        ground_truth=gt_map,
        resample_mode=resample_mode,
        n_replicates=n_replicates,
        sample_fraction=sample_fraction,
        seed=seed,
    )

    pair_key = "__".join(gt_bias_types)
    scalar_logs = {
        "fit/train_mse": train_mse,
        "fit/train_loss": train_loss,
        "fit/val_mse": val_mse,
        "fit/val_loss": val_loss,
        "fit/test_mse": test_mse,
        "fit/test_loss": test_loss,
        "resampling/full_pair_cosine": summary.full_pair_cosine,
        "resampling/full_design_condition": summary.full_design_condition,
        "resampling/full_ols_mean_rel_error": summary.full_ols_mean_rel_error,
        "resampling/full_ols_max_rel_error": summary.full_ols_max_rel_error,
        "resampling/single_resample_mean_rel_error": summary.single_resample_mean_rel_error,
        "resampling/single_resample_max_rel_error": summary.single_resample_max_rel_error,
        "resampling/aggregate_mean_rel_error": summary.aggregate_mean_rel_error,
        "resampling/aggregate_max_rel_error": summary.aggregate_max_rel_error,
        "resampling/stacked_mean_rel_error": summary.stacked_mean_rel_error,
        "resampling/stacked_max_rel_error": summary.stacked_max_rel_error,
        "resampling/stacked_relative_residual": summary.stacked_relative_residual,
        "resampling/stacked_design_condition": summary.stacked_design_condition,
        "resampling/replicate_pair_cosine_mean": summary.replicate_pair_cosine_mean,
        "resampling/replicate_pair_cosine_std": summary.replicate_pair_cosine_std,
        "resampling/replicate_design_condition_mean": summary.replicate_design_condition_mean,
        "resampling/replicate_design_condition_std": summary.replicate_design_condition_std,
        "data/teacher_effective_rank": metadata.teacher_effective_rank,
        "data/teacher_condition_number": metadata.teacher_condition_number,
        "data/input_effective_rank": metadata.input_effective_rank,
        "data/input_condition_number": metadata.input_condition_number,
        "config/resample_mode": resample_mode,
        "config/n_replicates": n_replicates,
        "config/sample_fraction": sample_fraction,
        "config/pair": pair_key,
    }

    LOGGER.info(
        "Fit | pair=%s teacher_spectrum=%s train_mse=%.6f val_mse=%.6f test_mse=%.6f train_loss=%.6f val_loss=%.6f test_loss=%.6f",
        pair_key,
        metadata.teacher_spectrum,
        train_mse,
        val_mse,
        test_mse,
        train_loss,
        val_loss,
        test_loss,
    )
    LOGGER.info(
        "Resample | pair=%s mode=%s full_cos=%.6f full_cond=%.6f single_mean_rel_error=%.6f aggregate_mean_rel_error=%.6f stacked_mean_rel_error=%.6f stacked_cond=%.6f stacked_relative_residual=%.6f replicate_cos_mean=%.6f",
        pair_key,
        resample_mode,
        summary.full_pair_cosine,
        summary.full_design_condition,
        summary.single_resample_mean_rel_error,
        summary.aggregate_mean_rel_error,
        summary.stacked_mean_rel_error,
        summary.stacked_design_condition,
        summary.stacked_relative_residual,
        summary.replicate_pair_cosine_mean,
    )

    wandb.log(scalar_logs)
    run.summary.update(scalar_logs)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
