#!/usr/bin/env python3
"""Fixed-weight resampling study on the structured nonlinear teacher-student pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, Sequence

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from function_class_identifiability import (  # noqa: E402
    DeepReLURegressor,
    calibrate_explicit_lambdas,
    compute_target_and_bias_gradients,
    cosine_between,
    generate_dataset,
    parse_csv_floats,
    parse_csv_list,
    set_seed,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResamplingSummary:
    test_mse: float
    test_loss: float
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


def build_design_matrix(
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
) -> Tensor:
    return torch.stack([bias_gradients[name] for name in bias_names], dim=1)


def design_condition_number(design: Tensor) -> float:
    gram = design.T @ design
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return float(torch.linalg.cond(gram + 1e-8 * eye).item())


def mean_relative_error(
    estimated: Dict[str, float],
    ground_truth: Dict[str, float],
    bias_names: Sequence[str],
) -> tuple[float, float]:
    total_rel_error = 0.0
    max_rel_error = 0.0
    for bias_name in bias_names:
        true_value = float(ground_truth.get(bias_name, 0.0))
        est_value = float(estimated.get(bias_name, 0.0))
        rel_error = abs(est_value - true_value) / max(abs(true_value), 1e-8)
        total_rel_error += rel_error
        max_rel_error = max(max_rel_error, rel_error)
    return total_rel_error / max(len(bias_names), 1), max_rel_error


def solve_least_squares_lambdas(
    target_gradient: Tensor,
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
) -> tuple[Dict[str, float], float]:
    design = build_design_matrix(bias_gradients, bias_names)
    solution = torch.linalg.lstsq(design, target_gradient).solution
    reconstructed = design @ solution
    residual = float(
        (target_gradient - reconstructed).norm().item() / max(target_gradient.norm().item(), 1e-8)
    )
    return (
        {name: float(solution[idx].item()) for idx, name in enumerate(bias_names)},
        residual,
    )


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
    return {
        name: float(sum(entry.get(name, 0.0) for entry in lambda_dicts) / len(lambda_dicts))
        for name in bias_names
    }


def evaluate_split_metrics(model: DeepReLURegressor, dataset: TensorDataset) -> tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs, targets = dataset.tensors
        predictions = model(inputs.to(device))
        mse_loss = F.mse_loss(predictions, targets.to(device))
        regularization, _ = model._explicit_regularization()
        total_loss = mse_loss + regularization
    return float(mse_loss.item()), float(total_loss.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-spectrum", type=str, default="identity")
    parser.add_argument("--input-rank", type=int, default=0)
    parser.add_argument("--input-spectrum-decay", type=float, default=1.5)
    parser.add_argument("--teacher-spectrum", type=str, default="identity")
    parser.add_argument("--teacher-rank", type=int, default=0)
    parser.add_argument("--teacher-spectrum-decay", type=float, default=0.0)
    parser.add_argument("--teacher-hidden-dim", type=int, default=16)
    parser.add_argument("--teacher-activation", type=str, default="relu", choices=["relu", "tanh", "linear"])
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--gt-bias-types", type=str, default="ridge,nuclear_norm")
    parser.add_argument("--gt-lambdas", type=str, default="0.05,0.05")
    parser.add_argument("--auto-balance-gt-lambdas", action="store_true", default=False)
    parser.add_argument("--gt-lambda-scale", type=float, default=0.05)
    parser.add_argument("--resample-mode", type=str, default="bootstrap", choices=["subsample", "bootstrap"])
    parser.add_argument("--n-replicates", type=int, default=16)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    gt_bias_types = parse_csv_list(args.gt_bias_types)
    gt_lambdas = parse_csv_floats(args.gt_lambdas)
    dataset, _ = generate_dataset(
        n_samples=args.n_samples,
        input_dim=args.input_dim,
        function_class="teacher_relu",
        noise_std=args.noise_std,
        seed=args.seed,
        data_mode="structured",
        input_rank=args.input_rank if args.input_rank > 0 else args.input_dim,
        input_spectrum=args.input_spectrum,
        input_spectrum_decay=args.input_spectrum_decay,
        teacher_rank=args.teacher_rank if args.teacher_rank > 0 else min(args.input_dim, args.teacher_hidden_dim),
        teacher_spectrum=args.teacher_spectrum,
        teacher_spectrum_decay=args.teacher_spectrum_decay,
        teacher_hidden_dim=args.teacher_hidden_dim,
        teacher_activation=args.teacher_activation,
        target_scale=args.target_scale,
    )
    test_size = int(len(dataset) * args.test_fraction)
    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size - test_size
    train_split, val_split, test_split = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = DeepReLURegressor(
        input_dim=args.input_dim,
        depth=args.depth,
        width=args.width,
        lr=args.lr,
        explicit_bias_types=gt_bias_types,
        explicit_lambdas=gt_lambdas,
    )
    if args.auto_balance_gt_lambdas:
        gt_lambdas = calibrate_explicit_lambdas(model.network, gt_bias_types, args.gt_lambda_scale)
        model.explicit_lambdas = list(gt_lambdas)
        model.hparams.explicit_lambdas = list(gt_lambdas)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=args.patience),
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True),
        ],
        log_every_n_steps=25,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    full_inputs, full_targets = dataset.tensors
    train_indices = torch.as_tensor(train_split.indices, dtype=torch.long)
    test_indices = torch.as_tensor(test_split.indices, dtype=torch.long)
    train_dataset = TensorDataset(full_inputs[train_indices], full_targets[train_indices])
    test_dataset = TensorDataset(full_inputs[test_indices], full_targets[test_indices])
    test_mse, test_loss = evaluate_split_metrics(model, test_dataset)

    gt_map = {name: value for name, value in zip(gt_bias_types, gt_lambdas)}
    full_target_gradient, full_bias_gradients, _ = compute_target_and_bias_gradients(
        model.network.eval(),
        train_dataset.tensors,
        gt_bias_types,
    )
    full_pair_cosine = abs(cosine_between(full_bias_gradients[gt_bias_types[0]], full_bias_gradients[gt_bias_types[1]]))
    full_estimated, _ = solve_least_squares_lambdas(full_target_gradient, full_bias_gradients, gt_bias_types)
    full_ols_mean_rel_error, full_ols_max_rel_error = mean_relative_error(full_estimated, gt_map, gt_bias_types)

    per_replicate_estimated: list[Dict[str, float]] = []
    stacked_targets: list[Tensor] = []
    stacked_designs: list[Tensor] = []
    n_total = len(train_dataset)
    base_generator = torch.Generator().manual_seed(args.seed + 100_000)
    for _ in range(args.n_replicates):
        replicate_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=base_generator).item())
        indices = make_resample_indices(
            n_total=n_total,
            mode=args.resample_mode,
            sample_fraction=args.sample_fraction,
            generator=torch.Generator().manual_seed(replicate_seed),
        )
        batch = dataset_from_indices(train_dataset, indices)
        target_gradient, bias_gradients, _ = compute_target_and_bias_gradients(
            model.network.eval(),
            batch,
            gt_bias_types,
        )
        design = build_design_matrix(bias_gradients, gt_bias_types)
        estimated, _ = solve_least_squares_lambdas(target_gradient, bias_gradients, gt_bias_types)
        per_replicate_estimated.append(estimated)
        stacked_targets.append(target_gradient)
        stacked_designs.append(design)

    single_errors = [mean_relative_error(estimated, gt_map, gt_bias_types) for estimated in per_replicate_estimated]
    single_mean_rel_error = float(sum(item[0] for item in single_errors) / len(single_errors))
    single_max_rel_error = float(sum(item[1] for item in single_errors) / len(single_errors))
    aggregate_estimated = average_lambda_dicts(per_replicate_estimated, gt_bias_types)
    aggregate_mean_rel_error, aggregate_max_rel_error = mean_relative_error(
        aggregate_estimated,
        gt_map,
        gt_bias_types,
    )
    stacked_target = torch.cat(stacked_targets, dim=0)
    stacked_design = torch.cat(stacked_designs, dim=0)
    stacked_solution = torch.linalg.lstsq(stacked_design, stacked_target).solution
    stacked_reconstructed = stacked_design @ stacked_solution
    stacked_relative_residual = float(
        (stacked_target - stacked_reconstructed).norm().item() / max(stacked_target.norm().item(), 1e-8)
    )
    stacked_estimated = {name: float(stacked_solution[idx].item()) for idx, name in enumerate(gt_bias_types)}
    stacked_mean_rel_error, stacked_max_rel_error = mean_relative_error(stacked_estimated, gt_map, gt_bias_types)

    LOGGER.info(
        "NonlinearResample | teacher_activation=%s depth=%d width=%d teacher_spectrum=%s resample_mode=%s test_mse=%.6f full_cos=%.6f full_cond=%.6f full_ols_mean_rel_error=%.6f single_mean_rel_error=%.6f aggregate_mean_rel_error=%.6f stacked_mean_rel_error=%.6f stacked_relative_residual=%.6f",
        args.teacher_activation,
        args.depth,
        args.width,
        args.teacher_spectrum,
        args.resample_mode,
        test_mse,
        full_pair_cosine,
        design_condition_number(build_design_matrix(full_bias_gradients, gt_bias_types)),
        full_ols_mean_rel_error,
        single_mean_rel_error,
        aggregate_mean_rel_error,
        stacked_mean_rel_error,
        stacked_relative_residual,
    )
    for bias_name in gt_bias_types:
        LOGGER.info(
            "  %s | true=%.6f full_ols=%.6f aggregate=%.6f stacked=%.6f",
            bias_name,
            gt_map[bias_name],
            full_estimated[bias_name],
            aggregate_estimated[bias_name],
            stacked_estimated[bias_name],
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
