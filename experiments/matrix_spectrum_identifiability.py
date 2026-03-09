#!/usr/bin/env python3
"""Matrix-spectrum identifiability experiment with a multi-output linear student."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, Sequence, Tuple

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import JointBias  # noqa: E402
from function_class_identifiability import (  # noqa: E402
    BiasWithMSENormalized,
    build_bias_module,
    build_spectrum_values,
    calibrate_explicit_lambdas,
    compute_target_and_bias_gradients,
    cosine_between,
    geometry_stats,
    get_scale_params,
    pairwise_cosine_matrix,
    parse_bool,
    parse_csv_floats,
    parse_csv_list,
    participation_ratio,
    positive_condition_number,
    random_orthogonal_matrix,
    set_seed,
)
from core.estimators import BiasWithMSE, vector_to_parameter_views  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixDatasetMetadata:
    input_rank: int
    input_effective_rank: float
    input_condition_number: float
    input_spectrum: str
    input_spectrum_decay: float
    teacher_rank: int
    teacher_effective_rank: float
    teacher_condition_number: float
    teacher_spectrum: str
    teacher_spectrum_decay: float
    output_dim: int
    target_std: float


def build_teacher_matrix(
    input_dim: int,
    output_dim: int,
    teacher_rank: int,
    teacher_spectrum: str,
    teacher_spectrum_decay: float,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    max_rank = min(input_dim, output_dim)
    active_rank = max(1, min(teacher_rank, max_rank))
    singular_values = build_spectrum_values(
        size=max_rank,
        active_rank=active_rank,
        family=teacher_spectrum,
        decay=teacher_spectrum_decay,
        target_sum=float(active_rank),
    )
    if teacher_spectrum == "identity":
        weight = torch.zeros(output_dim, input_dim, dtype=torch.float32)
        weight[:active_rank, :active_rank] = torch.diag(singular_values[:active_rank])
        return weight, singular_values

    left = random_orthogonal_matrix(output_dim, max_rank, generator)
    right = random_orthogonal_matrix(input_dim, max_rank, generator)
    weight = left @ torch.diag(singular_values) @ right.T
    return weight, singular_values


def generate_matrix_dataset(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise_std: float,
    seed: int,
    input_rank: int,
    input_spectrum: str,
    input_spectrum_decay: float,
    teacher_rank: int,
    teacher_spectrum: str,
    teacher_spectrum_decay: float,
    target_scale: float,
) -> Tuple[TensorDataset, MatrixDatasetMetadata]:
    generator = torch.Generator().manual_seed(seed)

    resolved_input_rank = max(1, min(input_rank, input_dim))
    input_eigenvalues = build_spectrum_values(
        size=input_dim,
        active_rank=resolved_input_rank,
        family=input_spectrum,
        decay=input_spectrum_decay,
        target_sum=float(input_dim),
    )
    input_basis = random_orthogonal_matrix(input_dim, input_dim, generator)
    latent = torch.randn(n_samples, input_dim, generator=generator, dtype=torch.float32)
    features = (latent * torch.sqrt(input_eigenvalues).unsqueeze(0)) @ input_basis.T

    resolved_teacher_rank = max(1, min(teacher_rank, min(input_dim, output_dim)))
    teacher_weight, teacher_singular_values = build_teacher_matrix(
        input_dim=input_dim,
        output_dim=output_dim,
        teacher_rank=resolved_teacher_rank,
        teacher_spectrum=teacher_spectrum,
        teacher_spectrum_decay=teacher_spectrum_decay,
        generator=generator,
    )
    targets = features @ teacher_weight.T
    target_std = float(targets.std().item())
    if target_std > 1e-8:
        targets = targets * (target_scale / target_std)
    if noise_std > 0:
        targets = targets + noise_std * torch.randn(
            n_samples,
            output_dim,
            generator=generator,
            dtype=torch.float32,
        )

    metadata = MatrixDatasetMetadata(
        input_rank=resolved_input_rank,
        input_effective_rank=participation_ratio(input_eigenvalues),
        input_condition_number=positive_condition_number(input_eigenvalues),
        input_spectrum=input_spectrum,
        input_spectrum_decay=input_spectrum_decay,
        teacher_rank=resolved_teacher_rank,
        teacher_effective_rank=participation_ratio(teacher_singular_values),
        teacher_condition_number=positive_condition_number(teacher_singular_values),
        teacher_spectrum=teacher_spectrum,
        teacher_spectrum_decay=teacher_spectrum_decay,
        output_dim=output_dim,
        target_std=float(targets.std().item()),
    )
    return TensorDataset(features.float(), targets.float()), metadata


class LinearMatrixRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float,
        optimizer_name: str,
        explicit_bias_types: Sequence[str],
        explicit_lambdas: Sequence[float],
    ) -> None:
        super().__init__()
        if len(explicit_bias_types) != len(explicit_lambdas):
            raise ValueError("explicit_bias_types and explicit_lambdas must have same length.")

        self.network = nn.Linear(input_dim, output_dim, bias=False)
        self.loss_fn = nn.MSELoss()
        self.optimizer_name = optimizer_name
        self.explicit_bias_types = list(explicit_bias_types)
        self.explicit_lambdas = [float(value) for value in explicit_lambdas]
        self.explicit_bias_modules: Dict[str, nn.Module] = {
            bias_name: build_bias_module(bias_name, trainable=False)
            for bias_name in self.explicit_bias_types
        }
        self.save_hyperparameters()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)

    def _explicit_regularization(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        flat_params = torch.nn.utils.parameters_to_vector(list(self.network.parameters()))
        structured_params = vector_to_parameter_views(flat_params, self.network)
        total_penalty = torch.zeros((), device=flat_params.device, dtype=flat_params.dtype)
        components: Dict[str, Tensor] = {}
        for bias_name, coefficient in zip(self.explicit_bias_types, self.explicit_lambdas):
            bias_module = self.explicit_bias_modules[bias_name]
            first_param = next(bias_module.parameters(), None)
            if first_param is not None and first_param.device != flat_params.device:
                bias_module = bias_module.to(flat_params.device)
                self.explicit_bias_modules[bias_name] = bias_module
            penalty_value = bias_module(flat_params, structured_params)
            weighted_penalty = coefficient * penalty_value
            components[bias_name] = weighted_penalty
            total_penalty = total_penalty + weighted_penalty
        return total_penalty, components

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        predictions = self(inputs)
        mse_loss = self.loss_fn(predictions, targets)
        regularization, components = self._explicit_regularization()
        loss = mse_loss + regularization
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/regularization", regularization, on_step=False, on_epoch=True, prog_bar=False)
        for name, value in components.items():
            self.log(f"train/penalty/{name}", value, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        predictions = self(inputs)
        mse_loss = self.loss_fn(predictions, targets)
        regularization, _ = self._explicit_regularization()
        loss = mse_loss + regularization
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        predictions = self(inputs)
        mse_loss = self.loss_fn(predictions, targets)
        regularization, _ = self._explicit_regularization()
        loss = mse_loss + regularization
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/mse", mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "lbfgs":
            return torch.optim.LBFGS(
                self.network.parameters(),
                lr=self.hparams.lr,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


def evaluate_split_metrics(model: LinearMatrixRegressor, dataset: TensorDataset) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs, targets = dataset.tensors
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        mse_loss = F.mse_loss(predictions, targets)
        regularization, _ = model._explicit_regularization()
        total_loss = mse_loss + regularization
    return float(mse_loss.item()), float(total_loss.item())


def solve_least_squares_lambdas(
    target_gradient: Tensor,
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
) -> Tuple[Dict[str, float], float]:
    design = torch.stack([bias_gradients[name] for name in bias_names], dim=1)
    solution = torch.linalg.lstsq(design, target_gradient).solution
    reconstructed = design @ solution
    relative_residual = float(
        (target_gradient - reconstructed).norm().item() / max(target_gradient.norm().item(), 1e-8)
    )
    return (
        {name: float(solution[idx].item()) for idx, name in enumerate(bias_names)},
        relative_residual,
    )


def mean_relative_error(
    estimated: Dict[str, float],
    ground_truth: Dict[str, float],
    bias_names: Sequence[str],
) -> Tuple[float, float]:
    total_rel_error = 0.0
    max_rel_error = 0.0
    for bias_name in bias_names:
        true_value = float(ground_truth.get(bias_name, 0.0))
        est_value = float(estimated.get(bias_name, 0.0))
        rel_error = abs(est_value - true_value) / max(abs(true_value), 1e-8)
        total_rel_error += rel_error
        max_rel_error = max(max_rel_error, rel_error)
    return total_rel_error / max(len(bias_names), 1), max_rel_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["adam", "lbfgs"])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
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
        default="identity",
        choices=["identity", "uniform", "spiked", "power_law"],
    )
    parser.add_argument("--teacher-rank", type=int, default=0)
    parser.add_argument("--teacher-spectrum-decay", type=float, default=1.5)
    parser.add_argument("--target-scale", type=float, default=1.0)

    parser.add_argument("--gt-bias-types", type=str, default="ridge,nuclear_norm")
    parser.add_argument("--gt-lambdas", type=str, default="0.05,0.05")
    parser.add_argument("--auto-balance-gt-lambdas", action="store_true", default=False)
    parser.add_argument("--gt-lambda-scale", type=float, default=0.05)

    parser.add_argument(
        "--candidate-bias-types",
        type=str,
        default="ridge,nuclear_norm,stable_rank,spectral_entropy",
    )
    parser.add_argument("--estimation-bias-types", type=str, default="")
    parser.add_argument("--screen-batch-size", type=int, default=0)

    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-max-epochs", type=int, default=2000)
    parser.add_argument("--bias-patience", type=int, default=100)
    parser.add_argument(
        "--estimator-mode",
        type=str,
        default="normalized",
        choices=["standard", "normalized"],
    )
    parser.add_argument("--recovery-noise-std", type=float, default=0.0)
    parser.add_argument("--recovery-noise-draws", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="inductive-bias-experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run = wandb.init(project=args.wandb_project, job_type="matrix_spectrum_identifiability")
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
    candidate_bias_types = parse_csv_list(str(cfg.get("candidate_bias_types", args.candidate_bias_types)))
    estimation_bias_types_raw = str(cfg.get("estimation_bias_types", args.estimation_bias_types))
    screen_batch_size = int(cfg.get("screen_batch_size", args.screen_batch_size))
    bias_lr = float(cfg.get("bias_lr", args.bias_lr))
    bias_max_epochs = int(cfg.get("bias_max_epochs", args.bias_max_epochs))
    bias_patience = int(cfg.get("bias_patience", args.bias_patience))
    estimator_mode = str(cfg.get("estimator_mode", args.estimator_mode))
    recovery_noise_std = float(cfg.get("recovery_noise_std", args.recovery_noise_std))
    recovery_noise_draws = int(cfg.get("recovery_noise_draws", args.recovery_noise_draws))

    if len(gt_bias_types) != len(gt_lambdas):
        raise ValueError("gt_bias_types and gt_lambdas must have matching lengths.")

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
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LinearMatrixRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        lr=lr,
        optimizer_name=optimizer_name,
        explicit_bias_types=gt_bias_types,
        explicit_lambdas=gt_lambdas,
    )
    if auto_balance_gt_lambdas:
        gt_lambdas = calibrate_explicit_lambdas(model.network, gt_bias_types, gt_lambda_scale)
        model.explicit_lambdas = list(gt_lambdas)
        model.hparams.explicit_lambdas = list(gt_lambdas)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    model_logger = WandbLogger(
        project=run.project,
        name=f"matrix-train-{teacher_spectrum}-s{seed}",
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
    train_dataset = TensorDataset(*train_split.dataset.tensors[0:2])
    train_indices = torch.as_tensor(train_split.indices, dtype=torch.long)
    val_indices = torch.as_tensor(val_split.indices, dtype=torch.long)
    test_indices = torch.as_tensor(test_split.indices, dtype=torch.long)
    full_inputs, full_targets = dataset.tensors
    train_dataset = TensorDataset(full_inputs[train_indices], full_targets[train_indices])
    val_dataset = TensorDataset(full_inputs[val_indices], full_targets[val_indices])
    test_dataset = TensorDataset(full_inputs[test_indices], full_targets[test_indices])
    train_mse, train_loss = evaluate_split_metrics(model, train_dataset)
    val_mse, val_loss = evaluate_split_metrics(model, val_dataset)
    test_mse, test_loss = evaluate_split_metrics(model, test_dataset)

    if screen_batch_size > 0:
        screening_loader = DataLoader(
            train_split,
            batch_size=min(screen_batch_size, len(train_split)),
            shuffle=False,
            num_workers=0,
        )
        screening_batch = next(iter(screening_loader))
    else:
        screening_batch = train_dataset.tensors
    target_gradient, bias_gradients, target_alignments = compute_target_and_bias_gradients(
        model.network.eval(),
        screening_batch,
        candidate_bias_types,
    )
    cosine_matrix = pairwise_cosine_matrix(bias_gradients, candidate_bias_types)
    max_offdiag_all, condition_all = geometry_stats(cosine_matrix)

    estimation_bias_types = (
        parse_csv_list(estimation_bias_types_raw) if estimation_bias_types_raw else list(gt_bias_types)
    )
    index_lookup = {name: idx for idx, name in enumerate(candidate_bias_types)}
    selected_indices = [index_lookup[name] for name in estimation_bias_types]
    selected_matrix = cosine_matrix[selected_indices][:, selected_indices]
    max_offdiag_selected, condition_selected = geometry_stats(selected_matrix)
    pair_key = "__".join(estimation_bias_types)
    primary_pair_cosine = float("nan")
    if len(estimation_bias_types) == 2:
        primary_pair_cosine = abs(
            cosine_between(
                bias_gradients[estimation_bias_types[0]],
                bias_gradients[estimation_bias_types[1]],
            )
        )
    gt_gradient = torch.zeros_like(target_gradient)
    for bias_name, lambda_value in zip(gt_bias_types, gt_lambdas):
        gt_gradient = gt_gradient + float(lambda_value) * bias_gradients[bias_name]
    gt_relative_residual = float(
        (target_gradient - gt_gradient).norm().item() / max(target_gradient.norm().item(), 1e-8)
    )
    ols_estimated, ols_relative_residual = solve_least_squares_lambdas(
        target_gradient,
        bias_gradients,
        estimation_bias_types,
    )

    estimators = [build_bias_module(name, trainable=True) for name in estimation_bias_types]
    joint_bias = JointBias(estimators)
    if estimator_mode == "normalized":
        bias_estimator = BiasWithMSENormalized(
            predictive_model=model.network.eval(),
            bias_model=joint_bias,
            grad_match_loss_fn=F.mse_loss,
            optimizer_cls=torch.optim.Adam,
            lr=bias_lr,
            bias_lr=bias_lr,
        )
    else:
        bias_estimator = BiasWithMSE(
            predictive_model=model.network.eval(),
            bias_model=joint_bias,
            grad_match_loss_fn=F.mse_loss,
            optimizer_cls=torch.optim.Adam,
            lr=bias_lr,
        )

    bias_logger = WandbLogger(
        project=run.project,
        name=f"matrix-estimate-{pair_key}-s{seed}",
        experiment=run,
        log_model=False,
    )
    bias_trainer = pl.Trainer(
        max_epochs=bias_max_epochs,
        logger=bias_logger,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="train_bias/loss", mode="min", patience=bias_patience)],
        log_every_n_steps=25,
    )
    bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)

    if estimator_mode == "normalized":
        estimated = {name: float(value) for name, value in bias_estimator.get_estimated_lambdas().items()}
    else:
        estimated = get_scale_params(joint_bias, estimation_bias_types)

    gt_map = {name: value for name, value in zip(gt_bias_types, gt_lambdas)}
    gt_mean_rel_error, max_rel_error = mean_relative_error(estimated, gt_map, estimation_bias_types)
    ols_mean_rel_error, ols_max_rel_error = mean_relative_error(
        ols_estimated,
        gt_map,
        estimation_bias_types,
    )
    for bias_name in estimation_bias_types:
        true_value = float(gt_map.get(bias_name, 0.0))
        est_value = float(estimated.get(bias_name, 0.0))
        rel_error = abs(est_value - true_value) / max(abs(true_value), 1e-8)
        ols_value = float(ols_estimated.get(bias_name, 0.0))
        ols_rel_error = abs(ols_value - true_value) / max(abs(true_value), 1e-8)
        wandb.log(
            {
                f"recovery/{bias_name}/true": true_value,
                f"recovery/{bias_name}/estimated": est_value,
                f"recovery/{bias_name}/rel_error": rel_error,
                f"recovery_ols/{bias_name}/estimated": ols_value,
                f"recovery_ols/{bias_name}/rel_error": ols_rel_error,
            }
        )
    noisy_ols_mean_rel_error = float("nan")
    noisy_ols_max_rel_error = float("nan")
    noisy_ols_relative_residual = float("nan")
    if recovery_noise_std > 0 and recovery_noise_draws > 0:
        noise_generator = torch.Generator(device=target_gradient.device).manual_seed(seed + 17)
        noise_scale = recovery_noise_std * target_gradient.norm() / max(target_gradient.numel() ** 0.5, 1.0)
        noisy_mean_errors = []
        noisy_max_errors = []
        noisy_residuals = []
        for _ in range(recovery_noise_draws):
            noisy_target_gradient = target_gradient + noise_scale * torch.randn(
                target_gradient.shape,
                device=target_gradient.device,
                dtype=target_gradient.dtype,
                generator=noise_generator,
            )
            noisy_estimated, noisy_residual = solve_least_squares_lambdas(
                noisy_target_gradient,
                bias_gradients,
                estimation_bias_types,
            )
            mean_err, max_err = mean_relative_error(noisy_estimated, gt_map, estimation_bias_types)
            noisy_mean_errors.append(mean_err)
            noisy_max_errors.append(max_err)
            noisy_residuals.append(noisy_residual)
        noisy_ols_mean_rel_error = float(sum(noisy_mean_errors) / len(noisy_mean_errors))
        noisy_ols_max_rel_error = float(sum(noisy_max_errors) / len(noisy_max_errors))
        noisy_ols_relative_residual = float(sum(noisy_residuals) / len(noisy_residuals))

    scalar_logs = {
        "fit/train_loss": train_loss,
        "fit/train_mse": train_mse,
        "fit/val_loss": val_loss,
        "fit/val_mse": val_mse,
        "fit/test_loss": test_loss,
        "fit/test_mse": test_mse,
        "identifiability/gt_mean_rel_error": gt_mean_rel_error,
        "identifiability/max_rel_error": max_rel_error,
        "identifiability/ols_mean_rel_error": ols_mean_rel_error,
        "identifiability/ols_max_rel_error": ols_max_rel_error,
        "identifiability/gt_relative_residual": gt_relative_residual,
        "identifiability/ols_relative_residual": ols_relative_residual,
        "identifiability/noisy_ols_mean_rel_error": noisy_ols_mean_rel_error,
        "identifiability/noisy_ols_max_rel_error": noisy_ols_max_rel_error,
        "identifiability/noisy_ols_relative_residual": noisy_ols_relative_residual,
        "geometry/all/max_abs_pairwise_cos": max_offdiag_all,
        "geometry/all/condition_number": condition_all,
        "geometry/selected/max_abs_pairwise_cos": max_offdiag_selected,
        "geometry/selected/condition_number": condition_selected,
        f"geometry/{pair_key}/cosine": primary_pair_cosine,
        "config/gt_bias_types": ",".join(gt_bias_types),
        "config/estimation_bias_types": ",".join(estimation_bias_types),
        "config/teacher_spectrum": metadata.teacher_spectrum,
        "config/input_spectrum": metadata.input_spectrum,
        "data/input_rank": metadata.input_rank,
        "data/teacher_rank": metadata.teacher_rank,
        "data/input_effective_rank": metadata.input_effective_rank,
        "data/teacher_effective_rank": metadata.teacher_effective_rank,
    }
    for name, value in target_alignments.items():
        scalar_logs[f"geometry/target_alignment/{name}"] = value

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
        "Geometry | pair=%s cosine=%.6f input_spectrum=%s teacher_spectrum=%s input_rank=%d teacher_rank=%d",
        pair_key,
        primary_pair_cosine,
        metadata.input_spectrum,
        metadata.teacher_spectrum,
        metadata.input_rank,
        metadata.teacher_rank,
    )
    LOGGER.info(
        "Result | pair=%s estimator_mode=%s gt_mean_rel_error=%.4f max_rel_error=%.4f ols_mean_rel_error=%.4f noisy_ols_mean_rel_error=%.4f gt_relative_residual=%.4f ols_relative_residual=%.4f noisy_ols_relative_residual=%.4f selected_max_abs_cos=%.4f selected_cond=%.4f",
        pair_key,
        estimator_mode,
        gt_mean_rel_error,
        max_rel_error,
        ols_mean_rel_error,
        noisy_ols_mean_rel_error,
        gt_relative_residual,
        ols_relative_residual,
        noisy_ols_relative_residual,
        max_offdiag_selected,
        condition_selected,
    )
    for bias_name in estimation_bias_types:
        LOGGER.info(
            "  %s | true=%.6f estimated=%.6f ols=%.6f",
            bias_name,
            float(gt_map.get(bias_name, 0.0)),
            float(estimated.get(bias_name, 0.0)),
            float(ols_estimated.get(bias_name, 0.0)),
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
