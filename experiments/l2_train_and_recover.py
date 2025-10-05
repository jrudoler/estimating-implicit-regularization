#!/usr/bin/env python3
"""Recover scalar L2 penalties across architectures and function classes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple
from dataclasses import asdict

import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import RidgeBias
from core.estimators import BiasWithMSE

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging once for console output."""
    if logging.getLogger().handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def add_file_handler(log_path: Path) -> None:
    """Attach a file handler for persistent logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class DatasetConfig:
    n_samples: int = 10000
    input_dim: int = 10
    noise_std: float = 0.5
    function_class: str = "linear"
    seed: int = 42


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs_model: int = 200
    max_epochs_bias: int = 1000
    lr_model: float = 1e-3
    lr_bias: float = 1e-2
    depth: int = 2
    width: int = 128
    l2_lambda: float = 0.01
    patience_model: int = 20
    patience_bias: int = 50


def build_dataset(cfg: DatasetConfig) -> TensorDataset:
    generator = torch.Generator().manual_seed(cfg.seed)
    X = torch.randn(cfg.n_samples, cfg.input_dim, generator=generator)

    weights = torch.randn(cfg.input_dim, 1, generator=generator)
    linear_response = X @ weights

    if cfg.function_class == "linear":
        y_clean = linear_response
    elif cfg.function_class == "sine":
        y_clean = torch.sin(linear_response)
    elif cfg.function_class == "polynomial":
        # random degree k polynomial
        k = 10
        coeffs = torch.randn(k + 1, 1, generator=generator)  # including constant term
        y_clean = sum(coeffs[i] * linear_response**i for i in range(k + 1))
    else:
        raise ValueError(
            f"Unsupported function class '{cfg.function_class}'."
            " Choose from ['linear', 'sine', 'polynomial']."
        )

    noise = cfg.noise_std * torch.randn(cfg.n_samples, 1, generator=generator)
    y = y_clean + noise
    return TensorDataset(X, y)


def make_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_fraction: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    lengths = [int(len(dataset) * train_fraction), len(dataset)]
    lengths[0] = min(lengths[0], len(dataset))
    lengths[1] = len(dataset) - lengths[0]

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, lengths, generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, val_loader


class FeedforwardRegressor(pl.LightningModule):
    """Simple MLP regressor with configurable depth and width."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
        l2_lambda: float,
        lr: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for layer_idx in range(depth):
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        loss += self.weight_regularization()
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        loss += self.weight_regularization()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def weight_regularization(self) -> Tensor:
        flattened = torch.cat([param.view(-1) for param in self.parameters()])
        l2_penalty = torch.sum(flattened.pow(2))
        return self.hparams.l2_lambda * l2_penalty

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def log_run_configuration(config: Dict[str, object]) -> None:
    printable = {
        key: value
        for key, value in config.items()
        if isinstance(value, (int, float, str))
    }
    LOGGER.info("Run configuration: %s", printable)


def main() -> None:
    configure_logging()

    run = wandb.init(project="inductive-bias", job_type="l2_sweep")
    cfg = run.config

    log_dir = REPO_ROOT / "logs" / "l2_recovery"
    add_file_handler(log_dir / f"l2_recovery_{run.id}.log")

    default_dataset_cfg = DatasetConfig()
    default_training_cfg = TrainingConfig()

    dataset_cfg = DatasetConfig(
        n_samples=int(cfg.get("n_samples", default_dataset_cfg.n_samples)),
        input_dim=int(cfg.get("input_dim", default_dataset_cfg.input_dim)),
        noise_std=float(cfg.get("noise_std", default_dataset_cfg.noise_std)),
        function_class=str(
            cfg.get("function_class", default_dataset_cfg.function_class)
        ),
        seed=int(cfg.get("seed", default_dataset_cfg.seed)),
    )
    training_cfg = TrainingConfig(
        batch_size=int(cfg.get("batch_size", default_training_cfg.batch_size)),
        max_epochs_model=int(
            cfg.get("max_epochs_model", default_training_cfg.max_epochs_model)
        ),
        max_epochs_bias=int(
            cfg.get("max_epochs_bias", default_training_cfg.max_epochs_bias)
        ),
        lr_model=float(cfg.get("lr_model", default_training_cfg.lr_model)),
        lr_bias=float(cfg.get("lr_bias", default_training_cfg.lr_bias)),
        depth=int(cfg.get("depth", default_training_cfg.depth)),
        width=int(cfg.get("width", default_training_cfg.width)),
        l2_lambda=float(cfg.get("l2_lambda", default_training_cfg.l2_lambda)),
        patience_model=int(
            cfg.get("patience_model", default_training_cfg.patience_model)
        ),
        patience_bias=int(
            cfg.get("patience_bias", default_training_cfg.patience_bias)
        ),
    )
    train_fraction = float(cfg.get("train_fraction", 0.8))

    set_seed(dataset_cfg.seed)
    LOGGER.info("Initialized W&B run: %s", run.name)
    log_run_configuration(
        {
            **asdict(dataset_cfg),
            **asdict(training_cfg),
            "train_fraction": train_fraction,
        }
    )

    dataset = build_dataset(dataset_cfg)
    train_loader, val_loader = make_dataloaders(
        dataset=dataset,
        batch_size=training_cfg.batch_size,
        train_fraction=train_fraction,
        seed=dataset_cfg.seed,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    model = FeedforwardRegressor(
        input_dim=dataset_cfg.input_dim,
        output_dim=1,
        depth=training_cfg.depth,
        width=training_cfg.width,
        l2_lambda=training_cfg.l2_lambda,
        lr=training_cfg.lr_model,
    )

    model_logger = WandbLogger(
        project="inductive-bias",
        name=f"l2-train-{training_cfg.depth}x{training_cfg.width}-{dataset_cfg.function_class}",
        experiment=run,
        save_dir=run.dir,
        log_model=False,
    )

    model_trainer = Trainer(
        max_epochs=training_cfg.max_epochs_model,
        logger=model_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            EarlyStopping(
                monitor="val/loss", mode="min", patience=training_cfg.patience_model
            )
        ],
        log_every_n_steps=10,
    )
    model_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    LOGGER.info("Finished training predictive model.")

    bias_model = RidgeBias()
    bias_model.beta.data = torch.tensor(
        [training_cfg.l2_lambda],
        dtype=bias_model.beta.dtype,
        device=bias_model.beta.device,
    )

    bias_estimator = BiasWithMSE(
        predictive_model=model,
        bias_model=bias_model,
        grad_match_loss_fn=torch.nn.functional.mse_loss,
        lr=training_cfg.lr_bias,
        optimizer_cls=torch.optim.Adam,
    )

    bias_logger = WandbLogger(
        project="inductive-bias",
        name=f"l2-bias-{training_cfg.depth}x{training_cfg.width}-{dataset_cfg.function_class}",
        experiment=run,
        save_dir=run.dir,
        log_model=False,
    )

    bias_trainer = Trainer(
        max_epochs=training_cfg.max_epochs_bias,
        logger=bias_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            EarlyStopping(
                monitor="train/loss", mode="min", patience=training_cfg.patience_bias
            )
        ],
        log_every_n_steps=10,
    )
    bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)

    estimated_l2 = bias_estimator.bias_model.beta.detach().cpu().item()
    true_l2 = training_cfg.l2_lambda
    abs_error = abs(estimated_l2 - true_l2)
    rel_error = abs_error / max(true_l2, 1e-12)

    wandb.log(
        {
            "l2/true": true_l2,
            "l2/estimated": estimated_l2,
            "l2/abs_error": abs_error,
            "l2/rel_error": rel_error,
        }
    )
    run.summary["l2_true"] = true_l2
    run.summary["l2_estimated"] = estimated_l2
    run.summary["l2_abs_error"] = abs_error
    run.summary["l2_rel_error"] = rel_error

    LOGGER.info(
        "Recovered L2 penalty: true=%.6f | estimated=%.6f | abs_error=%.6f | rel_error=%.6f",
        true_l2,
        estimated_l2,
        abs_error,
        rel_error,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
