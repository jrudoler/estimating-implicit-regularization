"""
Elastic-net recovery: train a small MLP with smoothed L1 + L2, then recover (λ1, λ2) via gradient matching.

W&B sweep should vary l1, l2, smooth, and seed (independent synthetic data + init per seed).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from core.bias import ElasticNet
from core.estimators import BiasWithMSE
from core.models import NonLinearNetwork


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_elasticnet_recovery(
    l1: float,
    l2: float,
    smooth: float,
    seed: int,
    *,
    input_dim: int = 10,
    output_dim: int = 1,
    hidden_dim: int = 200,
    n_samples: int = 5000,
    batch_size: int = 32,
    lr_model: float = 1e-3,
    lr_bias: float = 1e-2,
    max_epochs_train: int = 200,
    max_epochs_bias: int = 1000,
    train_patience: int = 15,
    bias_patience: int = 50,
    accelerator: str = "auto",
    devices: int = 1,
    wandb_run: Optional[Any] = None,
    log_prefix: str = "recovery",
) -> Dict[str, float]:
    """
    One full train + recover pipeline. Resamples synthetic data for each seed (fresh X, β, ε, split).

    Returns dict with true/estimated lambdas and log multiplicative errors log(λ̂/λ).
    """
    set_seed(seed)

    X = torch.randn(n_samples, input_dim)
    true_betas = 5 * torch.randn(input_dim)
    y = (X @ true_betas).view(-1, 1) + 0.5 * torch.randn(n_samples, 1)

    dataset = TensorDataset(X, y)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = NonLinearNetwork(
        input_dim,
        output_dim,
        hidden_dim,
        l1_lambda=l1,
        l1_smooth=smooth,
        l2_lambda=l2,
        lr=lr_model,
    )

    loggers_train = []
    if wandb_run is not None:
        loggers_train.append(
            WandbLogger(
                project=wandb_run.project,
                name=f"train-l1-{l1}_l2-{l2}_s{seed}",
                experiment=wandb_run,
                save_dir=wandb_run.dir,
            )
        )

    trainer = Trainer(
        max_epochs=max_epochs_train,
        logger=loggers_train if loggers_train else False,
        callbacks=[EarlyStopping(monitor="train/loss", patience=train_patience)],
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    bias_model = ElasticNet(smooth=smooth)
    bias_estimator = BiasWithMSE(
        predictive_model=model,
        bias_model=bias_model,
        grad_match_loss_fn=torch.nn.functional.mse_loss,
        lr=lr_bias,
        optimizer_cls=torch.optim.Adam,
    )

    loggers_bias = []
    if wandb_run is not None:
        loggers_bias.append(
            WandbLogger(
                project=wandb_run.project,
                name=f"bias-l1-{l1}_l2-{l2}_s{seed}",
                experiment=wandb_run,
                save_dir=wandb_run.dir,
                log_model=False,
            )
        )

    bias_trainer = Trainer(
        max_epochs=max_epochs_bias,
        logger=loggers_bias if loggers_bias else False,
        callbacks=[
            EarlyStopping(
                monitor="train_bias/loss", patience=bias_patience, mode="min"
            ),
        ],
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
    )
    bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)

    with torch.no_grad():
        l1_hat = float(bias_estimator.bias_model.lambda_1.item())
        l2_hat = float(bias_estimator.bias_model.lambda_2.item())

    eps = 1e-15
    log_m1 = float(np.log(max(l1_hat, eps) / max(l1, eps)))
    log_m2 = float(np.log(max(l2_hat, eps) / max(l2, eps)))

    metrics: Dict[str, float] = {
        f"{log_prefix}/lambda_1_hat": l1_hat,
        f"{log_prefix}/lambda_2_hat": l2_hat,
        f"{log_prefix}/log_mult_l1": log_m1,
        f"{log_prefix}/log_mult_l2": log_m2,
        "true_l1": float(l1),
        "true_l2": float(l2),
        "smooth": float(smooth),
        "seed": float(seed),
    }

    if wandb_run is not None:
        wandb_run.log(metrics)

    return metrics


def _fast_settings_from_env() -> Dict[str, int]:
    """Optional ELASTICNET_FAST=1 for local smoke tests (shorter epochs)."""
    if os.environ.get("ELASTICNET_ULTRA", "").lower() in ("1", "true", "yes"):
        return {"max_epochs_train": 25, "max_epochs_bias": 80}
    if os.environ.get("ELASTICNET_FAST", "").lower() in ("1", "true", "yes"):
        return {"max_epochs_train": 40, "max_epochs_bias": 150}
    return {}


def main() -> None:
    import wandb

    run = wandb.init(project="inductive-bias", job_type="sweep")
    cfg = wandb.config
    l1 = float(cfg.get("l1", 0.1))
    l2 = float(cfg.get("l2", 0.01))
    smooth = float(cfg.get("smooth", 0.01))
    seed = int(cfg.get("seed", 42))
    accelerator = str(cfg.get("accelerator", "gpu"))

    fast = _fast_settings_from_env()
    run_elasticnet_recovery(
        l1,
        l2,
        smooth,
        seed,
        accelerator=accelerator,
        wandb_run=run,
        **fast,
    )
    run.finish()


if __name__ == "__main__":
    main()
