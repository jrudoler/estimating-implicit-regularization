#!/usr/bin/env python3
"""
Mixed Bias Recovery Experiment

Train networks with known ground-truth regularization (Ridge and/or WeightCoherence),
then use JointBias estimation to recover the true coefficients.

Tests identifiability: can we correctly identify which regularizers are active
and estimate their coefficients?
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import RidgeBias, WeightCoherenceBias, JointBias
from core.estimators import (
    BiasWithCrossEntropy,
    BiasWithCrossEntropyScheduled,
    BiasWithCrossEntropyNormalized,
)

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepReLUClassifierMixed(pl.LightningModule):
    """Deep ReLU MLP classifier with configurable Ridge and WeightCoherence regularization."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        depth: int,
        width: int,
        ridge_lambda: float,
        coherence_lambda: float,
        lr: float,
        linear_layer_bias: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width, bias=linear_layer_bias))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Add Ridge (L2) regularization
        ridge_penalty = self.ridge_regularization()
        loss += ridge_penalty

        # Add WeightCoherence regularization
        coherence_penalty = self.coherence_regularization()
        loss += coherence_penalty

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self._accuracy(logits, y), on_step=False, on_epoch=True)
        self.log("train/ridge_penalty", ridge_penalty, on_step=False, on_epoch=True)
        self.log("train/coherence_penalty", coherence_penalty, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        loss += self.ridge_regularization()
        loss += self.coherence_regularization()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self._accuracy(logits, y), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        loss += self.ridge_regularization()
        loss += self.coherence_regularization()
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self._accuracy(logits, y), on_step=False, on_epoch=True)
        return loss

    def ridge_regularization(self) -> Tensor:
        """L2 penalty on all parameters: λ * ||W||²"""
        if self.hparams.ridge_lambda <= 0:
            return torch.zeros((), device=self.device)
        flattened = torch.cat([param.view(-1) for param in self.parameters()])
        penalty = torch.sum(flattened.pow(2))
        return self.hparams.ridge_lambda * penalty

    def coherence_regularization(self) -> Tensor:
        """
        Weight coherence penalty: λ * ||W^T W - diag(W^T W)||_F² / ||W||_F⁴
        
        Scale-invariant measure of column correlation in weight matrices.
        """
        if self.hparams.coherence_lambda <= 0:
            return torch.zeros((), device=self.device)
        
        penalty = torch.zeros((), device=self.device)
        count = 0
        for param in self.parameters():
            if param.ndim >= 2:
                matrix = param if param.ndim == 2 else param.reshape(param.shape[0], -1)
                # Gram matrix: W^T W
                gram = matrix.T @ matrix
                # Off-diagonal part
                diag_gram = torch.diag(torch.diag(gram))
                off_diag = gram - diag_gram
                # Normalized coherence
                fro_sq = torch.sum(matrix**2)
                coherence = torch.sum(off_diag**2) / (fro_sq**2 + 1e-8)
                penalty = penalty + coherence
                count += 1
        
        if count > 0:
            penalty = penalty / count
        return self.hparams.coherence_lambda * penalty

    def _accuracy(self, logits: Tensor, targets: Tensor) -> Tensor:
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.0
        )
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


def get_dataset(dataset_name: str, root: Path, train: bool):
    """Load dataset by name."""
    transform = transforms.ToTensor()

    if dataset_name.lower() == "mnist":
        return datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == "fashionmnist":
        return datasets.FashionMNIST(
            root=root, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_input_dim(dataset_name: str) -> int:
    if dataset_name.lower() in ("mnist", "fashionmnist"):
        return 28 * 28
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_num_classes(dataset_name: str) -> int:
    if dataset_name.lower() in ("mnist", "fashionmnist"):
        return 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed Bias Recovery: Train with known regularization, recover with JointBias"
    )
    # Dataset
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    
    # Network architecture
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    
    # Ground truth regularization coefficients
    parser.add_argument("--gt-ridge-lambda", type=float, default=0.05,
                        help="Ground truth Ridge (L2) coefficient")
    parser.add_argument("--gt-coherence-lambda", type=float, default=0.0,
                        help="Ground truth WeightCoherence coefficient")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    
    # Bias estimation hyperparameters
    parser.add_argument("--bias-lr", type=float, default=0.1)
    parser.add_argument("--bias-max-epochs", type=int, default=2000)
    parser.add_argument("--bias-patience", type=int, default=100)
    parser.add_argument(
        "--use-normalized",
        action="store_true",
        default=False,
        help="Use normalized gradient matching (fixes magnitude imbalance)",
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main() -> None:
    # Initialize W&B
    run = wandb.init(project="inductive-bias-experiments", job_type="mixed_bias_recovery")
    cfg = run.config

    # Parse args, override with W&B config if available
    args = parse_args()
    dataset_name = cfg.get("dataset", args.dataset)
    data_root = Path(cfg.get("data_root", str(args.data_root)))
    depth = cfg.get("depth", args.depth)
    width = cfg.get("width", args.width)
    gt_ridge_lambda = cfg.get("gt_ridge_lambda", args.gt_ridge_lambda)
    gt_coherence_lambda = cfg.get("gt_coherence_lambda", args.gt_coherence_lambda)
    batch_size = cfg.get("batch_size", args.batch_size)
    lr = cfg.get("lr", args.lr)
    max_epochs = cfg.get("max_epochs", args.max_epochs)
    patience = cfg.get("patience", args.patience)
    val_fraction = cfg.get("val_fraction", args.val_fraction)
    bias_lr = cfg.get("bias_lr", args.bias_lr)
    bias_max_epochs = cfg.get("bias_max_epochs", args.bias_max_epochs)
    bias_patience = cfg.get("bias_patience", args.bias_patience)
    use_normalized = cfg.get("use_normalized", args.use_normalized)
    seed = cfg.get("seed", args.seed)

    set_seed(seed)

    LOGGER.info("=" * 70)
    LOGGER.info("Mixed Bias Recovery Experiment")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Ground Truth: Ridge λ={gt_ridge_lambda:.6f}, Coherence λ={gt_coherence_lambda:.6f}")
    LOGGER.info(f"Network: depth={depth}, width={width}")
    LOGGER.info(f"Estimator: {'Normalized' if use_normalized else 'Standard'}")
    LOGGER.info(f"Seed: {seed}")

    # Load datasets
    train_dataset = get_dataset(dataset_name, data_root, train=True)
    test_dataset = get_dataset(dataset_name, data_root, train=False)

    # Split train into train/val
    val_size = int(len(train_dataset) * val_fraction)
    train_size = len(train_dataset) - val_size
    train_split, val_split = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    input_dim = get_input_dim(dataset_name)
    num_classes = get_num_classes(dataset_name)

    # =========================================================================
    # Phase 1: Train model with ground truth regularization
    # =========================================================================
    LOGGER.info("\n[Phase 1] Training model with ground truth regularization...")

    model = DeepReLUClassifierMixed(
        input_dim=input_dim,
        num_classes=num_classes,
        depth=depth,
        width=width,
        ridge_lambda=gt_ridge_lambda,
        coherence_lambda=gt_coherence_lambda,
        lr=lr,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    model_logger = WandbLogger(
        project="inductive-bias-experiments",
        name=f"train-ridge{gt_ridge_lambda:.4f}-coh{gt_coherence_lambda:.4f}-s{seed}",
        experiment=run,
        log_model=False,
    )

    model_trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=model_logger,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=patience),
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True),
        ],
        log_every_n_steps=50,
    )

    model_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model_trainer.test(model, dataloaders=test_loader)

    test_acc = model_trainer.callback_metrics.get("test/acc", 0.0)
    LOGGER.info(f"Model trained. Test accuracy: {test_acc:.4f}")

    # =========================================================================
    # Phase 2: Estimate bias using JointBias (Ridge + WeightCoherence)
    # =========================================================================
    estimator_type = "Normalized" if use_normalized else "Standard"
    LOGGER.info(f"\n[Phase 2] Estimating bias with {estimator_type} JointBias(Ridge, WeightCoherence)...")

    # Create bias models for estimation
    ridge_bias = RidgeBias(enforce_positive=True, init_value=0.01)
    coherence_bias = WeightCoherenceBias(enforce_positive=True, init_value=0.01)
    
    joint_bias = JointBias([ridge_bias, coherence_bias])

    if use_normalized:
        # Use normalized gradient matching (preconditioned regression)
        bias_estimator = BiasWithCrossEntropyNormalized(
            predictive_model=model,
            bias_model=joint_bias,
            grad_match_loss_fn=nn.functional.mse_loss,
            lr=bias_lr,
            optimizer_cls=torch.optim.Adam,
            bias_lr=bias_lr,
        )
    else:
        # Use standard gradient matching
        bias_estimator = BiasWithCrossEntropyScheduled(
            predictive_model=model,
            bias_model=joint_bias,
            grad_match_loss_fn=nn.functional.mse_loss,
            lr=bias_lr,
            optimizer_cls=torch.optim.Adam,
            bias_lr=bias_lr,
        )

    bias_logger = WandbLogger(
        project="inductive-bias-experiments",
        name=f"estimate-{estimator_type.lower()}-ridge{gt_ridge_lambda:.4f}-coh{gt_coherence_lambda:.4f}-s{seed}",
        experiment=run,
        log_model=False,
    )

    bias_trainer = pl.Trainer(
        max_epochs=bias_max_epochs,
        logger=bias_logger,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        callbacks=[
            EarlyStopping(monitor="train_bias/loss", mode="min", patience=bias_patience),
        ],
        log_every_n_steps=50,
    )

    bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)

    # =========================================================================
    # Phase 3: Extract and report results
    # =========================================================================
    LOGGER.info("\n[Phase 3] Results")
    LOGGER.info("=" * 70)

    # Get estimated parameters - different extraction for normalized vs standard
    if use_normalized:
        # For normalized estimator, use get_estimated_lambdas() which accounts for gradient norms
        estimated_lambdas = bias_estimator.get_estimated_lambdas()
        estimated_ridge = estimated_lambdas.get("ridge", 0.0)
        estimated_coherence = estimated_lambdas.get("weight_coherence", 0.0)
    else:
        # For standard estimator, use the raw bias params
        estimated_params = joint_bias.get_bias_params()
        estimated_ridge = estimated_params.get(
            "ridge/scale", estimated_params.get("ridge", 0.0)
        )
        estimated_coherence = estimated_params.get(
            "weight_coherence/scale", estimated_params.get("weight_coherence", 0.0)
        )

    # Calculate errors
    ridge_abs_error = abs(estimated_ridge - gt_ridge_lambda)
    ridge_rel_error = ridge_abs_error / max(gt_ridge_lambda, 1e-8) if gt_ridge_lambda > 0 else ridge_abs_error

    coherence_abs_error = abs(estimated_coherence - gt_coherence_lambda)
    coherence_rel_error = coherence_abs_error / max(gt_coherence_lambda, 1e-8) if gt_coherence_lambda > 0 else coherence_abs_error

    # Determine which regularizers are "active" (ground truth > 0)
    ridge_active = gt_ridge_lambda > 0
    coherence_active = gt_coherence_lambda > 0

    # Log results
    LOGGER.info(f"Ridge:     true={gt_ridge_lambda:.6f} | estimated={estimated_ridge:.6f} | rel_error={ridge_rel_error:.4f}")
    LOGGER.info(f"Coherence: true={gt_coherence_lambda:.6f} | estimated={estimated_coherence:.6f} | rel_error={coherence_rel_error:.4f}")

    # Log to W&B
    wandb.log({
        # Ground truth
        "gt/ridge_lambda": gt_ridge_lambda,
        "gt/coherence_lambda": gt_coherence_lambda,
        
        # Estimated values
        "estimated/ridge": estimated_ridge,
        "estimated/coherence": estimated_coherence,
        
        # Errors
        "error/ridge_abs": ridge_abs_error,
        "error/ridge_rel": ridge_rel_error,
        "error/coherence_abs": coherence_abs_error,
        "error/coherence_rel": coherence_rel_error,
        
        # Recovery metrics
        "recovery/ridge_success": ridge_rel_error < 0.3 if ridge_active else estimated_ridge < 0.01,
        "recovery/coherence_success": coherence_rel_error < 0.3 if coherence_active else estimated_coherence < 0.01,
        
        # Configuration
        "config/use_normalized": use_normalized,
    })

    # Update summary
    run.summary.update({
        "gt_ridge_lambda": gt_ridge_lambda,
        "gt_coherence_lambda": gt_coherence_lambda,
        "estimated_ridge": estimated_ridge,
        "estimated_coherence": estimated_coherence,
        "ridge_rel_error": ridge_rel_error,
        "coherence_rel_error": coherence_rel_error,
        "test_acc": float(test_acc) if hasattr(test_acc, 'item') else test_acc,
        "use_normalized": use_normalized,
    })

    # Print summary table
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 70)
    LOGGER.info(f"{'Regularizer':<20} {'True':<12} {'Estimated':<12} {'Rel Error':<12} {'Status':<10}")
    LOGGER.info("-" * 70)
    
    ridge_status = "✓" if (ridge_rel_error < 0.3 if ridge_active else estimated_ridge < 0.01) else "✗"
    coherence_status = "✓" if (coherence_rel_error < 0.3 if coherence_active else estimated_coherence < 0.01) else "✗"
    
    LOGGER.info(f"{'Ridge':<20} {gt_ridge_lambda:<12.6f} {estimated_ridge:<12.6f} {ridge_rel_error:<12.4f} {ridge_status:<10}")
    LOGGER.info(f"{'WeightCoherence':<20} {gt_coherence_lambda:<12.6f} {estimated_coherence:<12.6f} {coherence_rel_error:<12.4f} {coherence_status:<10}")
    LOGGER.info("=" * 70)

    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

