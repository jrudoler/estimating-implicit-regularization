#!/usr/bin/env python3
"""Recover L2 and orthogonal penalties from trained deep ReLU networks to test identifiability."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb

from core.bias import RidgeBias, OrthogonalBias, JointBias
from core.estimators import BiasWithCrossEntropy

LOGGER = logging.getLogger(__name__)


class BiasWithCrossEntropyScheduled(BiasWithCrossEntropy):
    """Bias estimator with adaptive LR scheduling."""

    def __init__(
        self, *args, bias_lr: float = 0.1, bias_weight_decay: float = 0.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bias_lr = bias_lr
        self.bias_weight_decay = bias_weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.bias_model.parameters(),
            lr=self.bias_lr,
            weight_decay=self.bias_weight_decay,
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
                "monitor": "train_bias/loss",
            },
        }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepReLUClassifierL2Orthogonal(pl.LightningModule):
    """Deep ReLU MLP classifier with L2 AND orthogonal regularization."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        depth: int,
        width: int,
        l2_lambda: float,
        orthogonal_lambda: float,
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
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Add L2 regularization
        l2_penalty = self.weight_regularization()
        loss += l2_penalty

        # Add orthogonal regularization
        orthogonal_penalty = self.orthogonal_regularization()
        loss += orthogonal_penalty

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self._accuracy(logits, y), on_step=False, on_epoch=True)
        self.log("train/l2_penalty", l2_penalty, on_step=False, on_epoch=True)
        self.log(
            "train/orthogonal_penalty", orthogonal_penalty, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        loss += self.weight_regularization()
        loss += self.orthogonal_regularization()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self._accuracy(logits, y), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        loss += self.weight_regularization()
        loss += self.orthogonal_regularization()
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self._accuracy(logits, y), on_step=False, on_epoch=True)
        return loss

    def weight_regularization(self) -> Tensor:
        """L2 penalty on all parameters."""
        flattened = torch.cat([param.view(-1) for param in self.parameters()])
        l2_penalty = torch.sum(flattened.pow(2))
        return self.hparams.l2_lambda * l2_penalty

    def orthogonal_regularization(self) -> Tensor:
        """Orthogonal penalty on weight matrices."""
        penalty = torch.zeros((), device=self.device)
        # Iterate over modules to find Linear layers easily, or inspect parameters
        # Using parameters() and checking shape is simpler given the structure
        for param in self.parameters():
            if param.ndim == 2:
                # Linear layer weights: (out_features, in_features)
                w = param
                rows, cols = w.shape
                if rows > cols:
                    # Cannot be orthogonal if rows > cols
                    continue

                gram = torch.mm(w, w.t())
                eye = torch.eye(rows, device=w.device)
                penalty = penalty + torch.norm(gram - eye, p="fro") ** 2

        return self.hparams.orthogonal_lambda * penalty

    def _accuracy(self, logits: Tensor, targets: Tensor) -> Tensor:
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean()

    def configure_optimizers(self):
        # No weight decay - penalties only in loss
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
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        return datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_input_dim(dataset_name: str) -> int:
    """Get input dimension for dataset."""
    if dataset_name.lower() == "mnist":
        return 28 * 28
    elif dataset_name.lower() == "fashionmnist":
        return 28 * 28
    elif dataset_name.lower() == "cifar10":
        return 32 * 32 * 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for dataset."""
    if dataset_name.lower() in ("mnist", "fashionmnist"):
        return 10
    elif dataset_name.lower() == "cifar10":
        return 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashionmnist", "cifar10"],
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--l2-lambda", type=float, default=0.01)
    parser.add_argument("--orthogonal-lambda", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--bias-max-epochs", type=int, default=2000)
    parser.add_argument("--bias-patience", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    # Initialize W&B
    run = wandb.init(project="inductive-bias", job_type="l2_orthogonal_recovery")
    cfg = run.config

    # Parse args, but override with W&B config if available
    args = parse_args()
    dataset_name = cfg.get("dataset", args.dataset)
    data_root = Path(cfg.get("data_root", str(args.data_root)))
    batch_size = cfg.get("batch_size", args.batch_size)
    depth = cfg.get("depth", args.depth)
    width = cfg.get("width", args.width)
    l2_lambda = cfg.get("l2_lambda", args.l2_lambda)
    orthogonal_lambda = cfg.get("orthogonal_lambda", args.orthogonal_lambda)
    lr = cfg.get("lr", args.lr)
    max_epochs = cfg.get("max_epochs", args.max_epochs)
    patience = cfg.get("patience", args.patience)
    seed = cfg.get("seed", args.seed)
    bias_lr = cfg.get("bias_lr", args.bias_lr)
    bias_weight_decay = cfg.get("bias_weight_decay", args.bias_weight_decay)
    bias_max_epochs = cfg.get("bias_max_epochs", args.bias_max_epochs)
    bias_patience = cfg.get("bias_patience", args.bias_patience)
    val_fraction = cfg.get("val_fraction", args.val_fraction)

    set_seed(seed)

    LOGGER.info("Starting L2 + Orthogonal recovery experiment")
    LOGGER.info(
        "Config: dataset=%s, depth=%d, width=%d, l2_lambda=%.6f, orthogonal_lambda=%.6f, seed=%d",
        dataset_name,
        depth,
        width,
        l2_lambda,
        orthogonal_lambda,
        seed,
    )

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

    train_loader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_split, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    input_dim = get_input_dim(dataset_name)
    num_classes = get_num_classes(dataset_name)

    # Train predictive model with BOTH penalties
    model = DeepReLUClassifierL2Orthogonal(
        input_dim=input_dim,
        num_classes=num_classes,
        depth=depth,
        width=width,
        l2_lambda=l2_lambda,
        orthogonal_lambda=orthogonal_lambda,
        lr=lr,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    model_logger = WandbLogger(
        project="inductive-bias",
        name=f"model-{dataset_name}-d{depth}w{width}-l2{l2_lambda:.4f}-orth{orthogonal_lambda:.4f}-s{seed}",
        experiment=run,
        log_model=False,
    )

    log_every_n_steps = 50
    model_trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=model_logger,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=patience),
            ModelCheckpoint(
                monitor="val/loss", mode="min", save_top_k=1, save_last=True
            ),
        ],
        log_every_n_steps=log_every_n_steps,
    )

    model_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate on test set
    model_trainer.test(model, dataloaders=test_loader)
    test_results = model_trainer.callback_metrics
    wandb.log(
        {
            "test/acc": test_results.get("test/acc", 0.0),
            "test/loss": test_results.get("test/loss", float("inf")),
        }
    )

    LOGGER.info("Finished training predictive model")

    # Estimate BOTH L2 and orthogonal penalties using JointBias
    l2_bias = RidgeBias(enforce_positive=True)
    orthogonal_bias = OrthogonalBias(enforce_positive=True)

    joint_bias = JointBias([l2_bias, orthogonal_bias])

    bias_estimator = BiasWithCrossEntropyScheduled(
        predictive_model=model,
        bias_model=joint_bias,
        grad_match_loss_fn=nn.functional.mse_loss,
        lr=bias_lr,  # Will be ignored due to custom configure_optimizers
        optimizer_cls=torch.optim.Adam,
        bias_lr=bias_lr,
        bias_weight_decay=bias_weight_decay,
    )

    bias_logger = WandbLogger(
        project="inductive-bias",
        name=f"bias-{dataset_name}-d{depth}w{width}-l2{l2_lambda:.4f}-orth{orthogonal_lambda:.4f}-s{seed}",
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
            EarlyStopping(
                monitor="train_bias/loss", mode="min", patience=bias_patience
            ),
        ],
        log_every_n_steps=log_every_n_steps,
    )

    bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)

    # Extract estimated parameters
    estimated_params = joint_bias.get_bias_params()
    estimated_l2 = float(estimated_params.get("ridge/scale", estimated_params.get("ridge", 0.0)))
    estimated_orthogonal = float(
        estimated_params.get("orthogonal/scale", estimated_params.get("orthogonal", 0.0))
    )

    # Update run summary
    run.summary.update(
        {
            "l2_true": l2_lambda,
            "l2_estimated": estimated_l2,
            "orthogonal_true": orthogonal_lambda,
            "orthogonal_estimated": estimated_orthogonal,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
