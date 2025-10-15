#!/usr/bin/env python3
"""Train a deep ReLU network on MNIST with Lightning and estimate inductive biases."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import Callback
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy

LOGGER = logging.getLogger(__name__)
MNIST_ROOT = Path(__file__).resolve().parents[1] / "MNIST"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass(slots=True)
class ExperimentConfig:
    batch_size: int = 256
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    width: int = 512
    depth: int = 5
    dropout: float = 0.0
    seed: int = 123
    num_workers: int = 0
    accelerator: str = "auto"
    devices: str = "auto"
    output: Path = RESULTS_DIR / "mnist_deep_relu_bias.json"


DEFAULT_CONFIG = ExperimentConfig()


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay
    )
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG.width)
    parser.add_argument("--depth", type=int, default=DEFAULT_CONFIG.depth)
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG.dropout)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_CONFIG.num_workers,
        help="Number of DataLoader workers. Use 0 for compatibility with strict sandboxes.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=DEFAULT_CONFIG.accelerator,
        help="Lightning accelerator: cpu, cuda, mps, or auto.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=DEFAULT_CONFIG.devices,
        help="Devices to use (e.g. 'auto', '1', '0,1').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CONFIG.output,
        help="Path to the JSON file where results will be stored.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging (for sweeps or manual tracking).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="inductive-bias",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team) name. Defaults to the current user if omitted.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group name for organizing runs.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Explicit W&B run name. If omitted, W&B assigns one.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional tags to attach to the W&B run.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="Set the W&B mode explicitly.",
    )
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="Enable model checkpoint logging to W&B (disabled by default).",
    )
    parser.add_argument(
        "--wandb-watch",
        action="store_true",
        help="Call wandb.watch on the Lightning module to log gradients/parameters.",
    )
    parser.add_argument(
        "--wandb-save-code",
        action="store_true",
        help="Ask W&B to snapshot source code for the run.",
    )
    args = parser.parse_args()
    return args


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        seed=args.seed,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        devices=str(args.devices),
        output=args.output,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int | float | str | bytes | bytearray | None) -> None:
    if seed is None:
        return

    numpy_integers = (np.integer,) if hasattr(np, "integer") else tuple()
    numpy_floats = (np.floating,) if hasattr(np, "floating") else tuple()

    if isinstance(seed, (float, str) + numpy_floats):
        seed_value = int(float(seed))
    elif isinstance(seed, (bytes, bytearray)):
        seed_value = int.from_bytes(seed, byteorder="little", signed=False)
    elif isinstance(seed, (int,) + numpy_integers):
        seed_value = int(seed)
    else:
        raise TypeError(
            "Seed must be one of {None, int, float, str, bytes, bytearray, numpy number}; "
            f"got type {type(seed)}."
        )

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    seed_everything(seed_value, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MNISTLightningDataModule(LightningDataModule):
    def __init__(
        self, root: Path, batch_size: int, num_workers: int, val_fraction: float = 0.1
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.transform = transforms.ToTensor()
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def prepare_data(self) -> None:  # type: ignore[override]
        datasets.MNIST(root=self.root, train=True, download=True)
        datasets.MNIST(root=self.root, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:  # type: ignore[override]
        if stage == "fit" or stage is None:
            full_train = datasets.MNIST(
                root=self.root,
                train=True,
                download=False,
                transform=self.transform,
            )
            val_size = int(len(full_train) * self.val_fraction)
            train_size = len(full_train) - val_size
            self._train_dataset, self._val_dataset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        if stage == "test" or stage is None:
            self._test_dataset = datasets.MNIST(
                root=self.root,
                train=False,
                download=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class DeepReLULightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        dropout: float,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        if depth < 1:
            raise ValueError("Depth must be >= 1")
        layers: List[nn.Module] = [nn.Flatten(), nn.Linear(input_dim, width), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(width, 10))
        self.network = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.network(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = self.train_accuracy(preds, labels)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy_epoch", acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = self.val_accuracy(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = self.test_accuracy(preds, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:  # type: ignore[override]
        self.train_accuracy.reset()

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        self.val_accuracy.reset()

    def on_test_epoch_end(self) -> None:  # type: ignore[override]
        self.test_accuracy.reset()

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class MetricHistoryCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.history: List[Dict[str, Any]] = []
        self._last_train_metrics: Dict[str, float] = {}

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # type: ignore[override]
        metrics = trainer.callback_metrics
        self._last_train_metrics = {
            "train_loss": _metric_to_float(metrics, "train/loss_epoch"),
            "train_accuracy": _metric_to_float(metrics, "train/accuracy_epoch"),
        }

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:  # type: ignore[override]
        metrics = trainer.callback_metrics
        record = {
            "epoch": int(trainer.current_epoch + 1),
            "train_loss": _metric_to_float(metrics, "train/loss_epoch"),
            "train_accuracy": _metric_to_float(metrics, "train/accuracy_epoch"),
            "val_loss": _metric_to_float(metrics, "val/loss"),
            "val_accuracy": _metric_to_float(metrics, "val/accuracy"),
        }
        if (
            math.isnan(record["train_loss"])
            and "train_loss" in self._last_train_metrics
        ):
            record["train_loss"] = self._last_train_metrics["train_loss"]
        if (
            math.isnan(record["train_accuracy"])
            and "train_accuracy" in self._last_train_metrics
        ):
            record["train_accuracy"] = self._last_train_metrics["train_accuracy"]

        small_weight_bias = compute_small_weight_bias(pl_module)
        low_rank_bias = compute_low_rank_bias(pl_module)

        record["small_weight_bias_total"] = small_weight_bias["total"]
        record["small_weight_bias_per_layer"] = small_weight_bias["per_layer"]
        record["low_rank_bias_total"] = low_rank_bias["total"]
        record["low_rank_bias_per_layer"] = low_rank_bias["per_layer"]

        bias_metrics_to_log = {
            "bias/small_weight_total": float(small_weight_bias["total"]),
            "bias/low_rank_total": float(low_rank_bias["total"]),
            **_flatten_bias_metrics(
                "bias/small_weight_per_layer", small_weight_bias["per_layer"]
            ),
            **_flatten_bias_metrics(
                "bias/low_rank_per_layer", low_rank_bias["per_layer"]
            ),
        }

        if trainer.logger is not None and hasattr(trainer.logger, "log_metrics"):
            numeric_bias_metrics = {
                key: value
                for key, value in bias_metrics_to_log.items()
                if not math.isnan(value)
            }
            if numeric_bias_metrics:
                trainer.logger.log_metrics(
                    numeric_bias_metrics, step=trainer.global_step
                )

        self.history.append(record)


def _metric_to_float(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("nan")
    if isinstance(value, Tensor):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _flatten_bias_metrics(prefix: str, per_layer: Dict[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for name, value in per_layer.items():
        safe_name = name.replace(".", "_")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        flattened[f"{prefix}/{safe_name}"] = numeric_value
    return flattened


def _apply_config_overrides(
    cfg: ExperimentConfig, overrides: Dict[str, Any]
) -> ExperimentConfig:
    if not overrides:
        return cfg
    valid_fields = {field.name for field in fields(ExperimentConfig)}
    updates: Dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid_fields:
            continue
        if key == "output" and value is not None:
            value = Path(value)
        if key == "devices" and value is not None:
            value = str(value)
        updates[key] = value
    if not updates:
        return cfg
    return replace(cfg, **updates)


def iter_weight_matrices(model: nn.Module) -> Iterable[Tuple[str, Tensor]]:
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2:
            yield name, parameter.detach().float().cpu()


def compute_small_weight_bias(model: nn.Module) -> Dict[str, Any]:
    layer_values: Dict[str, float] = {}
    total = 0.0
    for name, weight in iter_weight_matrices(model):
        value = torch.sum(weight.pow(2)).item()
        layer_values[name] = value
        total += value
    return {"per_layer": layer_values, "total": total}


def estimate_rank(matrix: Tensor, tol: float = 1e-4) -> int:
    if matrix.numel() == 0:
        return 0
    flat = matrix.detach().cpu()
    norm = torch.linalg.norm(flat)
    if norm == 0:
        return 0
    normalized = flat / norm
    singular_values = torch.linalg.svdvals(normalized)
    rank = torch.count_nonzero(torch.abs(singular_values) > tol).item()
    return int(rank)


def compute_low_rank_bias(model: nn.Module, tol: float = 1e-4) -> Dict[str, Any]:
    layer_values: Dict[str, int] = {}
    total_rank = 0
    for name, weight in iter_weight_matrices(model):
        rank = estimate_rank(weight, tol)
        layer_values[name] = rank
        total_rank += rank
    return {"per_layer": layer_values, "total": float(total_rank)}


def run_experiment(
    cfg: ExperimentConfig, logger: Optional[Any] = None, watch_model: bool = False
) -> Dict[str, Any]:
    datamodule = MNISTLightningDataModule(
        root=MNIST_ROOT,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = DeepReLULightningModule(
        input_dim=28 * 28,
        width=cfg.width,
        depth=cfg.depth,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history_callback = MetricHistoryCallback()

    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        deterministic=True,
        logger=logger if logger is not None else False,
        enable_checkpointing=False,
        log_every_n_steps=50,
        callbacks=[history_callback],
    )

    if watch_model and logger is not None and hasattr(logger, "watch"):
        try:
            logger.watch(model, log="all")  # type: ignore[attr-defined]
        except TypeError:
            logger.watch(model)  # type: ignore[attr-defined]

    trainer.fit(model, datamodule=datamodule)
    test_metrics_list = trainer.test(model, datamodule=datamodule, verbose=False)
    test_metrics = test_metrics_list[0] if test_metrics_list else {}

    device_used = str(trainer.strategy.root_device)

    model_cpu = model.cpu().eval()
    small_weight_bias = compute_small_weight_bias(model_cpu)
    low_rank_bias = compute_low_rank_bias(model_cpu)

    config_dict = asdict(cfg)
    config_dict["output"] = str(config_dict["output"])

    test_metrics_clean = {k: _metric_to_float(test_metrics, k) for k in test_metrics}

    results: Dict[str, Any] = {
        "config": config_dict,
        "device": device_used,
        "history": history_callback.history,
        "test_metrics": test_metrics_clean,
        "final_test_loss": test_metrics_clean.get("test/loss"),
        "final_test_accuracy": test_metrics_clean.get("test/accuracy"),
        "small_weight_bias": small_weight_bias,
        "low_rank_bias": low_rank_bias,
    }
    return results


def main() -> None:
    configure_logging()
    args = parse_cli_args()
    cfg = build_config_from_args(args)

    wandb_run = None
    wandb_logger = None

    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb must be installed to use --use-wandb or run sweeps."
            ) from exc

        wandb_settings = wandb.Settings(save_code=args.wandb_save_code)

        init_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "name": args.wandb_run_name,
            "tags": args.wandb_tags,
            "config": asdict(cfg),
        }
        if args.wandb_mode is not None:
            init_kwargs["mode"] = args.wandb_mode
        init_kwargs["settings"] = wandb_settings

        wandb_run = wandb.init(**init_kwargs)
        if wandb_run is None:
            raise RuntimeError("wandb.init returned None; failed to start a W&B run.")

        cfg = _apply_config_overrides(cfg, dict(wandb_run.config))

        if cfg.output == DEFAULT_CONFIG.output:
            cfg = replace(
                cfg,
                output=cfg.output.with_name(
                    f"{cfg.output.stem}_{wandb_run.id or wandb.util.generate_id()}.json"
                ),
            )

        wandb_run.config.update(asdict(cfg), allow_val_change=True)

        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=wandb_run.name,
            tags=args.wandb_tags,
            log_model=args.wandb_log_model,
            save_dir=str(cfg.output.parent),
        )

    set_seed(cfg.seed)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting experiment with config: %s", cfg)

    results = run_experiment(
        cfg,
        logger=wandb_logger,
        watch_model=bool(
            args.use_wandb and args.wandb_watch and wandb_logger is not None
        ),
    )
    with cfg.output.open("w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Results written to %s", cfg.output)

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "final_test_loss": results.get("final_test_loss"),
                "final_test_accuracy": results.get("final_test_accuracy"),
                "small_weight_bias_total": results.get("small_weight_bias", {}).get(
                    "total"
                ),
                "low_rank_bias_total": results.get("low_rank_bias", {}).get("total"),
                "result_path": str(cfg.output),
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
