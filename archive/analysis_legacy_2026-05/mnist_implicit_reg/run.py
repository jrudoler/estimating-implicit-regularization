#!/usr/bin/env python3
"""Train MNIST MLPs and measure implicit gradient regularisation with PyTorch Lightning."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning_lite.utilities.seed import seed_everything
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_ROOT = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results" / "mnist_implicit_reg"
CHECKPOINT_DIR = REPO_ROOT / "saved_models" / "mnist_implicit_reg"
CSV_LOG_DIR = REPO_ROOT / "logs" / "lightning"


@dataclass(slots=True)
class RunConfig:
    width: int = 200
    depth: int = 5
    learning_rate: float = 0.01
    batch_size: int = 256
    max_epochs: int = 100
    weight_decay: float = 0.0
    momentum: float = 0.0
    nesterov: bool = False
    seed: int = 17
    num_workers: int = 4
    device: str = "auto"
    log_interval: int = 10
    patience: Optional[int] = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: Tuple[str, ...] | None = None
    wandb_mode: str | None = None
    wandb_run_name: str | None = None
    wandb_watch: bool = False
    wandb_log_model: bool = False
    wandb_save_code: bool = False
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    mnist_root: Path = MNIST_ROOT
    mnist_download: bool = False
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    accumulate_rig: bool = True


def config_to_dict(cfg: RunConfig) -> Dict[str, Any]:
    mapping: Dict[str, Any] = asdict(cfg)
    for key in ("output_path", "checkpoint_path", "mnist_root"):
        value = mapping.get(key)
        mapping[key] = str(value) if value is not None else None
    if mapping.get("wandb_tags") is not None:
        mapping["wandb_tags"] = list(mapping["wandb_tags"])
    return mapping


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = RunConfig()
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--depth", type=int, default=defaults.depth)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--momentum", type=float, default=defaults.momentum)
    parser.add_argument("--nesterov", action="store_true", default=defaults.nesterov)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-watch", action="store_true")
    parser.add_argument("--wandb-log-model", action="store_true")
    parser.add_argument("--wandb-save-code", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save a JSON summary for this run.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional path to store the best model weights.",
    )
    parser.add_argument(
        "--mnist-root",
        type=Path,
        default=MNIST_ROOT,
        help="Directory where the MNIST dataset is stored.",
    )
    parser.add_argument(
        "--mnist-download",
        action="store_true",
        help="Download MNIST if the dataset is not found locally.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit number of training batches per epoch (debug only).",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Limit number of eval batches (debug only).",
    )
    parser.add_argument(
        "--no-rig",
        action="store_false",
        dest="accumulate_rig",
        help="Skip computing R_IG even when the model fits the training set.",
    )
    args = parser.parse_args()
    return RunConfig(
        width=args.width,
        depth=args.depth,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.nesterov,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        log_interval=args.log_interval,
        patience=args.patience,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=tuple(args.wandb_tags) if args.wandb_tags else None,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.wandb_run_name,
        wandb_watch=args.wandb_watch,
        wandb_log_model=args.wandb_log_model,
        wandb_save_code=args.wandb_save_code,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        mnist_root=args.mnist_root,
        mnist_download=args.mnist_download,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        accumulate_rig=args.accumulate_rig,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_request: str) -> torch.device:
    if device_request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_request)


def resolve_trainer_devices(device: torch.device) -> tuple[str, int | list[int]]:
    if device.type == "cuda":
        index = device.index
        devices: int | list[int] = [index] if index is not None else 1
        return "gpu", devices
    if device.type == "mps":
        return "mps", 1
    return "cpu", 1


def build_model(input_dim: int, cfg: RunConfig) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Flatten()]
    in_dim = input_dim
    for _ in range(cfg.depth):
        layers.append(nn.Linear(in_dim, cfg.width))
        layers.append(nn.ReLU())
        in_dim = cfg.width
    layers.append(nn.Linear(in_dim, 10))
    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mnist_data_present(root: Path) -> bool:
    processed_dir = root / "processed"
    required_files = ("training.pt", "test.pt")
    return all((processed_dir / name).exists() for name in required_files)


class MNISTImplicitDataModule(LightningDataModule):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_dataset: datasets.MNIST | None = None
        self.test_dataset: datasets.MNIST | None = None

    def prepare_data(self) -> None:
        should_download = self.cfg.mnist_download or not mnist_data_present(self.cfg.mnist_root)
        if should_download and not self.cfg.mnist_download:
            LOGGER.info(
                "MNIST dataset not found at %s; enabling download.",
                self.cfg.mnist_root,
            )
            self.cfg.mnist_download = True
        if self.cfg.mnist_download:
            datasets.MNIST(root=self.cfg.mnist_root, train=True, download=True)
            datasets.MNIST(root=self.cfg.mnist_root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is not None and stage not in ("fit", "validate", "test"):
            return
        self.train_dataset = datasets.MNIST(
            root=self.cfg.mnist_root,
            train=True,
            download=self.cfg.mnist_download,
            transform=self.transform,
        )
        self.test_dataset = datasets.MNIST(
            root=self.cfg.mnist_root,
            train=False,
            download=self.cfg.mnist_download,
            transform=self.transform,
        )

    @property
    def train_dataset_size(self) -> int:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset has not been initialised.")
        return len(self.train_dataset)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the train dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the val dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting the test dataloader.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def train_eval_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the eval dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )


class MNISTImplicitModule(LightningModule):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_model(28 * 28, cfg)
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.num_params = count_parameters(self.model)
        self.theoretical_lambda = cfg.learning_rate * self.num_params / 4.0
        self.best_test_accuracy = -math.inf
        self.best_metrics: Dict[str, Any] = {}
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.summary: Dict[str, Any] | None = None
        self.last_train_eval: Dict[str, float] | None = None
        self.last_test_eval: Dict[str, float] | None = None
        self.train_eval_dataset_size: int = 0
        self.last_evaluated_epoch: int | None = None
        self.save_hyperparameters(config_to_dict(cfg))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean()
        batch_size = inputs.size(0)
        self.log(
            "train/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/accuracy",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean()
        batch_size = inputs.size(0)
        self.log(
            "val/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/accuracy",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        datamodule = self._get_datamodule()
        train_eval_loader = datamodule.train_eval_dataloader()
        test_loader = datamodule.test_dataloader()
        device = self.device

        self.last_train_eval = evaluate_model(
            self,
            train_eval_loader,
            self.loss_fn,
            device,
            self.cfg.max_eval_batches,
        )
        self.last_test_eval = evaluate_model(
            self,
            test_loader,
            self.loss_fn,
            device,
            self.cfg.max_eval_batches,
        )
        self.train_eval_dataset_size = datamodule.train_dataset_size
        self.last_evaluated_epoch = self.trainer.current_epoch + 1

    def test_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean()
        batch_size = inputs.size(0)
        self.log(
            "test/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/accuracy",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.learning_rate,
            momentum=self.cfg.momentum,
            nesterov=self.cfg.nesterov,
            weight_decay=self.cfg.weight_decay,
        )

    def on_fit_end(self) -> None:
        wandb_logger = self._get_wandb_logger()
        run_identifier: Optional[str] = None
        if wandb_logger is not None:
            run = wandb_logger.experiment
            if run is not None:
                if self.best_metrics:
                    run.summary.update(
                        {
                            "rig_at_best": self.best_metrics.get("rig", math.nan),
                            "test_accuracy_at_best": self.best_metrics.get("test_accuracy", math.nan),
                            "epoch_at_best": self.best_metrics.get("epoch", math.nan),
                        }
                    )
                else:
                    run.summary.update({"rig_at_best": math.nan})
                run_identifier = getattr(run, "id", None)

        summary_payload = {
            "config": config_to_dict(self.cfg),
            "num_params": self.num_params,
            "lambda_theoretical": self.theoretical_lambda,
            "best_metrics": self.best_metrics,
        }
        self.summary = summary_payload

        output_path = self.cfg.output_path
        if output_path is None:
            if run_identifier is None:
                slurm_job = os.environ.get("SLURM_JOB_ID")
                run_identifier = slurm_job or f"seed{self.cfg.seed}"
            default_name = f"mnist_width{self.cfg.width}_lr{self.cfg.learning_rate:g}_{run_identifier}.json"
            output_path = RESULTS_DIR / default_name
        save_summary(output_path, summary_payload)
        LOGGER.info("Saved summary to %s", output_path)


        checkpoint_path = self.cfg.checkpoint_path
        identifier = run_identifier
        if identifier is None:
            slurm_job = os.environ.get("SLURM_JOB_ID")
            identifier = slurm_job or f"seed{self.cfg.seed}"
        if checkpoint_path is None:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_name = f"mnist_width{self.cfg.width}_lr{self.cfg.learning_rate:g}_{identifier}.pt"
            checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        else:
            checkpoint_path = checkpoint_path.resolve()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self.best_state_dict is not None:
            state_dict = self.best_state_dict
        else:
            with torch.no_grad():
                state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
        torch.save(
            {
                "state_dict": state_dict,
                "config": config_to_dict(self.cfg),
                "best_metrics": self.best_metrics,
            },
            checkpoint_path,
        )
        LOGGER.info("Saved best model to %s", checkpoint_path)

    def _get_datamodule(self) -> MNISTImplicitDataModule:
        if self.trainer is None or self.trainer.datamodule is None:
            raise RuntimeError("Trainer datamodule is not available.")
        return cast(MNISTImplicitDataModule, self.trainer.datamodule)

    def _get_wandb_logger(self) -> Optional[WandbLogger]:
        if self.trainer is None:
            return None
        logger = self.trainer.logger
        return logger if isinstance(logger, WandbLogger) else None


class RigComputationCallback(Callback):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_eval_loader: DataLoader | None = None
        self.train_dataset_size: int = 0

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage not in ("fit", None):
            return
        self._refresh_loaders(trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        module = cast(MNISTImplicitModule, pl_module)
        if module.last_train_eval is None or module.last_test_eval is None:
            return
        if self.train_eval_loader is None:
            self._refresh_loaders(trainer)

        train_eval = module.last_train_eval
        test_eval = module.last_test_eval
        current_epoch = module.last_evaluated_epoch or (trainer.current_epoch + 1)
        rig_value, grad_norm = self._compute_rig(module)

        current_metrics = {
            "epoch": current_epoch,
            "train_loss": train_eval["loss"],
            "train_accuracy": train_eval["accuracy"],
            "test_loss": test_eval["loss"],
            "test_accuracy": test_eval["accuracy"],
            "rig": rig_value,
            "grad_norm": grad_norm,
            "lambda_theoretical": module.theoretical_lambda,
            "num_params": module.num_params,
        }

        module.log(
            "full/train_loss",
            train_eval["loss"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        module.log(
            "full/train_accuracy",
            train_eval["accuracy"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        module.log(
            "full/test_loss",
            test_eval["loss"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        module.log(
            "full/test_accuracy",
            test_eval["accuracy"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        module.log(
            "metrics/rig",
            rig_value,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        module.log(
            "metrics/grad_norm",
            grad_norm,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        module.log(
            "metrics/lambda_theoretical",
            module.theoretical_lambda,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        LOGGER.info(
            "Epoch %03d | train_acc=%.4f | test_acc=%.4f | R_IG=%.3e",
            current_epoch,
            train_eval["accuracy"],
            test_eval["accuracy"],
            rig_value,
        )

        if not module.best_metrics:
            module.best_metrics = current_metrics
            module.best_state_dict = self._copy_state_dict(module)
            if math.isfinite(test_eval["accuracy"]):
                module.best_test_accuracy = test_eval["accuracy"]
        elif math.isfinite(test_eval["accuracy"]) and (
            not math.isfinite(module.best_test_accuracy)
            or test_eval["accuracy"] > module.best_test_accuracy
        ):
            module.best_metrics = current_metrics
            module.best_test_accuracy = test_eval["accuracy"]
            module.best_state_dict = self._copy_state_dict(module)

    @staticmethod
    def _copy_state_dict(module: MNISTImplicitModule) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _refresh_loaders(self, trainer: Trainer) -> None:
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("Trainer is missing the MNIST data module.")
        mnist_datamodule = cast(MNISTImplicitDataModule, datamodule)
        self.train_eval_loader = mnist_datamodule.train_eval_dataloader()
        self.train_dataset_size = mnist_datamodule.train_dataset_size

    def _compute_rig(self, module: MNISTImplicitModule) -> tuple[float, float]:
        if not self.cfg.accumulate_rig:
            return math.nan, math.nan
        loader = self.train_eval_loader
        dataset_size = self.train_dataset_size or module.train_eval_dataset_size
        if loader is None or dataset_size == 0:
            return math.nan, math.nan

        was_training = module.training
        module.eval()
        module.zero_grad(set_to_none=True)
        total_processed = 0
        with torch.enable_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                if self.cfg.max_eval_batches is not None and batch_idx >= self.cfg.max_eval_batches:
                    break
                inputs = inputs.to(module.device, non_blocking=True)
                targets = targets.to(module.device, non_blocking=True)
                logits = module(inputs)
                loss = module.loss_fn(logits, targets)
                batch_size = inputs.size(0)
                scale = batch_size / dataset_size
                (loss * scale).backward()
                total_processed += batch_size
        if total_processed == 0:
            module.zero_grad(set_to_none=True)
            if was_training:
                module.train()
            return math.nan, math.nan

        grad_sq_sum = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            grad_sq_sum += float(param.grad.detach().pow(2).sum().item())

        rig_value = grad_sq_sum / float(module.num_params)
        grad_norm = math.sqrt(grad_sq_sum)
        module.zero_grad(set_to_none=True)
        if was_training:
            module.train()
        return rig_value, grad_norm


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return float(correct)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            correct += accuracy_from_logits(logits, targets)
            total += batch_size
    if was_training:
        model.train()
    avg_loss = total_loss / total if total > 0 else math.nan
    accuracy = correct / total if total > 0 else math.nan
    return {"loss": avg_loss, "accuracy": accuracy}


def save_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def maybe_create_wandb_logger(cfg: RunConfig, module: MNISTImplicitModule) -> WandbLogger | None:
    if cfg.wandb_project is None:
        return None
    try:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            mode=cfg.wandb_mode,
            save_dir=str(REPO_ROOT / "logs" / "wandb"),
            name=cfg.wandb_run_name,
            save_code=cfg.wandb_save_code,
            log_model=cfg.wandb_log_model,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - wandb optional
        raise RuntimeError(
            "wandb is not installed but --wandb-project was specified. Install wandb or drop the flag."
        ) from exc

    config_payload = {
        **config_to_dict(cfg),
        "num_params": module.num_params,
        "lambda_theoretical": module.theoretical_lambda,
    }
    wandb_logger.experiment.config.update(config_payload, allow_val_change=True)
    if cfg.wandb_watch:
        wandb_logger.watch(module, log="all", log_freq=cfg.log_interval, log_graph=False)
    return wandb_logger


def train(cfg: RunConfig) -> Dict[str, Any]:
    configure_logging()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    accelerator, devices = resolve_trainer_devices(device)
    LOGGER.info("Using device: %s (accelerator=%s)", device, accelerator)

    datamodule = MNISTImplicitDataModule(cfg)
    model = MNISTImplicitModule(cfg)
    LOGGER.info(
        "Model with width=%d, depth=%d has %d parameters | theoretical λ=%.3e",
        cfg.width,
        cfg.depth,
        model.num_params,
        model.theoretical_lambda,
    )

    wandb_logger = maybe_create_wandb_logger(cfg, model)
    if wandb_logger is None:
        CSV_LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger_arg = CSVLogger(save_dir=str(CSV_LOG_DIR), name="mnist_implicit_reg")
    else:
        logger_arg = wandb_logger

    rig_callback = RigComputationCallback(cfg)
    callbacks: list[Callback] = [rig_callback]
    if cfg.patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/accuracy",
                mode="max",
                patience=cfg.patience,
                check_on_train_epoch_end=False,
            )
        )

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger_arg,
        callbacks=callbacks,
        log_every_n_steps=cfg.log_interval,
        limit_train_batches=cfg.max_train_batches if cfg.max_train_batches is not None else 1.0,
        limit_val_batches=cfg.max_eval_batches if cfg.max_eval_batches is not None else 1.0,
        limit_test_batches=cfg.max_eval_batches if cfg.max_eval_batches is not None else 1.0,
        enable_checkpointing=False,
        deterministic=True,
        gradient_clip_val=1.0,               # safely above typical grad norm
        gradient_clip_algorithm="norm",      # ("norm" is default) or "value" for element-wise clipping
    )

    trainer.fit(model, datamodule=datamodule)

    if model.summary is None:
        raise RuntimeError("Training summary was not generated.")
    return model.summary


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
