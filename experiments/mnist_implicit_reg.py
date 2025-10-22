#!/usr/bin/env python3
"""Train MNIST MLPs and measure implicit gradient regularisation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
MNIST_ROOT = REPO_ROOT / "MNIST"
RESULTS_DIR = REPO_ROOT / "results" / "mnist_implicit_reg"

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional at test time
    wandb = None  # type: ignore[assignment]


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
    mnist_root: Path = MNIST_ROOT
    mnist_download: bool = False
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    accumulate_rig: bool = True


def config_to_dict(cfg: RunConfig) -> Dict[str, Any]:
    mapping: Dict[str, Any] = asdict(cfg)
    for key in ("output_path", "mnist_root"):
        value = mapping.get(key)
        mapping[key] = str(value) if value is not None else None
    if mapping.get("wandb_tags") is not None:
        mapping["wandb_tags"] = list(mapping["wandb_tags"])
    return mapping


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=RunConfig.width)
    parser.add_argument("--depth", type=int, default=RunConfig.depth)
    parser.add_argument("--learning-rate", type=float, default=RunConfig.learning_rate)
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--max-epochs", type=int, default=RunConfig.max_epochs)
    parser.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=RunConfig.momentum)
    parser.add_argument("--nesterov", action="store_true", default=RunConfig.nesterov)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--num-workers", type=int, default=RunConfig.num_workers)
    parser.add_argument("--device", type=str, default=RunConfig.device)
    parser.add_argument("--log-interval", type=int, default=RunConfig.log_interval)
    parser.add_argument("--patience", type=int, default=None)
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
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


def prepare_dataloaders(cfg: RunConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_set = datasets.MNIST(
        root=cfg.mnist_root,
        train=True,
        download=cfg.mnist_download,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=cfg.mnist_root,
        train=False,
        download=cfg.mnist_download,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Evaluation loader uses deterministic shuffling for gradient computation.
    train_eval_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, train_eval_loader, test_loader


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
    avg_loss = total_loss / total if total > 0 else math.nan
    accuracy = correct / total if total > 0 else math.nan
    return {"loss": avg_loss, "accuracy": accuracy}


def compute_rig(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_params: int,
    dataset_size: int,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    model.zero_grad(set_to_none=True)
    total_processed = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        batch_size = inputs.size(0)
        scale = batch_size / dataset_size
        (loss * scale).backward()
        total_processed += batch_size
    if total_processed == 0:
        return math.nan, math.nan

    grad_sq_sum = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_sq_sum += float(param.grad.detach().pow(2).sum().item())

    rig_value = grad_sq_sum / float(num_params)
    grad_norm = math.sqrt(grad_sq_sum)
    model.zero_grad(set_to_none=True)
    return rig_value, grad_norm


def maybe_init_wandb(cfg: RunConfig, num_params: int) -> Any:
    if cfg.wandb_project is None:
        return None
    if wandb is None:
        raise RuntimeError(
            "wandb is not installed but --wandb-project was specified. "
            "Install wandb or drop the flag."
        )
    wandb_kwargs: Dict[str, Any] = {
        "project": cfg.wandb_project,
        "entity": cfg.wandb_entity,
        "group": cfg.wandb_group,
        "tags": list(cfg.wandb_tags) if cfg.wandb_tags else None,
        "mode": cfg.wandb_mode,
        "name": cfg.wandb_run_name,
        "save_code": cfg.wandb_save_code,
        "config": {
            **config_to_dict(cfg),
            "num_params": num_params,
            "lambda_theoretical": cfg.learning_rate * num_params / 4.0,
        },
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    run = wandb.init(**wandb_kwargs)
    return run


def save_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def train(cfg: RunConfig) -> Dict[str, Any]:
    configure_logging()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    LOGGER.info("Using device: %s", device)

    train_loader, train_eval_loader, test_loader = prepare_dataloaders(cfg)
    input_dim = 28 * 28
    model = build_model(input_dim, cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
        weight_decay=cfg.weight_decay,
    )

    num_params = count_parameters(model)
    theoretical_lambda = cfg.learning_rate * num_params / 4.0
    wandb_run = maybe_init_wandb(cfg, num_params)
    if cfg.wandb_watch and wandb_run is not None:
        wandb.watch_called = False  # silence repeated warnings
        wandb.watch(model, log="all", log_freq=cfg.log_interval)

    LOGGER.info(
        "Model with width=%d, depth=%d has %d parameters | theoretical λ=%.3e",
        cfg.width,
        cfg.depth,
        num_params,
        theoretical_lambda,
    )

    train_size = len(train_loader.dataset)
    best_test_accuracy_fit = -math.inf
    best_metrics: Dict[str, Any] = {}
    epochs_without_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if cfg.max_train_batches is not None and batch_idx >= cfg.max_train_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_correct += accuracy_from_logits(logits, targets)
            epoch_samples += batch_size

        train_loss = epoch_loss / epoch_samples if epoch_samples else math.nan
        train_accuracy = epoch_correct / epoch_samples if epoch_samples else math.nan

        train_eval = evaluate_model(
            model, train_eval_loader, loss_fn, device, cfg.max_eval_batches
        )
        test_eval = evaluate_model(
            model, test_loader, loss_fn, device, cfg.max_eval_batches
        )

        rig_value = math.nan
        grad_norm = math.nan
        if (
            cfg.accumulate_rig
            and train_eval["accuracy"] >= 0.99999
            and math.isfinite(train_eval["accuracy"])
        ):
            rig_value, grad_norm = compute_rig(
                model,
                train_eval_loader,
                loss_fn,
                device,
                num_params=num_params,
                dataset_size=train_size,
                max_batches=cfg.max_eval_batches,
            )

            if test_eval["accuracy"] > best_test_accuracy_fit:
                best_test_accuracy_fit = test_eval["accuracy"]
                best_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_eval["accuracy"],
                    "test_loss": test_eval["loss"],
                    "test_accuracy": test_eval["accuracy"],
                    "rig": rig_value,
                    "grad_norm": grad_norm,
                    "lambda_theoretical": theoretical_lambda,
                    "num_params": num_params,
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
        else:
            epochs_without_improve += 1

        log_payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_eval["accuracy"],
            "test/loss": test_eval["loss"],
            "test/accuracy": test_eval["accuracy"],
            "metrics/rig": rig_value,
            "metrics/grad_norm": grad_norm,
            "metrics/lambda_theoretical": theoretical_lambda,
        }

        LOGGER.info(
            "Epoch %03d | train_acc=%.4f | test_acc=%.4f | R_IG=%.3e",
            epoch,
            train_eval["accuracy"],
            test_eval["accuracy"],
            rig_value,
        )

        if wandb_run is not None:
            wandb.log(log_payload)

        if cfg.patience is not None and epochs_without_improve >= cfg.patience:
            LOGGER.info(
                "Stopping early after %d epochs without improvement (patience=%d).",
                epochs_without_improve,
                cfg.patience,
            )
            break

    run_identifier: Optional[str] = None
    if wandb_run is not None:
        if best_metrics:
            wandb_run.summary.update(
                {
                    "rig_at_best": best_metrics.get("rig", math.nan),
                    "test_accuracy_at_best": best_metrics.get(
                        "test_accuracy", math.nan
                    ),
                    "epoch_at_best": best_metrics.get("epoch", math.nan),
                }
            )
        else:
            wandb_run.summary.update({"rig_at_best": math.nan})
        run_identifier = wandb_run.id
        wandb_run.finish()

    summary = {
        "config": config_to_dict(cfg),
        "num_params": num_params,
        "lambda_theoretical": theoretical_lambda,
        "best_metrics": best_metrics,
    }
    if cfg.output_path is not None:
        save_summary(cfg.output_path, summary)
    else:
        if run_identifier is None:
            slurm_job = os.environ.get("SLURM_JOB_ID")
            run_identifier = slurm_job or f"seed{cfg.seed}"
        default_name = (
            f"mnist_width{cfg.width}_lr{cfg.learning_rate:g}_{run_identifier}.json"
        )
        default_path = RESULTS_DIR / default_name
        save_summary(default_path, summary)
        LOGGER.info("Saved summary to %s", default_path)

    return summary


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
