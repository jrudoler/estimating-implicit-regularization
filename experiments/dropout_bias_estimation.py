#!/usr/bin/env python3
"""Train MNIST dropout models and estimate inductive biases, optionally reusing a pre-trained checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

from core.bias import RidgeBias, NuclearNormBias, LowRankBias, JointBias
from core.estimators import BiasWithCrossEntropyScheduled
from core.models import DeepReLUClassifier
from core.data import MNISTLightningDataModule

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MNIST dropout models and estimate inductive biases, "
            "optionally reusing a pre-trained checkpoint."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Load an existing predictive-model checkpoint and skip retraining.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("~/inductive-bias/logs/checkpoints").expanduser(),
        help="Directory for saving predictive-model checkpoints when training.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "inductive-bias" / "data",
        help="Local directory for MNIST data (downloaded if missing).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="JSON file containing wandb config values for bias estimation.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override wandb entity when launching outside a sweep.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override wandb project when launching outside a sweep.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Name to use for the wandb run when not managed by a sweep.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Completely disable wandb logging and run locally only.",
    )
    args, _ = parser.parse_known_args()
    args.checkpoint_dir = args.checkpoint_dir.expanduser()
    args.data_root = args.data_root.expanduser()
    if args.checkpoint_path is not None:
        args.checkpoint_path = args.checkpoint_path.expanduser()
    if args.config_path is not None:
        args.config_path = args.config_path.expanduser()
    return args


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, key):
        return getattr(config, key)
    if default is not None:
        return default
    raise KeyError(f"Missing required config value/attribute: {key}")


def main() -> None:
    args = parse_args()

    config_from_file: Optional[Dict[str, Any]] = None
    if args.config_path is not None:
        with args.config_path.open("r", encoding="utf-8") as fp:
            config_from_file = json.load(fp)

    if args.disable_wandb and config_from_file is None:
        raise ValueError(
            "--config-path is required when wandb logging is disabled to provide hyper-parameters."
        )

    init_kwargs: Dict[str, Any] = {}
    if config_from_file is not None:
        init_kwargs["config"] = config_from_file
    if args.wandb_project is not None:
        init_kwargs["project"] = args.wandb_project
    if args.wandb_entity is not None:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        init_kwargs["name"] = args.wandb_run_name
    if not args.disable_wandb:
        init_kwargs.setdefault("project", "inductive-bias")
        init_kwargs.setdefault("job_type", "dropout_bias_estimation")

    wandb_context = (
        nullcontext(None) if args.disable_wandb else wandb.init(**init_kwargs)
    )

    with wandb_context as run:
        if config_from_file is not None:
            config_source: Any = config_from_file
            if run is not None and not args.disable_wandb:
                run.config.update(config_from_file, allow_val_change=True)
        else:
            config_source = wandb.config if not args.disable_wandb else config_from_file

        # Extract hyperparameters with defaults
        seed = int(get_config_value(config_source, "seed", 42))
        set_seed(seed)

        input_dim = int(get_config_value(config_source, "input_dim", 28 * 28))
        num_classes = int(get_config_value(config_source, "num_classes", 10))
        depth = int(get_config_value(config_source, "depth", 5))
        width = int(get_config_value(config_source, "width", 128))
        dropout = float(get_config_value(config_source, "dropout", 0.2))
        batchnorm = bool(get_config_value(config_source, "batchnorm", False))
        l2_lambda = float(get_config_value(config_source, "l2_penalty", 0.01))
        lr = float(get_config_value(config_source, "lr", 0.001))
        momentum = float(get_config_value(config_source, "momentum", 0.9))
        batch_size = int(get_config_value(config_source, "batch_size", 64))
        num_data_workers = int(get_config_value(config_source, "num_data_workers", 4))
        val_fraction = float(get_config_value(config_source, "val_fraction", 0.1))
        max_epochs = int(get_config_value(config_source, "max_epochs", 500))
        patience = int(get_config_value(config_source, "patience", 50))

        bias_lr = float(get_config_value(config_source, "bias_lr", 0.1))
        bias_max_epochs = int(get_config_value(config_source, "bias_max_epochs", 2000))
        bias_patience = int(get_config_value(config_source, "bias_patience", 100))

        # Parse bias types (comma-separated list, e.g., "ridge,nuclear_norm")
        bias_types_str = get_config_value(config_source, "bias_types", "ridge")
        bias_types = [b.strip() for b in bias_types_str.split(",")]

        LOGGER.info("Starting dropout bias estimation experiment")
        LOGGER.info(
            "Config: depth=%d, width=%d, dropout=%.3f, batchnorm=%s, l2_lambda=%.6f, seed=%d",
            depth,
            width,
            dropout,
            batchnorm,
            l2_lambda,
            seed,
        )
        LOGGER.info("Bias types: %s", bias_types)

        # Build datamodule
        datamodule = MNISTLightningDataModule(
            root=args.data_root,
            batch_size=batch_size,
            num_workers=num_data_workers,
            val_fraction=val_fraction,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")

        # Load or train predictive model
        checkpoint_path: Optional[Path]
        if args.checkpoint_path is not None:
            LOGGER.info(
                "[bias-estimation] Loading predictive model from %s",
                args.checkpoint_path,
            )
            model = DeepReLUClassifier.load_from_checkpoint(str(args.checkpoint_path))
            model.eval()
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_dir = args.checkpoint_dir
            if run is not None and not args.disable_wandb:
                checkpoint_dir = checkpoint_dir / run.id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model = DeepReLUClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                depth=depth,
                width=width,
                dropout=dropout,
                batchnorm=batchnorm,
                l2_lambda=l2_lambda,
                lr=lr,
                momentum=momentum,
            )

            model_logger = None
            if not args.disable_wandb and run is not None:
                model_logger = WandbLogger(
                    project=run.project,
                    name=f"model-dropout{dropout:.2f}-d{depth}w{width}-l2{l2_lambda:.4f}-s{seed}",
                    experiment=run,
                    log_model=True,
                    save_dir="./logs/",
                )

            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            devices = 1

            checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="model_{epoch:03d}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )

            model_trainer = Trainer(
                max_epochs=max_epochs,
                logger=model_logger,
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="val/loss", mode="min", patience=patience),
                    checkpoint_callback,
                ],
                log_every_n_steps=50,
            )

            model_trainer.fit(model, datamodule=datamodule)

            # Evaluate on test set
            datamodule.setup("test")
            model_trainer.test(model, datamodule=datamodule)
            test_results = model_trainer.callback_metrics
            if not args.disable_wandb and run is not None:
                wandb.log(
                    {
                        "test/acc": test_results.get("test/acc", 0.0),
                        "test/loss": test_results.get("test/loss", float("inf")),
                    }
                )

            best_model_path_str = (
                checkpoint_callback.best_model_path
                or checkpoint_callback.last_model_path
            )
            checkpoint_path = Path(best_model_path_str) if best_model_path_str else None
            model.eval()

            LOGGER.info("Finished training predictive model")

        if run is not None and checkpoint_path is not None:
            run.summary["predictive_model_checkpoint"] = str(checkpoint_path)
        if run is not None:
            run.summary["bias_estimation_uses_pretrained_model"] = (
                args.checkpoint_path is not None
            )

        # Build bias models
        bias_model_list: List[torch.nn.Module] = []
        for bias_type in bias_types:
            if bias_type == "ridge":
                bias_model_list.append(RidgeBias(enforce_positive=True))
            elif bias_type == "nuclear_norm":
                bias_model_list.append(NuclearNormBias(enforce_positive=True))
            elif bias_type == "low_rank":
                bias_model_list.append(LowRankBias(enforce_positive=True))
            else:
                raise ValueError(f"Unknown bias type: {bias_type}")

        joint_bias_model = JointBias(bias_model_list)

        bias_estimator = BiasWithCrossEntropyScheduled(
            predictive_model=model.eval(),
            bias_model=joint_bias_model,
            grad_match_loss_fn=torch.nn.functional.mse_loss,
            lr=bias_lr,  # This will be ignored, but kept for compatibility
            optimizer_cls=torch.optim.Adam,
            bias_lr=bias_lr,
        )

        bias_logger = None
        if not args.disable_wandb and run is not None:
            bias_logger = WandbLogger(
                project=run.project,
                name=f"bias-dropout{dropout:.2f}-d{depth}w{width}-l2{l2_lambda:.4f}-s{seed}",
                experiment=run,
                log_model=False,
            )

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1

        bias_trainer = Trainer(
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
            log_every_n_steps=10,
        )

        train_dataloader = datamodule.train_dataloader()
        bias_trainer.fit(bias_estimator, train_dataloaders=train_dataloader)

        # Log results
        bias_parameter_metrics = {
            f"estimated_{name}": float(value)
            for name, value in joint_bias_model.report_parameters().items()
        }

        if not args.disable_wandb and run is not None:
            wandb.log(bias_parameter_metrics)
            for key, value in bias_parameter_metrics.items():
                run.summary[key] = value
        else:
            LOGGER.info("[bias-estimation] Estimated bias parameters:")
            for key, value in bias_parameter_metrics.items():
                LOGGER.info("  %s: %f", key, value)

        if not args.disable_wandb and run is not None:
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
