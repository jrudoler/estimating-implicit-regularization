import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import datasets, transforms
import wandb

# get parent directory and add to sys.path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))


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
        default=Path("checkpoints"),
        help="Directory for saving predictive-model checkpoints when training.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "inductive-bias" / "MNIST",
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


def get_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        if key not in config:
            raise KeyError(f"Missing required config value: {key}")
        return config[key]
    if hasattr(config, key):
        return getattr(config, key)
    raise KeyError(f"Missing required config attribute: {key}")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "on"}
    return bool(value)


# estimate L2 bias
from core.estimators import BiasWithCrossEntropy  # noqa: E402


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
        batchnorm: bool,
        lr: float,
        momentum: float,
        l2_penalty: float,
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
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
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

    def _compute_l2_penalty(self) -> Tensor:
        coefficient: float = float(self.hparams.l2_penalty)
        if coefficient <= 0:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        for param in self.parameters():
            if param.requires_grad:
                penalty = penalty + param.pow(2).sum()
        return 0.5 * coefficient * penalty

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        if float(self.hparams.l2_penalty) > 0:
            l2_penalty = self._compute_l2_penalty()
            loss = loss + l2_penalty
            self.log(
                "train/l2_penalty_epoch",
                l2_penalty,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
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
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
        )
        # reduce lr by a factor of 10 every at epochs 60, 100, and 200
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 100, 200],
            gamma=0.1,
        )
        return [optimizer], [scheduler]


class RidgeBias(nn.Module):
    bias_name = "ridge"

    def __init__(self, enforce_positive: bool = True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([1.0]))  # Initialize parameter
        self.enforce_positive = enforce_positive

    def forward(
        self,
        flattened_params: torch.Tensor,
        named_params: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        penalty = torch.sum(flattened_params.pow(2))
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty

    def get_bias_params(self) -> float:
        if self.enforce_positive:
            return torch.exp(self.beta).item()
        else:
            return self.beta.item()


class NuclearNormBias(nn.Module):
    bias_name = "nuclear_norm"

    def __init__(self, enforce_positive: bool = True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([1.0]))  # Initialize parameter
        self.enforce_positive = enforce_positive

    def forward(
        self,
        flattened_params: torch.Tensor,
        named_params: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        for weight in named_params.values():
            if weight.ndim >= 2:
                matrix = (
                    weight if weight.ndim == 2 else weight.reshape(weight.shape[0], -1)
                )
                penalty = penalty + torch.linalg.matrix_norm(matrix, ord="nuc")
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty

    def get_bias_params(self) -> float:
        if self.enforce_positive:
            return torch.exp(self.beta).item()
        else:
            return self.beta.item()


## TODO: implement a low-rank bias model that penalizes the rank of each weight matrix in the model
## it should compute the rank of each weight matrix in the model and then penalize the average rank


class LowRankBias(nn.Module):
    bias_name = "low_rank"

    def __init__(self, enforce_positive: bool = True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([1.0]))  # Initialize parameter
        self.enforce_positive = enforce_positive

    def compute_rank(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        rank = torch.linalg.matrix_rank(weight_matrix)
        return rank.to(weight_matrix.dtype)

    def compute_average_rank(
        self,
        named_params: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        total_rank = torch.zeros((), device=device, dtype=dtype)
        count = 0
        for _, weight in iter_weight_matrices(named_params):
            matrix = weight if weight.ndim == 2 else weight.reshape(weight.shape[0], -1)
            total_rank = total_rank + self.compute_rank(matrix)
            count += 1
        if count == 0:
            return total_rank
        return total_rank / float(count)

    def forward(
        self,
        flattened_params: torch.Tensor,
        named_params: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        average_rank = self.compute_average_rank(
            named_params, device=flattened_params.device, dtype=flattened_params.dtype
        )
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * average_rank

    def get_bias_params(self) -> float:
        if self.enforce_positive:
            return torch.exp(self.beta).item()
        else:
            return self.beta.item()


def iter_weight_matrices(
    named_params: Dict[str, torch.Tensor],
) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, parameter in named_params.items():
        if parameter.ndim >= 2:
            yield name, parameter


# TODO: abstraction for a list of bias models that adds ALL of their losses together and ensures
# that gradients are backpropagated through all of them to jointly estimate their parameters / strengths


class JointBias(nn.Module):
    def __init__(self, bias_models: Iterable[nn.Module]):
        super().__init__()
        self.bias_models = nn.ModuleList(list(bias_models))

    def forward(
        self,
        flattened_params: torch.Tensor,
        named_params: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        total = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        for bias_model in self.bias_models:
            total = total + bias_model(flattened_params, named_params, **kwargs)
        return total

    def report_parameters(self) -> Dict[str, float]:
        report: Dict[str, float] = {}
        for idx, bias_model in enumerate(self.bias_models):
            key = getattr(bias_model, "bias_name", bias_model.__class__.__name__)
            if key in report:
                key = f"{key}_{idx}"
            report[key] = bias_model.get_bias_params()
        return report


# train a model and then estimate the L2 regularization strength with an InductiveBiasEstimator


class TrainConfig:
    input_dim: int = 28 * 28
    width: int = 128
    depth: int = 5
    dropout: float = 0.2
    batchnorm: bool = False
    lr: float = 0.001
    momentum: float = 0.9
    l2_penalty: float = 0.01
    batch_size: int = 64
    num_data_workers: int = 4


def train_config_from_config(config: Any) -> TrainConfig:
    return TrainConfig(
        input_dim=int(get_config_value(config, "input_dim"))
        if (hasattr(config, "input_dim") or (isinstance(config, dict) and "input_dim" in config))
        else 28 * 28,
        width=int(get_config_value(config, "width")),
        depth=int(get_config_value(config, "depth")),
        dropout=float(get_config_value(config, "dropout")),
        batchnorm=as_bool(get_config_value(config, "batchnorm")),
        lr=float(get_config_value(config, "lr")),
        momentum=float(get_config_value(config, "momentum")),
        l2_penalty=float(get_config_value(config, "l2_penalty")),
        batch_size=int(get_config_value(config, "batch_size")),
        num_data_workers=int(get_config_value(config, "num_data_workers")),
    )


def build_datamodule(
    train_config: TrainConfig, data_root: Path
) -> MNISTLightningDataModule:
    datamodule = MNISTLightningDataModule(
        root=data_root,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_data_workers,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    return datamodule


def fit_predictive_model(
    train_config: TrainConfig,
    datamodule: MNISTLightningDataModule,
    checkpoint_dir: Path,
    logger: Optional[object] = None,
) -> Tuple[DeepReLULightningModule, Optional[Path]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = DeepReLULightningModule(
        input_dim=train_config.input_dim,
        width=train_config.width,
        depth=train_config.depth,
        dropout=train_config.dropout,
        batchnorm=train_config.batchnorm,
        lr=train_config.lr,
        momentum=train_config.momentum,
        l2_penalty=train_config.l2_penalty,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="model_{epoch:03d}",
        monitor="train/loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        max_epochs=500,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="train/loss_epoch", mode="min", patience=5),
            checkpoint_callback,
        ],
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
    best_model_path_str = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    best_model_path = Path(best_model_path_str) if best_model_path_str else None
    model.eval()
    return model, best_model_path


def load_trained_model(checkpoint_path: Path) -> DeepReLULightningModule:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = DeepReLULightningModule.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    return model


if __name__ == "__main__":
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

    wandb_context = nullcontext(None) if args.disable_wandb else wandb.init(**init_kwargs)

    with wandb_context as run:
        if config_from_file is not None:
            config_source: Any = config_from_file
            if run is not None and not args.disable_wandb:
                run.config.update(config_from_file, allow_val_change=True)
        else:
            config_source = wandb.config

        train_config = train_config_from_config(config_source)
        bias_lr = float(get_config_value(config_source, "bias_lr"))

        datamodule = build_datamodule(train_config, args.data_root)

        checkpoint_path: Optional[Path]
        if args.checkpoint_path is not None:
            print(
                f"[bias-estimation] Loading predictive model from {args.checkpoint_path}"
            )
            model = load_trained_model(args.checkpoint_path)
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_dir = args.checkpoint_dir
            if run is not None and not args.disable_wandb:
                checkpoint_dir = checkpoint_dir / run.id
            train_logger = None
            if not args.disable_wandb and run is not None:
                train_logger = WandbLogger(
                    name="train_deep_ReLU_model",
                    group=getattr(run, "name", None),
                    project=run.project,
                    entity=getattr(run, "entity", None),
                    log_model="all",
                )
            model, checkpoint_path = fit_predictive_model(
                train_config, datamodule, checkpoint_dir, logger=train_logger
            )
            if checkpoint_path is None:
                print(
                    f"[bias-estimation] No checkpoint saved in {checkpoint_dir}; continuing with in-memory weights."
                )

        if run is not None and checkpoint_path is not None:
            run.summary["predictive_model_checkpoint"] = str(checkpoint_path)
        if run is not None:
            run.summary["bias_estimation_uses_pretrained_model"] = (
                args.checkpoint_path is not None
            )

        l2_bias_model = RidgeBias()
        nuclear_norm_bias_model = NuclearNormBias()
        # low_rank_bias_model = LowRankBias()
        joint_bias_model = JointBias([l2_bias_model, nuclear_norm_bias_model])
        joint_estimator = BiasWithCrossEntropy(
            bias_model=joint_bias_model,
            predictive_model=model.eval(),
            lr=bias_lr,
        )

        bias_logger = None
        if not args.disable_wandb and run is not None:
            bias_logger = WandbLogger(
                name="dropout_bias_l2_estimation",
                group=getattr(run, "name", None),
                project=run.project,
                entity=getattr(run, "entity", None),
                prefix="bias",
            )

        bias_trainer = Trainer(
            max_epochs=500,
            accelerator="auto",
            devices=1,
            logger=bias_logger,
            log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor="train/loss", mode="min", patience=50)],
        )
        train_dataloader = datamodule.train_dataloader()

        bias_trainer.fit(joint_estimator, train_dataloaders=train_dataloader)

        bias_parameter_metrics = {
            f"estimated_{name}": float(value)
            for name, value in joint_bias_model.report_parameters().items()
        }

        if not args.disable_wandb and run is not None:
            wandb.log(bias_parameter_metrics)
            for key, value in bias_parameter_metrics.items():
                run.summary[key] = value
        else:
            print("[bias-estimation] Estimated bias parameters:")
            for key, value in bias_parameter_metrics.items():
                print(f"  {key}: {value}")
