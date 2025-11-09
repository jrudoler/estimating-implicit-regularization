from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

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


@dataclass
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


def fit_predictive_model(
    train_config: TrainConfig,
) -> Tuple[DeepReLULightningModule, Trainer]:
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
    trainer = Trainer(
        max_epochs=500,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="train/loss_epoch", mode="min", patience=5),
            ModelCheckpoint(
                dirpath="checkpoints/",
                filename="model_{epoch:03d}",
                monitor="train/loss_epoch", 
                mode="min",
                ),
        ],
        logger=WandbLogger(
            name="train_deep_ReLU_model",
            group=wandb.run.name,
            project=wandb.run.project,
        ),
        log_every_n_steps=1,
    )
    train_dm = MNISTLightningDataModule(
        root=os.path.expanduser("~/inductive-bias/MNIST"),
        batch_size=train_config.batch_size,
        num_workers=train_config.num_data_workers,
    )
    train_dm.prepare_data()
    train_dm.setup("fit")
    trainer.fit(model, train_dm)
    return model, trainer


if __name__ == "__main__":
    ## get config from wandb sweep
    with wandb.init() as run:
        config = wandb.config
        train_config = TrainConfig(
            width=config.width,
            depth=config.depth,
            dropout=config.dropout,
            batchnorm=config.batchnorm,
            lr=config.lr,
            momentum=config.momentum,
            l2_penalty=config.l2_penalty,
            batch_size=config.batch_size,
            num_data_workers=config.num_data_workers,
        )
        model, trainer = fit_predictive_model(train_config)

        l2_bias_model = RidgeBias()
        nuclear_norm_bias_model = NuclearNormBias()
        # low_rank_bias_model = LowRankBias()
        joint_bias_model = JointBias([l2_bias_model, nuclear_norm_bias_model])
        joint_estimator = BiasWithCrossEntropy(
            bias_model=joint_bias_model,
            predictive_model=model.eval(),
            lr=config.bias_lr,
        )
        bias_logger = WandbLogger(
            name="dropout_bias_l2_estimation",
            group=run.name,
            project=run.project,
        )
        bias_trainer = Trainer(
            max_epochs=500,
            accelerator="auto",
            devices=1,
            logger=bias_logger,
            log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor="train/loss", mode="min", patience=50)],
        )
        train_dataloader = trainer.datamodule.train_dataloader()

        bias_trainer.fit(joint_estimator, train_dataloaders=train_dataloader)

        # log estimated bias parameters as summary metrics
        bias_parameter_metrics = {
            f"estimated_{name}": value
            for name, value in joint_bias_model.report_parameters().items()
        }
        wandb.log(bias_parameter_metrics)
        for key, value in bias_parameter_metrics.items():
            run.summary[key] = value
