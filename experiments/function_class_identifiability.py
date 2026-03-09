#!/usr/bin/env python3
"""Function-class identifiability experiment with configurable bias bases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import (  # noqa: E402
    JointBias,
    LayerNormBalanceBias,
    LayerNormProductBias,
    NuclearNormBias,
    OrthogonalBias,
    RidgeBias,
    RowNormVarianceBias,
    SpectralEntropyBias,
    StableRankBias,
    WeightCoherenceBias,
)
from core.estimators import BiasWithMSE, vector_to_parameter_views  # noqa: E402
from core.estimators import BiasWithCrossEntropyNormalized  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticDatasetMetadata:
    data_mode: str
    function_class: str
    input_rank: int
    input_effective_rank: float
    input_condition_number: float
    input_spectrum: str
    input_spectrum_decay: float
    teacher_rank: int
    teacher_effective_rank: float
    teacher_condition_number: float
    teacher_spectrum: str
    teacher_spectrum_decay: float
    teacher_hidden_dim: int
    teacher_activation: str
    target_std: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_csv_list(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_csv_floats(raw_value: str) -> List[float]:
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_bool(raw_value: object) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw_value)


def participation_ratio(values: Tensor) -> float:
    safe_values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(safe_values.sum().item())
    squared_total = float(safe_values.pow(2).sum().item())
    if total <= 1e-12 or squared_total <= 1e-12:
        return 0.0
    return (total * total) / squared_total


def positive_condition_number(values: Tensor) -> float:
    safe_values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    positive_values = safe_values[safe_values > 1e-12]
    if positive_values.numel() <= 1:
        return 1.0
    return float((positive_values.max() / positive_values.min()).item())


def normalize_spectrum(values: Tensor, target_sum: float) -> Tensor:
    safe_values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    total = safe_values.sum()
    if float(total.item()) <= 1e-12:
        return torch.zeros_like(safe_values)
    return safe_values * (target_sum / total)


def build_spectrum_values(
    size: int,
    active_rank: int,
    family: str,
    decay: float,
    *,
    target_sum: float,
) -> Tensor:
    if size <= 0:
        raise ValueError("Spectrum size must be positive.")

    if family == "identity":
        active_rank = size

    rank = max(1, min(active_rank, size))
    values = torch.zeros(size, dtype=torch.float32)
    index = torch.arange(rank, dtype=torch.float32)

    if family in {"identity", "uniform"}:
        active_values = torch.ones(rank, dtype=torch.float32)
    elif family == "spiked":
        active_values = torch.ones(rank, dtype=torch.float32)
        if rank > 1:
            active_values[1:] = math.exp(-max(decay, 0.0))
    elif family == "power_law":
        active_values = torch.pow(index + 1.0, -max(decay, 1e-3))
    else:
        raise ValueError(f"Unknown spectrum family: {family}")

    values[:rank] = normalize_spectrum(active_values, target_sum=target_sum)
    return values


def random_orthogonal_matrix(
    rows: int, cols: int, generator: torch.Generator
) -> Tensor:
    matrix = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
    q_matrix, _ = torch.linalg.qr(matrix, mode="reduced")
    return q_matrix


def build_teacher_first_layer(
    input_dim: int,
    hidden_dim: int,
    teacher_rank: int,
    teacher_spectrum: str,
    teacher_spectrum_decay: float,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    max_rank = min(input_dim, hidden_dim)
    active_rank = max(1, min(teacher_rank, max_rank))
    singular_values = build_spectrum_values(
        size=max_rank,
        active_rank=active_rank,
        family=teacher_spectrum,
        decay=teacher_spectrum_decay,
        target_sum=float(active_rank),
    )
    if teacher_spectrum == "identity":
        weight = torch.zeros(hidden_dim, input_dim, dtype=torch.float32)
        weight[:active_rank, :active_rank] = torch.diag(singular_values[:active_rank])
        return weight, singular_values
    left = random_orthogonal_matrix(hidden_dim, max_rank, generator)
    right = random_orthogonal_matrix(input_dim, max_rank, generator)
    weight = left @ torch.diag(singular_values) @ right.T
    return weight, singular_values


def apply_teacher_activation(activation: str, values: Tensor) -> Tensor:
    if activation == "relu":
        return F.relu(values)
    if activation == "tanh":
        return torch.tanh(values)
    if activation == "linear":
        return values
    raise ValueError(f"Unknown teacher activation: {activation}")


def build_bias_module(name: str, trainable: bool) -> nn.Module:
    bias_map = {
        "ridge": RidgeBias,
        "nuclear_norm": NuclearNormBias,
        "orthogonal": OrthogonalBias,
        "weight_coherence": WeightCoherenceBias,
        "row_norm_variance": RowNormVarianceBias,
        "layer_norm_product": LayerNormProductBias,
        "layer_norm_balance": LayerNormBalanceBias,
        "stable_rank": StableRankBias,
        "spectral_entropy": SpectralEntropyBias,
    }
    if name not in bias_map:
        raise ValueError(f"Unknown bias type: {name}")

    module = bias_map[name](enforce_positive=True, init_value=0.0)
    if not trainable:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return module


def calibrate_explicit_lambdas(
    predictive_model: nn.Module,
    bias_types: Sequence[str],
    lambda_scale: float,
) -> List[float]:
    flat_params = (
        torch.nn.utils.parameters_to_vector(list(predictive_model.parameters()))
        .detach()
        .requires_grad_(True)
    )
    structured_params = vector_to_parameter_views(flat_params, predictive_model)

    balanced_lambdas: List[float] = []
    for bias_name in bias_types:
        bias_module = build_bias_module(bias_name, trainable=False).to(flat_params.device)
        penalty = bias_module(flat_params, structured_params)
        grad_vec = torch.autograd.grad(
            penalty, flat_params, create_graph=False, retain_graph=True
        )[0]
        grad_norm = float(torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0).norm().item())
        balanced_lambdas.append(lambda_scale / max(grad_norm, 1e-8))
    return balanced_lambdas


def generate_dataset(
    n_samples: int,
    input_dim: int,
    function_class: str,
    noise_std: float,
    seed: int,
    data_mode: str = "legacy",
    input_rank: int | None = None,
    input_spectrum: str = "identity",
    input_spectrum_decay: float = 1.5,
    teacher_rank: int | None = None,
    teacher_spectrum: str = "uniform",
    teacher_spectrum_decay: float = 1.5,
    teacher_hidden_dim: int = 16,
    teacher_activation: str = "relu",
    target_scale: float = 1.0,
) -> Tuple[TensorDataset, SyntheticDatasetMetadata]:
    generator = torch.Generator().manual_seed(seed)
    if data_mode == "legacy":
        features = torch.randn(n_samples, input_dim, generator=generator)
        weights = torch.randn(input_dim, 1, generator=generator)
        linear_response = features @ weights

        if function_class == "linear":
            targets = linear_response
        elif function_class == "polynomial":
            targets = 0.5 * linear_response + 0.25 * linear_response.pow(2)
        elif function_class == "sine":
            targets = torch.sin(linear_response)
        else:
            raise ValueError(f"Unknown function class: {function_class}")

        if noise_std > 0:
            targets = targets + noise_std * torch.randn(
                n_samples, 1, generator=generator
            )
        metadata = SyntheticDatasetMetadata(
            data_mode="legacy",
            function_class=function_class,
            input_rank=input_dim,
            input_effective_rank=float(input_dim),
            input_condition_number=1.0,
            input_spectrum="identity",
            input_spectrum_decay=0.0,
            teacher_rank=0,
            teacher_effective_rank=0.0,
            teacher_condition_number=1.0,
            teacher_spectrum="none",
            teacher_spectrum_decay=0.0,
            teacher_hidden_dim=0,
            teacher_activation="none",
            target_std=float(targets.std().item()),
        )
        return TensorDataset(features.float(), targets.float()), metadata

    if data_mode != "structured":
        raise ValueError(f"Unknown data_mode: {data_mode}")
    if function_class != "teacher_relu":
        raise ValueError(
            "Structured data_mode requires function_class='teacher_relu'."
        )

    resolved_input_rank = max(1, min(input_rank or input_dim, input_dim))
    input_eigenvalues = build_spectrum_values(
        size=input_dim,
        active_rank=resolved_input_rank,
        family=input_spectrum,
        decay=input_spectrum_decay,
        target_sum=float(input_dim),
    )
    input_basis = random_orthogonal_matrix(input_dim, input_dim, generator)
    latent = torch.randn(n_samples, input_dim, generator=generator, dtype=torch.float32)
    scaled_latent = latent * torch.sqrt(input_eigenvalues).unsqueeze(0)
    features = scaled_latent @ input_basis.T

    resolved_teacher_hidden_dim = max(1, teacher_hidden_dim)
    max_teacher_rank = min(input_dim, resolved_teacher_hidden_dim)
    resolved_teacher_rank = max(1, min(teacher_rank or max_teacher_rank, max_teacher_rank))
    teacher_first_layer, teacher_singular_values = build_teacher_first_layer(
        input_dim=input_dim,
        hidden_dim=resolved_teacher_hidden_dim,
        teacher_rank=resolved_teacher_rank,
        teacher_spectrum=teacher_spectrum,
        teacher_spectrum_decay=teacher_spectrum_decay,
        generator=generator,
    )
    teacher_second_layer = torch.randn(
        resolved_teacher_hidden_dim,
        1,
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(resolved_teacher_hidden_dim)

    hidden_activations = apply_teacher_activation(
        teacher_activation,
        features @ teacher_first_layer.T,
    )
    targets = hidden_activations @ teacher_second_layer
    target_std = float(targets.std().item())
    if target_std > 1e-8:
        targets = targets * (target_scale / target_std)
    if noise_std > 0:
        targets = targets + noise_std * torch.randn(
            n_samples, 1, generator=generator, dtype=torch.float32
        )

    metadata = SyntheticDatasetMetadata(
        data_mode="structured",
        function_class=function_class,
        input_rank=resolved_input_rank,
        input_effective_rank=participation_ratio(input_eigenvalues),
        input_condition_number=positive_condition_number(input_eigenvalues),
        input_spectrum=input_spectrum,
        input_spectrum_decay=input_spectrum_decay,
        teacher_rank=resolved_teacher_rank,
        teacher_effective_rank=participation_ratio(teacher_singular_values),
        teacher_condition_number=positive_condition_number(teacher_singular_values),
        teacher_spectrum=teacher_spectrum,
        teacher_spectrum_decay=teacher_spectrum_decay,
        teacher_hidden_dim=resolved_teacher_hidden_dim,
        teacher_activation=teacher_activation,
        target_std=float(targets.std().item()),
    )
    return TensorDataset(features.float(), targets.float()), metadata


class DeepReLURegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        depth: int,
        width: int,
        lr: float,
        explicit_bias_types: Sequence[str],
        explicit_lambdas: Sequence[float],
    ) -> None:
        super().__init__()
        if len(explicit_bias_types) != len(explicit_lambdas):
            raise ValueError("explicit_bias_types and explicit_lambdas must have same length.")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        self.explicit_bias_types = list(explicit_bias_types)
        self.explicit_lambdas = [float(value) for value in explicit_lambdas]
        self.explicit_bias_modules: Dict[str, nn.Module] = {
            bias_name: build_bias_module(bias_name, trainable=False)
            for bias_name in self.explicit_bias_types
        }
        self.save_hyperparameters()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)

    def _explicit_regularization(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        flat_params = torch.nn.utils.parameters_to_vector(list(self.network.parameters()))
        structured_params = vector_to_parameter_views(flat_params, self.network)
        total_penalty = torch.zeros(
            (), device=flat_params.device, dtype=flat_params.dtype
        )
        components: Dict[str, Tensor] = {}

        for bias_name, coefficient in zip(
            self.explicit_bias_types, self.explicit_lambdas
        ):
            bias_module = self.explicit_bias_modules[bias_name]
            first_param = next(bias_module.parameters(), None)
            if first_param is not None and first_param.device != flat_params.device:
                bias_module = bias_module.to(flat_params.device)
                self.explicit_bias_modules[bias_name] = bias_module

            penalty_value = bias_module(flat_params, structured_params)
            weighted_penalty = coefficient * penalty_value
            components[bias_name] = weighted_penalty
            total_penalty = total_penalty + weighted_penalty

        return total_penalty, components

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        predictions = self(inputs)
        mse_loss = self.loss_fn(predictions, targets)
        regularization, components = self._explicit_regularization()
        loss = mse_loss + regularization

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/regularization",
            regularization,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        for name, value in components.items():
            self.log(
                f"train/penalty/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        predictions = self(inputs)
        mse_loss = self.loss_fn(predictions, targets)
        regularization, _ = self._explicit_regularization()
        loss = mse_loss + regularization
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        predictions = self(inputs)
        mse_loss = self.loss_fn(predictions, targets)
        regularization, _ = self._explicit_regularization()
        loss = mse_loss + regularization
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/mse", mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
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


class BiasWithMSENormalized(BiasWithCrossEntropyNormalized):
    """Normalized gradient matching specialized for MSE targets."""

    def predictive_loss_grad(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if targets.ndim == 1:
            targets = targets.view(-1, 1)
        return -2 * (targets - predictions)


def cosine_between(vec_a: Tensor, vec_b: Tensor) -> float:
    safe_a = torch.nan_to_num(vec_a.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_b = torch.nan_to_num(vec_b.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(safe_a.norm().item() * safe_b.norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(safe_a, safe_b).item() / denom)


def compute_target_and_bias_gradients(
    predictive_model: nn.Module,
    batch: Tuple[Tensor, Tensor],
    candidate_bias_types: Sequence[str],
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, float]]:
    inputs, targets = batch
    device = next(predictive_model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)

    predictions = predictive_model(inputs)
    data_loss = F.mse_loss(predictions, targets)
    loss_grads = torch.autograd.grad(
        data_loss, tuple(predictive_model.parameters()), create_graph=False
    )
    target_gradient = -torch.cat([grad.reshape(-1) for grad in loss_grads]).detach()

    flat_params = (
        torch.nn.utils.parameters_to_vector(list(predictive_model.parameters()))
        .detach()
        .requires_grad_(True)
        .to(device)
    )
    structured_params = vector_to_parameter_views(flat_params, predictive_model)

    bias_gradients: Dict[str, Tensor] = {}
    target_alignments: Dict[str, float] = {}
    for bias_name in candidate_bias_types:
        bias_module = build_bias_module(bias_name, trainable=False).to(device)
        penalty = bias_module(flat_params, structured_params)
        grad_vec = torch.autograd.grad(
            penalty, flat_params, create_graph=False, retain_graph=True
        )[0].detach()
        grad_vec = torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
        bias_gradients[bias_name] = grad_vec
        target_alignments[bias_name] = cosine_between(target_gradient, grad_vec)

    return target_gradient, bias_gradients, target_alignments


def pairwise_cosine_matrix(
    bias_gradients: Dict[str, Tensor], ordered_bias_names: Sequence[str]
) -> Tensor:
    count = len(ordered_bias_names)
    matrix = torch.eye(count, dtype=torch.float32)
    for i in range(count):
        for j in range(i + 1, count):
            name_i = ordered_bias_names[i]
            name_j = ordered_bias_names[j]
            cosine_value = cosine_between(bias_gradients[name_i], bias_gradients[name_j])
            matrix[i, j] = cosine_value
            matrix[j, i] = cosine_value
    return matrix


def geometry_stats(cosine_matrix: Tensor) -> Tuple[float, float]:
    safe_matrix = torch.nan_to_num(
        cosine_matrix, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(min=-1.0, max=1.0)
    count = safe_matrix.shape[0]
    if count <= 1:
        return 0.0, 1.0

    eye = torch.eye(count, device=safe_matrix.device, dtype=safe_matrix.dtype)
    off_diag = (safe_matrix - eye).abs()
    max_abs_offdiag = float(off_diag.max().item())
    gram = safe_matrix + 1e-6 * eye
    condition_number = float(torch.linalg.cond(gram).item())
    return max_abs_offdiag, condition_number


def select_noncollinear_subset(
    candidate_bias_types: Sequence[str],
    bias_gradients: Dict[str, Tensor],
    target_gradient: Tensor,
    subset_size: int,
    max_pairwise_cos: float,
) -> List[str]:
    ranked = sorted(
        candidate_bias_types,
        key=lambda name: abs(cosine_between(target_gradient, bias_gradients[name])),
        reverse=True,
    )

    selected: List[str] = []
    for bias_name in ranked:
        if len(selected) >= subset_size:
            break
        is_compatible = all(
            abs(cosine_between(bias_gradients[bias_name], bias_gradients[chosen]))
            <= max_pairwise_cos
            for chosen in selected
        )
        if is_compatible:
            selected.append(bias_name)

    if len(selected) < subset_size:
        for bias_name in ranked:
            if bias_name not in selected:
                selected.append(bias_name)
                if len(selected) >= subset_size:
                    break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Function-class identifiability experiment using gradient-based bias recovery."
    )
    parser.add_argument(
        "--function-class",
        type=str,
        default="linear",
        choices=["linear", "polynomial", "sine", "teacher_relu"],
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="legacy",
        choices=["legacy", "structured"],
    )
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-spectrum",
        type=str,
        default="identity",
        choices=["identity", "uniform", "spiked", "power_law"],
    )
    parser.add_argument("--input-rank", type=int, default=0)
    parser.add_argument("--input-spectrum-decay", type=float, default=1.5)
    parser.add_argument(
        "--teacher-spectrum",
        type=str,
        default="uniform",
        choices=["identity", "uniform", "spiked", "power_law"],
    )
    parser.add_argument("--teacher-rank", type=int, default=0)
    parser.add_argument("--teacher-spectrum-decay", type=float, default=1.5)
    parser.add_argument("--teacher-hidden-dim", type=int, default=0)
    parser.add_argument(
        "--teacher-activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "linear"],
    )
    parser.add_argument("--target-scale", type=float, default=1.0)

    parser.add_argument("--gt-bias-types", type=str, default="ridge,nuclear_norm")
    parser.add_argument("--gt-lambdas", type=str, default="0.05,0.05")
    parser.add_argument(
        "--auto-balance-gt-lambdas",
        action="store_true",
        default=False,
        help="Rescale ground-truth lambdas by inverse initial gradient norms.",
    )
    parser.add_argument(
        "--gt-lambda-scale",
        type=float,
        default=0.05,
        help="Target scale used when auto-balancing ground-truth lambdas.",
    )

    parser.add_argument(
        "--candidate-bias-types",
        type=str,
        default=(
            "ridge,nuclear_norm,orthogonal,weight_coherence,"
            "layer_norm_product,layer_norm_balance"
        ),
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="fixed",
        choices=["fixed", "auto_noncollinear"],
    )
    parser.add_argument("--estimation-bias-types", type=str, default="")
    parser.add_argument("--max-pairwise-cos", type=float, default=0.3)
    parser.add_argument("--screen-batch-size", type=int, default=512)

    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-max-epochs", type=int, default=2000)
    parser.add_argument("--bias-patience", type=int, default=100)
    parser.add_argument(
        "--estimator-mode",
        type=str,
        default="normalized",
        choices=["standard", "normalized"],
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="inductive-bias-experiments",
    )
    return parser.parse_args()


def get_scale_params(joint_bias: JointBias, bias_types: Sequence[str]) -> Dict[str, float]:
    raw = joint_bias.get_bias_params()
    values: Dict[str, float] = {}
    for bias_name in bias_types:
        values[bias_name] = float(
            raw.get(f"{bias_name}/scale", raw.get(bias_name, 0.0))
        )
    return values


def main() -> None:
    args = parse_args()

    run = wandb.init(
        project=args.wandb_project,
        job_type="function_class_identifiability",
    )
    cfg = run.config

    function_class = str(cfg.get("function_class", args.function_class))
    data_mode = str(cfg.get("data_mode", args.data_mode))
    n_samples = int(cfg.get("n_samples", args.n_samples))
    input_dim = int(cfg.get("input_dim", args.input_dim))
    noise_std = float(cfg.get("noise_std", args.noise_std))
    depth = int(cfg.get("depth", args.depth))
    width = int(cfg.get("width", args.width))
    batch_size = int(cfg.get("batch_size", args.batch_size))
    lr = float(cfg.get("lr", args.lr))
    max_epochs = int(cfg.get("max_epochs", args.max_epochs))
    patience = int(cfg.get("patience", args.patience))
    val_fraction = float(cfg.get("val_fraction", args.val_fraction))
    test_fraction = float(cfg.get("test_fraction", args.test_fraction))
    seed = int(cfg.get("seed", args.seed))
    input_spectrum = str(cfg.get("input_spectrum", args.input_spectrum))
    input_rank_raw = int(cfg.get("input_rank", args.input_rank))
    input_spectrum_decay = float(
        cfg.get("input_spectrum_decay", args.input_spectrum_decay)
    )
    teacher_spectrum = str(cfg.get("teacher_spectrum", args.teacher_spectrum))
    teacher_rank_raw = int(cfg.get("teacher_rank", args.teacher_rank))
    teacher_spectrum_decay = float(
        cfg.get("teacher_spectrum_decay", args.teacher_spectrum_decay)
    )
    teacher_hidden_dim_raw = int(
        cfg.get("teacher_hidden_dim", args.teacher_hidden_dim)
    )
    teacher_activation = str(cfg.get("teacher_activation", args.teacher_activation))
    target_scale = float(cfg.get("target_scale", args.target_scale))

    gt_bias_types = parse_csv_list(str(cfg.get("gt_bias_types", args.gt_bias_types)))
    gt_lambdas = parse_csv_floats(str(cfg.get("gt_lambdas", args.gt_lambdas)))
    auto_balance_gt_lambdas = parse_bool(
        cfg.get("auto_balance_gt_lambdas", args.auto_balance_gt_lambdas)
    )
    gt_lambda_scale = float(cfg.get("gt_lambda_scale", args.gt_lambda_scale))
    if len(gt_bias_types) != len(gt_lambdas):
        raise ValueError("gt_bias_types and gt_lambdas must have matching lengths.")

    candidate_bias_types = parse_csv_list(
        str(cfg.get("candidate_bias_types", args.candidate_bias_types))
    )
    selection_mode = str(cfg.get("selection_mode", args.selection_mode))
    estimation_bias_types_raw = str(
        cfg.get("estimation_bias_types", args.estimation_bias_types)
    )
    max_pairwise_cos = float(cfg.get("max_pairwise_cos", args.max_pairwise_cos))
    screen_batch_size = int(cfg.get("screen_batch_size", args.screen_batch_size))

    bias_lr = float(cfg.get("bias_lr", args.bias_lr))
    bias_max_epochs = int(cfg.get("bias_max_epochs", args.bias_max_epochs))
    bias_patience = int(cfg.get("bias_patience", args.bias_patience))
    estimator_mode = str(cfg.get("estimator_mode", args.estimator_mode))

    set_seed(seed)

    teacher_hidden_dim = teacher_hidden_dim_raw if teacher_hidden_dim_raw > 0 else width
    dataset, dataset_metadata = generate_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        function_class=function_class,
        noise_std=noise_std,
        seed=seed,
        data_mode=data_mode,
        input_rank=input_rank_raw if input_rank_raw > 0 else input_dim,
        input_spectrum=input_spectrum,
        input_spectrum_decay=input_spectrum_decay,
        teacher_rank=teacher_rank_raw if teacher_rank_raw > 0 else min(input_dim, teacher_hidden_dim),
        teacher_spectrum=teacher_spectrum,
        teacher_spectrum_decay=teacher_spectrum_decay,
        teacher_hidden_dim=teacher_hidden_dim,
        teacher_activation=teacher_activation,
        target_scale=target_scale,
    )
    test_size = int(len(dataset) * test_fraction)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size - test_size
    if train_size <= 0:
        raise ValueError("Invalid split fractions; train size must be positive.")

    train_split, val_split, test_split = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DeepReLURegressor(
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        explicit_bias_types=gt_bias_types,
        explicit_lambdas=gt_lambdas,
    )
    if auto_balance_gt_lambdas:
        gt_lambdas = calibrate_explicit_lambdas(
            predictive_model=model.network,
            bias_types=gt_bias_types,
            lambda_scale=gt_lambda_scale,
        )
        model.explicit_lambdas = list(gt_lambdas)
        model.hparams.explicit_lambdas = list(gt_lambdas)
        LOGGER.info(
            "Auto-balanced gt lambdas (scale=%.4f): %s",
            gt_lambda_scale,
            ", ".join(f"{name}={value:.6f}" for name, value in zip(gt_bias_types, gt_lambdas)),
        )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    model_logger = WandbLogger(
        project=run.project,
        name=f"train-{function_class}-d{depth}w{width}-s{seed}",
        experiment=run,
        log_model=True,
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
        log_every_n_steps=25,
    )
    model_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_metrics_list = model_trainer.test(model, dataloaders=test_loader, verbose=False)
    test_metrics = test_metrics_list[0] if test_metrics_list else {}
    test_mse = float(test_metrics.get("test/mse", float("nan")))
    test_loss = float(test_metrics.get("test/loss", float("nan")))

    screening_loader = DataLoader(
        train_split,
        batch_size=min(screen_batch_size, len(train_split)),
        shuffle=False,
        num_workers=0,
    )
    screening_batch = next(iter(screening_loader))
    target_gradient, bias_gradients, target_alignments = compute_target_and_bias_gradients(
        model.network.eval(), screening_batch, candidate_bias_types
    )
    cosine_matrix = pairwise_cosine_matrix(bias_gradients, candidate_bias_types)
    max_offdiag_all, condition_all = geometry_stats(cosine_matrix)
    ridge_nuclear_cosine = float("nan")
    if "ridge" in bias_gradients and "nuclear_norm" in bias_gradients:
        ridge_nuclear_cosine = abs(
            cosine_between(
                bias_gradients["ridge"],
                bias_gradients["nuclear_norm"],
            )
        )

    if selection_mode == "auto_noncollinear":
        estimation_bias_types = select_noncollinear_subset(
            candidate_bias_types=candidate_bias_types,
            bias_gradients=bias_gradients,
            target_gradient=target_gradient,
            subset_size=len(gt_bias_types),
            max_pairwise_cos=max_pairwise_cos,
        )
    else:
        estimation_bias_types = (
            parse_csv_list(estimation_bias_types_raw)
            if estimation_bias_types_raw
            else list(gt_bias_types)
        )

    for name in estimation_bias_types:
        if name not in candidate_bias_types:
            raise ValueError(
                f"Estimation bias '{name}' must be included in candidate_bias_types."
            )

    index_lookup = {name: idx for idx, name in enumerate(candidate_bias_types)}
    selected_indices = [index_lookup[name] for name in estimation_bias_types]
    selected_matrix = cosine_matrix[selected_indices][:, selected_indices]
    max_offdiag_selected, condition_selected = geometry_stats(selected_matrix)

    estimators = [build_bias_module(name, trainable=True) for name in estimation_bias_types]
    joint_bias = JointBias(estimators)
    if estimator_mode == "normalized":
        bias_estimator = BiasWithMSENormalized(
            predictive_model=model.network.eval(),
            bias_model=joint_bias,
            grad_match_loss_fn=F.mse_loss,
            optimizer_cls=torch.optim.Adam,
            lr=bias_lr,
            bias_lr=bias_lr,
        )
    else:
        bias_estimator = BiasWithMSE(
            predictive_model=model.network.eval(),
            bias_model=joint_bias,
            grad_match_loss_fn=F.mse_loss,
            optimizer_cls=torch.optim.Adam,
            lr=bias_lr,
        )

    bias_logger = WandbLogger(
        project=run.project,
        name=f"estimate-{function_class}-{selection_mode}-s{seed}",
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
        log_every_n_steps=25,
    )
    bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)

    if estimator_mode == "normalized":
        estimated = {
            name: float(value)
            for name, value in bias_estimator.get_estimated_lambdas().items()
        }
    else:
        estimated = get_scale_params(joint_bias, estimation_bias_types)
    gt_map = {name: value for name, value in zip(gt_bias_types, gt_lambdas)}
    considered_biases = sorted(set(gt_bias_types) | set(estimation_bias_types))
    gt_bias_set = set(gt_bias_types)

    total_rel_error = 0.0
    max_rel_error = 0.0
    gt_rel_error_total = 0.0
    gt_count = 0
    false_positive_abs_total = 0.0
    false_positive_count = 0
    tp = fp = fn = 0
    activity_threshold = 1e-3

    for bias_name in considered_biases:
        true_value = float(gt_map.get(bias_name, 0.0))
        est_value = float(estimated.get(bias_name, 0.0))
        abs_error = abs(est_value - true_value)
        rel_error = (
            abs_error / max(abs(true_value), 1e-8) if abs(true_value) > 0 else abs_error
        )
        total_rel_error += rel_error
        max_rel_error = max(max_rel_error, rel_error)
        if bias_name in gt_bias_set:
            gt_rel_error_total += rel_error
            gt_count += 1
        else:
            false_positive_abs_total += abs(est_value)
            false_positive_count += 1

        is_true_active = abs(true_value) > activity_threshold
        is_est_active = abs(est_value) > activity_threshold
        if is_true_active and is_est_active:
            tp += 1
        elif (not is_true_active) and is_est_active:
            fp += 1
        elif is_true_active and (not is_est_active):
            fn += 1

        wandb.log(
            {
                f"recovery/{bias_name}/true": true_value,
                f"recovery/{bias_name}/estimated": est_value,
                f"recovery/{bias_name}/abs_error": abs_error,
                f"recovery/{bias_name}/rel_error": rel_error,
            }
        )

    mean_rel_error = total_rel_error / max(len(considered_biases), 1)
    gt_mean_rel_error = gt_rel_error_total / max(gt_count, 1)
    false_positive_mean_abs = false_positive_abs_total / max(false_positive_count, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    support_f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    scalar_logs = {
        "identifiability/mean_rel_error": mean_rel_error,
        "identifiability/gt_mean_rel_error": gt_mean_rel_error,
        "identifiability/max_rel_error": max_rel_error,
        "identifiability/false_positive_mean_abs": false_positive_mean_abs,
        "identifiability/support_precision": precision,
        "identifiability/support_recall": recall,
        "identifiability/support_f1": support_f1,
        "fit/test_loss": test_loss,
        "fit/test_mse": test_mse,
        "geometry/all/max_abs_pairwise_cos": max_offdiag_all,
        "geometry/all/condition_number": condition_all,
        "geometry/selected/max_abs_pairwise_cos": max_offdiag_selected,
        "geometry/selected/condition_number": condition_selected,
        "geometry/ridge_nuclear/cosine": ridge_nuclear_cosine,
        "config/data_mode": data_mode,
        "config/selection_mode": selection_mode,
        "config/estimator_mode": estimator_mode,
        "config/auto_balance_gt_lambdas": int(auto_balance_gt_lambdas),
        "config/gt_lambda_scale": gt_lambda_scale,
        "config/function_class": function_class,
        "config/gt_bias_types": ",".join(gt_bias_types),
        "config/gt_lambdas": ",".join(f"{value:.8f}" for value in gt_lambdas),
        "config/estimation_bias_types": ",".join(estimation_bias_types),
    }
    scalar_logs.update(
        {
            "data/input_rank": dataset_metadata.input_rank,
            "data/input_effective_rank": dataset_metadata.input_effective_rank,
            "data/input_condition_number": dataset_metadata.input_condition_number,
            "data/input_spectrum_decay": dataset_metadata.input_spectrum_decay,
            "data/teacher_rank": dataset_metadata.teacher_rank,
            "data/teacher_effective_rank": dataset_metadata.teacher_effective_rank,
            "data/teacher_condition_number": dataset_metadata.teacher_condition_number,
            "data/teacher_spectrum_decay": dataset_metadata.teacher_spectrum_decay,
            "data/teacher_hidden_dim": dataset_metadata.teacher_hidden_dim,
            "data/target_std": dataset_metadata.target_std,
        }
    )
    scalar_logs["config/input_spectrum"] = dataset_metadata.input_spectrum
    scalar_logs["config/teacher_spectrum"] = dataset_metadata.teacher_spectrum
    scalar_logs["config/teacher_activation"] = dataset_metadata.teacher_activation
    for name, value in target_alignments.items():
        scalar_logs[f"geometry/target_alignment/{name}"] = value
    for i, name_i in enumerate(candidate_bias_types):
        for j, name_j in enumerate(candidate_bias_types):
            scalar_logs[f"geometry/pairwise_cos/{name_i}__{name_j}"] = float(
                cosine_matrix[i, j].item()
            )

    LOGGER.info(
        "Fit | function_class=%s data_mode=%s test_mse=%.6f test_loss=%.6f",
        function_class,
        data_mode,
        test_mse,
        test_loss,
    )
    LOGGER.info(
        "Geometry | ridge_nuclear_cosine=%.6f input_spectrum=%s teacher_spectrum=%s input_rank=%d teacher_rank=%d",
        ridge_nuclear_cosine,
        dataset_metadata.input_spectrum,
        dataset_metadata.teacher_spectrum,
        dataset_metadata.input_rank,
        dataset_metadata.teacher_rank,
    )
    LOGGER.info(
        (
            "Result | function_class=%s selection_mode=%s "
            "estimator_mode=%s "
            "mean_rel_error=%.4f gt_mean_rel_error=%.4f max_rel_error=%.4f "
            "support_f1=%.4f "
            "selected_max_abs_cos=%.4f selected_cond=%.4f"
        ),
        function_class,
        selection_mode,
        estimator_mode,
        mean_rel_error,
        gt_mean_rel_error,
        max_rel_error,
        support_f1,
        max_offdiag_selected,
        condition_selected,
    )
    for bias_name in considered_biases:
        LOGGER.info(
            "  %s | true=%.6f estimated=%.6f",
            bias_name,
            float(gt_map.get(bias_name, 0.0)),
            float(estimated.get(bias_name, 0.0)),
        )

    wandb.log(scalar_logs)
    run.summary.update(scalar_logs)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
