"""Retrain recovered elastic-net endpoints with explicit estimated penalties.

Each task starts from one retained elastic-net recovery endpoint for which both
estimated coefficients are close to ground truth.  The synthetic data, split,
initialization, minibatch order, optimizer, and number of epochs are reproduced
from the original seed.  Three models are trained:

* ``true``: the original ground-truth elastic-net coefficients;
* ``estimated``: the recovered coefficients;
* ``ablated``: no explicit regularization.

The saved original model is the target.  Because every retrain uses the same
initialization and minibatch randomness, both parameter and prediction distances
are meaningful training-dynamics diagnostics rather than cross-seed comparisons.
Only training-set quantities are reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from core.models import NonLinearNetwork  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticData:
    train_loader: DataLoader
    validation_loader: DataLoader
    train_x: Tensor
    train_y: Tensor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_data(seed: int, batch_size: int) -> SyntheticData:
    """Reproduce the original data generation and global-RNG split exactly."""
    set_seed(seed)
    features = torch.randn(5_000, 10)
    true_betas = 5.0 * torch.randn(10)
    targets = (features @ true_betas).view(-1, 1)
    targets = targets + 0.5 * torch.randn(5_000, 1)
    dataset = TensorDataset(features, targets)
    train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])
    train_indices = torch.tensor(train_dataset.indices)
    return SyntheticData(
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),
        validation_loader=DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        ),
        train_x=features[train_indices],
        train_y=targets[train_indices],
    )


def parameter_vector(model: LightningModule) -> Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()


def smooth_l1_basis_gradient(parameters: Tensor, smooth: float) -> Tensor:
    return torch.where(
        parameters.abs() < smooth,
        parameters / smooth,
        parameters.sign(),
    )


def penalty_values(parameters: Tensor, smooth: float) -> dict[str, float]:
    return {
        "smooth_l1": float(
            F.smooth_l1_loss(
                parameters,
                torch.zeros_like(parameters),
                beta=smooth,
                reduction="sum",
            )
        ),
        "l2": float(parameters.square().sum()),
    }


def endpoint_metrics(
    model: NonLinearNetwork,
    train_x: Tensor,
    train_y: Tensor,
    l1: float,
    l2: float,
    smooth: float,
) -> dict[str, Any]:
    """Full-training endpoint losses and stationarity under a stated objective."""
    device = next(model.parameters()).device
    features = train_x.to(device)
    targets = train_y.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    predictions = model(features)
    loss = F.mse_loss(predictions, targets)
    parameters = list(model.parameters())
    loss_gradients = torch.autograd.grad(loss, parameters)
    loss_gradient = torch.cat([gradient.reshape(-1) for gradient in loss_gradients])
    flat = parameter_vector(model)
    l1_gradient = smooth_l1_basis_gradient(flat, smooth)
    l2_gradient = 2.0 * flat
    objective_gradient = loss_gradient + l1 * l1_gradient + l2 * l2_gradient
    penalties = penalty_values(flat, smooth)
    parameter_norm = float(flat.norm())
    loss_gradient_norm = float(loss_gradient.norm())
    return {
        "train_mse": float(loss.detach()),
        "parameter_norm": parameter_norm,
        "prediction_mean": float(predictions.detach().mean()),
        "prediction_std": float(predictions.detach().std()),
        "loss_gradient_norm": loss_gradient_norm,
        "objective_gradient_norm": float(objective_gradient.norm()),
        "relative_objective_gradient_norm": float(
            objective_gradient.norm() / flat.norm()
        ),
        "objective_to_loss_gradient_ratio": float(
            objective_gradient.norm() / loss_gradient.norm()
        ),
        "smooth_l1_value": penalties["smooth_l1"],
        "l2_value": penalties["l2"],
        "penalty_term": l1 * penalties["smooth_l1"] + l2 * penalties["l2"],
        "total_objective": (
            float(loss.detach())
            + l1 * penalties["smooth_l1"]
            + l2 * penalties["l2"]
        ),
        "_parameters": flat.cpu(),
        "_predictions": predictions.detach().cpu(),
    }


def compare_endpoints(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float]:
    candidate_parameters = candidate["_parameters"]
    reference_parameters = reference["_parameters"]
    candidate_predictions = candidate["_predictions"]
    reference_predictions = reference["_predictions"]
    parameter_distance = float((candidate_parameters - reference_parameters).norm())
    prediction_rmse = float(
        (candidate_predictions - reference_predictions).square().mean().sqrt()
    )
    reference_prediction_rms = float(reference_predictions.square().mean().sqrt())
    reference_prediction_std = float(reference_predictions.std())
    return {
        "parameter_relative_l2": parameter_distance
        / max(float(reference_parameters.norm()), 1e-30),
        "parameter_cosine": float(
            F.cosine_similarity(
                candidate_parameters,
                reference_parameters,
                dim=0,
            )
        ),
        "prediction_rmse": prediction_rmse,
        "prediction_relative_rmse": prediction_rmse
        / max(reference_prediction_rms, 1e-30),
        "prediction_rmse_over_reference_std": prediction_rmse
        / max(reference_prediction_std, 1e-30),
        "prediction_correlation": float(
            torch.corrcoef(
                torch.stack(
                    [
                        candidate_predictions.reshape(-1),
                        reference_predictions.reshape(-1),
                    ]
                )
            )[0, 1]
        ),
    }


def public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


class DynamicsTrace(Callback):
    """Record matched training trajectories without changing optimizer state."""

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        target_parameters: Tensor,
        target_predictions: Tensor,
        every: int = 5,
    ) -> None:
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.target_parameters = target_parameters
        self.target_predictions = target_predictions
        self.every = every
        self.records: list[dict[str, float | int]] = []

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        epoch = trainer.current_epoch + 1
        if epoch != trainer.max_epochs and epoch % self.every != 0:
            return
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module(self.train_x.to(device)).cpu()
            flat = parameter_vector(pl_module).cpu()
            train_mse = float(
                F.mse_loss(predictions, self.train_y)
            )
            parameter_relative_l2 = float(
                (flat - self.target_parameters).norm()
                / self.target_parameters.norm()
            )
            prediction_relative_rmse = float(
                (predictions - self.target_predictions).square().mean().sqrt()
                / self.target_predictions.square().mean().sqrt()
            )
        if was_training:
            pl_module.train()
        self.records.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "parameter_norm": float(flat.norm()),
                "parameter_relative_l2_to_target": parameter_relative_l2,
                "prediction_relative_rmse_to_target": prediction_relative_rmse,
            }
        )


def load_target_model(
    checkpoint_path: Path,
    true_l1: float,
    true_l2: float,
    smooth: float,
) -> NonLinearNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = NonLinearNetwork(
        10,
        1,
        200,
        l1_lambda=true_l1,
        l2_lambda=true_l2,
        l1_smooth=smooth,
        lr=1e-3,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train_condition(
    *,
    seed: int,
    l1: float,
    l2: float,
    smooth: float,
    batch_size: int,
    epochs: int,
    target_parameters: Tensor,
    target_predictions: Tensor,
) -> tuple[NonLinearNetwork, SyntheticData, list[dict[str, float | int]]]:
    data = make_data(seed, batch_size)
    # make_data deliberately leaves the global RNG at the exact state preceding
    # model construction in the original experiment.
    model = NonLinearNetwork(
        10,
        1,
        200,
        l1_lambda=l1,
        l2_lambda=l2,
        l1_smooth=smooth,
        lr=1e-3,
    )
    trace = DynamicsTrace(
        data.train_x,
        data.train_y,
        target_parameters,
        target_predictions,
    )
    trainer = Trainer(
        max_epochs=epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        callbacks=[trace],
    )
    trainer.fit(
        model,
        train_dataloaders=data.train_loader,
        val_dataloaders=data.validation_loader,
    )
    return model, data, trace.records


def parse_checkpoint_epochs(path: Path) -> int:
    match = re.search(r"epoch=(\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot parse epoch from {path}")
    return int(match.group(1)) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--true-l1", type=float, required=True)
    parser.add_argument("--true-l2", type=float, required=True)
    parser.add_argument("--estimated-l1", type=float, required=True)
    parser.add_argument("--estimated-l2", type=float, required=True)
    parser.add_argument("--smooth", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    epochs = (
        args.epochs
        if args.epochs is not None
        else parse_checkpoint_epochs(args.target_checkpoint)
    )
    start = time.time()

    target_data = make_data(args.seed, args.batch_size)
    target_model = load_target_model(
        args.target_checkpoint,
        args.true_l1,
        args.true_l2,
        args.smooth,
    )
    target_metrics = endpoint_metrics(
        target_model,
        target_data.train_x,
        target_data.train_y,
        args.true_l1,
        args.true_l2,
        args.smooth,
    )
    target_parameters = target_metrics["_parameters"]
    target_predictions = target_metrics["_predictions"]

    specifications = {
        "true": (args.true_l1, args.true_l2),
        "estimated": (args.estimated_l1, args.estimated_l2),
        "ablated": (0.0, 0.0),
    }
    conditions: dict[str, Any] = {}
    private_metrics: dict[str, dict[str, Any]] = {"target": target_metrics}
    for name, (l1, l2) in specifications.items():
        LOGGER.info(
            "%s: training seed=%d for %d epochs with l1=%g l2=%g",
            name,
            args.seed,
            epochs,
            l1,
            l2,
        )
        model, data, trace = train_condition(
            seed=args.seed,
            l1=l1,
            l2=l2,
            smooth=args.smooth,
            batch_size=args.batch_size,
            epochs=epochs,
            target_parameters=target_parameters,
            target_predictions=target_predictions,
        )
        metrics = endpoint_metrics(
            model,
            data.train_x,
            data.train_y,
            l1,
            l2,
            args.smooth,
        )
        private_metrics[name] = metrics
        conditions[name] = {
            "l1": l1,
            "l2": l2,
            "endpoint": public_metrics(metrics),
            "to_target": compare_endpoints(metrics, target_metrics),
            "trajectory": trace,
        }

    pairwise = {
        "estimated_to_true": compare_endpoints(
            private_metrics["estimated"],
            private_metrics["true"],
        ),
        "ablated_to_true": compare_endpoints(
            private_metrics["ablated"],
            private_metrics["true"],
        ),
    }
    output = {
        "config": {
            "tag": args.tag,
            "seed": args.seed,
            "true_l1": args.true_l1,
            "true_l2": args.true_l2,
            "estimated_l1": args.estimated_l1,
            "estimated_l2": args.estimated_l2,
            "smooth": args.smooth,
            "batch_size": args.batch_size,
            "epochs": epochs,
            "target_checkpoint": str(args.target_checkpoint),
        },
        "target": public_metrics(target_metrics),
        "conditions": conditions,
        "pairwise": pairwise,
        "wall_seconds": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    LOGGER.info(
        "wrote %s; estimated target prediction relative RMSE=%.4g, "
        "ablated=%.4g",
        args.output,
        conditions["estimated"]["to_target"]["prediction_relative_rmse"],
        conditions["ablated"]["to_target"]["prediction_relative_rmse"],
    )


if __name__ == "__main__":
    main()
