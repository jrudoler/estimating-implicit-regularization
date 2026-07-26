#!/usr/bin/env python3
"""Rerun one Figure-5 model and retain exact Lightning trajectory checkpoints.

The original corrected Figure-5 grid disabled checkpointing. This experiment
replays the same Lightning model/data/optimizer path to the run's recorded
endpoint T, saves full checkpoints at 0.5T, 0.8T, and T, and evaluates the exact
closed-form regularizer coefficients at each saved state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.data import MNISTLightningDataModule  # noqa: E402
from core.models import DeepReLUClassifier  # noqa: E402


LOGGER = logging.getLogger(__name__)
FRACTIONS = (0.5, 0.8, 1.0)


def load_closed_form_module() -> ModuleType:
    """Load the exact estimator used by the corrected Figure-5 grid."""
    path = REPO_ROOT / "analysis" / "closed_form_dropout_sweep" / "run.py"
    spec = importlib.util.spec_from_file_location("closed_form_dropout_sweep_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import closed-form estimator from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FractionalCheckpoint(Callback):
    """Save full Lightning checkpoints after selected completed epochs."""

    def __init__(self, targets: dict[int, float], checkpoint_dir: Path) -> None:
        super().__init__()
        self.targets = targets
        self.checkpoint_dir = checkpoint_dir
        self.saved: dict[int, Path] = {}

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch not in self.targets or completed_epoch in self.saved:
            return
        fraction = self.targets[completed_epoch]
        path = self.checkpoint_dir / (
            f"fraction_{fraction:.1f}_epoch_{completed_epoch:04d}.ckpt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(path)
        self.saved[completed_epoch] = path
        LOGGER.info(
            "Saved fraction %.1f checkpoint after epoch %d to %s",
            fraction,
            completed_epoch,
            path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--endpoint-epochs", type=int, required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def target_epochs(endpoint: int) -> dict[int, float]:
    """Map rounded completed epochs to their requested trajectory fractions."""
    targets: dict[int, float] = {}
    for fraction in FRACTIONS:
        epoch = max(1, min(endpoint, int(round(fraction * endpoint))))
        targets[epoch] = fraction
    if len(targets) != len(FRACTIONS):
        raise ValueError(f"Endpoint {endpoint} is too short for distinct checkpoints.")
    return targets


def evaluate_checkpoint(
    checkpoint: Path,
    datamodule: MNISTLightningDataModule,
    closed_form_module: ModuleType,
) -> dict[str, Any]:
    """Evaluate stationarity and all scalar families without changing training."""
    model = DeepReLUClassifier.load_from_checkpoint(str(checkpoint), map_location="cpu")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loss_gradient = closed_form_module.full_batch_loss_grad(
        model,
        datamodule.train_dataloader(),
    )
    parameters = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    stationarity = {
        "full_batch_grad_norm": float(loss_gradient.norm()),
        "param_norm": float(parameters.norm()),
        "relative_grad_norm": float(loss_gradient.norm() / parameters.norm()),
    }
    closed_form = {
        name: closed_form_module.closed_form(name, model, loss_gradient)
        for name in closed_form_module.FAMILIES
    }
    LOGGER.info(
        "Evaluated %s on %s: ridge lambda=%.6g, rel_grad=%.6g",
        checkpoint.name,
        accelerator,
        closed_form["ridge"]["scale_star"],
        stationarity["relative_grad_norm"],
    )
    return {"stationarity": stationarity, "closed_form": closed_form}


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.endpoint_epochs <= 0:
        raise ValueError("--endpoint-epochs must be positive.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datamodule = MNISTLightningDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = DeepReLUClassifier(
        input_dim=784,
        num_classes=10,
        depth=args.depth,
        width=args.width,
        dropout=args.dropout,
        batchnorm=False,
        l2_lambda=0.0,
        lr=0.05,
        momentum=0.9,
    )
    targets = target_epochs(args.endpoint_epochs)
    checkpoint_callback = FractionalCheckpoint(targets, args.checkpoint_dir)
    trainer = Trainer(
        max_epochs=args.endpoint_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)

    missing = sorted(set(targets) - set(checkpoint_callback.saved))
    if missing:
        raise RuntimeError(f"Training ended without saving target epochs: {missing}.")

    closed_form_module = load_closed_form_module()
    records: list[dict[str, Any]] = []
    for epoch, fraction in sorted(targets.items(), key=lambda item: item[1]):
        checkpoint = checkpoint_callback.saved[epoch]
        evaluation = evaluate_checkpoint(checkpoint, datamodule, closed_form_module)
        records.append(
            {
                "fraction": fraction,
                "epoch": epoch,
                "checkpoint": str(checkpoint.resolve().relative_to(REPO_ROOT)),
                **evaluation,
            }
        )

    payload = {
        "config": {
            "dropout": args.dropout,
            "seed": args.seed,
            "depth": args.depth,
            "width": args.width,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "endpoint_epochs": args.endpoint_epochs,
            "fractions": list(FRACTIONS),
            "training_backend": "Lightning Trainer + MNISTLightningDataModule",
        },
        "trajectory": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
