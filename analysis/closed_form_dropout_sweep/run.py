"""Re-estimate the dropout implicit-regularization trend using the EXACT optimum.

Motivation: the iterative gradient-matching fit used previously (Adam on
beta with scale = exp(beta), mse_loss reduction="mean") does not converge -- it
overshoots the true least-squares optimum by 38-61x on this architecture, because
d(loss)/d(beta) underflows Adam's eps before the optimum is reached.  See
analysis/verify_estimator_convergence/.

Every family here is R(theta) = s * P(theta) with a single scalar s, so gradient
matching is one-dimensional least squares with an exact solution against the
full-batch (deterministic) loss gradient:

    s* = <grad P, -grad L> / ||grad P||^2

This script trains one model per (dropout, seed) and reports s* for every family,
plus scale-free adequacy (cosine, residual_ratio) -- no iterative fitting, so the
estimate cannot be optimizer-limited.  Purpose: test whether the reported
"lambda increases with dropout rate" trend survives a correct estimator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

sys.path.insert(0, "src")

from core.bias import (
    RidgeBias,
    NuclearNormBias,
    StableRankBias,
    SpectralEntropyBias,
    SpectralGapBias,
)
from core.data import MNISTLightningDataModule
from core.estimators import vector_to_parameter_views
from core.models import DeepReLUClassifier

FAMILIES = {
    "ridge": RidgeBias,
    "nuclear_norm": NuclearNormBias,
    "stable_rank": StableRankBias,
    "spectral_entropy": SpectralEntropyBias,
    "spectral_gap": SpectralGapBias,
}


def full_batch_loss_grad(model, loader):
    """Deterministic mean cross-entropy gradient over the whole loader."""
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        F.cross_entropy(model(x), y, reduction="sum").backward()
        total += y.numel()
    grad = torch.cat(
        [
            (p.grad.detach().reshape(-1) / total)
            if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in model.parameters()
        ]
    )
    model.zero_grad(set_to_none=True)
    return grad


def closed_form(family_name, model, loss_grad):
    """Exact 1-D least-squares optimum for R = s * P(theta), plus adequacy."""
    fam = FAMILIES[family_name](enforce_positive=True, init_value=0.0).to(loss_grad.device)
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach().requires_grad_()
    structured = vector_to_parameter_views(flat, model)
    # init_value=0.0 -> scale = exp(0) = 1, so this gradient is exactly grad P.
    (grad_p,) = torch.autograd.grad(fam(flat, structured), flat)
    with torch.no_grad():
        target = -loss_grad
        denom = float(grad_p.pow(2).sum())
        if denom == 0:
            return {"scale_star": None}
        s = float(grad_p @ target) / denom
        tn = float(target.norm())
        return {
            "scale_star": s,
            "scale_star_clamped": max(s, 0.0),
            "residual_ratio": float((s * grad_p - target).norm()) / tn,
            "residual_ratio_clamped": float((max(s, 0.0) * grad_p - target).norm()) / tn,
            "grad_cosine": float(F.cosine_similarity(grad_p, target, dim=0)),
            "grad_p_norm": denom**0.5,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batchnorm", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dm = MNISTLightningDataModule(root=args.data_root, batch_size=2048, num_workers=2)
    dm.prepare_data()
    dm.setup("fit")

    model = DeepReLUClassifier(
        input_dim=784, num_classes=10, depth=args.depth, width=args.width,
        dropout=args.dropout, batchnorm=args.batchnorm, l2_lambda=0.0,
        lr=0.05, momentum=0.9,
    )
    trainer = Trainer(
        max_epochs=args.max_epochs, logger=False, enable_progress_bar=False,
        enable_checkpointing=False, accelerator="auto", devices=1,
        callbacks=[EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)],
    )
    trainer.fit(model, datamodule=dm)
    model.eval()
    test = trainer.test(model, datamodule=dm, verbose=False)

    grad_l = full_batch_loss_grad(model, dm.train_dataloader())
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    out = {
        "config": {
            "dropout": args.dropout, "seed": args.seed, "depth": args.depth,
            "width": args.width, "batchnorm": args.batchnorm,
            "param_count": int(theta.numel()),
        },
        "stationarity": {
            "full_batch_grad_norm": float(grad_l.norm()),
            "param_norm": float(theta.norm()),
            "relative_grad_norm": float(grad_l.norm() / theta.norm()),
            "epochs_run": int(trainer.current_epoch),
        },
        "test": {k: float(v) for k, v in (test[0] if test else {}).items()},
        "closed_form": {name: closed_form(name, model, grad_l) for name in FAMILIES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["closed_form"]["ridge"], indent=2))


if __name__ == "__main__":
    main()
