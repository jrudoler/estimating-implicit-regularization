"""Does the iterative gradient-matching fit actually reach its least-squares optimum?

For every ScalarBias family the regularizer is R = s * P(theta) with a single scalar
s = exp(beta), so gradient matching is a one-dimensional least-squares problem with a
closed-form solution.  For ridge, P = sum(theta^2), grad R = 2 s theta, and minimizing
||grad R + grad L||^2 over s gives

    s* = -<theta, grad L> / (2 ||theta||^2)

evaluated at the FULL-BATCH grad L (the estimator's minibatch target has this as its
mean).  This script trains the exact A1(c) configuration, then compares:

  (a) s from the iterative fit used by analysis/dropout_bias_estimation/run.py
      (BiasWithCrossEntropyScheduled, mse_loss reduction="mean", Adam, bias_lr=0.05)
  (b) s* from the closed form

If (a) != (b) the reported coefficients are optimizer artifacts rather than estimates.
Also reports residual_ratio computed against the full-batch gradient (the meaningful
version) alongside the per-minibatch value the estimator logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

import sys

sys.path.insert(0, "src")

from core.bias import RidgeBias
from core.data import MNISTLightningDataModule
from core.estimators import BiasWithCrossEntropyScheduled
from core.models import DeepReLUClassifier


def full_batch_loss_grad(model: torch.nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic mean cross-entropy gradient over the whole loader."""
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # sum reduction so unequal final batches combine into the exact mean
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss.backward()
        total += y.numel()
    grads = torch.cat(
        [
            (p.grad.detach().reshape(-1) / total)
            if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in model.parameters()
        ]
    )
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    return theta, grads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=("baseline", "dropout", "batchnorm"), required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--bias-max-epochs", type=int, default=2000)
    ap.add_argument("--bias-patience", type=int, default=100)
    ap.add_argument("--bias-lr", type=float, default=0.05)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dropout = 0.3 if args.condition == "dropout" else 0.0
    batchnorm = args.condition == "batchnorm"

    dm = MNISTLightningDataModule(root=args.data_root, batch_size=2048, num_workers=2)
    dm.prepare_data()
    dm.setup("fit")

    model = DeepReLUClassifier(
        input_dim=784, num_classes=10, depth=3, width=256,
        dropout=dropout, batchnorm=batchnorm, l2_lambda=0.0, lr=0.05, momentum=0.9,
    )
    Trainer(
        max_epochs=args.max_epochs, logger=False, enable_progress_bar=False,
        enable_checkpointing=False, accelerator="auto", devices=1,
        callbacks=[EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)],
    ).fit(model, datamodule=dm)
    model.eval()

    # ---- closed form on the full-batch gradient -------------------------------
    loader = dm.train_dataloader()
    theta, grad_l = full_batch_loss_grad(model, loader)
    theta_sq = float(theta.dot(theta))
    theta_dot_g = float(theta.dot(grad_l))
    s_star = -theta_dot_g / (2.0 * theta_sq)
    grad_l_norm = float(grad_l.norm())
    cos_full = -theta_dot_g / ((theta_sq**0.5) * grad_l_norm)

    def residual_at(s: float) -> float:
        return float((2.0 * s * theta + grad_l).norm()) / grad_l_norm

    # ---- iterative fit, exactly as the A1(c) runner does it -------------------
    bias = RidgeBias(enforce_positive=True)
    beta_init = float(bias.beta.detach().reshape(-1)[0])
    estimator = BiasWithCrossEntropyScheduled(
        predictive_model=model.eval(), bias_model=bias,
        grad_match_loss_fn=F.mse_loss, lr=args.bias_lr,
        optimizer_cls=torch.optim.Adam, bias_lr=args.bias_lr,
    )
    bias_trainer = Trainer(
        max_epochs=args.bias_max_epochs, logger=False, enable_progress_bar=False,
        enable_checkpointing=False, accelerator="auto", devices=1,
        callbacks=[EarlyStopping(monitor="train_bias/loss", mode="min", patience=args.bias_patience)],
    )
    bias_trainer.fit(estimator, train_dataloaders=dm.train_dataloader())
    s_iter = float(bias.get_bias_params()["scale"])
    logged = {k: float(v) for k, v in bias_trainer.callback_metrics.items()}

    out = {
        "condition": args.condition,
        "seed": args.seed,
        "scale_init": float(torch.exp(torch.tensor(beta_init))),
        "scale_iterative": s_iter,
        "scale_closed_form": s_star,
        "ratio_iterative_over_closed_form": (s_iter / s_star) if s_star != 0 else None,
        "full_batch": {
            "theta_sq_norm": theta_sq,
            "theta_dot_grad": theta_dot_g,
            "grad_l_norm": grad_l_norm,
            "cosine_full_batch": cos_full,
            "residual_ratio_at_closed_form": residual_at(s_star),
            "residual_ratio_at_iterative": residual_at(s_iter),
            "residual_ratio_at_zero": residual_at(0.0),
        },
        "estimator_logged_minibatch": logged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
