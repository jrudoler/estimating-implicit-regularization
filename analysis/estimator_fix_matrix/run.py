"""Can the iterative gradient-matching fit be FIXED to agree with the closed form?

The closed form is the exact minimizer of the very objective the iterative fit descends,
so if the iterative fit is merely mis-numericked, some set of numerical fixes must make it
converge to that value from a cold start. This script tests that directly.

Diagnosis being tested. RidgeBias uses s = exp(beta) with beta ~ N(0, 0.1^2), so s starts
near 1.0 while the optimum is ~4e-5. The objective with mse_loss(reduction="mean") is
   Loss(s) = (1/p) * sum_i (2*s*theta_i + g_i)^2 ,
   dLoss/dbeta = (4*s/p) * [2*s*||theta||^2 + <theta,g>] .
The 1/p factor (p ~ 3.4e5) drives this below Adam's eps=1e-8 at s ~ 9e-4, so descent
stalls two orders of magnitude above the optimum. A second effect compounds it: the
s-dependent signal rides on the constant ||g||^2, and its relative size ~4s|<theta,g>|/||g||^2
shrinks as dropout inflates ||g||, so the stall happens earlier at higher dropout.

Fixes tested (each isolates one mechanism):
  sum reduction   -- removes the 1/p factor, lifting the gradient p-fold above eps
  adam eps 1e-16  -- removes the eps floor directly
  float64         -- removes float32 cancellation of the s-signal against ||g||^2
  direct s        -- drops the exp(beta) reparametrization (uniform step size in s)
  full batch      -- deterministic gradient, matching the closed form's target exactly
  warm start      -- init at the closed form; checks it is actually a fixed point
If the diagnosis is right, the combined fix reproduces the closed form and the baseline
reproduces the published lambda_hat (~1e-3).
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

from core.data import MNISTLightningDataModule
from core.models import DeepReLUClassifier


def flat_grad(model, x, y, dtype):
    """grad of mean cross-entropy on one batch, flattened."""
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    g = torch.cat([
        (p.grad.detach().reshape(-1) if p.grad is not None
         else torch.zeros(p.numel(), device=x.device))
        for p in model.parameters()
    ]).to(dtype)
    model.zero_grad(set_to_none=True)
    return g


def full_batch_grad(model, loader, dtype):
    model.zero_grad(set_to_none=True)
    dev = next(model.parameters()).device
    n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        F.cross_entropy(model(x), y, reduction="sum").backward()
        n += y.numel()
    g = torch.cat([
        (p.grad.detach().reshape(-1) / n if p.grad is not None
         else torch.zeros(p.numel(), device=dev))
        for p in model.parameters()
    ]).to(dtype)
    model.zero_grad(set_to_none=True)
    return g


def fit(model, loader, theta, s_star, *, reduction, dtype, adam_eps, lr, epochs,
        direct_s, full_batch, warm_start, fb_grad):
    """One iterative fit variant. Returns final s and its trajectory."""
    th = theta.to(dtype)
    init = s_star if warm_start else float(torch.exp(torch.randn(1) * 0.1))
    if direct_s:
        par = torch.nn.Parameter(torch.tensor(float(init), dtype=dtype, device=th.device))

        def get_s():
            return par

    else:
        par = torch.nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init))),
                                              dtype=dtype, device=th.device))

        def get_s():
            return torch.exp(par)

    opt = torch.optim.Adam([par], lr=lr, eps=adam_eps)
    traj, step = [], 0
    for _ in range(epochs):
        batches = [(None, None)] if full_batch else loader
        for batch in batches:
            g = fb_grad.to(dtype) if full_batch else flat_grad(
                model, batch[0].to(th.device), batch[1].to(th.device), dtype)
            grad_r = 2.0 * get_s() * th
            loss = F.mse_loss(grad_r, -g, reduction=reduction)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 25 == 0:
                traj.append(float(get_s().detach()))
            step += 1
    return float(get_s().detach()), traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--fit-epochs", type=int, default=60)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dm = MNISTLightningDataModule(root=args.data_root, batch_size=2048, num_workers=2)
    dm.prepare_data()
    dm.setup("fit")
    model = DeepReLUClassifier(input_dim=784, num_classes=10, depth=3, width=256,
                               dropout=args.dropout, batchnorm=False, l2_lambda=0.0,
                               lr=0.05, momentum=0.9)
    Trainer(max_epochs=args.epochs, logger=False, enable_progress_bar=False,
            enable_checkpointing=False, accelerator="auto", devices=1,
            callbacks=[EarlyStopping(monitor="val/loss", mode="min", patience=50)]
            ).fit(model, datamodule=dm)
    model.eval()

    loader = dm.train_dataloader()
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gfb = full_batch_grad(model, loader, torch.float64)
    th64 = theta.to(torch.float64)
    s_star = float(-(th64 @ gfb) / (2.0 * (th64 @ th64)))
    p = theta.numel()

    base = dict(reduction="mean", dtype=torch.float32, adam_eps=1e-8, lr=0.05,
                epochs=args.fit_epochs, direct_s=False, full_batch=False,
                warm_start=False, fb_grad=gfb)
    variants = {
        "baseline (as published)":      dict(),
        "+ sum reduction":             dict(reduction="sum"),
        "+ adam eps=1e-16":            dict(adam_eps=1e-16),
        "+ float64":                   dict(dtype=torch.float64),
        "+ direct s (no exp)":         dict(direct_s=True, lr=1e-5),
        "+ full-batch grad":           dict(full_batch=True),
        "ALL FIXES (sum+fp64+eps16)":  dict(reduction="sum", dtype=torch.float64,
                                            adam_eps=1e-16),
        "ALL FIXES + full-batch":      dict(reduction="sum", dtype=torch.float64,
                                            adam_eps=1e-16, full_batch=True),
        "warm start at s* (baseline)": dict(warm_start=True),
    }
    out = {"config": {"dropout": args.dropout, "seed": args.seed, "param_count": p,
                      "theta_norm": float(theta.norm()),
                      "full_batch_grad_norm": float(gfb.norm())},
           "closed_form_s_star": s_star, "variants": {}}
    for name, over in variants.items():
        kw = {**base, **over}
        torch.manual_seed(args.seed)  # same cold-start init across variants
        s, traj = fit(model, loader, theta, s_star, **kw)
        out["variants"][name] = {"s_final": s, "ratio_to_closed_form": s / s_star,
                                 "trajectory": traj[::4]}
        print(f"{name:32s} s={s:.5e}  ratio={s/s_star:8.2f}x")
    print(f"{'CLOSED FORM (target)':32s} s={s_star:.5e}  ratio=    1.00x")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
