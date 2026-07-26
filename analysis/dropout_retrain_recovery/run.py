"""Success mode 2 for dropout: estimate the implicit regularizer, then ablate
dropout and retrain with that regularizer added EXPLICITLY.

One SLURM task = one (condition, seed) training run + full endpoint measurement.

Protocols
---------
The closed-form coefficient ``s*`` measured at a trained endpoint is the *total*
effective regularization there -- it also absorbs the implicit bias of SGD and of
early stopping, which the dropout=0 reference has too.  It is therefore NOT the
marginal contribution of dropout.  (This differs from the early-stopping
demonstration in the paper, where the ablated reference is trained to convergence
and so has zero early-stopping regularization by construction.)  Both readings
are run and reported separately:

    protocol T ("total"):     s = s*(dropout=0.3)
    protocol M ("marginal"):  s = s*(dropout=0.3) - s*(dropout=0.0)

Measurement at the endpoint
---------------------------
``full_batch_loss_grad`` and ``closed_form`` are imported verbatim from
``analysis/closed_form_dropout_sweep/run.py`` (loaded by path so that file is not
touched), so the re-estimated coefficients are produced by exactly the same code
that produced the coefficients being injected.  The re-estimation always uses the
**pure cross-entropy** full-batch gradient, never the penalized objective, so
``s*`` retains its meaning of "total effective regularization at this endpoint"
and is comparable across conditions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

_REPO = Path(__file__).resolve().parents[2]
_SRC = str(_REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.data import MNISTLightningDataModule  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FAMILIES, PenalizedDeepReLUClassifier, unit_scale_penalty_module  # noqa: E402


def _load_sweep_helpers():
    """Import ``full_batch_loss_grad`` / ``closed_form`` from the sweep script.

    Loaded by file path rather than copied so the estimator used here is provably
    identical to the one that produced the injected coefficients, and so that
    ``analysis/closed_form_dropout_sweep/run.py`` is not modified.
    """
    path = _REPO / "analysis" / "closed_form_dropout_sweep" / "run.py"
    spec = importlib.util.spec_from_file_location("_cf_dropout_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.full_batch_loss_grad, module.closed_form


full_batch_loss_grad, closed_form = _load_sweep_helpers()


def spectral_summaries(model) -> Dict[str, Any]:
    """Per-weight-matrix spectral descriptors + whole-model aggregates."""
    per_layer = {}
    sranks, entropies = [], []
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        matrix = param.detach() if param.ndim == 2 else param.detach().reshape(param.shape[0], -1)
        sig = torch.linalg.svdvals(matrix.float())
        sig_sq = sig.pow(2)
        fro_sq = float(sig_sq.sum())
        s1 = float(sig[0])
        srank = fro_sq / max(s1**2, 1e-30)
        probs = (sig_sq / max(fro_sq, 1e-30)).clamp_min(1e-12)
        entropy = float(-(probs * probs.log()).sum())
        per_layer[name] = {
            "shape": list(matrix.shape),
            "fro_norm": math.sqrt(fro_sq),
            "spectral_norm": s1,
            "nuclear_norm": float(sig.sum()),
            "stable_rank": srank,
            "spectral_entropy": entropy,
            "spectral_gap_log": float(torch.log(sig[0] / (sig[1] + 1e-8) + 1e-8)) if sig.numel() > 1 else None,
        }
        sranks.append(srank)
        entropies.append(entropy)
    return {
        "per_layer": per_layer,
        "mean_stable_rank": sum(sranks) / len(sranks),
        "mean_spectral_entropy": sum(entropies) / len(entropies),
    }


def penalty_values(model) -> Dict[str, float]:
    """``P(theta)`` for every family at the endpoint (the estimator's own ``P``)."""
    device = next(model.parameters()).device
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    from core.estimators import vector_to_parameter_views

    structured = vector_to_parameter_views(flat, model)
    out = {}
    with torch.no_grad():
        for name in FAMILIES:
            module = unit_scale_penalty_module(name).to(device)
            out[name] = float(module(flat, structured))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="none")
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--penalty-family", type=str, default="none")
    ap.add_argument("--penalty-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--data-root", type=Path, default=_REPO / "data" / "raw")
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)

    dm = MNISTLightningDataModule(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    dm.prepare_data()
    dm.setup("fit")

    model = PenalizedDeepReLUClassifier(
        input_dim=784,
        num_classes=10,
        depth=args.depth,
        width=args.width,
        dropout=args.dropout,
        batchnorm=False,
        l2_lambda=0.0,
        lr=args.lr,
        momentum=args.momentum,
        penalty_family=args.penalty_family,
        penalty_coef=args.penalty_coef,
    )
    n_params_before = sum(p.numel() for p in model.parameters())

    # Each run gets its own root: a shared checkpoint dir / shared CSVLogger has
    # cost us runs before.  logger=False and enable_checkpointing=False on top.
    ckpt_dir = args.checkpoint_dir / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        default_root_dir=str(ckpt_dir),
        accelerator="auto",
        devices=1,
        callbacks=[EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)],
    )
    trainer.fit(model, datamodule=dm)
    train_seconds = time.time() - t0
    model.eval()
    test = trainer.test(model, datamodule=dm, verbose=False)

    # Endpoint measurement.  grad_l is the PURE cross-entropy full-batch gradient
    # (no explicit penalty), which is what makes s* comparable across conditions.
    grad_l = full_batch_loss_grad(model, dm.train_dataloader())
    theta = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    assert theta.numel() == n_params_before, "parameter vector changed size"

    out = {
        "config": {
            "tag": args.tag,
            "condition": args.condition,
            "protocol": args.protocol,
            "dropout": args.dropout,
            "penalty_family": args.penalty_family,
            "penalty_coef": args.penalty_coef,
            "seed": args.seed,
            "depth": args.depth,
            "width": args.width,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "param_count": int(theta.numel()),
        },
        "stationarity": {
            "full_batch_grad_norm": float(grad_l.norm()),
            "param_norm": float(theta.norm()),
            "relative_grad_norm": float(grad_l.norm() / theta.norm()),
            "epochs_run": int(trainer.current_epoch),
            "train_seconds": train_seconds,
        },
        "test": {k: float(v) for k, v in (test[0] if test else {}).items()},
        "penalty_values": penalty_values(model),
        "spectral": spectral_summaries(model),
        "closed_form": {name: closed_form(name, model, grad_l) for name in FAMILIES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "tag": args.tag,
                "test_acc": out["test"].get("test/acc"),
                "param_norm": out["stationarity"]["param_norm"],
                "mean_stable_rank": out["spectral"]["mean_stable_rank"],
                "ridge_s_star": out["closed_form"]["ridge"]["scale_star"],
                "stable_rank_s_star": out["closed_form"]["stable_rank"]["scale_star"],
                "epochs": out["stationarity"]["epochs_run"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
