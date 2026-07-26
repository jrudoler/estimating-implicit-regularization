"""One (condition, seed) run of the activation-quadratic retrain-recovery test.

The question: ``analysis/dropout_retrain_recovery`` ablated dropout and retrained
with an injected *global ridge* / *stable rank* penalty and failed to recover the
dropout endpoint -- but ``analysis/dropout_structured_regularizers`` shows both of
those are poor fits to that endpoint's gradient, while the per-layer
activation-weighted quadratic fits substantially better.  This script injects that
better-fitting family and retrains, so the two experiments can be compared row by
row.

Conditions are driven entirely by the command line, so the same script produces:

* ``target``   -- dropout 0.3, no penalty (the endpoint to be reinstated);
* ``ablated``  -- dropout 0.0, no penalty (the "do nothing" null);
* ``actquad_*``-- dropout 0.0 plus a coefficient vector taken from a per-seed
  injection spec written by ``build_specs.py`` from the ``target``/``ablated`` runs.

Measurement at the endpoint reuses the upstream code verbatim (see ``reuse.py``):
the scalar closed-form ``s*`` for all five families from the retrain harness, and
the exact-NNLS fit of all four structured families from the structured-regularizer
harness.  Both are computed from the **pure cross-entropy** full-batch gradient,
never the penalized objective, so every coefficient keeps its meaning of "total
effective regularization at this endpoint" and stays comparable across conditions.

The run also saves the endpoint itself (parameter vector, test-set softmax, the
full-batch gradient, and the activation moments) so that function-space and
parameter-space distances to the seed-matched ``target`` can be computed
afterwards in ``analyze.py`` without re-running any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reuse import (  # noqa: E402
    REPO,
    activation_quadratic_bases,
    closed_form,
    combine_summaries,
    exact_nnls,
    fit_all_candidates,
    full_batch_loss_grad,
    linear_layers,
    loss_gradient_and_moments,
    make_split_loaders,
    penalty_values,
    SCALAR_FAMILIES,
    spectral_summaries,
)

from core.data import MNISTLightningDataModule  # noqa: E402
from penalty import (  # noqa: E402
    ActivationQuadraticPenalty,
    ActQuadPenalizedClassifier,
    MomentRefresh,
    moment_summary,
    moments_by_name,
    ordinal_moments,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="none")
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="per-seed injection spec from build_specs.py; omit for target/ablated",
    )
    ap.add_argument(
        "--coef-key",
        type=str,
        default="coef_total",
        help="which coefficient vector in the spec to inject",
    )
    ap.add_argument(
        "--refresh-moments-every",
        type=int,
        default=0,
        help="0 = fixed-moment variant (primary); N>0 re-measures E[h^2] every N epochs",
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument(
        "--fixed-epochs",
        type=int,
        default=0,
        help="matched-budget arm: train exactly N epochs with early stopping disabled",
    )
    ap.add_argument("--split-seed", type=int, default=2027)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    return ap.parse_args()


def load_injection(spec_path: Path, coef_key: str) -> Dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    moments = torch.load(spec_path.with_name(spec["moment_file"]), weights_only=True)
    if coef_key not in spec:
        raise KeyError(f"{spec_path} has no coefficient vector named {coef_key!r}")
    coefficients = [float(c) for c in spec[coef_key]]
    if len(coefficients) != len(moments):
        raise ValueError("spec coefficient/moment lengths disagree")
    return {"spec": spec, "coefficients": coefficients, "moments": moments}


def verify_penalty_gradient(model, coefficients: List[float]) -> Dict[str, float]:
    """Check ``grad P`` equals the upstream activation-quadratic basis combination.

    The penalty is implemented here but the basis it must reproduce lives in the
    structured-regularizer harness.  Asserting agreement once, before training,
    catches a wrong coefficient/moment ordering or a layer-index mismatch for the
    price of one backward pass.
    """
    frozen = [model.act_quad.moment(i) for i in range(model.act_quad.n_layers)]
    parameters = list(model.parameters())
    penalty = model.act_quad_penalty()
    grads = torch.autograd.grad(penalty, parameters, allow_unused=True)
    mine = torch.cat(
        [
            (g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(parameters, grads)
        ]
    )
    bases = activation_quadratic_bases(model, moments_by_name(model, frozen))
    reference = torch.zeros_like(mine)
    for coefficient, basis in zip(coefficients, bases.values()):
        reference = reference + coefficient * basis
    model.zero_grad(set_to_none=True)
    denominator = float(reference.norm())
    return {
        "penalty_value": float(penalty.detach()),
        "grad_norm": denominator,
        "relative_error": float((mine - reference).norm()) / max(denominator, 1e-30),
    }


@torch.no_grad()
def test_predictions(model, loader) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    probabilities, labels = [], []
    model.eval()
    for features, targets in loader:
        logits = model(features.to(device))
        probabilities.append(F.softmax(logits.float(), dim=1).cpu())
        labels.append(targets.cpu())
    return {
        "test_probs": torch.cat(probabilities),
        "test_labels": torch.cat(labels),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)

    dm = MNISTLightningDataModule(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    injection = load_injection(args.spec, args.coef_key) if args.spec else None
    act_quad = (
        ActivationQuadraticPenalty(injection["coefficients"], injection["moments"])
        if injection
        else None
    )

    model = ActQuadPenalizedClassifier(
        input_dim=784,
        num_classes=10,
        depth=args.depth,
        width=args.width,
        dropout=args.dropout,
        batchnorm=False,
        l2_lambda=0.0,
        lr=args.lr,
        momentum=args.momentum,
        penalty_family="none",
        penalty_coef=0.0,
        act_quad=act_quad,
    )
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    n_params = initial.numel()
    initial_digest = hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest()[:16]

    check = verify_penalty_gradient(model, injection["coefficients"]) if injection else None
    if check is not None and check["relative_error"] > 1e-5:
        raise AssertionError(f"penalty gradient mismatch: {check}")

    callbacks: List[Any] = []
    if args.fixed_epochs:
        max_epochs = args.fixed_epochs
    else:
        max_epochs = args.max_epochs
        callbacks.append(
            EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)
        )
    refresher = None
    if args.refresh_moments_every and act_quad is not None:
        refresher = MomentRefresh(dm.train_dataloader, args.refresh_moments_every)
        callbacks.append(refresher)

    checkpoint_dir = args.checkpoint_dir / args.tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        default_root_dir=str(checkpoint_dir),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)
    train_seconds = time.time() - started
    model.eval()
    test = trainer.test(model, datamodule=dm, verbose=False)

    # Endpoint measurement.  Both gradients below are pure cross-entropy: the
    # first reproduces the retrain harness's estimand, the second (split into
    # halves) reproduces the structured-regularizer harness's estimand.
    grad_l = full_batch_loss_grad(model, dm.train_dataloader())
    loader_a, loader_b = make_split_loaders(
        dm._train_dataset, args.batch_size, args.num_workers, args.split_seed
    )
    summary_a = loss_gradient_and_moments(model, loader_a)
    summary_b = loss_gradient_and_moments(model, loader_b)
    summary_full = combine_summaries(summary_a, summary_b)
    structured_fits = fit_all_candidates(model, summary_full, summary_a, summary_b)

    endpoint_moments = ordinal_moments(model, summary_full.moments)
    layer_names = [name for name, _module in linear_layers(model)]
    theta = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    if theta.numel() != n_params:
        raise AssertionError("parameter vector changed size during training")

    frozen_refit: Optional[Dict[str, Any]] = None
    if injection is not None:
        # With the injected quadratic form held fixed, which coefficient vector
        # does the retrained endpoint imply?  This is the direct analogue of the
        # scalar experiment's re-estimated s*, in the injected basis.
        frozen_refit = exact_nnls(
            -summary_full.gradient,
            activation_quadratic_bases(
                model, moments_by_name(model, injection["moments"])
            ),
        )

    record: Dict[str, Any] = {
        "config": {
            "tag": args.tag,
            "condition": args.condition,
            "protocol": args.protocol,
            "dropout": args.dropout,
            "penalty_family": "activation_quadratic" if injection else "none",
            "coef_key": args.coef_key if injection else None,
            "injected_coefficients": injection["coefficients"] if injection else None,
            "refresh_moments_every": args.refresh_moments_every,
            "moment_refreshes": refresher.refreshes if refresher else 0,
            "fixed_epochs": args.fixed_epochs,
            "early_stopping": not bool(args.fixed_epochs),
            "seed": args.seed,
            "depth": args.depth,
            "width": args.width,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "max_epochs": max_epochs,
            "patience": args.patience,
            "split_seed": args.split_seed,
            "param_count": int(theta.numel()),
            "layer_names": layer_names,
            "init_param_norm": float(initial.norm()),
            "init_param_sha256_16": initial_digest,
        },
        "penalty_gradient_check": check,
        "stationarity": {
            "full_batch_grad_norm": float(grad_l.norm()),
            "param_norm": float(theta.norm()),
            "relative_grad_norm": float(grad_l.norm() / theta.norm()),
            "epochs_run": int(trainer.current_epoch),
            "train_seconds": train_seconds,
            "split_gradient_norm": float(summary_full.gradient.norm()),
            "train_mean_loss": summary_full.mean_loss,
        },
        "test": {k: float(v) for k, v in (test[0] if test else {}).items()},
        "penalty_values": penalty_values(model),
        "spectral": spectral_summaries(model),
        "closed_form": {name: closed_form(name, model, grad_l) for name in SCALAR_FAMILIES},
        "structured": structured_fits,
        "moment_summary": moment_summary(endpoint_moments),
        "frozen_form_refit": frozen_refit,
    }
    if injection is not None:
        record["injected_penalty_value"] = float(model.act_quad_penalty().detach())
        record["injected_spec"] = {
            "path": str(args.spec),
            "coef_key": args.coef_key,
            "source_moment_summary": moment_summary(injection["moments"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tag": args.tag,
            "condition": args.condition,
            "seed": args.seed,
            "param_vector": theta.cpu(),
            "param_names": [name for name, _p in model.named_parameters()],
            "param_shapes": [list(p.shape) for _n, p in model.named_parameters()],
            "grad_vector": summary_full.gradient.detach().cpu(),
            "moments": [m.detach().cpu() for m in endpoint_moments],
            "layer_names": layer_names,
            **{k: v for k, v in test_predictions(model, dm.test_dataloader()).items()},
        },
        args.artifact,
    )

    print(
        json.dumps(
            {
                "tag": args.tag,
                "test_acc": record["test"].get("test/acc"),
                "param_norm": record["stationarity"]["param_norm"],
                "epochs": record["stationarity"]["epochs_run"],
                "actquad_coefficients": structured_fits["activation_quadratic"]["full"][
                    "coefficients"
                ],
                "actquad_projection_r2": structured_fits["activation_quadratic"]["full"][
                    "projection_r2"
                ],
                "ridge_s_star": record["closed_form"]["ridge"]["scale_star"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
