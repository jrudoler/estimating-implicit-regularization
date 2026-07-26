#!/usr/bin/env python3
"""Experiment A3: measure the effective-regularizer profile of an MLP vs a CNN.

One run = (architecture, dataset, candidate regularizer family, seed).  The run
trains a predictive model to (near-)stationarity, then fits a *single* candidate
regularizer family by gradient matching and records the fitted coefficient
together with scale-free goodness-of-fit diagnostics.

Scope of the claim
------------------
An architecture *class* is not an ablatable module: there is no "train the same
model without being a CNN" reference condition, so nothing here is a validation
of the estimator.  This is a descriptive measurement that uses the estimator as
an instrument.  The reportable object is the per-family
(fitted coefficient, goodness-of-fit) profile for each architecture, compared
across architectures on the same task.

Caveats worth carrying into any writeup
---------------------------------------
* Exactly one family is fitted per run.  Several families are *not* fitted
  jointly because their gradients are strongly collinear at a trained optimum,
  which makes the coefficients non-identifiable (a joint fit redistributes
  weight between families arbitrarily).  The profile is therefore a set of
  independent single-family fits, and the coefficients are not additive.
* The matrix/spectral families in ``core.bias`` iterate the structured
  parameters and act on every tensor with ``ndim >= 2``, reshaping a conv kernel
  ``(out, in, kh, kw)`` to ``(out, in * kh * kw)``.  That is a valid but
  non-canonical matricization of a convolution operator (it is not the doubly
  block-circulant matrix the conv actually applies), so spectral quantities are
  comparable across runs but should not be read as spectral properties of the
  convolution operator itself.
* Gradient matching assumes the predictive model sits at an optimum of its
  training objective.  The run therefore records the endpoint training loss and
  the norm of the (near) full-batch loss gradient so runs that were not close to
  stationarity can be flagged rather than silently averaged in.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from core.bias import (
    JointBias,
    NuclearNormBias,
    RidgeBias,
    SpectralEntropyBias,
    SpectralGapBias,
    StableRankBias,
)
from core.data import CIFAR10LightningDataModule, MNISTLightningDataModule
from core.estimators import BiasWithCrossEntropyScheduled
from core.models import DeepReLUClassifier, SmallCNN

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

ARCHITECTURES = ("mlp", "cnn")
DATASETS = ("mnist", "cifar10")
FAMILIES = (
    "ridge",
    "nuclear_norm",
    "stable_rank",
    "spectral_entropy",
    "spectral_gap",
)

FAMILY_FACTORIES = {
    "ridge": RidgeBias,
    "nuclear_norm": NuclearNormBias,
    "stable_rank": StableRankBias,
    "spectral_entropy": SpectralEntropyBias,
    "spectral_gap": SpectralGapBias,
}

# Dataset geometry: (input_dim for the MLP, in_channels, image_size, num_classes)
DATASET_SHAPES = {
    "mnist": (28 * 28, 1, 28, 10),
    "cifar10": (3 * 32 * 32, 3, 32, 10),
}

ADEQUACY_KEYS = (
    "train_bias/loss",
    "train_bias/residual_ratio",
    "train_bias/grad_cosine",
    "train_bias/projection_r2",
    "train_bias/loss_grad_norm",
)


def _sum_squared_error(
    prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Squared error SUMMED over parameters, ignoring the requested reduction.

    Rescaling the gradient-matching objective by a constant (the parameter count)
    leaves the minimizing coefficient unchanged but fixes an optimizer pathology:
    with ``reduction="mean"`` over ~3e5 parameters, d(loss)/d(beta) drops below
    Adam's ``eps`` (1e-8) at coefficients around 1e-3, i.e. *above* the actual
    optimum (~1e-4 for ridge here).  Adam's step size then collapses and the fit
    freezes at an initialization-dependent value instead of converging.  Measured
    on a trained MNIST MLP: dL/dbeta = 9.8e-7 at scale 1e-2, 8.7e-9 at 1e-3,
    1.8e-11 at 1e-4 with mean reduction, versus 3.3e-1, 2.9e-3, 6.1e-6 summed.

    The estimator calls ``grad_match_loss_fn(..., reduction="mean")`` explicitly,
    hence the ignored keyword.  Note this puts ``train_bias/loss`` on a different
    scale from analyses that use plain ``mse_loss``; the adequacy diagnostics
    (residual_ratio, grad_cosine, projection_r2) are computed from gradients and
    are unaffected.
    """
    del reduction
    return F.mse_loss(prediction, target, reduction="sum")


GRAD_MATCH_LOSSES = {
    "sum": _sum_squared_error,
    "mean": torch.nn.functional.mse_loss,
}


def _resolve_accelerator() -> str:
    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_channels(spec: Any) -> Tuple[int, ...]:
    if isinstance(spec, (list, tuple)):
        return tuple(int(c) for c in spec)
    return tuple(int(part) for part in str(spec).split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one candidate regularizer family to one trained model "
            "(architecture x dataset x family x seed) and write a JSON record."
        ),
        allow_abbrev=False,
    )
    # --- what to run -------------------------------------------------------
    parser.add_argument("--arch", type=str, default=None, choices=ARCHITECTURES)
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS)
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=FAMILIES,
        help="Single candidate regularizer family to fit (see module docstring).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON file whose keys supply defaults for any flag below "
            "(underscores or dashes); explicit command-line flags still win."
        ),
    )

    # --- architecture ------------------------------------------------------
    parser.add_argument("--depth", type=int, default=3, help="MLP hidden depth.")
    parser.add_argument("--width", type=int, default=256, help="MLP hidden width.")
    parser.add_argument(
        "--channels",
        type=str,
        default="32,64",
        help="Comma-separated conv widths for the CNN.",
    )
    parser.add_argument(
        "--fc-width",
        type=int,
        default=96,
        help=(
            "CNN classifier-head width.  96 matches the MLP (depth 3, width 256) "
            "parameter budget on MNIST to within ~4%%."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.0,
        help=(
            "Explicit L2 in the training objective.  Default 0 so the fitted "
            "coefficient reflects only architecture/optimizer-induced bias and "
            "the gradient-matching target is the plain cross-entropy gradient."
        ),
    )

    # --- predictive-model training ----------------------------------------
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--model-monitor",
        type=str,
        default="train/loss",
        help=(
            "Early-stopping monitor for the predictive model.  Defaults to the "
            "TRAINING loss, not val/loss: gradient matching is only meaningful at "
            "an optimum of the training objective, so stopping at best validation "
            "loss would hand the estimator a non-stationary point."
        ),
    )
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--train-max-minutes",
        type=float,
        default=45.0,
        help="Wall-clock cap on predictive-model training (0 disables).",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Trainer limit_train_batches/limit_val_batches (smoke tests).",
    )

    # --- bias estimation ---------------------------------------------------
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument(
        "--bias-init",
        type=str,
        default="closed_form",
        choices=("closed_form", "random"),
        help=(
            "How to initialize the family's raw coefficient beta (scale = "
            "exp(beta)).  'closed_form' starts at the analytic least-squares "
            "optimum, which matters because Adam on beta crawls when the fitted "
            "coefficient is many decades away from the init; 'random' reproduces "
            "the other analyses' small random init."
        ),
    )
    parser.add_argument(
        "--bias-init-value",
        type=float,
        default=None,
        help="Explicit beta init (log-space); overrides --bias-init.",
    )
    parser.add_argument(
        "--bias-lr-patience",
        type=int,
        default=10,
        help=(
            "ReduceLROnPlateau patience (epochs) inside the estimator.  The "
            "gradient-matching loss flattens long before the coefficient has "
            "converged, so the default of 10 can freeze the fit prematurely."
        ),
    )
    parser.add_argument("--bias-lr-factor", type=float, default=0.5)
    parser.add_argument(
        "--grad-match-reduction",
        type=str,
        default="sum",
        choices=tuple(GRAD_MATCH_LOSSES),
        help=(
            "Reduction of the gradient-matching squared error.  'sum' (default) "
            "keeps the same minimizer as 'mean' but avoids Adam stalling on "
            "1e-9-scale gradients; 'mean' reproduces the other analyses exactly."
        ),
    )
    parser.add_argument("--bias-max-epochs", type=int, default=500)
    parser.add_argument("--bias-patience", type=int, default=50)
    parser.add_argument(
        "--bias-max-minutes",
        type=float,
        default=45.0,
        help="Wall-clock cap on the gradient-matching fit (0 disables).",
    )
    parser.add_argument(
        "--limit-bias-batches",
        type=float,
        default=1.0,
        help="Trainer limit_train_batches for the bias fit (smoke tests).",
    )

    # --- stationarity check ------------------------------------------------
    parser.add_argument(
        "--stationarity-max-batches",
        type=int,
        default=0,
        help=(
            "Number of training batches used for the endpoint full-batch "
            "gradient/loss (0 = the whole training split)."
        ),
    )
    parser.add_argument(
        "--stationarity-tol",
        type=float,
        default=1e-3,
        help=(
            "Heuristic threshold on ||grad L|| / ||theta|| below which the run is "
            "flagged near_stationary.  Diagnostic only; the raw numbers are "
            "always written out."
        ),
    )

    # --- plumbing ----------------------------------------------------------
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "data"
            / "generated"
            / "arch_effective_regularizer"
            / "checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Reuse an existing predictive-model checkpoint instead of training.",
    )
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="inductive-bias")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    # Two-pass parse so a --config-path JSON can supply defaults while explicit
    # command-line flags still take precedence.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config-path", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    if pre_args.config_path is not None:
        config = json.loads(pre_args.config_path.expanduser().read_text())
        known = {action.dest for action in parser._actions}
        defaults: Dict[str, Any] = {}
        for key, value in config.items():
            dest = key.replace("-", "_")
            if dest not in known:
                LOGGER.warning("Ignoring unknown config key %r", key)
                continue
            defaults[dest] = value
        parser.set_defaults(**defaults)

    args = parser.parse_args()

    # set_defaults bypasses argparse's own choices/type checks, so validate here.
    for name, allowed in (
        ("arch", ARCHITECTURES),
        ("dataset", DATASETS),
        ("family", FAMILIES),
    ):
        value = getattr(args, name)
        if value is None:
            raise ValueError(f"--{name} is required (via CLI or --config-path)")
        if value not in allowed:
            raise ValueError(f"--{name}={value!r} must be one of {allowed}")

    args.channels = _parse_channels(args.channels)
    args.data_root = Path(args.data_root).expanduser()
    args.checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if args.output is not None:
        args.output = Path(args.output).expanduser()
    if args.checkpoint_path is not None:
        args.checkpoint_path = Path(args.checkpoint_path).expanduser()
    return args


def build_datamodule(args: argparse.Namespace):
    cls = (
        MNISTLightningDataModule
        if args.dataset == "mnist"
        else CIFAR10LightningDataModule
    )
    return cls(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
    )


def build_model(args: argparse.Namespace):
    input_dim, in_channels, image_size, num_classes = DATASET_SHAPES[args.dataset]
    if args.arch == "mlp":
        return DeepReLUClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            depth=args.depth,
            width=args.width,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            l2_lambda=args.l2_lambda,
            lr=args.lr,
            momentum=args.momentum,
        )
    return SmallCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        image_size=image_size,
        channels=args.channels,
        fc_width=args.fc_width,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
        l2_lambda=args.l2_lambda,
        lr=args.lr,
        momentum=args.momentum,
    )


def _max_time(minutes: float) -> Optional[timedelta]:
    return timedelta(minutes=minutes) if minutes and minutes > 0 else None


def _limit_batches(value: float):
    """Lightning reads a float as a fraction and an int as a batch count.

    ``1.0`` therefore means "all batches" while ``2`` means "two batches", so
    values above one are converted to ints.
    """
    if value > 1.0 and float(value).is_integer():
        return int(value)
    return float(value)


def stationarity_diagnostics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
    """Endpoint loss and (near) full-batch loss-gradient norm at the trained model.

    Gradient matching solves ``grad R(theta) = -grad L(theta)``, which only has a
    meaningful solution if ``grad L`` is small, i.e. the model really sits at an
    optimum.  The gradient is accumulated in eval mode over the training split
    (the same conditions the estimator sees) for the plain cross-entropy loss,
    which is exactly the target the estimator matches.

    Returns the diagnostics and the mean loss gradient itself, which is reused as
    the target of the closed-form coefficient below.
    """
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    accumulated = [torch.zeros_like(p) for p in params]
    total_examples = 0
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        # reduction="sum" so batches of unequal size combine into the exact mean.
        loss = F.cross_entropy(logits, targets, reduction="sum")
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for slot, grad in zip(accumulated, grads):
                slot.add_(grad)
        total_loss += float(loss.detach())
        total_examples += int(inputs.size(0))

    if was_training:
        model.train()
    if total_examples == 0:
        return {}, None

    with torch.no_grad():
        grad_vector = torch.cat([g.reshape(-1) for g in accumulated]) / total_examples
        param_vector = torch.nn.utils.parameters_to_vector(params)
        grad_norm = float(grad_vector.norm())
        param_norm = float(param_vector.norm())
    diagnostics = {
        "train_ce_loss": total_loss / total_examples,
        "full_batch_grad_norm": grad_norm,
        "param_norm": param_norm,
        "relative_grad_norm": grad_norm / param_norm if param_norm > 0 else float("nan"),
        "num_examples": float(total_examples),
    }
    return diagnostics, grad_vector


def closed_form_coefficient(
    family_name: str,
    model: torch.nn.Module,
    loss_grad: torch.Tensor,
) -> Dict[str, float]:
    """Exact optimum of the gradient-matching objective for one family.

    Every family used here is ``R(theta) = scale * P(theta)`` with a fixed penalty
    ``P``, so ``grad R = scale * grad P`` and the objective
    ``|| scale * grad P + grad L ||^2`` is a one-dimensional least-squares problem
    with the analytic solution

        scale* = <grad P, -grad L> / ||grad P||^2 .

    Two uses: (i) an exact reference the iterative fit must reproduce, which is
    what makes the reported coefficient trustworthy rather than
    optimizer-limited; (ii) an initialization for the iterative fit, whose
    ``scale = exp(beta)`` parametrization otherwise needs Adam to crawl across
    several decades of ``beta``.

    ``scale*`` can be negative, meaning no *positive* coefficient beats using no
    regularizer at all; ``enforce_positive=True`` then drives the fitted scale
    toward 0.  Both the raw and the clamped residual ratios are reported.
    """
    from core.estimators import vector_to_parameter_views  # read-only helper

    family = FAMILY_FACTORIES[family_name](enforce_positive=True, init_value=0.0)
    family = family.to(loss_grad.device)
    flat = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .requires_grad_()
    )
    structured = vector_to_parameter_views(flat, model)
    # init_value=0.0 -> scale = exp(0) = 1, so this gradient is exactly grad P.
    penalty = family(flat, structured)
    (grad_penalty,) = torch.autograd.grad(penalty, flat)

    with torch.no_grad():
        target = -loss_grad
        denominator = float(grad_penalty.pow(2).sum())
        if denominator == 0:
            return {"penalty_grad_norm": 0.0}
        scale_star = float(grad_penalty @ target) / denominator
        target_norm = float(target.norm())
        residual = float((scale_star * grad_penalty - target).norm()) / target_norm
        clamped = max(scale_star, 0.0)
        residual_clamped = (
            float((clamped * grad_penalty - target).norm()) / target_norm
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(grad_penalty, target, dim=0)
        )
    return {
        "penalty": float(penalty.detach()),
        "penalty_grad_norm": denominator**0.5,
        "closed_form_scale": scale_star,
        "closed_form_residual_ratio": residual,
        "closed_form_residual_ratio_clamped": residual_clamped,
        "closed_form_grad_cosine": cosine,
        "closed_form_projection_r2": cosine**2,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)

    run = None
    if not args.disable_wandb:
        import wandb  # lazy: only needed when logging remotely

        run = wandb.init(
            project=args.wandb_project,
            job_type="arch_effective_regularizer",
            name=args.wandb_run_name
            or f"{args.arch}-{args.dataset}-{args.family}-s{args.seed}",
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )

    accelerator = _resolve_accelerator()
    LOGGER.info(
        "A3 run: arch=%s dataset=%s family=%s seed=%d accelerator=%s",
        args.arch,
        args.dataset,
        args.family,
        args.seed,
        accelerator,
    )

    datamodule = build_datamodule(args)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # ------------------------------------------------------------------ model
    # Epoch-level train metrics are only aggregated in on_train_epoch_end, which
    # runs after the validation loop, so a train/* monitor has to be checked there.
    model_early_stop = EarlyStopping(
        monitor=args.model_monitor,
        mode="min",
        patience=args.patience,
        min_delta=args.min_delta,
        check_on_train_epoch_end=args.model_monitor.startswith("train"),
    )
    predictive_metrics: Dict[str, float] = {}
    model_epochs_run: Optional[int] = None

    if args.checkpoint_path is not None:
        cls = DeepReLUClassifier if args.arch == "mlp" else SmallCNN
        model = cls.load_from_checkpoint(str(args.checkpoint_path))
        checkpoint_path: Optional[Path] = args.checkpoint_path
        model_trainer = Trainer(
            accelerator=accelerator, devices=1, logger=False, enable_progress_bar=False
        )
    else:
        model = build_model(args)
        checkpoint_dir = (
            args.checkpoint_dir
            / f"{args.arch}_{args.dataset}_{args.family}_s{args.seed}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Only the LAST weights are kept: the bias fit uses the in-memory model at
        # the end of training (the training-objective endpoint), so a best-val
        # checkpoint would not correspond to the point being measured.
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir), save_top_k=0, save_last=True
        )
        model_trainer = Trainer(
            max_epochs=args.max_epochs,
            max_time=_max_time(args.train_max_minutes),
            logger=False,
            accelerator=accelerator,
            devices=1,
            enable_progress_bar=False,
            limit_train_batches=_limit_batches(args.limit_train_batches),
            limit_val_batches=_limit_batches(args.limit_train_batches),
            callbacks=[model_early_stop, checkpoint_callback],
            log_every_n_steps=50,
        )
        model_trainer.fit(model, datamodule=datamodule)
        model_epochs_run = int(model_trainer.current_epoch)
        # Grab train/val metrics before .test() overwrites callback_metrics.
        predictive_metrics = {
            str(key): float(value)
            for key, value in model_trainer.callback_metrics.items()
        }
        last = checkpoint_callback.last_model_path
        checkpoint_path = Path(last) if last else None

    param_count = sum(p.numel() for p in model.parameters())
    LOGGER.info("Parameter count (%s/%s): %d", args.arch, args.dataset, param_count)

    datamodule.setup("test")
    model_trainer.test(model, datamodule=datamodule)
    predictive_metrics.update(
        {
            str(key): float(value)
            for key, value in model_trainer.callback_metrics.items()
        }
    )

    # The estimator evaluates the model in eval mode; do the same everywhere so
    # the stationarity check and the fit see the identical objective.
    model.eval()
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")
    model.to(device)

    stationarity, loss_grad = stationarity_diagnostics(
        model,
        datamodule.train_dataloader(),
        device=device,
        max_batches=args.stationarity_max_batches,
    )
    if stationarity:
        stationarity["near_stationary"] = bool(
            stationarity["relative_grad_norm"] < args.stationarity_tol
        )
        LOGGER.info(
            "Stationarity: train CE=%.6g ||grad L||=%.6g relative=%.3g",
            stationarity["train_ce_loss"],
            stationarity["full_batch_grad_norm"],
            stationarity["relative_grad_norm"],
        )

    # ---------------------------------------------------- closed-form reference
    closed_form: Dict[str, float] = {}
    if loss_grad is not None:
        closed_form = closed_form_coefficient(args.family, model, loss_grad)
        LOGGER.info("Closed-form optimum: %s", closed_form)

    # ------------------------------------------------------------------- bias
    # Exactly ONE family per run.  Fitting several families jointly is
    # non-identifiable here: their gradients at a trained optimum are strongly
    # collinear, so the joint solution splits the coefficient between families
    # arbitrarily.  The single-element JointBias wrapper is only for naming --
    # it yields keys like "ridge/scale", matching the other analyses' JSON.
    init_value = args.bias_init_value
    if init_value is None and args.bias_init == "closed_form":
        # scale = exp(beta), so start beta at log of the analytic optimum when that
        # optimum is positive; a non-positive optimum means the positivity
        # constraint pushes the fit toward zero, so start small instead.
        scale_star = closed_form.get("closed_form_scale", 0.0)
        init_value = (
            float(torch.log(torch.tensor(max(scale_star, 1e-12))))
            if scale_star > 0
            else -12.0
        )
        LOGGER.info("Initializing beta at %.4g (mode=closed_form)", init_value)
    family = FAMILY_FACTORIES[args.family](
        enforce_positive=True, init_value=init_value
    )
    bias_model = JointBias([family])

    bias_estimator = BiasWithCrossEntropyScheduled(
        predictive_model=model.eval(),
        bias_model=bias_model,
        grad_match_loss_fn=GRAD_MATCH_LOSSES[args.grad_match_reduction],
        lr=args.bias_lr,
        optimizer_cls=torch.optim.Adam,
        bias_lr=args.bias_lr,
        patience_lr=args.bias_lr_patience,
        factor_lr=args.bias_lr_factor,
    )
    bias_early_stop = EarlyStopping(
        monitor="train_bias/loss", mode="min", patience=args.bias_patience
    )
    bias_trainer = Trainer(
        max_epochs=args.bias_max_epochs,
        max_time=_max_time(args.bias_max_minutes),
        logger=False,
        # Nothing to checkpoint here (one scalar), and Lightning's default
        # checkpointer would otherwise dump the whole predictive model each epoch.
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        limit_train_batches=_limit_batches(args.limit_bias_batches),
        callbacks=[bias_early_stop],
        log_every_n_steps=10,
    )
    bias_trainer.fit(bias_estimator, train_dataloaders=datamodule.train_dataloader())

    bias_params = {
        name: float(value) for name, value in bias_model.get_bias_params().items()
    }
    adequacy = {
        key: float(value)
        for key, value in bias_trainer.callback_metrics.items()
        if key in ADEQUACY_KEYS
    }

    LOGGER.info("Fitted bias params: %s", bias_params)
    LOGGER.info("Adequacy: %s", adequacy)

    record: Dict[str, Any] = {
        "config": {
            "arch": args.arch,
            "dataset": args.dataset,
            "family": args.family,
            "seed": args.seed,
            "param_count": param_count,
            "accelerator": accelerator,
            "depth": args.depth,
            "width": args.width,
            "channels": list(args.channels),
            "fc_width": args.fc_width,
            "dropout": args.dropout,
            "batchnorm": bool(args.batchnorm),
            "l2_lambda": args.l2_lambda,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "model_monitor": args.model_monitor,
            "min_delta": args.min_delta,
            "val_fraction": args.val_fraction,
            "bias_lr": args.bias_lr,
            "grad_match_reduction": args.grad_match_reduction,
            "bias_init": args.bias_init,
            "bias_init_value": args.bias_init_value,
            "bias_lr_patience": args.bias_lr_patience,
            "bias_lr_factor": args.bias_lr_factor,
            "bias_max_epochs": args.bias_max_epochs,
            "bias_patience": args.bias_patience,
            "limit_train_batches": args.limit_train_batches,
            "limit_bias_batches": args.limit_bias_batches,
        },
        "bias_params": bias_params,
        "adequacy": adequacy,
        "closed_form": closed_form,
        "stationarity": stationarity,
        "predictive_metrics": predictive_metrics,
        "fit_status": {
            "model_epochs_run": model_epochs_run,
            "model_early_stopped": bool(model_early_stop.stopped_epoch)
            if model_epochs_run is not None
            else None,
            "bias_epochs_run": int(bias_trainer.current_epoch),
            "bias_early_stopped": bool(bias_early_stop.stopped_epoch),
            "bias_init_beta": init_value,
            # ~1 means the iterative fit reproduced the analytic optimum, i.e. the
            # reported coefficient is converged rather than optimizer-limited.
            "fit_over_closed_form": (
                next(iter(bias_params.values()))
                / closed_form["closed_form_scale"]
                if closed_form.get("closed_form_scale", 0.0) > 0
                else None
            ),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "runtime_seconds": round(time.time() - started, 2),
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2))
        LOGGER.info("Wrote %s", args.output)

    if run is not None:
        import wandb

        flat = {f"bias/{k}": v for k, v in bias_params.items()}
        flat.update(adequacy)
        flat.update({f"stationarity/{k}": v for k, v in stationarity.items()})
        flat.update({f"closed_form/{k}": v for k, v in closed_form.items()})
        flat["param_count"] = param_count
        wandb.log(flat)
        for key, value in flat.items():
            run.summary[key] = value
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
