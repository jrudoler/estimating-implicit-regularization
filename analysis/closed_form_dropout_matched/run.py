"""Dropout implicit-bias sweep with MATCHED stationarity (convergence control).

Why this exists
---------------
`analysis/closed_form_dropout_sweep/` found that as the dropout rate rises, an L2
(ridge) explanation of the residual full-batch loss gradient gets steadily worse
(grad_cosine 0.236 -> 0.064) while scale-invariant spectral families get steadily
better.  But higher dropout ALSO left the models further from stationarity
(Spearman(dropout, relative_grad_norm) = +0.914), and relative_grad_norm is itself
~-0.99 correlated with the ridge cosine.  Gradient matching assumes a stationary
point, so a larger residual gradient mechanically inflates the "unexplained"
component and could produce the whole trend as an artifact.

This script removes that confound by measuring the SAME closed-form quantities
along the WHOLE optimisation trajectory rather than only at a val-loss early-stop
point.  Every `--probe-every` epochs it records

  * eval-mode (deterministic, dropout-free) full-batch grad L, its norm, and
    relative_grad_norm = ||grad L|| / ||theta||          <- the stationarity axis
  * an unbiased estimate of the norm of the gradient of the *objective actually
    being optimised* (the dropout-expected loss), see `train_objective_grad_norm`
  * the exact closed-form optimum s* + adequacy (residual_ratio, grad_cosine) for
    all five families

Two convergence-controlled analyses then become possible from one launch:

  (A) matched TARGET: for each run take the first probe whose relative_grad_norm
      is <= a common target -> every dropout rate is compared at (just below) the
      same stationarity level.
  (B) matched BAND: for each run take the probe whose relative_grad_norm is
      closest to a common value g* -> exact matching by selection, valid even if
      some dropout rates can never reach a low target (which is expected, since
      dropout training converges to a stationary point of the *dropout* objective,
      where the dropout-free gradient need not vanish).

Both are done in `analyze.py`; this script only produces the trajectory.

Faithfulness to the original sweep
----------------------------------
`full_batch_loss_grad`, `closed_form` and `FAMILIES` are imported directly from
`analysis/closed_form_dropout_sweep/run.py` (not copied), so the estimator is
bit-identical.  The model is the same `DeepReLUClassifier` (depth 3, width 256,
l2_lambda=0, lr=0.05, momentum=0.9, batch_size=2048) with the same
SGD+MultiStepLR([60,100,200], gamma=0.1) returned by its own
`configure_optimizers()`.  The only changes are (a) MNIST is held in memory on the
device instead of going through a DataLoader each epoch (numerically the same
tensors and the same seed-42 train/val split, ~20x faster per epoch, which is what
makes a multi-thousand-epoch budget affordable), and (b) training runs on a long
fixed budget instead of val-loss patience.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from core.models import DeepReLUClassifier  # noqa: E402


def _load_original_sweep_module():
    """Import analysis/closed_form_dropout_sweep/run.py without editing it."""
    path = REPO_ROOT / "analysis" / "closed_form_dropout_sweep" / "run.py"
    spec = importlib.util.spec_from_file_location("cf_dropout_sweep_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ORIG = _load_original_sweep_module()
FAMILIES = _ORIG.FAMILIES
full_batch_loss_grad = _ORIG.full_batch_loss_grad
closed_form = _ORIG.closed_form


# --------------------------------------------------------------------------- data


def load_mnist_in_memory(root: Path, device: torch.device, val_fraction: float = 0.1):
    """Same tensors and same seed-42 split as core.data.MNISTLightningDataModule.

    ToTensor() on an MNIST PIL image is exactly uint8/255 reshaped to (1,28,28),
    and random_split(generator=manual_seed(42)) is randperm(60000) sliced in order,
    so this reproduces the original pipeline's data exactly.
    """
    train_raw = datasets.MNIST(root=root, train=True, download=False)
    test_raw = datasets.MNIST(root=root, train=False, download=False)

    x_all = train_raw.data.to(torch.float32).div_(255.0).unsqueeze(1)
    y_all = train_raw.targets.clone()
    n = x_all.shape[0]
    val_size = int(n * val_fraction)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = perm[: n - val_size], perm[n - val_size :]

    return {
        "x_train": x_all[train_idx].to(device),
        "y_train": y_all[train_idx].to(device),
        "x_val": x_all[val_idx].to(device),
        "y_val": y_all[val_idx].to(device),
        "x_test": test_raw.data.to(torch.float32).div_(255.0).unsqueeze(1).to(device),
        "y_test": test_raw.targets.clone().to(device),
    }


def chunks(x, y, batch_size):
    """A minimal (x, y) iterable standing in for a DataLoader (sequential, no shuffle)."""
    for i in range(0, x.shape[0], batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]


class ChunkLoader:
    """Re-iterable sequential loader over in-memory tensors."""

    def __init__(self, x, y, batch_size):
        self.x, self.y, self.batch_size = x, y, batch_size

    def __iter__(self):
        return chunks(self.x, self.y, self.batch_size)


# ------------------------------------------------------------------- stationarity


def _flat_grad(model, total):
    device = next(model.parameters()).device
    return torch.cat(
        [
            (p.grad.detach().reshape(-1) / total)
            if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in model.parameters()
        ]
    )


def _train_mode_grad_sample(model, loader, total, reps):
    """Mean of `reps` Monte-Carlo samples of grad E_mask[L_dropout] over the train set.

    Runs in train() mode, so each minibatch draws fresh dropout masks; the sum over
    all batches divided by N is an unbiased estimate of the gradient of the
    dropout-expected training objective.
    """
    device = next(model.parameters()).device
    acc = None
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        for x, y in loader:
            F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum").backward()
        g = _flat_grad(model, total)
        acc = g if acc is None else acc + g
    model.zero_grad(set_to_none=True)
    return acc / reps


def train_objective_grad_norm(model, loader, total, reps=6):
    """Unbiased ||grad E_mask[L_dropout]|| via two independent halves of mask draws.

    THIS, not the eval-mode gradient, is the quantity that must vanish at
    convergence.  Writing the implicit regulariser as
        R(theta) := E_mask[L_dropout(theta)] - L(theta),
    the optimiser follows grad(L + R), so the measured train-objective gradient is
    exactly the non-stationarity residual eps = grad L + grad R.  At an exact
    stationary point grad L = -grad R, i.e. the eval-mode gradient the estimator
    explains IS the implicit-regulariser gradient and is *supposed* to grow with
    dropout.  So convergence must be matched on ||eps||, not on ||grad L||.

    A single MC estimate has E||g_hat||^2 = ||g||^2 + tr(Sigma)/K, badly biased
    upward at 3.4e5 parameters.  Two independent half-averages give the unbiased
    <g_A, g_B> -> ||g||^2 with the mask-noise term cancelling in expectation;
    `reps` samples per half shrink its variance by ~reps.
    """
    was_training = model.training
    model.train()
    g_a = _train_mode_grad_sample(model, loader, total, reps)
    g_b = _train_mode_grad_sample(model, loader, total, reps)
    if not was_training:
        model.eval()
    dot = float(g_a @ g_b)
    return {
        "train_grad_norm_sq_unbiased": dot,
        "train_grad_norm_unbiased": dot**0.5 if dot > 0 else 0.0,
        "train_grad_norm_naive": float(((g_a + g_b) / 2).norm()),
        "train_grad_mask_reps": 2 * reps,
    }


@torch.no_grad()
def eval_metrics(model, x, y, batch_size):
    """Deterministic (eval-mode) mean CE and accuracy."""
    was_training = model.training
    model.eval()
    loss_sum, correct = 0.0, 0
    for xb, yb in chunks(x, y, batch_size):
        logits = model(xb)
        loss_sum += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(dim=1) == yb).sum())
    if was_training:
        model.train()
    n = int(y.numel())
    return {"loss": loss_sum / n, "acc": correct / n}


def probe(model, data, batch_size, epoch, with_train_grad=True, mask_reps=6):
    """One full measurement at the current theta: stationarity + all 5 families."""
    model.eval()
    train_loader = ChunkLoader(data["x_train"], data["y_train"], batch_size)
    n_train = int(data["y_train"].numel())

    grad_l = full_batch_loss_grad(model, train_loader)
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    grad_norm = float(grad_l.norm())
    param_norm = float(theta.norm())

    rec = {
        "epoch": int(epoch),
        "full_batch_grad_norm": grad_norm,
        "param_norm": param_norm,
        "relative_grad_norm": grad_norm / param_norm,
        "train_loss": eval_metrics(model, data["x_train"], data["y_train"], batch_size)["loss"],
        "closed_form": {name: closed_form(name, model, grad_l) for name in FAMILIES},
    }
    val = eval_metrics(model, data["x_val"], data["y_val"], batch_size)
    rec["val_loss"], rec["val_acc"] = val["loss"], val["acc"]
    if with_train_grad:
        stat = train_objective_grad_norm(model, train_loader, n_train, reps=mask_reps)
        rec.update(stat)
        rec["relative_train_grad_norm"] = stat["train_grad_norm_unbiased"] / param_norm
        # fraction of the measured eval-mode gradient that could be non-stationarity
        rec["contamination_ratio"] = stat["train_grad_norm_unbiased"] / grad_norm
    model.eval()
    return rec


# ----------------------------------------------------------------------- training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batchnorm", action="store_true")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--probe-every", type=int, default=25)
    # Dense early probing: low-dropout runs sweep through the whole range of
    # relative_grad_norm within the first ~150 epochs, and the matched-band analysis
    # needs a probe near the common level g* (~1e-3) for those runs too.
    ap.add_argument("--probe-every-early", type=int, default=5)
    ap.add_argument("--early-until", type=int, default=200)
    ap.add_argument("--mask-reps", type=int, default=12,
                    help="mask draws per half for the unbiased train-objective grad norm")
    # Extra tail LR decay, applied identically to every condition.  The model's own
    # MultiStepLR([60,100,200], 0.1) bottoms out at lr=5e-5, where SGD + dropout noise
    # leaves a non-zero equilibrium ||eps||.  Shrinking lr further shrinks that floor
    # and so drives every condition closer to exact stationarity of its OWN objective.
    ap.add_argument("--tail-decay-start", type=int, default=600)
    ap.add_argument("--tail-decay-every", type=int, default=300)
    ap.add_argument("--tail-decay-gamma", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument(
        "--target-rel-grad",
        type=float,
        default=3e-4,
        help="common stationarity target on eval-mode ||grad L||/||theta||; the first "
        "probe at or below it is recorded as the matched-target endpoint.",
    )
    ap.add_argument(
        "--stop-on-target",
        action="store_true",
        help="stop training once the target is met (default: run the full budget, "
        "which also supports the matched-band analysis).",
    )
    ap.add_argument("--max-minutes", type=float, default=200.0)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    data = load_mnist_in_memory(args.data_root, device)
    x_train, y_train = data["x_train"], data["y_train"]
    n_train = int(y_train.numel())

    model = DeepReLUClassifier(
        input_dim=784,
        num_classes=10,
        depth=args.depth,
        width=args.width,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
        l2_lambda=0.0,
        lr=args.lr,
        momentum=args.momentum,
    ).to(device)
    optimizers, schedulers = model.configure_optimizers()
    optimizer, scheduler = optimizers[0], schedulers[0]

    t0 = time.time()
    trajectory = [probe(model, data, args.batch_size, epoch=0, mask_reps=args.mask_reps)]
    target_epoch = None
    if trajectory[0]["relative_grad_norm"] <= args.target_rel_grad:
        target_epoch = 0
    stopped_reason = "max_epochs"
    epochs_run = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(n_train, device=device)
        for i in range(0, n_train, args.batch_size):
            idx = order[i : i + args.batch_size]
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (
            args.tail_decay_gamma < 1.0
            and epoch >= args.tail_decay_start
            and (epoch - args.tail_decay_start) % args.tail_decay_every == 0
        ):
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * args.tail_decay_gamma, args.min_lr)
        epochs_run = epoch

        cadence = args.probe_every_early if epoch <= args.early_until else args.probe_every
        if epoch % cadence == 0 or epoch == args.max_epochs:
            rec = probe(model, data, args.batch_size, epoch=epoch, mask_reps=args.mask_reps)
            rec["lr"] = float(optimizer.param_groups[0]["lr"])
            trajectory.append(rec)
            if target_epoch is None and rec["relative_grad_norm"] <= args.target_rel_grad:
                target_epoch = epoch
                if args.stop_on_target:
                    stopped_reason = "target_met"
                    break
            if (time.time() - t0) / 60.0 > args.max_minutes:
                stopped_reason = "time_budget"
                break

    if trajectory[-1]["epoch"] != epochs_run:
        rec = probe(model, data, args.batch_size, epoch=epochs_run, mask_reps=args.mask_reps)
        rec["lr"] = float(optimizer.param_groups[0]["lr"])
        trajectory.append(rec)

    final = trajectory[-1]
    matched = next(
        (r for r in trajectory if r["relative_grad_norm"] <= args.target_rel_grad), None
    )
    test = eval_metrics(model, data["x_test"], data["y_test"], args.batch_size)

    out = {
        "config": {
            "dropout": args.dropout,
            "seed": args.seed,
            "depth": args.depth,
            "width": args.width,
            "batchnorm": args.batchnorm,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "probe_every": args.probe_every,
            "probe_every_early": args.probe_every_early,
            "early_until": args.early_until,
            "target_rel_grad": args.target_rel_grad,
            "stop_on_target": bool(args.stop_on_target),
            "mask_reps": args.mask_reps,
            "tail_decay_start": args.tail_decay_start,
            "tail_decay_every": args.tail_decay_every,
            "tail_decay_gamma": args.tail_decay_gamma,
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "device": str(device),
        },
        "run": {
            "epochs_run": epochs_run,
            "stopped_reason": stopped_reason,
            "wall_minutes": (time.time() - t0) / 60.0,
            "target_met": matched is not None,
            "target_epoch": target_epoch,
            "n_probes": len(trajectory),
        },
        "test": {"test/loss": test["loss"], "test/acc": test["acc"]},
        # endpoint of the fixed budget
        "final": final,
        # first probe at or below the common stationarity target (matched-target design)
        "matched": matched,
        "trajectory": trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "dropout": args.dropout,
                "seed": args.seed,
                "epochs_run": epochs_run,
                "stopped_reason": stopped_reason,
                "wall_minutes": round(out["run"]["wall_minutes"], 2),
                "target_met": matched is not None,
                "target_epoch": target_epoch,
                "final_rel_grad": final["relative_grad_norm"],
                "final_rel_train_grad": final.get("relative_train_grad_norm"),
                "matched_rel_grad": matched["relative_grad_norm"] if matched else None,
                "final_ridge_cosine": final["closed_form"]["ridge"]["grad_cosine"],
                "test_acc": test["acc"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
