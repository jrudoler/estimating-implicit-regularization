#!/usr/bin/env python3
"""Reproduction of Barrett & Dherin (2022) Figure 2.

Trains MLPs on MNIST across a grid of (learning_rate, width). For each trained
model, records at the time of maximum test accuracy:
  - R_IG = (1/p) * ||grad L||^2 on full train set (Barrett's definition)
  - lambda_hat via the trajectory flow-ref estimator (our contribution)
  - lambda_theoretical = h * p / 4 (Barrett's analytic prediction)
  - max test accuracy

Writes one .pt file; analysis/barrett_igr_figure2_plot.py produces the figure.

Unlike experiments/barrett_igr_trajectory.py which sweeps for theoretical
recovery, this script follows Barrett's Figure 2 protocol: it trains deep nets
to solve the task and reports the implicit-regularization quantity at the
max-test-accuracy iterate.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Sequence

import torch
from torch import nn, Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.igr_trajectory import (
    compute_full_batch_grad_and_hvp,
    fit_igr_lambda_closed_form,
    integrate_gradient_flow_rk4,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "barrett_igr_figure2.pt")
    parser.add_argument("--mnist-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--activation", choices=("relu", "tanh", "gelu"), default="tanh")
    parser.add_argument(
        "--widths",
        type=str,
        default="32,64,128,256",
        help="Comma-separated hidden widths; architecture is [w, w, w] (3 hidden layers).",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="0.003,0.01,0.03,0.1",
    )
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate train/test accuracy every N epochs.",
    )
    parser.add_argument(
        "--train-acc-threshold",
        type=float,
        default=0.9,
        help="Require train accuracy >= this when selecting max-test-acc iterate. "
        "Barrett requires 100%; we relax since small-width tanh nets may not reach 100% on subsamples.",
    )
    parser.add_argument(
        "--estimator-steps",
        type=int,
        default=30,
        help="Number of full-batch GD steps to run from max-test-acc state for the lambda estimator.",
    )
    parser.add_argument("--flow-k", type=int, default=20)
    parser.add_argument(
        "--reference-method",
        choices=("euler", "rk4"),
        default="rk4",
        help="Method for the near-flow reference in the lambda_hat probe.",
    )
    parser.add_argument(
        "--estimator-eta",
        type=float,
        default=None,
        help="Step size for the lambda estimator probe. Default: match training lr.",
    )
    parser.add_argument("--double-precision", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist(
    root: Path, train_samples: int, test_samples: int, seed: int, dtype: torch.dtype
) -> tuple[TensorDataset, TensorDataset]:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    def _subsample(split_train: bool, n: int) -> TensorDataset:
        ds = datasets.MNIST(
            root=root, train=split_train, download=False, transform=transform
        )
        if n < len(ds):
            gen = torch.Generator().manual_seed(seed + (0 if split_train else 1))
            idx = torch.randperm(len(ds), generator=gen)[:n].tolist()
            ds = Subset(ds, idx)
        loader = DataLoader(ds, batch_size=min(2048, len(ds)))
        xs, ys = [], []
        for images, labels in loader:
            xs.append(images.view(images.size(0), -1))
            ys.append(labels)
        return TensorDataset(torch.cat(xs).to(dtype), torch.cat(ys).long())

    return _subsample(True, train_samples), _subsample(False, test_samples)


def build_mlp(
    input_dim: int, widths: Sequence[int], num_classes: int, activation: str, dtype: torch.dtype
) -> nn.Module:
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
    layers: list[nn.Module] = []
    prev = input_dim
    for w in widths:
        layers.append(nn.Linear(prev, w, bias=False).to(dtype))
        layers.append(act())
        prev = w
    layers.append(nn.Linear(prev, num_classes).to(dtype))
    return nn.Sequential(*layers)


def evaluate(model: nn.Module, features: Tensor, targets: Tensor, batch_size: int = 2048) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            xb = features[i : i + batch_size]
            yb = targets[i : i + batch_size]
            preds = model(xb).argmax(dim=-1)
            correct += int((preds == yb).sum().item())
            total += yb.numel()
    return correct / max(1, total)


def flat_params(model: nn.Module) -> Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def full_batch_gd_step(
    model: nn.Module, loss_fn: nn.Module, features: Tensor, targets: Tensor, eta: float
) -> None:
    model.zero_grad(set_to_none=True)
    preds = model(features)
    loss = loss_fn(preds, targets)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.sub_(eta * p.grad)


def estimate_lambda_flow_ref(
    model: nn.Module,
    loss_fn: nn.Module,
    features: Tensor,
    targets: Tensor,
    eta: float,
    num_steps: int,
    flow_k: int,
    reference_method: str = "rk4",
) -> dict:
    """Run num_steps full-batch GD steps from current model state, collecting
    flow-ref (Δθ_flow - Δθ_GD)/η targets. Fit scalar λ in closed form."""
    hg_list: list[Tensor] = []
    target_list: list[Tensor] = []
    stats: list[dict] = []

    for t in range(1, num_steps + 1):
        theta_before = flat_params(model)

        if reference_method == "rk4":
            theta_after_flow = integrate_gradient_flow_rk4(
                model,
                loss_fn,
                features,
                targets,
                theta_before,
                t_end=eta,
                n_steps=flow_k,
            )
        else:  # euler
            ref_model = copy.deepcopy(model)
            sub_h = eta / flow_k
            for _ in range(flow_k):
                full_batch_gd_step(ref_model, loss_fn, features, targets, sub_h)
            theta_after_flow = flat_params(ref_model)

        flat_grad, flat_hvp = compute_full_batch_grad_and_hvp(
            model, loss_fn, features, targets
        )

        # One coarse GD step on the primary model.
        full_batch_gd_step(model, loss_fn, features, targets, eta)
        theta_after_gd = flat_params(model)

        delta_gd = theta_after_gd - theta_before
        delta_flow = theta_after_flow - theta_before
        target = (delta_flow - delta_gd) / eta

        hg_list.append(flat_hvp.detach().cpu())
        target_list.append(target.detach().cpu())
        stats.append(
            {
                "step": t,
                "grad_norm": float(flat_grad.norm()),
                "hg_norm": float(flat_hvp.norm()),
                "target_norm": float(target.norm()),
            }
        )

    num_params = sum(p.numel() for p in model.parameters())
    fit = fit_igr_lambda_closed_form(hg_list, target_list, num_params)
    return {
        "lambda_hat": fit.lambda_hat,
        "lambda_hat_per_param": fit.lambda_hat_per_param,
        "residual_ratio": fit.residual_ratio,
        "num_checkpoints": fit.num_checkpoints,
        "step_stats": stats,
    }


def run_one(
    args: argparse.Namespace,
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    width: int,
    eta: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    torch.manual_seed(seed)

    train_features, train_targets = train_ds.tensors
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    test_features, test_targets = test_ds.tensors
    test_features = test_features.to(device)
    test_targets = test_targets.to(device)

    model = build_mlp(
        input_dim=train_features.shape[1],
        widths=[width, width, width],
        num_classes=10,
        activation=args.activation,
        dtype=dtype,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters())

    # Mini-batch SGD training. Track (train_acc, test_acc) at each eval epoch,
    # save model state at every eval epoch.
    gen = torch.Generator().manual_seed(seed + 17)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=gen,
        drop_last=False,
    )

    history: list[dict] = []
    saved_states: dict[int, dict] = {}  # keyed by eval epoch

    # Initial eval.
    train_acc = evaluate(model, train_features, train_targets)
    test_acc = evaluate(model, test_features, test_targets)
    history.append({"epoch": 0, "train_acc": train_acc, "test_acc": test_acc})

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            model.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.sub_(eta * p.grad)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            train_acc = evaluate(model, train_features, train_targets)
            test_acc = evaluate(model, test_features, test_targets)
            history.append({"epoch": epoch, "train_acc": train_acc, "test_acc": test_acc})
            LOGGER.info(
                "w=%d eta=%.3g seed=%d epoch=%d train_acc=%.4f test_acc=%.4f",
                width,
                eta,
                seed,
                epoch,
                train_acc,
                test_acc,
            )
            saved_states[epoch] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

    # Barrett's protocol: max test acc with train acc >= threshold. If no eval
    # epoch passes the threshold, fall back to max test acc unconditionally.
    threshold = args.train_acc_threshold
    candidates = [h for h in history if h["epoch"] > 0 and h["train_acc"] >= threshold]
    used_fallback = False
    if candidates:
        best = max(candidates, key=lambda h: h["test_acc"])
    else:
        used_fallback = True
        best = max((h for h in history if h["epoch"] > 0), key=lambda h: h["test_acc"])
        LOGGER.warning(
            "w=%d eta=%.3g seed=%d did not hit train_acc>=%.2f; using max test-acc state",
            width,
            eta,
            seed,
            threshold,
        )
    best_epoch = best["epoch"]
    best_test = best["test_acc"]
    best_state = saved_states[best_epoch]

    # Restore max-test-acc state for R_IG + lambda estimation.
    model.load_state_dict(best_state)

    # R_IG = (1/m) * ||grad L||^2 on full train set.
    flat_grad, flat_hvp = compute_full_batch_grad_and_hvp(
        model, loss_fn, train_features, train_targets
    )
    r_ig = float(flat_grad.pow(2).sum() / num_params)
    grad_norm_sq = float(flat_grad.pow(2).sum())
    hg_norm = float(flat_hvp.norm())

    # Lambda estimation via flow-ref from current (best) state.
    est_eta = args.estimator_eta if args.estimator_eta is not None else eta
    est = estimate_lambda_flow_ref(
        model=model,
        loss_fn=loss_fn,
        features=train_features,
        targets=train_targets,
        eta=est_eta,
        num_steps=args.estimator_steps,
        flow_k=args.flow_k,
        reference_method=args.reference_method,
    )

    return {
        "width": width,
        "eta": eta,
        "seed": seed,
        "num_params": num_params,
        "lambda_theoretical": eta * num_params / 4.0,
        "lambda_hat": est["lambda_hat"],
        "lambda_hat_per_param": est["lambda_hat_per_param"],
        "residual_ratio": est["residual_ratio"],
        "r_ig": r_ig,
        "grad_norm_sq": grad_norm_sq,
        "hg_norm_best": hg_norm,
        "best_epoch": best_epoch,
        "best_test_acc": best_test,
        "used_fallback": used_fallback,
        "estimator_eta": est_eta,
        "history": history,
        "activation": args.activation,
    }


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    return obj


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    dtype = torch.float64 if args.double_precision else torch.float32
    device = resolve_device(args.device)
    LOGGER.info("device=%s dtype=%s", device, dtype)

    widths = [int(x) for x in args.widths.split(",")]
    etas = [float(x) for x in args.learning_rates.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    # Load MNIST once; reuse across runs (seed affects subsample ordering minimally).
    train_ds, test_ds = load_mnist(
        args.mnist_root, args.train_samples, args.test_samples, seed=0, dtype=dtype
    )

    results: list[dict] = []
    combos = list(product(widths, etas, seeds))
    for i, (width, eta, seed) in enumerate(combos, 1):
        LOGGER.info("[%d/%d] width=%d eta=%.4g seed=%d", i, len(combos), width, eta, seed)
        try:
            res = run_one(args, train_ds, test_ds, width, eta, seed, device, dtype)
            results.append(res)
            LOGGER.info(
                "  -> lambda_hat=%.3g lambda_theory=%.3g R_IG=%.3e test_acc=%.4f resid=%.3g",
                res["lambda_hat"],
                res["lambda_theoretical"],
                res["r_ig"],
                res["best_test_acc"],
                res["residual_ratio"],
            )
        except Exception as e:
            LOGGER.exception("Run failed for width=%d eta=%g seed=%d: %s", width, eta, seed, e)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"results": results, "config": vars(args)}, args.out)
    json_path = args.out.with_suffix(".json")
    with json_path.open("w") as f:
        # Strip history to keep JSON compact.
        compact = [
            {k: v for k, v in r.items() if k not in {"history"}} for r in results
        ]
        json.dump(_json_safe({"results": compact, "config": vars(args)}), f, indent=2)
    LOGGER.info("Saved %d results to %s and %s", len(results), args.out, json_path)


if __name__ == "__main__":
    main()
