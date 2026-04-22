#!/usr/bin/env python3
"""Trajectory-based empirical reproduction of Barrett & Dherin (2022) IGR.

Fits the scalar lambda in R(theta) = (lambda / p) * ||grad L||^2 against
observed trajectory targets. Two modes:

  flow_ref : train GD at step eta alongside a k-substep near-flow reference;
             target = (Delta_theta_flow - Delta_theta_GD) / eta.
             One GD step is -eta*g exactly; one true-flow step over time eta is
             -eta*g + (eta^2/2)*Hg + O(eta^3). The per-unit-time gap is therefore
             (eta/2)*Hg, which equals grad R with lambda = eta*p/4 (Barrett 2022).

  sgd      : train mini-batch SGD at step eta;
             target = -Delta_theta_SGD / eta - grad L_full(theta_before).
             Theory predicts lambda > 0, magnitude depends on batch noise.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
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
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="synthetic")
    parser.add_argument("--mode", choices=("flow_ref", "sgd"), default="flow_ref")
    parser.add_argument("--eta", type=float, default=1e-2, help="GD step size eta.")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated list of step indices to collect trajectory data at. "
            "Default: log-spaced grid in [1, num_steps]."
        ),
    )

    # Synthetic regression settings
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--p-features", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.1)

    # MNIST settings
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[256, 128],
        help="Hidden layer sizes for MNIST MLP.",
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "tanh", "gelu"),
        default="relu",
        help="Activation function for MNIST MLP. Smooth activations (tanh/gelu) "
        "better satisfy Barrett's backward-error analysis assumptions.",
    )
    parser.add_argument(
        "--mnist-root",
        type=Path,
        default=REPO_ROOT / "MNIST",
    )
    parser.add_argument("--mnist-max-samples", type=int, default=10000)
    parser.add_argument("--mnist-download", action="store_true")

    # Mode-specific
    parser.add_argument(
        "--flow-k",
        type=int,
        default=20,
        help="Number of sub-steps for the near-flow reference (flow_ref mode).",
    )
    parser.add_argument(
        "--reference-method",
        choices=("euler", "rk4"),
        default="rk4",
        help="Method for the near-flow reference. 'rk4' (recommended) gives "
        "O(h^5) local error per substep and avoids the k-Euler 'two-Euler "
        "tautology' artifact where the reference and primary trajectories "
        "are both Euler discretizations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for SGD mode. Ignored in flow_ref.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--double-precision",
        action="store_true",
        help="Run in float64 (recommended for synthetic, slow for MNIST).",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps override.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output path for the results .pt file.",
    )
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


def default_checkpoints(num_steps: int) -> list[int]:
    """Log-spaced checkpoint grid inclusive of first and last steps."""
    if num_steps < 2:
        return [num_steps]
    n_points = min(20, max(8, int(torch.log10(torch.tensor(float(num_steps))).item() * 6)))
    raw = torch.logspace(0, torch.log10(torch.tensor(float(num_steps))), n_points)
    grid = sorted({int(round(float(x))) for x in raw})
    grid = [t for t in grid if 1 <= t <= num_steps]
    if grid[-1] != num_steps:
        grid.append(num_steps)
    return grid


def make_synthetic(args: argparse.Namespace, dtype: torch.dtype) -> tuple[TensorDataset, nn.Module, nn.Module]:
    gen = torch.Generator().manual_seed(args.seed)
    X = torch.randn(args.n_samples, args.p_features, generator=gen, dtype=dtype)
    beta = 3.0 * torch.randn(args.p_features, generator=gen, dtype=dtype)
    y = X @ beta
    if args.noise_std > 0:
        y = y + args.noise_std * torch.randn(y.shape, generator=gen, dtype=dtype)
    y = y.unsqueeze(-1)
    dataset = TensorDataset(X, y)

    model = nn.Linear(args.p_features, 1, bias=False).to(dtype)
    with torch.no_grad():
        model.weight.zero_()
    loss_fn = nn.MSELoss(reduction="mean")
    return dataset, model, loss_fn


def make_mnist(args: argparse.Namespace, dtype: torch.dtype) -> tuple[TensorDataset, nn.Module, nn.Module]:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root=args.mnist_root,
        train=True,
        download=args.mnist_download,
        transform=transform,
    )
    total = len(dataset)
    if args.mnist_max_samples and args.mnist_max_samples < total:
        gen = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(total, generator=gen)[: args.mnist_max_samples].tolist()
        dataset = Subset(dataset, indices)
        total = len(dataset)

    loader = DataLoader(dataset, batch_size=min(2048, total))
    feats, tgts = [], []
    for images, labels in loader:
        feats.append(images.view(images.size(0), -1))
        tgts.append(labels)
    features = torch.cat(feats, dim=0).to(dtype)
    targets = torch.cat(tgts, dim=0).long()
    dataset = TensorDataset(features, targets)

    activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    activation_cls = activation_map[args.activation]
    layers: list[nn.Module] = []
    prev = features.shape[1]
    for h in args.hidden_dims:
        layers.append(nn.Linear(prev, h, bias=False).to(dtype))
        layers.append(activation_cls())
        prev = h
    layers.append(nn.Linear(prev, 10).to(dtype))
    model = nn.Sequential(*layers)
    loss_fn = nn.CrossEntropyLoss()
    return dataset, model, loss_fn


def flat_params(model: nn.Module) -> Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_flat_params(model: nn.Module, flat: Tensor) -> None:
    vector_to_parameters(flat, model.parameters())


def full_batch_grad_step(
    model: nn.Module,
    loss_fn: nn.Module,
    features: Tensor,
    targets: Tensor,
    step_size: float,
) -> Tensor:
    """In-place full-batch GD step; returns the new flat parameter vector."""
    model.zero_grad(set_to_none=True)
    preds = model(features)
    loss = loss_fn(preds, targets)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.sub_(step_size * p.grad)
    return flat_params(model)


def run_flow_ref(
    args: argparse.Namespace,
    dataset: TensorDataset,
    model: nn.Module,
    loss_fn: nn.Module,
    checkpoints: Sequence[int],
    device: torch.device,
) -> dict:
    features, targets = dataset.tensors
    features = features.to(device)
    targets = targets.to(device)
    model = model.to(device)

    cp_set = set(checkpoints)
    records: list[dict] = []

    # Primary full-batch GD trajectory at step eta.
    theta = flat_params(model)
    for t in range(1, args.num_steps + 1):
        theta_before = theta.clone()

        if t in cp_set:
            # Near-flow reference from theta_before over time eta.
            if args.reference_method == "rk4":
                theta_after_flow = integrate_gradient_flow_rk4(
                    model,
                    loss_fn,
                    features,
                    targets,
                    theta_before,
                    t_end=args.eta,
                    n_steps=args.flow_k,
                )
            else:  # euler
                ref_model = copy.deepcopy(model)
                sub_h = args.eta / args.flow_k
                for _ in range(args.flow_k):
                    full_batch_grad_step(ref_model, loss_fn, features, targets, sub_h)
                theta_after_flow = flat_params(ref_model)

            # Compute full-batch grad and Hg at theta_before.
            flat_grad, flat_hvp = compute_full_batch_grad_and_hvp(
                model, loss_fn, features, targets
            )

        # Advance primary trajectory one GD step.
        theta_after_gd = full_batch_grad_step(
            model, loss_fn, features, targets, args.eta
        )
        theta = theta_after_gd

        if t in cp_set:
            delta_gd = theta_after_gd - theta_before
            delta_flow = theta_after_flow - theta_before
            target = (delta_flow - delta_gd) / args.eta

            records.append(
                {
                    "step": t,
                    "theta": theta_before.detach().cpu(),
                    "grad": flat_grad.detach().cpu(),
                    "hg": flat_hvp.detach().cpu(),
                    "target": target.detach().cpu(),
                    "grad_norm": float(flat_grad.norm()),
                    "hg_norm": float(flat_hvp.norm()),
                    "target_norm": float(target.norm()),
                    "delta_gd_norm": float(delta_gd.norm()),
                    "delta_flow_norm": float(delta_flow.norm()),
                }
            )
            LOGGER.info(
                "step=%d ||g||=%.3e ||Hg||=%.3e ||target||=%.3e ||Δ_gd-Δ_flow||=%.3e",
                t,
                float(flat_grad.norm()),
                float(flat_hvp.norm()),
                float(target.norm()),
                float((delta_gd - delta_flow).norm()),
            )

    return {"records": records}


def run_sgd(
    args: argparse.Namespace,
    dataset: TensorDataset,
    model: nn.Module,
    loss_fn: nn.Module,
    checkpoints: Sequence[int],
    device: torch.device,
) -> dict:
    features, targets = dataset.tensors
    features = features.to(device)
    targets = targets.to(device)
    model = model.to(device)

    gen = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=gen,
        drop_last=False,
    )
    cp_set = set(checkpoints)
    records: list[dict] = []

    step = 0
    epoch = 0
    while step < args.num_steps:
        epoch += 1
        for batch_features, batch_targets in loader:
            step += 1
            if step > args.num_steps:
                break
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            theta_before = flat_params(model)

            if step in cp_set:
                flat_grad, flat_hvp = compute_full_batch_grad_and_hvp(
                    model, loss_fn, features, targets
                )

            # One SGD step on the mini-batch.
            model.zero_grad(set_to_none=True)
            preds = model(batch_features)
            loss = loss_fn(preds, batch_targets)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.sub_(args.eta * p.grad)
            theta_after = flat_params(model)

            if step in cp_set:
                delta = theta_after - theta_before
                target = -delta / args.eta - flat_grad
                records.append(
                    {
                        "step": step,
                        "theta": theta_before.detach().cpu(),
                        "grad": flat_grad.detach().cpu(),
                        "hg": flat_hvp.detach().cpu(),
                        "target": target.detach().cpu(),
                        "grad_norm": float(flat_grad.norm()),
                        "hg_norm": float(flat_hvp.norm()),
                        "target_norm": float(target.norm()),
                        "delta_norm": float(delta.norm()),
                    }
                )
                LOGGER.info(
                    "step=%d epoch=%d ||g_full||=%.3e ||Hg||=%.3e ||target||=%.3e",
                    step,
                    epoch,
                    float(flat_grad.norm()),
                    float(flat_hvp.norm()),
                    float(target.norm()),
                )

    return {"records": records}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.double_precision else torch.float32
    device = resolve_device(args.device)
    LOGGER.info(
        "dataset=%s mode=%s eta=%.4g steps=%d dtype=%s device=%s",
        args.dataset,
        args.mode,
        args.eta,
        args.num_steps,
        dtype,
        device,
    )

    if args.dataset == "synthetic":
        dataset, model, loss_fn = make_synthetic(args, dtype)
    else:
        dataset, model, loss_fn = make_mnist(args, dtype)

    num_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("num_params=%d", num_params)

    checkpoints = (
        sorted(int(x) for x in args.checkpoints.split(","))
        if args.checkpoints
        else default_checkpoints(args.num_steps)
    )
    LOGGER.info("checkpoints=%s (n=%d)", checkpoints, len(checkpoints))

    if args.mode == "flow_ref":
        run_results = run_flow_ref(args, dataset, model, loss_fn, checkpoints, device)
    else:
        run_results = run_sgd(args, dataset, model, loss_fn, checkpoints, device)

    records = run_results["records"]

    # Per-checkpoint fit (single-step lambda estimate).
    per_step = []
    for rec in records:
        fit = fit_igr_lambda_closed_form([rec["hg"]], [rec["target"]], num_params)
        per_step.append(
            {
                "step": rec["step"],
                "lambda_hat": fit.lambda_hat,
                "lambda_hat_per_param": fit.lambda_hat_per_param,
                "residual_ratio": fit.residual_ratio,
                "hg_norm": rec["hg_norm"],
                "target_norm": rec["target_norm"],
                "grad_norm": rec["grad_norm"],
            }
        )

    # Pooled fit across all checkpoints.
    pooled = fit_igr_lambda_closed_form(
        [rec["hg"] for rec in records],
        [rec["target"] for rec in records],
        num_params,
    )

    theoretical_lambda = args.eta * num_params / 4.0
    LOGGER.info(
        "pooled lambda_hat=%.6g | lambda_hat/p=%.6g | theory=eta*p/4=%.6g | ratio=%.3f | residual_ratio=%.3f",
        pooled.lambda_hat,
        pooled.lambda_hat_per_param,
        theoretical_lambda,
        pooled.lambda_hat / theoretical_lambda if theoretical_lambda else float("nan"),
        pooled.residual_ratio,
    )

    payload = {
        "config": vars(args),
        "num_params": num_params,
        "theoretical_lambda": theoretical_lambda,
        "pooled": {
            "lambda_hat": pooled.lambda_hat,
            "lambda_hat_per_param": pooled.lambda_hat_per_param,
            "residual_ratio": pooled.residual_ratio,
            "num_checkpoints": pooled.num_checkpoints,
        },
        "per_step": per_step,
    }

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Persist raw tensors separately for post-hoc analysis.
        tensor_payload = {
            "records": records,
            "summary": payload,
        }
        torch.save(tensor_payload, args.save)
        json_path = args.save.with_suffix(".json")
        with json_path.open("w") as f:
            json.dump(_json_safe(payload), f, indent=2)
        LOGGER.info("Saved results to %s and %s", args.save, json_path)


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


if __name__ == "__main__":
    main()
