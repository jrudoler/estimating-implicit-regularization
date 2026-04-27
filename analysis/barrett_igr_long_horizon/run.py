#!/usr/bin/env python3
"""Long-horizon 3-trajectory test of Barrett & Dherin (2022) Thm 3.1.

From a shared initial state theta_0, compute three trajectories over time
[0, N*eta]:
  1. DISCRETE GD at step size eta, for N steps: theta_k = theta_{k-1} - eta*g.
  2. ORIGINAL-LOSS GRADIENT FLOW, integrated via RK4: dtheta/dt = -grad L(theta).
  3. MODIFIED-LOSS FLOW, integrated via RK4:
        dtheta/dt = -grad L(theta) - (eta/2)*H(theta)@grad L(theta).

Barrett's Theorem 3.1 predicts that trajectory 1 tracks trajectory 3 to
O(eta^2) per step (so accumulated drift over N steps stays small), while it
drifts from trajectory 2 by O(eta) per step (accumulated drift grows).

This is the non-tautological version of our flow-ref estimator: we're not
constructing targets from the same Taylor expansion, we're measuring how well
GD's discrete iterates match a numerically-integrated continuous ODE over
many steps. The Taylor identity is still the theoretical basis, but the
*integrability* of the modified flow over long horizons is a substantive
prediction with its own failure modes.

Outputs per-step L2 distances ||theta_GD(k*eta) - theta_orig(k*eta)|| and
||theta_GD(k*eta) - theta_mod(k*eta)|| for k = 1..N.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.igr_trajectory import (
    _compute_flat_grad_at,
    _compute_flat_hg_at,
    integrate_gradient_flow_rk4,
    integrate_modified_flow_rk4,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("synthetic", "mnist"), default="synthetic"
    )
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of discrete GD steps, and number of sample points on the ODE trajectories.",
    )
    parser.add_argument(
        "--rk4-substeps",
        type=int,
        default=20,
        help="RK4 substeps per eta interval when integrating the ODEs. "
        "RK4 has O(h^5) local error, so modest values are sufficient.",
    )
    # Synthetic
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--p-features", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.1)
    # MNIST
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 32])
    parser.add_argument(
        "--activation", choices=("relu", "tanh", "gelu"), default="tanh"
    )
    parser.add_argument("--mnist-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--mnist-max-samples", type=int, default=2000)
    # General
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--double-precision", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--save", type=Path, default=None, help="Output path for results .pt file."
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


def resolve_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype:
    if device.type == "mps":
        if args.double_precision:
            LOGGER.warning(
                "MPS does not reliably support float64; using float32 instead."
            )
        return torch.float32
    return torch.float64 if args.double_precision else torch.float32


def make_synthetic(args, dtype) -> tuple[TensorDataset, nn.Module, nn.Module]:
    gen = torch.Generator().manual_seed(args.seed)
    X = torch.randn(args.n_samples, args.p_features, generator=gen, dtype=dtype)
    beta = 3.0 * torch.randn(args.p_features, generator=gen, dtype=dtype)
    y = X @ beta
    if args.noise_std > 0:
        y = y + args.noise_std * torch.randn(y.shape, generator=gen, dtype=dtype)
    y = y.unsqueeze(-1)
    ds = TensorDataset(X, y)
    model = nn.Linear(args.p_features, 1, bias=False).to(dtype)
    with torch.no_grad():
        model.weight.zero_()
    return ds, model, nn.MSELoss(reduction="mean")


def make_mnist(args, dtype) -> tuple[TensorDataset, nn.Module, nn.Module]:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST(
        root=args.mnist_root, train=True, download=False, transform=transform
    )
    if args.mnist_max_samples and args.mnist_max_samples < len(ds):
        gen = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(ds), generator=gen)[: args.mnist_max_samples].tolist()
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=min(2048, len(ds)))
    xs, ys = [], []
    for images, labels in loader:
        xs.append(images.view(images.size(0), -1))
        ys.append(labels)
    features = torch.cat(xs).to(dtype)
    targets = torch.cat(ys).long()
    dataset = TensorDataset(features, targets)

    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[args.activation]
    layers: list[nn.Module] = []
    prev = features.shape[1]
    for h in args.hidden_dims:
        layers.append(nn.Linear(prev, h, bias=False).to(dtype))
        layers.append(act())
        prev = h
    layers.append(nn.Linear(prev, 10).to(dtype))
    model = nn.Sequential(*layers)
    return dataset, model, nn.CrossEntropyLoss()


def flat_params(model: nn.Module) -> Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_flat(model: nn.Module, flat: Tensor) -> None:
    vector_to_parameters(flat, model.parameters())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args, device)
    LOGGER.info(
        "dataset=%s eta=%g N=%d dtype=%s device=%s",
        args.dataset,
        args.eta,
        args.num_steps,
        dtype,
        device,
    )

    if args.dataset == "synthetic":
        ds, model, loss_fn = make_synthetic(args, dtype)
    else:
        ds, model, loss_fn = make_mnist(args, dtype)

    features, targets = ds.tensors
    features = features.to(device)
    targets = targets.to(device)
    model = model.to(device)

    theta0 = flat_params(model)
    num_params = theta0.numel()
    LOGGER.info("num_params=%d ||theta_0||=%.4g", num_params, float(theta0.norm()))

    # --- Trajectory 1: discrete GD ---
    LOGGER.info("Computing discrete GD trajectory (N=%d)...", args.num_steps)
    theta_gd_snapshots: list[Tensor] = [theta0.clone()]
    set_flat(model, theta0)
    for k in range(1, args.num_steps + 1):
        g = _compute_flat_grad_at(model, loss_fn, features, targets, flat_params(model))
        new_theta = flat_params(model) - args.eta * g
        set_flat(model, new_theta)
        theta_gd_snapshots.append(new_theta.detach().clone())

    # --- Trajectory 2: original-loss gradient flow via RK4, sampled at k*eta ---
    LOGGER.info(
        "Integrating original-loss gradient flow (RK4, substeps=%d per eta)...",
        args.rk4_substeps,
    )
    theta_orig_snapshots: list[Tensor] = [theta0.clone()]
    cur = theta0.clone()
    for k in range(1, args.num_steps + 1):
        cur = integrate_gradient_flow_rk4(
            model=model,
            loss_fn=loss_fn,
            features=features,
            targets=targets,
            theta0=cur,
            t_end=args.eta,
            n_steps=args.rk4_substeps,
        )
        theta_orig_snapshots.append(cur.detach().clone())
        if k % max(1, args.num_steps // 10) == 0 or k == args.num_steps:
            LOGGER.info("  original-flow step %d/%d", k, args.num_steps)

    # --- Trajectory 3: modified-loss flow via RK4, sampled at k*eta ---
    LOGGER.info(
        "Integrating modified-loss flow (RK4, substeps=%d per eta)...",
        args.rk4_substeps,
    )
    theta_mod_snapshots: list[Tensor] = [theta0.clone()]
    cur = theta0.clone()
    for k in range(1, args.num_steps + 1):
        cur = integrate_modified_flow_rk4(
            model=model,
            loss_fn=loss_fn,
            features=features,
            targets=targets,
            theta0=cur,
            t_end=args.eta,
            n_steps=args.rk4_substeps,
            eta=args.eta,
        )
        theta_mod_snapshots.append(cur.detach().clone())
        if k % max(1, args.num_steps // 10) == 0 or k == args.num_steps:
            LOGGER.info("  modified-flow step %d/%d", k, args.num_steps)

    # --- Drift measurements at each sample point ---
    drifts = []
    for k in range(args.num_steps + 1):
        d_orig = float((theta_gd_snapshots[k] - theta_orig_snapshots[k]).norm())
        d_mod = float((theta_gd_snapshots[k] - theta_mod_snapshots[k]).norm())
        d_theta0 = float((theta_gd_snapshots[k] - theta0).norm())
        drifts.append(
            {
                "step": k,
                "time": k * args.eta,
                "dist_gd_to_orig_flow": d_orig,
                "dist_gd_to_mod_flow": d_mod,
                "dist_gd_from_theta0": d_theta0,
                "gd_theta_norm": float(theta_gd_snapshots[k].norm()),
            }
        )

    LOGGER.info(
        "Final drifts at N=%d: GD-vs-orig_flow=%.3e, GD-vs-mod_flow=%.3e, GD-from-theta0=%.3e",
        args.num_steps,
        drifts[-1]["dist_gd_to_orig_flow"],
        drifts[-1]["dist_gd_to_mod_flow"],
        drifts[-1]["dist_gd_from_theta0"],
    )
    LOGGER.info(
        "  ratio drift_orig / drift_theta0 = %.3f (should grow with eta if GD leaves pure-flow path)",
        drifts[-1]["dist_gd_to_orig_flow"]
        / max(drifts[-1]["dist_gd_from_theta0"], 1e-30),
    )
    LOGGER.info(
        "  ratio drift_mod  / drift_orig   = %.3f (<<1 means modified flow tracks GD far better than original flow)",
        drifts[-1]["dist_gd_to_mod_flow"]
        / max(drifts[-1]["dist_gd_to_orig_flow"], 1e-30),
    )

    payload = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "num_params": num_params,
        "eta": args.eta,
        "num_steps": args.num_steps,
        "drifts": drifts,
    }

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Save trajectories so plotting can do richer analyses.
        torch.save(
            {
                **payload,
                "theta_gd": torch.stack(theta_gd_snapshots, dim=0),
                "theta_orig_flow": torch.stack(theta_orig_snapshots, dim=0),
                "theta_mod_flow": torch.stack(theta_mod_snapshots, dim=0),
            },
            args.save,
        )
        with args.save.with_suffix(".json").open("w") as f:
            json.dump(payload, f, indent=2)
        LOGGER.info("Saved results to %s", args.save)


if __name__ == "__main__":
    main()
