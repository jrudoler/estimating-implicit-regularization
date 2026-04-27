#!/usr/bin/env python3
"""Local sweep driver for barrett_igr_trajectory.

Runs a grid of (eta, p, seed, flow_k) for the synthetic flow-ref mode,
plus (eta, batch_size, seed) for SGD mode. Aggregates results into a
single .pt file consumed by analysis/barrett_igr_plot.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAJ_SCRIPT = REPO_ROOT / "experiments" / "barrett_igr_trajectory.py"


def run_one(tmp_dir: Path, **overrides) -> dict:
    save_path = tmp_dir / "run.pt"
    if save_path.exists():
        save_path.unlink()
    cmd = [
        sys.executable,
        str(TRAJ_SCRIPT),
        "--save",
        str(save_path),
        "--device",
        "cpu",
        "--double-precision",
        "--log-level",
        "WARNING",
    ]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        cmd.extend([flag, str(v)])
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    payload = torch.load(save_path, weights_only=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "barrett_igr_sweep.pt")
    parser.add_argument(
        "--etas",
        type=str,
        default="1e-3,3e-3,1e-2,3e-2",
        help="Comma-separated list of eta values.",
    )
    parser.add_argument("--ps", type=str, default="5,10,20")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--flow-k", type=int, default=100)
    parser.add_argument("--batch-sizes", type=str, default="32,128")
    parser.add_argument(
        "--sgd-n-steps",
        type=int,
        default=400,
        help="Number of steps for SGD runs (larger for noise averaging).",
    )
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-sgd", action="store_true")
    parser.add_argument(
        "--mnist-etas",
        type=str,
        default="3e-4,1e-3,3e-3",
        help="Eta grid for MNIST sweep. Empty string to skip MNIST.",
    )
    parser.add_argument(
        "--mnist-archs",
        type=str,
        default="linear,tanh,relu",
        help="Comma-separated arch keys for MNIST sweep: linear|tanh|relu|gelu.",
    )
    parser.add_argument("--mnist-num-steps", type=int, default=30)
    parser.add_argument("--mnist-flow-k", type=int, default=10)
    parser.add_argument("--mnist-seeds", type=str, default="0,1")
    parser.add_argument("--mnist-samples", type=int, default=2000)
    parser.add_argument("--mnist-hidden", type=str, default="64,32")
    args = parser.parse_args()

    etas = [float(x) for x in args.etas.split(",")]
    ps = [int(x) for x in args.ps.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    tmp_dir = args.out.parent / "_barrett_igr_sweep_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_runs = []

    if not args.skip_flow:
        for eta, p, seed in product(etas, ps, seeds):
            print(f"[flow_ref] eta={eta} p={p} seed={seed}")
            payload = run_one(
                tmp_dir,
                dataset="synthetic",
                mode="flow_ref",
                eta=eta,
                p_features=p,
                num_steps=args.num_steps,
                flow_k=args.flow_k,
                seed=seed,
            )
            payload["summary"]["sweep_axes"] = {
                "mode": "flow_ref",
                "eta": eta,
                "p": p,
                "seed": seed,
                "flow_k": args.flow_k,
            }
            all_runs.append(payload)

    if not args.skip_sgd:
        # SGD sweep uses a fixed p to limit combinatorial explosion.
        for eta, bs, seed in product(etas, batch_sizes, seeds):
            p = ps[0]
            print(f"[sgd] eta={eta} B={bs} p={p} seed={seed}")
            payload = run_one(
                tmp_dir,
                dataset="synthetic",
                mode="sgd",
                eta=eta,
                p_features=p,
                num_steps=args.sgd_n_steps,
                batch_size=bs,
                seed=seed,
            )
            payload["summary"]["sweep_axes"] = {
                "mode": "sgd",
                "eta": eta,
                "p": p,
                "batch_size": bs,
                "seed": seed,
            }
            all_runs.append(payload)

    if args.mnist_etas:
        mnist_etas = [float(x) for x in args.mnist_etas.split(",")]
        archs = [x.strip() for x in args.mnist_archs.split(",") if x.strip()]
        mnist_seeds = [int(x) for x in args.mnist_seeds.split(",")]
        hidden = [int(x) for x in args.mnist_hidden.split(",")]
        for arch, eta, seed in product(archs, mnist_etas, mnist_seeds):
            print(f"[mnist] arch={arch} eta={eta} seed={seed}")
            overrides: dict = {
                "dataset": "mnist",
                "mode": "flow_ref",
                "eta": eta,
                "num_steps": args.mnist_num_steps,
                "flow_k": args.mnist_flow_k,
                "seed": seed,
                "mnist_max_samples": args.mnist_samples,
                "mnist_root": "data",
            }
            # Inject architecture: "linear" -> empty hidden_dims; else use hidden + activation.
            arch_note = arch
            if arch == "linear":
                cmd_extras = ["--hidden-dims"]  # nargs=* with no values => empty list
            else:
                cmd_extras = ["--hidden-dims", *[str(h) for h in hidden], "--activation", arch]
            # run_one doesn't support nargs='*' directly, so build command manually here.
            tmp_save = tmp_dir / "run.pt"
            if tmp_save.exists():
                tmp_save.unlink()
            cmd = [
                sys.executable,
                str(TRAJ_SCRIPT),
                "--save",
                str(tmp_save),
                "--device",
                "cpu",
                "--double-precision",
                "--log-level",
                "WARNING",
            ]
            for k, v in overrides.items():
                cmd.extend(["--" + k.replace("_", "-"), str(v)])
            cmd.extend(cmd_extras)
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)
            payload = torch.load(tmp_save, weights_only=False)
            payload["summary"]["sweep_axes"] = {
                "mode": "flow_ref",
                "dataset": "mnist",
                "arch": arch_note,
                "eta": eta,
                "seed": seed,
                "flow_k": args.mnist_flow_k,
                "num_steps": args.mnist_num_steps,
            }
            all_runs.append(payload)

    torch.save({"runs": all_runs}, args.out)
    print(f"Saved {len(all_runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
