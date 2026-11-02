"""Confirm iterative ridge fixes on identical saved checkpoints.

Collect caches the exact scalar quadratic for each deterministic training batch
on Locust. Fit replays 54,000 Adam updates using those statistics, then measures
full-dataset objective gaps against float64 closed form. No network is trained.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.dropout_checkpoint_audit.run import InMemoryBatches
from core.data import MNISTLightningDataModule
from core.dropout_grid import SEEDS
from core.models import DeepReLUClassifier
from core.scalar_estimation import file_sha256, full_batch_loss_grad, model_sha256, parameters64, scalar_projection

RATES = (0.0, 0.3, 0.5)
SEED = 123
STEPS_PER_EPOCH = 27
EPOCHS = 2000


def resolve_checkpoint(path: str) -> Path:
    source = Path(path)
    if not source.exists() and "/data/generated/" in path:
        source = ROOT / "data/generated" / path.split("/data/generated/", 1)[1]
    if not source.is_file():
        raise FileNotFoundError(path)
    return source


def collect(results_dir: Path, data_root: Path, device: str) -> dict:
    torch.set_num_threads(2)
    dm = MNISTLightningDataModule(root=data_root, batch_size=2048, num_workers=0)
    dm.setup("fit")
    loader = InMemoryBatches(dm._train_dataset, device)
    records = []
    for rate, seed in product(RATES, SEEDS):
        source = results_dir / f"d3_w256_do{rate}_s{seed}.json"
        result = json.loads(source.read_text())
        if (result["config"]["dropout"], result["config"]["seed"]) != (rate, seed):
            raise ValueError(f"Result configuration mismatch: {source}")
        checkpoint_record = next(r for r in result["trajectory"] if r["fraction"] == 1.0)
        evidence = checkpoint_record["provenance"]
        checkpoint = resolve_checkpoint(evidence["checkpoint"])
        if file_sha256(checkpoint) != evidence["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash changed: {checkpoint}")
        model = DeepReLUClassifier.load_from_checkpoint(str(checkpoint), map_location=device).to(device).eval()
        if model_sha256(model) != evidence["model_sha256"]:
            raise ValueError(f"Model state hash changed: {checkpoint}")
        if (model.hparams["depth"], model.hparams["width"], model.hparams["dropout"]) != (3, 256, rate):
            raise ValueError(f"Checkpoint configuration mismatch: {checkpoint}")
        basis = 2 * parameters64(model)
        target = -full_batch_loss_grad(model, loader)
        fit = scalar_projection(basis, target)
        denom = float(basis @ basis)
        dots = []
        norms = []
        sizes = []
        for x, y in loader:
            gradient = -full_batch_loss_grad(model, [(x, y)])
            dots.append(float(basis @ gradient))
            norms.append(float(gradient @ gradient))
            sizes.append(len(y))
        if len(dots) != STEPS_PER_EPOCH or sum(sizes) != 54000:
            raise ValueError("Unexpected MNIST training split or batch count")
        existing = checkpoint_record.get("optimizer_audit")
        if existing and abs(existing["reference_nonnegative_lstsq"] - fit["estimated_ridge"]) > 1e-12:
            raise ValueError("Saved reference does not match the recovered checkpoint")
        records.append({
            "dropout": rate,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "results_sha256": file_sha256(source),
            "parameter_count": basis.numel(),
            "basis_sq_norm": denom,
            "full_dataset_basis_dot_target": float(basis @ target),
            "full_dataset_target_sq_norm": float(target @ target),
            "closed_form": fit["estimated_ridge"],
            "existing_direct_with_plateau": existing["variants"]["direct_coefficient"] if existing else None,
            "cached_batches": [
                {"basis_dot_target": dot, "target_sq_norm": norm, "sample_count": size}
                for dot, norm, size in zip(dots, norms, sizes)
            ],
        })
    return {
        "schema_version": 1,
        "design": "All 30 final checkpoints; fixed 27 minibatches; 2000 epochs; 54000 Adam updates.",
        "source_sha256": file_sha256(__file__),
        "records": records,
    }


def fit(data: dict) -> dict:
    """Vectorized float32 Adam, with one independent state per condition/variant."""
    records = data["records"]
    if [(r["dropout"], r["seed"]) for r in records] != [
        (rate, seed) for rate in RATES for seed in SEEDS
    ]:
        raise ValueError("Expected all thirty final checkpoints")
    if any(len(r["cached_batches"]) != STEPS_PER_EPOCH for r in records):
        raise ValueError("Incomplete cached batches")
    # This is the same random beta initialization used in the prior audit.
    generator = torch.Generator().manual_seed(SEED)
    initial = float(torch.exp(torch.randn(1, generator=generator) * .1))
    variants = {
        "direct_plateau_original": {"kind": "plateau", "initial_lr": .05, "full_batch": False},
        "direct_no_scheduler_lr_0.05": {"kind": "constant", "initial_lr": .05, "full_batch": False},
        "direct_no_scheduler_lr_0.001": {"kind": "constant", "initial_lr": .001, "full_batch": False},
        "direct_no_scheduler_lr_0.0001": {"kind": "constant", "initial_lr": .0001, "full_batch": False},
        "direct_milestones": {"kind": "milestones", "initial_lr": .05, "full_batch": False},
        "direct_sample_weighted_milestones": {"kind": "milestones", "initial_lr": .05,
                                               "full_batch": False, "sample_weighted": True},
        "direct_full_dataset_milestones": {"kind": "milestones", "initial_lr": .05, "full_batch": True},
    }
    positions = [(i, name) for i in range(len(records)) for name in variants]
    count = len(positions)
    value = torch.full((count,), initial, dtype=torch.float32)
    moment = torch.zeros_like(value)
    velocity = torch.zeros_like(value)
    learning_rate = torch.tensor(
        [variants[name]["initial_lr"] for _, name in positions], dtype=torch.float32
    )
    denom = torch.tensor([records[i]["basis_sq_norm"] for i, _ in positions], dtype=torch.float32)
    divisor = torch.tensor([records[i]["parameter_count"] for i, _ in positions], dtype=torch.float32)
    full_dot = [r["full_dataset_basis_dot_target"] for r in records]
    full_norm = [r["full_dataset_target_sq_norm"] for r in records]
    batch_dots = torch.tensor([
        [full_dot[i] if variants[name]["full_batch"] else records[i]["cached_batches"][batch]["basis_dot_target"]
         for i, name in positions]
        for batch in range(STEPS_PER_EPOCH)
    ], dtype=torch.float32)
    batch_norms = torch.tensor([
        [full_norm[i] if variants[name]["full_batch"] else records[i]["cached_batches"][batch]["target_sq_norm"]
         for i, name in positions]
        for batch in range(STEPS_PER_EPOCH)
    ], dtype=torch.float32)
    batch_weights = torch.tensor([
        [STEPS_PER_EPOCH * records[i]["cached_batches"][batch]["sample_count"] / 54000
         if variants[name].get("sample_weighted") else 1.0
         for i, name in positions]
        for batch in range(STEPS_PER_EPOCH)
    ], dtype=torch.float32)
    plateau = torch.tensor([variants[name]["kind"] == "plateau" for _, name in positions], dtype=torch.bool)
    milestones = torch.tensor([variants[name]["kind"] == "milestones" for _, name in positions], dtype=torch.bool)
    best = torch.full_like(value, float("inf"))
    bad_epochs = torch.zeros(count, dtype=torch.int32)
    trace = [[] for _ in positions]
    beta1, beta2, epsilon = .9, .999, 1e-8
    step = 0
    for epoch in range(EPOCHS):
        for batch in range(STEPS_PER_EPOCH):
            step += 1
            dot = batch_dots[batch]
            if batch == STEPS_PER_EPOCH - 1:
                final_batch_loss = (denom * value.square() - 2 * dot * value + batch_norms[batch]) / divisor
            gradient = batch_weights[batch] * (2 * denom * value - 2 * dot) / divisor
            moment = beta1 * moment + (1 - beta1) * gradient
            velocity = beta2 * velocity + (1 - beta2) * gradient.square()
            corrected_moment = moment / (1 - beta1**step)
            corrected_velocity = velocity / (1 - beta2**step)
            value = torch.clamp(value - learning_rate * corrected_moment /
                                (corrected_velocity.sqrt() + epsilon), min=0)
        # Match ReduceLROnPlateau(mode=min, patience=10, factor=.5) observing
        # the final minibatch loss, as in the old estimator.
        improved = final_batch_loss < best * (1 - 1e-4)
        best = torch.where(improved, final_batch_loss, best)
        bad_epochs = torch.where(improved, 0, bad_epochs + 1)
        reduce = plateau & (bad_epochs > 10)
        candidate = learning_rate * .5
        learning_rate = torch.where(reduce & ((learning_rate - candidate) > 1e-8),
                                    candidate, learning_rate)
        bad_epochs = torch.where(reduce, 0, bad_epochs)
        if epoch + 1 in (25, 100, 400, 1000, 1500):
            learning_rate = torch.where(milestones, learning_rate * .1, learning_rate)
        if epoch + 1 in (1, 60, 200, EPOCHS):
            for j in range(count):
                trace[j].append({"epoch": epoch + 1, "coefficient": float(value[j]),
                                 "learning_rate": float(learning_rate[j])})
    output = []
    for i, record in enumerate(records):
        exact = record["closed_form"]
        minibatch_optimum = max(0., sum(b["basis_dot_target"] for b in record["cached_batches"]) /
                                STEPS_PER_EPOCH / record["basis_sq_norm"])
        row = {"dropout": record["dropout"], "seed": record["seed"],
               "checkpoint_sha256": record["checkpoint_sha256"],
               "closed_form_full_dataset": exact,
               "fixed_equal_batch_optimum": minibatch_optimum, "variants": {}}
        for j, (source_i, name) in enumerate(positions):
            if source_i != i:
                continue
            estimate = float(value[j])
            row["variants"][name] = {
                "coefficient": estimate,
                "relative_error_to_closed_form": abs(estimate - exact) / exact if exact else None,
                "relative_error_to_fixed_batch_optimum": abs(estimate - minibatch_optimum) /
                                                         minibatch_optimum if minibatch_optimum else None,
                "full_dataset_sse_gap": record["basis_sq_norm"] * (estimate - exact)**2,
                "updates": EPOCHS * STEPS_PER_EPOCH,
                "trace": trace[j],
            }
        old = record["existing_direct_with_plateau"]
        replay = row["variants"]["direct_plateau_original"]["coefficient"]
        row["plateau_replay_relative_error"] = abs(replay - old["coefficient"]) / old["coefficient"] if old else None
        output.append(row)
    return {
        "schema_version": 1,
        "source_stats_sha256": None,  # Set by CLI after loading the input file.
        "method": "Float32 direct coefficient with nonnegative projection; independent vectorized Adam states.",
        "constraint": "lambda>=0",
        "scheduler_milestones": [25, 100, 400, 1000, 1500],
        "scheduler_factor": .1,
        "records": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--results-dir", type=Path, required=True)
    collect_parser.add_argument("--data-root", type=Path, required=True)
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--output", type=Path, required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--stats", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.results_dir, args.data_root, args.device)
    else:
        result = fit(json.loads(args.stats.read_text()))
        result["source_stats_sha256"] = file_sha256(args.stats)
        if any(r["plateau_replay_relative_error"] is not None and
               r["plateau_replay_relative_error"] > .05 for r in result["records"]):
            raise ValueError("Adam replay disagrees with the previously recorded plateau audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
