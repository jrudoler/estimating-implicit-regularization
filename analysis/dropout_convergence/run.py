"""Continue a verified saved endpoint and test practical dropout stationarity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import lightning
import torch

from core.data import InMemoryBatches, MNISTLightningDataModule
from core.models import DeepReLUClassifier
from core.dropout_convergence import assess_convergence, continuation_lr, train_epoch
from core.scalar_estimation import dropout_control, file_sha256, model_sha256, provenance


def atomic_json(path, content):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False))
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=4000)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.check_every < 1 or args.lr <= 0:
        raise ValueError("Training limits and learning rate must be positive")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = json.loads(args.source_result.read_text())
    cfg = source["config"]
    endpoint = next(r for r in source["trajectory"] if r["fraction"] == 1.0)
    expected = endpoint["provenance"]
    if file_sha256(args.checkpoint) != expected["checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash mismatch")
    model = DeepReLUClassifier.load_from_checkpoint(str(args.checkpoint), map_location=args.device).to(args.device)
    if model_sha256(model) != expected["model_sha256"]:
        raise ValueError("Source model hash mismatch")
    if model.hparams.batchnorm or model.hparams.l2_lambda != 0:
        raise ValueError("Continuation requires dropout-only regularization")
    if any(model.hparams[k] != cfg[k] for k in ("depth", "width", "dropout")):
        raise ValueError("Source configuration mismatch")
    dm = MNISTLightningDataModule(root=args.data_root, batch_size=2048, num_workers=0)
    dm.setup("fit")
    split_hash = hashlib.sha256(torch.tensor(dm._train_dataset.indices, dtype=torch.int64).numpy().tobytes()).hexdigest()
    if split_hash != expected["dataset"]["split_indices_sha256"]:
        raise ValueError("Training split changed")
    batches = InMemoryBatches(dm._train_dataset, args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=model.hparams.momentum)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    optimizer.load_state_dict(payload["optimizer_states"][0])
    if any(g.get("weight_decay", 0) != 0 for g in optimizer.param_groups):
        raise ValueError("Unexpected optimizer weight decay")
    protocol = {"max_epochs": args.max_epochs, "check_every": args.check_every,
                "initial_lr": args.lr, "lr_milestones": [1000, 2500, 3500], "lr_factor": .2,
                "batch_size": 2048, "screening_draws": 32, "refined_draws": [128, 512],
                "confirmation_draws": 512, "consecutive_passes": 2,
                "norm_absolute": 1e-4, "norm_relative": .1, "norm_reduction": .1,
                "coefficient_absolute": 1e-7, "coefficient_relative": .05,
                "optimizer": "SGD; restored momentum; explicitly restarted LR schedule",
                "rng": "new seeded training stream; diagnostics use independent forked streams"}
    signature = {"source_checkpoint_sha256": expected["checkpoint_sha256"],
                 "source_result_sha256": file_sha256(args.source_result), "config": cfg,
                 "protocol": protocol, "runner_sha256": file_sha256(__file__),
                 "methods_sha256": file_sha256(Path(__file__).parents[2] / "src/core/dropout_convergence.py"),
                 "gradient_sha256": file_sha256(Path(__file__).parents[2] / "src/core/scalar_estimation.py")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir / "latest.ckpt"
    seed = cfg["seed"] + 700000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    history, start, consecutive = [], 0, 0
    status = "running"
    baseline_norm = endpoint["control"]["train_gradient_norm_naive"]
    if latest.exists():
        resumed = torch.load(latest, map_location=args.device, weights_only=False)
        if resumed["signature"] != signature:
            raise ValueError("Existing continuation has a different protocol or source")
        model.load_state_dict(resumed["state_dict"])
        optimizer.load_state_dict(resumed["optimizer_states"][0])
        torch.set_rng_state(resumed["cpu_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in resumed["cuda_rng"]])
        start, history = resumed["additional_epochs"], resumed["history"]
        consecutive, baseline_norm = resumed["consecutive"], resumed["baseline_norm"]
        status = resumed.get("status", "running")
    begin = time.monotonic()

    def save(epoch, path):
        state = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                 "pytorch-lightning_version": lightning.__version__,
                 "epoch": expected["epoch"] + epoch,
                 "optimizer_states": [optimizer.state_dict()], "additional_epochs": epoch,
                 "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 "signature": signature, "history": history, "consecutive": consecutive,
                 "baseline_norm": baseline_norm, "status": status}
        pending = path.with_suffix(".tmp")
        torch.save(state, pending)
        pending.replace(path)

    def diagnose(epoch, draws, offset=0):
        control = dropout_control(model, batches, draws=draws, seed=cfg["seed"] + 1000000 + epoch * 10000 + offset)
        assessment = assess_convergence(control, baseline_norm=baseline_norm, dropout=cfg["dropout"])
        return {"additional_epochs": epoch, "draws": draws,
                "control": control, "assessment": assessment}

    for epoch in range(start, args.max_epochs + 1):
        if status.startswith("converged"):
            break
        if epoch > start:
            for group in optimizer.param_groups:
                group["lr"] = continuation_lr(epoch - 1, initial=args.lr)
            loss = train_epoch(model, batches, optimizer)
            if epoch % 50 == 0:
                print(json.dumps({"epoch": epoch, "train_loss": loss, "lr": optimizer.param_groups[0]["lr"],
                                  "elapsed_seconds": time.monotonic() - begin}), flush=True)
                save(epoch, latest)
        should_check = epoch == 0 or epoch % args.check_every == 0 or epoch == args.max_epochs
        if not should_check or (history and history[-1]["additional_epochs"] == epoch):
            continue
        record = diagnose(epoch, 32)
        for draws in (128, 512):
            if record["assessment"]["passes"] or not record["assessment"]["point_estimates_near_tolerance"]:
                break
            record = diagnose(epoch, draws, offset=draws)
        consecutive = consecutive + 1 if record["assessment"]["passes"] else 0
        if consecutive >= 2:
            record["confirmation"] = diagnose(epoch, 512, offset=5000)
            if record["confirmation"]["assessment"]["passes"]:
                status = "converged_to_declared_tolerance"
            else:
                consecutive = 0
        checkpoint = args.output_dir / f"epoch_{epoch:05d}.ckpt"
        record["checkpoint"] = str(checkpoint)
        record["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(record)
        save(epoch, checkpoint)
        record["provenance"] = provenance(model, checkpoint=checkpoint, epoch=expected["epoch"] + epoch,
                                          endpoint_kind="continued_dropout_training", dataset=dm._train_dataset)
        save(epoch, latest)
        atomic_json(args.output_dir / "progress.json", {**signature, "status": status, "trajectory": history})
        print(json.dumps({"epoch": epoch, "diagnostic": record["assessment"], "draws": record["draws"], "status": status}), flush=True)
        if status.startswith("converged"):
            break
    if status == "running":
        status = "budget_exhausted"
    atomic_json(args.output_dir / "result.json", {**signature, "status": status, "baseline_norm": baseline_norm,
                "trajectory": history, "elapsed_seconds": time.monotonic() - begin,
                "interpretation": "Approximate MC bounds and a practical stationarity tolerance; positive ridge fit is not assumed."})


if __name__ == "__main__":
    main()
