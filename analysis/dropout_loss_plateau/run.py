"""Train one Figure-5 model with loss-only stopping; optionally audit extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import lightning
import torch

from core.data import InMemoryBatches, MNISTLightningDataModule
from core.dropout_continuation import atomic_json, restore_rng
from core.dropout_convergence import train_epoch
from core.loss_plateau import LossPlateau, coefficient_stability
from core.models import DeepReLUClassifier
from core.scalar_estimation import dropout_control, file_sha256, provenance, ridge_estimate

LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=100000)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--lr-patience", type=int, default=500)
    parser.add_argument("--min-lr", type=float, default=.0005)
    parser.add_argument("--relative-improvement", type=float, default=.001)
    parser.add_argument("--absolute-improvement", type=float, default=0.)
    parser.add_argument("--study-name", default="dropout_loss_plateau")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-min-epochs", type=int, default=5000)
    parser.add_argument("--audit-draws", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.max_epochs < 1 or args.checkpoint_every < 1 or args.audit_min_epochs < 1:
        raise ValueError("Epoch limits must be positive")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    dm = MNISTLightningDataModule(root=args.data_root, batch_size=2048, num_workers=0)
    dm.setup("fit")
    batches = InMemoryBatches(dm._train_dataset, args.device)
    cfg = {"depth": args.depth, "width": args.width, "dropout": args.dropout,
           "seed": args.seed, "batchnorm": False, "batch_size": 2048}
    model = DeepReLUClassifier(input_dim=784, num_classes=10, depth=args.depth,
                              width=args.width, dropout=args.dropout, batchnorm=False,
                              l2_lambda=0., lr=.05, momentum=.9).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9)
    stopper = LossPlateau(window=args.window, patience=args.patience,
                          lr_patience=args.lr_patience, min_lr=args.min_lr,
                          relative_improvement=args.relative_improvement,
                          absolute_improvement=args.absolute_improvement)
    root = Path(__file__).resolve().parents[2]
    protocol = {"max_epochs": args.max_epochs, "stopping": stopper.state_dict(),
                "initial_lr": .05, "momentum": .9, "weight_decay": 0.,
                "initialization": "fresh seeded DeepReLUClassifier; not a historical weight replay",
                "loss": "sample-weighted, dropout-on training CE; nonoverlapping epoch blocks",
                "checkpoint_every": args.checkpoint_every}
    signature = {"study": args.study_name, "config": cfg, "protocol": protocol,
                 "audit": {"enabled": args.audit, "extra_epochs": "max(stop_epoch, audit_min_epochs)",
                           "audit_min_epochs": args.audit_min_epochs, "draws": args.audit_draws},
                 "split_indices_sha256": hashlib.sha256(torch.tensor(dm._train_dataset.indices, dtype=torch.int64).numpy().tobytes()).hexdigest(),
                 "code_sha256": {p: file_sha256(root / p) for p in (
                     "analysis/dropout_loss_plateau/run.py", "src/core/loss_plateau.py",
                     "src/core/dropout_convergence.py", "src/core/dropout_continuation.py",
                     "src/core/models.py", "src/core/data.py", "src/core/scalar_estimation.py", "uv.lock")}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path, latest = args.output_dir / "result.json", args.output_dir / "latest.ckpt"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["signature"] != signature:
            raise ValueError("Refusing to overwrite a different experiment")
        LOG.info("Preserving completed result: %s", result["status"])
        return
    epoch, previous_seconds = 0, 0.
    history: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    endpoint: dict[str, Any] | None = None
    if latest.exists():
        saved = torch.load(latest, map_location=args.device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Recovery signature mismatch")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_states"][0])
        stopper = LossPlateau(**saved["stopper"])
        epoch, previous_seconds = saved["completed_epochs"], saved["elapsed_seconds"]
        history, trajectory, endpoint = saved["history"], saved["trajectory"], saved["endpoint"]
        restore_rng(saved)
        LOG.info("Recovered epoch %d", epoch)
    begin = time.monotonic()

    def elapsed() -> float:
        return previous_seconds + time.monotonic() - begin

    def save(path: Path) -> None:
        payload = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                   "pytorch-lightning_version": lightning.__version__, "epoch": epoch - 1,
                   "completed_epochs": epoch, "optimizer_states": [optimizer.state_dict()],
                   "cpu_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   "signature": signature, "stopper": stopper.state_dict(),
                   "history": history, "trajectory": trajectory, "endpoint": endpoint,
                   "elapsed_seconds": elapsed()}
        pending = path.with_suffix(".tmp")
        torch.save(payload, pending)
        pending.replace(path)

    def measure(kind: str) -> dict[str, Any]:
        checkpoint = args.output_dir / f"{kind}_epoch_{epoch:06d}.ckpt"
        save(checkpoint)
        return {"epoch": epoch, "kind": kind, "ridge": ridge_estimate(model, batches),
                "lr": optimizer.param_groups[0]["lr"],
                "provenance": provenance(model, checkpoint=checkpoint, epoch=epoch,
                                         endpoint_kind=kind, dataset=dm._train_dataset)}

    if epoch == 0:
        trajectory.append(measure("initial"))
        save(latest)
    while endpoint is None and epoch < args.max_epochs:
        loss = train_epoch(model, batches, optimizer)
        epoch += 1
        block = stopper.update(loss, optimizer.param_groups[0]["lr"])
        if block is not None:
            history.append(block)
            for group in optimizer.param_groups:
                group["lr"] = block["next_lr"]
            atomic_json(args.output_dir / "progress.json", {"config": cfg, "epoch": epoch,
                        "phase": "training", "block": block, "elapsed_seconds": elapsed()})
            LOG.info("epoch=%d mean_loss=%.9g lr=%.5g action=%s elapsed=%.1fs", epoch,
                     block["mean_train_loss"], block["next_lr"], block["action"], elapsed())
        should_stop = block is not None and block["action"] == "stop"
        if should_stop or epoch == args.max_epochs:
            endpoint = measure("endpoint")
            endpoint["status"] = "loss_plateau" if should_stop else "budget_exhausted"
            trajectory.append(endpoint)
            if args.audit:
                endpoint["control"] = dropout_control(model, batches, draws=args.audit_draws,
                                                       seed=6000000000 + args.seed)
            save(latest)
        elif epoch % args.checkpoint_every == 0:
            trajectory.append(measure("trajectory"))
            save(latest)
    assert endpoint is not None
    extension = None
    if args.audit:
        target = endpoint["epoch"] + max(endpoint["epoch"], args.audit_min_epochs)
        # Fixed terminal LR; no further early stopping or coefficient-driven decisions.
        while epoch < target:
            loss = train_epoch(model, batches, optimizer)
            epoch += 1
            if epoch % args.window == 0:
                atomic_json(args.output_dir / "progress.json", {"config": cfg, "epoch": epoch,
                            "phase": "extension", "target_epoch": target, "elapsed_seconds": elapsed()})
                LOG.info("extension epoch=%d loss=%.9g elapsed=%.1fs", epoch, loss, elapsed())
            if epoch % args.checkpoint_every == 0 and epoch != target:
                trajectory.append(measure("extension"))
                save(latest)
        extension = measure("extended_endpoint")
        extension["control"] = dropout_control(model, batches, draws=args.audit_draws,
                                                seed=6100000000 + args.seed)
        extension["stability"] = coefficient_stability(endpoint["ridge"]["scale_star"],
                                                       extension["ridge"]["scale_star"])
        trajectory.append(extension)
    result = {"signature": signature, "config": cfg, "protocol": protocol,
              "status": endpoint["status"], "endpoint": endpoint, "extension": extension,
              "history": history, "trajectory": trajectory, "elapsed_seconds": elapsed(),
              "interpretation": "Loss-based stopping only; budget exhaustion is not convergence. Ridge is an endpoint gradient projection, not automatically dropout's causal penalty."}
    save(latest)
    atomic_json(result_path, result)
    atomic_json(args.output_dir / "progress.json", {"config": cfg, "epoch": epoch,
                "phase": "complete", "status": endpoint["status"], "elapsed_seconds": elapsed()})
    LOG.info("Completed epoch=%d status=%s extension_stability=%s", endpoint["epoch"],
             endpoint["status"], extension["stability"] if extension else None)


if __name__ == "__main__":
    main()
