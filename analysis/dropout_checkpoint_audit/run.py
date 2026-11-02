"""Fixed-checkpoint controls; replay only explicitly specified missing trajectories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.data import MNISTLightningDataModule, InMemoryBatches
from core.models import DeepReLUClassifier
from core.scalar_estimation import dropout_control, full_batch_loss_grad, parameters64, provenance, scalar_projection


def optimizer_audit(model, loader, *, epochs=2000, seed=123):
    """Matched update budgets on cached fixed batches; not a replay of old shuffles.

Sufficient statistics give exactly the scalar quadratic objective for each
cached batch. All one-factor ablations share the same data and initialization.
The scheduler observes the final batch each epoch, like the legacy step metric.
"""
    basis = 2 * parameters64(model)
    target = -full_batch_loss_grad(model, loader)
    fit = scalar_projection(basis, target)
    reference = float(torch.linalg.lstsq(basis[:, None], target[:, None]).solution[0, 0])
    reference = max(0., reference)
    denom = float(basis @ basis)
    stats = []
    for batch in loader:
        g = -full_batch_loss_grad(model, [batch])
        stats.append((float(basis @ g), float(g @ g)))
    generator = torch.Generator().manual_seed(seed)
    beta0 = float(torch.randn(1, generator=generator) * .1)
    variants = {
        "baseline_cached_batches": {},
        "sum_reduction": {"reduction": "sum"},
        "epsilon_1e16": {"eps": 1e-16},
        "direct_coefficient": {"exponential": False},
        "no_scheduler": {"scheduler": False},
        "float64": {"dtype": torch.float64},
        "full_batch_same_updates": {"full_batch": True},
    }
    output = {}
    for name, overrides in variants.items():
        cfg = dict(reduction="mean", eps=1e-8, exponential=True, scheduler=True,
                   dtype=torch.float32, full_batch=False)
        cfg.update(overrides)
        par = torch.nn.Parameter(torch.tensor(beta0 if cfg["exponential"] else float(torch.exp(torch.tensor(beta0))), dtype=cfg["dtype"]))
        opt = torch.optim.Adam([par], lr=.05, eps=cfg["eps"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=.5)
        divisor = basis.numel() if cfg["reduction"] == "mean" else 1
        selected = [(float(basis @ target), float(target @ target))] * len(stats) if cfg["full_batch"] else stats
        trace = []
        for epoch in range(epochs):
            for dot, norm_sq in selected:
                scale = par.exp() if cfg["exponential"] else par
                loss = (denom * scale.square() - 2 * dot * scale + norm_sq) / divisor
                opt.zero_grad()
                loss.backward()
                opt.step()
                if not cfg["exponential"]:
                    with torch.no_grad():
                        par.clamp_(min=0.)
            if cfg["scheduler"]:
                scheduler.step(float(loss.detach()))
            if epoch in (0, 59, 199, epochs - 1):
                trace.append({"epoch": epoch + 1, "coefficient": float(par.detach().exp() if cfg["exponential"] else par.detach()),
                              "lr": opt.param_groups[0]["lr"]})
        scale = float(par.detach().exp() if cfg["exponential"] else par.detach())
        gap = float(((scale * basis - target)**2).sum()) - fit["sse_clamped"]
        output[name] = {"coefficient": scale, "absolute_error": abs(scale - reference),
                        "full_dataset_sse_gap": gap, "updates": epochs * len(stats), "trace": trace}
    return {"reference_nonnegative_lstsq": reference, "closed_form": fit,
            "comparison": "same checkpoint, fixed cached batches, equal update budgets; not historical shuffle replay",
            "variants": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--mask-draws", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--optimizer-epochs", type=int, default=2000)
    args = p.parse_args()
    torch.set_num_threads(2)
    spec = json.loads(args.spec.read_text())
    cfg = spec["config"]
    if not spec.get("checkpoints"):
        if spec.get("endpoint_kind") != "explicit_replacement_replay":
            raise ValueError("Missing checkpoints: replacement training requires an explicit replay specification")
        replay_record = Path(spec["replay_record"])
        if not replay_record.exists() and "/data/generated/" in str(replay_record):
            replay_record = ROOT / "data/generated" / str(replay_record).split("/data/generated/", 1)[1]
        if not replay_record.is_file():
            raise FileNotFoundError(f"Run the dropout_checkpoint_replay workflow rule first: {replay_record}")
        replay = json.loads(replay_record.read_text())
        checkpoints = replay["trajectory"]
    else:
        checkpoints = spec["checkpoints"]
    dm = MNISTLightningDataModule(root=args.data_root, batch_size=2048, num_workers=0)
    dm.setup("fit")
    loader = InMemoryBatches(dm._train_dataset, args.device)
    records = []
    for checkpoint in checkpoints:
        path = Path(checkpoint["checkpoint"])
        if not path.exists() and "/data/generated/" in str(path):
            path = ROOT / "data/generated" / str(path).split("/data/generated/", 1)[1]
        model = DeepReLUClassifier.load_from_checkpoint(str(path), map_location=args.device).to(args.device).eval()
        for name in ("depth", "width", "dropout"):
            if model.hparams[name] != cfg[name]:
                raise ValueError(f"Checkpoint/config mismatch for {name}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        record = {"fraction": checkpoint["fraction"],
                  "provenance": provenance(model, checkpoint=path, epoch=payload.get("epoch"),
                                           endpoint_kind=spec["endpoint_kind"], dataset=dm._train_dataset),
                  "control": dropout_control(model, loader, draws=args.mask_draws, seed=cfg["seed"])}
        if cfg["seed"] == 123 and checkpoint["fraction"] == 1.0:
            record["optimizer_audit"] = optimizer_audit(model, loader, epochs=args.optimizer_epochs, seed=cfg["seed"])
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 2, "config": cfg, "trajectory": records}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
