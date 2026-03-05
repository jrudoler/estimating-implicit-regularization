#!/usr/bin/env python3
"""
Bootstrap / Subsampling Bias Recovery Experiment  (Phase 3, C2–C4)

Trains a network with known ground-truth regularization (Ridge + WeightCoherence),
then repeatedly estimates the bias coefficients on re-sampled subsets of the training
data.  Aggregates replicate estimates to evaluate whether resampling improves stability
or just adds variance.

Resampling modes
----------------
full        -- all replicates use the full training set (variance from optimiser noise only)
subsample   -- each replicate draws a random *without-replacement* subset of size
               floor(sample_fraction * N)
bootstrap   -- each replicate draws a random *with-replacement* sample of size
               floor(sample_fraction * N)

Metrics logged per replicate
-----------------------------
  bootstrap/replicate_{i}/ridge_lambda
  bootstrap/replicate_{i}/coherence_lambda
  bootstrap/replicate_{i}/loss_final

Aggregate metrics logged at the end
-------------------------------------
  bootstrap/ridge/mean, std, ci95, cv, sign_consistency
  bootstrap/coherence/mean, std, ci95, cv, sign_consistency
  bootstrap/ridge/rel_error  (of the aggregate mean vs ground truth)
  bootstrap/coherence/rel_error
  identifiability/gram_condition_number  (collinearity diagnostic)
  identifiability/cosine_similarity      (|cos(∇R_ridge, ∇R_coherence)|)
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler,
    random_split,
)
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision import datasets, transforms
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import RidgeBias, WeightCoherenceBias, JointBias
from core.estimators import (
    BiasWithCrossEntropyScheduled,
    BiasWithCrossEntropyNormalized,
    vector_to_parameter_views,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predictive model (same architecture as mixed_bias_recovery.py)
# ---------------------------------------------------------------------------

class DeepReLUClassifier(pl.LightningModule):
    """Deep ReLU MLP with configurable Ridge + WeightCoherence regularization."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        depth: int,
        width: int,
        ridge_lambda: float,
        coherence_lambda: float,
        lr: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.flatten(1))

    def _ridge(self) -> Tensor:
        if self.hparams.ridge_lambda <= 0:
            return torch.zeros((), device=self.device)
        flat = torch.cat([p.reshape(-1) for p in self.parameters()])
        return self.hparams.ridge_lambda * flat.pow(2).sum()

    def _coherence(self) -> Tensor:
        if self.hparams.coherence_lambda <= 0:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        count = 0
        for p in self.parameters():
            if p.ndim >= 2:
                W = p if p.ndim == 2 else p.reshape(p.shape[0], -1)
                gram = W.T @ W
                off = gram - torch.diag(gram.diag())
                fro2 = W.pow(2).sum()
                penalty = penalty + off.pow(2).sum() / (fro2.pow(2) + 1e-8)
                count += 1
        return self.hparams.coherence_lambda * (penalty / max(count, 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y) + self._ridge() + self._coherence()
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y) + self._ridge() + self._coherence()
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(name: str, root: Path, train: bool):
    tf = transforms.ToTensor()
    name = name.lower()
    if name == "mnist":
        return datasets.MNIST(root=root, train=train, download=True, transform=tf)
    if name == "fashionmnist":
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=tf)
    raise ValueError(f"Unknown dataset: {name}")


def make_resample_loader(
    base_dataset,
    mode: str,
    sample_fraction: float,
    seed: int,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Return a DataLoader for one replicate based on the resampling mode."""
    n_total = len(base_dataset)
    n_sample = max(1, int(math.floor(sample_fraction * n_total)))
    rng = torch.Generator().manual_seed(seed)

    if mode == "full":
        return DataLoader(base_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, generator=rng)

    if mode == "subsample":
        indices = torch.randperm(n_total, generator=rng)[:n_sample].tolist()
        return DataLoader(Subset(base_dataset, indices), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, generator=rng)

    if mode == "bootstrap":
        # With-replacement sample of the same size as n_sample
        weights = torch.ones(n_total)
        sampler = WeightedRandomSampler(
            weights, num_samples=n_sample, replacement=True, generator=rng
        )
        return DataLoader(base_dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers)

    raise ValueError(f"Unknown resample_mode: {mode!r}. Choose full | subsample | bootstrap")


# ---------------------------------------------------------------------------
# Identifiability diagnostics
# ---------------------------------------------------------------------------

def compute_identifiability_metrics(
    model: nn.Module,
    ridge_bias: RidgeBias,
    coherence_bias: WeightCoherenceBias,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute the pairwise collinearity of ridge vs coherence penalty gradients.

    Returns:
        gram_condition_number  -- κ(Ĝ^T Ĝ) where Ĝ = [ĝ_ridge | ĝ_coherence] (normalised cols)
        cosine_similarity      -- |cos(∇R_ridge, ∇R_coherence)|
    """
    flat = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .requires_grad_(True)
        .to(device)
    )
    struct = vector_to_parameter_views(flat, model)

    grads = []
    for bias_mod in (ridge_bias, coherence_bias):
        # Set scale to 1 to get pure penalty gradient
        if hasattr(bias_mod, "beta"):
            saved = bias_mod.beta.data.clone()
            bias_mod.beta.data.zero_()          # exp(0) = 1 for enforce_positive
        val = bias_mod(flat, struct)
        g = torch.autograd.grad(val, flat)[0].detach()
        if hasattr(bias_mod, "beta"):
            bias_mod.beta.data.copy_(saved)
        grads.append(g)

    g1, g2 = grads
    n1, n2 = g1.norm().clamp(min=1e-12), g2.norm().clamp(min=1e-12)
    cos_sim = (g1 @ g2 / (n1 * n2)).abs().item()

    # Gram matrix of normalised columns
    G = torch.stack([g1 / n1, g2 / n2], dim=1)  # (P, 2)
    gram = G.T @ G  # (2, 2)
    sv = torch.linalg.svdvals(gram)
    kappa = (sv.max() / sv.min().clamp(min=1e-12)).item()

    return {"gram_condition_number": kappa, "cosine_similarity": cos_sim}


# ---------------------------------------------------------------------------
# Per-replicate bias estimation
# ---------------------------------------------------------------------------

def estimate_one_replicate(
    model: nn.Module,
    loader: DataLoader,
    use_normalized: bool,
    bias_lr: float,
    bias_max_epochs: int,
    bias_patience: int,
    accelerator: str,
) -> Dict[str, float]:
    """Run one bias estimation replicate; return {"ridge": ..., "coherence": ...}."""
    ridge_bias = RidgeBias(enforce_positive=True, init_value=0.01)
    coherence_bias = WeightCoherenceBias(enforce_positive=True, init_value=0.01)
    joint_bias = JointBias([ridge_bias, coherence_bias])

    if use_normalized:
        estimator = BiasWithCrossEntropyNormalized(
            predictive_model=model,
            bias_model=joint_bias,
            bias_lr=bias_lr,
        )
    else:
        estimator = BiasWithCrossEntropyScheduled(
            predictive_model=model,
            bias_model=joint_bias,
            bias_lr=bias_lr,
        )

    trainer = pl.Trainer(
        max_epochs=bias_max_epochs,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,           # no per-replicate logging to keep W&B clean
        callbacks=[
            EarlyStopping(monitor="train_bias/loss", mode="min", patience=bias_patience),
        ],
        log_every_n_steps=50,
    )
    trainer.fit(estimator, train_dataloaders=loader)

    if use_normalized:
        lambdas = estimator.get_estimated_lambdas()
        return {
            "ridge": lambdas.get("ridge", 0.0),
            "coherence": lambdas.get("weight_coherence", 0.0),
        }
    else:
        params = joint_bias.get_bias_params()
        return {
            "ridge": params.get("ridge/scale", params.get("ridge", 0.0)),
            "coherence": params.get("weight_coherence/scale", params.get("weight_coherence", 0.0)),
        }


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _agg(values: List[float], gt: float) -> Dict[str, float]:
    """Compute mean/std/CI/CV/sign_consistency/rel_error for a list of estimates."""
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    cv = std / abs(mean) if abs(mean) > 1e-12 else float("inf")
    sign_consistency = float((arr > 0).mean()) if mean > 0 else float((arr < 0).mean())
    abs_err = abs(mean - gt)
    rel_err = abs_err / max(abs(gt), 1e-8)
    return dict(mean=mean, std=std, ci95=ci95, cv=cv,
                sign_consistency=sign_consistency, rel_error=rel_err)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap/subsampling stability evaluation for bias recovery"
    )
    # Dataset
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashionmnist"])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    # Architecture
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    # Ground truth
    p.add_argument("--gt-ridge-lambda", type=float, default=0.05)
    p.add_argument("--gt-coherence-lambda", type=float, default=0.0)
    # Model training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--val-fraction", type=float, default=0.1)
    # Bias estimation (single-run baseline)
    p.add_argument("--bias-lr", type=float, default=0.1)
    p.add_argument("--bias-max-epochs", type=int, default=2000)
    p.add_argument("--bias-patience", type=int, default=100)
    p.add_argument("--use-normalized", action="store_true", default=False)
    # Resampling
    p.add_argument("--resample-mode", default="full",
                   choices=["full", "subsample", "bootstrap"])
    p.add_argument("--n-replicates", type=int, default=10)
    p.add_argument("--sample-fraction", type=float, default=0.5,
                   help="Fraction of training data per replicate (ignored for mode=full)")
    p.add_argument("--resample-seed-base", type=int, default=0)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run = wandb.init(project="inductive-bias-bootstrap", job_type="bootstrap_bias_recovery")
    cfg = run.config

    args = parse_args()
    # W&B config overrides argparse defaults
    def g(key, default):
        return cfg.get(key, default)

    dataset_name    = g("dataset",            args.dataset)
    data_root       = Path(g("data_root",     str(args.data_root)))
    depth           = g("depth",              args.depth)
    width           = g("width",              args.width)
    gt_ridge        = g("gt_ridge_lambda",    args.gt_ridge_lambda)
    gt_coherence    = g("gt_coherence_lambda",args.gt_coherence_lambda)
    batch_size      = g("batch_size",         args.batch_size)
    lr              = g("lr",                 args.lr)
    max_epochs      = g("max_epochs",         args.max_epochs)
    patience        = g("patience",           args.patience)
    val_fraction    = g("val_fraction",       args.val_fraction)
    bias_lr         = g("bias_lr",            args.bias_lr)
    bias_max_epochs = g("bias_max_epochs",    args.bias_max_epochs)
    bias_patience   = g("bias_patience",      args.bias_patience)
    use_normalized  = g("use_normalized",     args.use_normalized)
    resample_mode   = g("resample_mode",      args.resample_mode)
    n_replicates    = g("n_replicates",       args.n_replicates)
    sample_fraction = g("sample_fraction",    args.sample_fraction)
    seed_base       = g("resample_seed_base", args.resample_seed_base)
    seed            = g("seed",               args.seed)

    torch.manual_seed(seed)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    LOGGER.info("Bootstrap Bias Recovery | mode=%s n_rep=%d frac=%.2f",
                resample_mode, n_replicates, sample_fraction)

    # ------------------------------------------------------------------
    # Phase 1: Train predictive model
    # ------------------------------------------------------------------
    train_ds = load_dataset(dataset_name, data_root, train=True)
    test_ds  = load_dataset(dataset_name, data_root, train=False)

    val_n   = int(len(train_ds) * val_fraction)
    train_n = len(train_ds) - val_n
    train_split, val_split = random_split(
        train_ds, [train_n, val_n], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_split,   batch_size=batch_size, shuffle=False, num_workers=4)

    input_dim   = {"mnist": 784, "fashionmnist": 784}[dataset_name.lower()]
    num_classes = 10

    model = DeepReLUClassifier(
        input_dim=input_dim, num_classes=num_classes,
        depth=depth, width=width,
        ridge_lambda=gt_ridge, coherence_lambda=gt_coherence, lr=lr,
    )

    model_logger = WandbLogger(
        project="inductive-bias-bootstrap",
        name=f"train-r{gt_ridge:.4f}-c{gt_coherence:.4f}-s{seed}",
        experiment=run, log_model=False,
    )
    model_trainer = pl.Trainer(
        max_epochs=max_epochs, logger=model_logger,
        accelerator=accelerator, devices=1, enable_progress_bar=False,
        callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=patience),
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True),
        ],
        log_every_n_steps=50,
    )
    model_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    LOGGER.info("Model training complete.")

    # ------------------------------------------------------------------
    # Identifiability diagnostics (before resampling loop)
    # ------------------------------------------------------------------
    device = next(model.parameters()).device
    ridge_probe    = RidgeBias(enforce_positive=True, init_value=0.01).to(device)
    coherence_probe = WeightCoherenceBias(enforce_positive=True, init_value=0.01).to(device)

    ident_metrics = compute_identifiability_metrics(model, ridge_probe, coherence_probe, device)
    LOGGER.info("Identifiability: kappa=%.2f  cos_sim=%.4f",
                ident_metrics["gram_condition_number"], ident_metrics["cosine_similarity"])
    wandb.log({f"identifiability/{k}": v for k, v in ident_metrics.items()})

    # ------------------------------------------------------------------
    # Phase 2: Resampling loop
    # ------------------------------------------------------------------
    ridge_estimates: List[float] = []
    coherence_estimates: List[float] = []

    for rep in range(n_replicates):
        rep_seed = seed_base + rep * 1000 + seed
        loader = make_resample_loader(
            train_split, mode=resample_mode,
            sample_fraction=sample_fraction if resample_mode != "full" else 1.0,
            seed=rep_seed, batch_size=batch_size,
        )
        LOGGER.info("Replicate %d / %d  (seed=%d, n=%d)", rep + 1, n_replicates,
                    rep_seed, len(loader.dataset))

        lambdas = estimate_one_replicate(
            model=model, loader=loader,
            use_normalized=use_normalized,
            bias_lr=bias_lr, bias_max_epochs=bias_max_epochs,
            bias_patience=bias_patience, accelerator=accelerator,
        )
        ridge_estimates.append(lambdas["ridge"])
        coherence_estimates.append(lambdas["coherence"])

        wandb.log({
            f"bootstrap/replicate_{rep}/ridge_lambda":     lambdas["ridge"],
            f"bootstrap/replicate_{rep}/coherence_lambda": lambdas["coherence"],
        })
        LOGGER.info("  ridge=%.6f  coherence=%.6f", lambdas["ridge"], lambdas["coherence"])

    # ------------------------------------------------------------------
    # Phase 3: Aggregate and decision metrics
    # ------------------------------------------------------------------
    ridge_agg    = _agg(ridge_estimates,    gt_ridge)
    coherence_agg = _agg(coherence_estimates, gt_coherence)

    agg_log: Dict[str, float] = {}
    for stat, val in ridge_agg.items():
        agg_log[f"bootstrap/ridge/{stat}"] = val
    for stat, val in coherence_agg.items():
        agg_log[f"bootstrap/coherence/{stat}"] = val

    wandb.log(agg_log)

    # W&B summary — human-readable final numbers
    run.summary.update({
        "gt_ridge_lambda":       gt_ridge,
        "gt_coherence_lambda":   gt_coherence,
        "resample_mode":         resample_mode,
        "n_replicates":          n_replicates,
        "sample_fraction":       sample_fraction,
        "use_normalized":        use_normalized,
        **{f"ridge_{k}":     v for k, v in ridge_agg.items()},
        **{f"coherence_{k}": v for k, v in coherence_agg.items()},
        **{f"identifiability_{k}": v for k, v in ident_metrics.items()},
    })

    LOGGER.info("=" * 70)
    LOGGER.info("AGGREGATE RESULTS  (mode=%s  n=%d)", resample_mode, n_replicates)
    LOGGER.info("=" * 70)
    LOGGER.info("Ridge:     true=%.6f | mean=%.6f±%.6f  CV=%.3f  rel_err=%.4f",
                gt_ridge, ridge_agg["mean"], ridge_agg["std"],
                ridge_agg["cv"], ridge_agg["rel_error"])
    LOGGER.info("Coherence: true=%.6f | mean=%.6f±%.6f  CV=%.3f  rel_err=%.4f",
                gt_coherence, coherence_agg["mean"], coherence_agg["std"],
                coherence_agg["cv"], coherence_agg["rel_error"])
    LOGGER.info("Gram κ=%.2f  cos_sim=%.4f",
                ident_metrics["gram_condition_number"], ident_metrics["cosine_similarity"])

    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
