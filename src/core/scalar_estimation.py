"""Deterministic scalar gradient matching and fixed-checkpoint dropout controls.

Model arithmetic retains its checkpoint dtype; batch gradients are accumulated and
all projections evaluated on CPU in float64 (also supported for MPS models).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def evaluation_mode(model, *, dropout=False):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if dropout:
            for module in model.modules():
                if isinstance(module, nn.modules.dropout._DropoutNd):
                    module.train()
        yield
    finally:
        for module, training in modes:
            module.training = training


def parameters64(model):
    return torch.cat([p.detach().reshape(-1).cpu().double() for p in model.parameters()])


def full_batch_loss_grad(model, loader, *, dropout=False):
    """Sample-weighted CE gradient; preserve modes, buffers and existing .grad."""
    params = list(model.parameters())
    if not params or any(not p.requires_grad for p in params):
        raise ValueError("All predictive parameters must require gradients for estimation")
    total = 0
    result = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    device = params[0].device
    with evaluation_mode(model, dropout=dropout), torch.enable_grad():
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            result += torch.cat([
                g.detach().reshape(-1).cpu().double() if g is not None
                else torch.zeros(p.numel(), dtype=torch.float64)
                for p, g in zip(params, grads)
            ])
            total += y.numel()
    if not total:
        raise ValueError("Cannot estimate a gradient from an empty dataset")
    result /= total
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite loss gradient")
    return result


def scalar_projection(basis, target):
    """Signed and nonnegative least squares, with consistent fit diagnostics."""
    basis, target = (v.detach().reshape(-1).cpu().double() for v in (basis, target))
    if basis.shape != target.shape or basis.numel() == 0:
        raise ValueError("Basis and target must have identical nonempty shapes")
    if not torch.isfinite(basis).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite basis or target")
    denom, target_sq = float(basis @ basis), float(target @ target)
    dot = float(basis @ target)
    signed = dot / denom if denom else None
    constrained = max(signed, 0.0) if signed is not None else None
    out = {
        "scale_star": signed, "scale_star_clamped": constrained,
        "estimated_ridge": constrained, "constraint": "nonnegative",
        "grad_cosine": dot / (denom * target_sq)**0.5 if denom and target_sq else None,
        "grad_p_norm": denom**0.5, "target_norm": target_sq**0.5,
        "status": "zero_basis" if not denom else "zero_target" if not target_sq else "ok",
    }
    for suffix, scale in (("", signed), ("_clamped", constrained)):
        sse = float(((scale or 0.0) * basis - target).square().sum()) if scale else target_sq
        out["sse" + suffix] = sse
        out["residual_ratio" + suffix] = (sse / target_sq)**0.5 if target_sq else None
        out["projection_r2" + suffix] = 1 - sse / target_sq if target_sq else None
    return out


def ridge_estimate(model, loader):
    gradient = full_batch_loss_grad(model, loader)
    return scalar_projection(2 * parameters64(model), -gradient)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def provenance(model, *, checkpoint=None, epoch=None, endpoint_kind="provided_checkpoint", dataset=None):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            return "unavailable"
    versions = {}
    for name in ("torch", "lightning", "numpy", "torchvision"):
        versions[name] = importlib.metadata.version(name)
    indices = getattr(dataset, "indices", None)
    diff = git("diff", "HEAD", "--", ".", ":(exclude)paper")
    return {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "model_sha256": model_sha256(model), "epoch": epoch,
        "endpoint_kind": endpoint_kind, "code_revision": os.environ.get("DROPOUT_CODE_REVISION") or git("rev-parse", "HEAD"),
        "source_bundle_sha256": os.environ.get("DROPOUT_SOURCE_SHA256"),
        "working_tree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff != "unavailable" else None,
        "estimator_source_sha256": file_sha256(__file__),
        "versions": versions, "model_hparams": dict(getattr(model, "hparams", {})),
        "dataset": {"name": "MNIST" if indices is not None else "provided",
                    "n": len(dataset) if dataset is not None else None,
                    "split_indices_sha256": hashlib.sha256(torch.tensor(indices, dtype=torch.int64).numpy().tobytes()).hexdigest() if indices is not None else None},
        "model_mode": "eval", "accumulation_dtype": "float64",
        "model_dtype": str(next(model.parameters()).dtype),
    }


def dropout_control(model, loader, *, draws=32, seed=123, bootstrap=2000):
    """Direct dropout gradient vs endpoint proxy on exactly the same weights.

The U statistic estimates squared training-gradient norm without the MC noise
bias. Its jackknife SE is reported without clipping negative estimates to zero.
Only dropout modules are stochastic; batch-normalized models are rejected since
this would not reproduce their training objective.
"""
    if draws < 4:
        raise ValueError("At least four independent mask draws are required")
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("Dropout-objective controls do not support BatchNorm")
    before = model_sha256(model)
    clean = full_batch_loss_grad(model, loader)
    basis = 2 * parameters64(model)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        samples = torch.stack([full_batch_loss_grad(model, loader, dropout=True) for _ in range(draws)])
    # Center before averaging: identical zero-dropout draws then give exactly
    # zero rather than a subtraction artifact from summing repeated vectors.
    direct = (samples - clean).mean(0)
    mean = clean + direct
    endpoint = scalar_projection(basis, -clean)
    measured = scalar_projection(basis, direct)
    denom = float(basis @ basis)
    coefficients = (samples - clean) @ basis / denom if denom else None
    gram = samples @ samples.T
    off_sum = gram.sum() - gram.diag().sum()
    norm_sq = float(off_sum / (draws * (draws - 1)))
    leave_one = (off_sum - 2 * (gram.sum(1) - gram.diag())) / ((draws - 1) * (draws - 2))
    norm_se = float(((draws - 1) / draws * (leave_one - leave_one.mean()).square().sum()).sqrt())
    ci = None
    if coefficients is not None:
        generator = torch.Generator().manual_seed(seed + 1)
        indexes = torch.randint(draws, (bootstrap, draws), generator=generator)
        means = coefficients[indexes].mean(1)
        ci = torch.quantile(means, torch.tensor([.025, .975], dtype=torch.float64)).tolist()
    if model_sha256(model) != before:
        raise RuntimeError("Diagnostic changed predictive parameters or buffers")
    return {
        "endpoint": endpoint, "direct_dropout": measured,
        "mask_draws": draws, "mask_seed": seed,
        "direct_signed_lambda_draws": coefficients.tolist() if coefficients is not None else None,
        "direct_signed_lambda_mc_ci95": ci,
        "train_gradient_norm_sq_unbiased": norm_sq,
        "train_gradient_norm_sq_jackknife_se": norm_se,
        "train_gradient_norm_sq_ci95": [norm_sq - 1.96 * norm_se, norm_sq + 1.96 * norm_se],
        "train_gradient_norm_naive": float(mean.norm()),
        "clean_gradient_norm": float(clean.norm()),
        "signed_coefficient_contamination": float(mean @ basis) / denom if denom else None,
        "interpretation": "Squared-norm interval overlapping zero is inconclusive, not proof of convergence.",
    }
