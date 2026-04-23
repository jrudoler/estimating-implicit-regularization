#!/usr/bin/env python3
"""
Ridge + NuclearNorm recovery test with normalized gradient matching.
Uses MSE loss for regression (not cross-entropy for classification).
Both regularizers have non-trivial alignment with the loss gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call, vjp
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
import pandas as pd
from itertools import product
from typing import Dict, Tuple, Callable
from torch.optim import Optimizer

from core.bias import RidgeBias, NuclearNormBias, JointBias
from core.estimators import BiasWithMSE, vector_to_parameter_views

torch.set_float32_matmul_precision("medium")


class BiasWithMSENormalized(BiasWithMSE):
    """
    MSE-based bias estimator with normalized gradient matching.

    Same as BiasWithCrossEntropyNormalized but for regression tasks.
    """

    def __init__(
        self,
        predictive_model: nn.Module,
        bias_model: nn.Module,
        grad_match_loss_fn: Callable[..., torch.Tensor] = F.mse_loss,
        optimizer_cls: Callable[..., Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        bias_lr: float = 0.1,
        monitor_lr: str = "train_bias/loss",
        patience_lr: int = 10,
        factor_lr: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__(
            predictive_model=predictive_model,
            bias_model=bias_model,
            grad_match_loss_fn=grad_match_loss_fn,
            optimizer_cls=optimizer_cls,
            lr=lr,
        )
        self.bias_lr = bias_lr
        self.monitor_lr = monitor_lr
        self.patience_lr = patience_lr
        self.factor_lr = factor_lr
        self.eps = eps

        if not hasattr(bias_model, "bias_models"):
            raise ValueError(
                "BiasWithMSENormalized requires a JointBias model "
                "with multiple bias models."
            )

        n_biases = len(bias_model.bias_models)
        self.normalized_coefs = nn.Parameter(torch.zeros(n_biases))

        self.save_hyperparameters(
            ignore=["predictive_model", "bias_model", "grad_match_loss_fn"]
        )

    def _compute_penalty_gradient(
        self, bias_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient of raw penalty (coefficient=1)."""
        flat_params = (
            torch.nn.utils.parameters_to_vector(self.predictive_model.parameters())
            .detach()
            .requires_grad_()
            .to(self.device)
        )
        struct_params = vector_to_parameter_views(flat_params, self.predictive_model)

        if hasattr(bias_model, "beta"):
            original_beta = bias_model.beta.data.clone()
            if hasattr(bias_model, "enforce_positive") and bias_model.enforce_positive:
                bias_model.beta.data = torch.zeros_like(bias_model.beta.data)
            else:
                bias_model.beta.data = torch.ones_like(bias_model.beta.data)
        elif hasattr(bias_model, "alpha"):
            original_alpha = bias_model.alpha.data.clone()
            bias_model.alpha.data = torch.ones_like(bias_model.alpha.data)

        R_i = bias_model(flat_params, struct_params)
        grad_i = torch.autograd.grad(R_i, flat_params, create_graph=True)[0]

        if hasattr(bias_model, "beta"):
            bias_model.beta.data = original_beta
        elif hasattr(bias_model, "alpha"):
            bias_model.alpha.data = original_alpha

        norm_i = grad_i.norm() + self.eps
        return grad_i, norm_i

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = batch
        if y.ndim == 1:
            y = y.view(-1, 1)

        params: Dict[str, torch.Tensor] = dict(self.predictive_model.named_parameters())

        def model_output(p: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.predictive_model, p, (x,))

        predictions, vjp_func = vjp(model_output, params, X)
        loss_gradient = -self.predictive_loss_grad(predictions, y)
        vjp_result = vjp_func(loss_gradient)[0]
        true_grad = torch.cat([v.view(-1) for v in vjp_result.values()]) / X.size(0)

        normalized_grads = []
        penalty_grad_norms = []
        bias_names = []

        for bias_model in self.bias_model.bias_models:
            name = getattr(bias_model, "bias_name", bias_model.__class__.__name__)
            bias_names.append(name)
            grad_i, norm_i = self._compute_penalty_gradient(bias_model)
            normalized_grads.append(grad_i / norm_i)
            penalty_grad_norms.append(norm_i)

        predicted_grad = torch.zeros_like(true_grad)
        for i, norm_grad in enumerate(normalized_grads):
            predicted_grad = predicted_grad + self.normalized_coefs[i] * norm_grad

        loss = self.grad_match_loss_fn(predicted_grad, true_grad, reduction="mean")

        self.log("train_bias/loss", loss, prog_bar=False)

        for i, name in enumerate(bias_names):
            coef = self.normalized_coefs[i].item()
            norm = penalty_grad_norms[i].item()
            estimated_lambda = coef / norm if norm > self.eps else 0.0
            self.log(f"train_bias/{name}/coef", coef, prog_bar=False)
            self.log(f"train_bias/{name}/grad_norm", norm, prog_bar=False)
            self.log(
                f"train_bias/{name}/estimated_lambda", estimated_lambda, prog_bar=False
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([self.normalized_coefs], lr=self.bias_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor_lr,
            patience=self.patience_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor_lr},
        }

    def get_estimated_lambdas(self) -> Dict[str, float]:
        """Get the recovered lambda values."""
        lambdas = {}
        for i, bias_model in enumerate(self.bias_model.bias_models):
            name = getattr(bias_model, "bias_name", bias_model.__class__.__name__)
            _, norm_i = self._compute_penalty_gradient(bias_model)
            coef = self.normalized_coefs[i].item()
            lambdas[name] = coef / norm_i.item()
        return lambdas


class BiasWithMSEScheduled(BiasWithMSE):
    """MSE-based bias estimator with LR scheduling (standard, non-normalized)."""

    def __init__(
        self,
        *args,
        bias_lr: float = 0.1,
        patience_lr: int = 10,
        factor_lr: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bias_lr = bias_lr
        self.patience_lr = patience_lr
        self.factor_lr = factor_lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.bias_model.parameters(), lr=self.bias_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor_lr,
            patience=self.patience_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_bias/loss"},
        }


def compute_nuclear_norm_penalty(model):
    """Compute nuclear norm penalty for training."""
    penalty = 0.0
    for p in model.parameters():
        if p.ndim >= 2:
            matrix = p if p.ndim == 2 else p.reshape(p.shape[0], -1)
            penalty += torch.linalg.matrix_norm(matrix, ord="nuc")
    return penalty


def train_network(gt_ridge, gt_nuclear, loader, device, seed=42):
    """Train network with ground truth regularization using MSE loss."""
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(100):  # More epochs for regression
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            # MSE loss for regression
            loss = F.mse_loss(net(X_b), y_b)

            if gt_ridge > 0:
                l2 = sum(p.pow(2).sum() for p in net.parameters())
                loss = loss + gt_ridge * l2

            if gt_nuclear > 0:
                nuc = compute_nuclear_norm_penalty(net)
                loss = loss + gt_nuclear * nuc

            loss.backward()
            optimizer.step()

    return net


def compute_geometry(net, X, y, device):
    """Compute geometric properties at trained network."""
    X, y = X.to(device), y.to(device)

    net.zero_grad()
    pred = net(X)
    # MSE gradient: d/dW MSE = d/dW (1/n) sum (f(x) - y)^2
    # We compute -∇L which should equal λ_r∇R + λ_n∇N at optimum
    loss = F.mse_loss(pred, y)
    loss.backward()
    grad_L = torch.cat([p.grad.flatten() for p in net.parameters()])
    target = -grad_L  # This is what we're trying to match

    flat = (
        torch.cat([p.flatten() for p in net.parameters()]).detach().requires_grad_(True)
    )
    struct = vector_to_parameter_views(flat, net)

    rb = RidgeBias(enforce_positive=True)
    rb.beta.data = torch.tensor(0.0, device=device)
    rb = rb.to(device)
    r_val = rb(flat, struct)
    grad_r = torch.autograd.grad(r_val, flat)[0]

    flat2 = (
        torch.cat([p.flatten() for p in net.parameters()]).detach().requires_grad_(True)
    )
    struct2 = vector_to_parameter_views(flat2, net)
    nucb = NuclearNormBias(enforce_positive=True)
    nucb.beta.data = torch.tensor(0.0, device=device)
    nucb = nucb.to(device)
    nuc_val = nucb(flat2, struct2)
    grad_nuc = torch.autograd.grad(nuc_val, flat2)[0]

    cos_r = F.cosine_similarity(target.unsqueeze(0), grad_r.unsqueeze(0)).item()
    cos_nuc = F.cosine_similarity(target.unsqueeze(0), grad_nuc.unsqueeze(0)).item()
    cos_r_nuc = F.cosine_similarity(grad_r.unsqueeze(0), grad_nuc.unsqueeze(0)).item()

    return {
        "target_norm": target.norm().item(),
        "grad_r_norm": grad_r.norm().item(),
        "grad_nuc_norm": grad_nuc.norm().item(),
        "cos_target_r": cos_r,
        "cos_target_nuc": cos_nuc,
        "cos_r_nuc": cos_r_nuc,
    }


def estimate_bias(net, loader, method="standard", device="cpu"):
    """Run bias estimation using MSE-based estimators."""
    ridge = RidgeBias(enforce_positive=True, init_value=0.01)
    nuc = NuclearNormBias(enforce_positive=True, init_value=0.01)
    joint = JointBias([ridge, nuc])

    if method == "standard":
        estimator = BiasWithMSEScheduled(
            predictive_model=net,
            bias_model=joint,
            grad_match_loss_fn=F.mse_loss,
            bias_lr=0.1,
        )
    else:  # normalized
        estimator = BiasWithMSENormalized(
            predictive_model=net,
            bias_model=joint,
            grad_match_loss_fn=F.mse_loss,
            bias_lr=0.1,
        )

    accelerator = "gpu" if device != "cpu" and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="train_bias/loss", patience=75)],
    )
    trainer.fit(estimator, loader)

    if method == "standard":
        params = joint.get_bias_params()
        return params["ridge/scale"], params["nuclear_norm/scale"]
    else:
        lambdas = estimator.get_estimated_lambdas()
        return lambdas["ridge"], lambdas["nuclear_norm"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create REGRESSION dataset (not classification!)
    torch.manual_seed(42)
    n_samples = 2000
    n_features = 100

    # Generate data with some structure
    X = torch.randn(n_samples, n_features)
    # True function: y = X @ w_true + noise
    w_true = torch.randn(n_features, 1) * 0.5
    y = X @ w_true + torch.randn(n_samples, 1) * 0.1

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    # Ground truth combinations to test
    ridge_values = [0.0, 0.01, 0.05]
    nuclear_values = [0.0, 0.01, 0.05]

    results = []

    for gt_ridge, gt_nuclear in product(ridge_values, nuclear_values):
        print(f"\n{'=' * 60}")
        print(f"Ground truth: Ridge={gt_ridge}, Nuclear={gt_nuclear}")
        print("=" * 60)

        # Train network with regularization
        net = train_network(gt_ridge, gt_nuclear, loader, device)

        # Compute geometry at trained network
        geom = compute_geometry(net, X[:256], y[:256], device)
        print(
            f"Geometry: cos(t,R)={geom['cos_target_r']:.3f}, cos(t,N)={geom['cos_target_nuc']:.3f}, cos(R,N)={geom['cos_r_nuc']:.3f}"
        )
        print(
            f"          ||target||={geom['target_norm']:.4f}, ||∇R||={geom['grad_r_norm']:.2f}, ||∇N||={geom['grad_nuc_norm']:.2f}"
        )

        # Standard estimator
        est_r_std, est_n_std = estimate_bias(
            net, loader, method="standard", device=device
        )
        r_err_std = (
            abs(est_r_std - gt_ridge) / max(gt_ridge, 0.001) * 100
            if gt_ridge > 0
            else abs(est_r_std) * 100
        )
        n_err_std = (
            abs(est_n_std - gt_nuclear) / max(gt_nuclear, 0.001) * 100
            if gt_nuclear > 0
            else abs(est_n_std) * 100
        )
        print(
            f"Standard:   Ridge={est_r_std:.4f} (err {r_err_std:.0f}%), Nuclear={est_n_std:.4f} (err {n_err_std:.0f}%)"
        )

        # Normalized estimator
        est_r_norm, est_n_norm = estimate_bias(
            net, loader, method="normalized", device=device
        )
        r_err_norm = (
            abs(est_r_norm - gt_ridge) / max(gt_ridge, 0.001) * 100
            if gt_ridge > 0
            else abs(est_r_norm) * 100
        )
        n_err_norm = (
            abs(est_n_norm - gt_nuclear) / max(gt_nuclear, 0.001) * 100
            if gt_nuclear > 0
            else abs(est_n_norm) * 100
        )
        print(
            f"Normalized: Ridge={est_r_norm:.4f} (err {r_err_norm:.0f}%), Nuclear={est_n_norm:.4f} (err {n_err_norm:.0f}%)"
        )

        results.append(
            {
                "gt_ridge": gt_ridge,
                "gt_nuclear": gt_nuclear,
                **geom,
                "std_ridge": est_r_std,
                "std_nuclear": est_n_std,
                "std_ridge_err": r_err_std,
                "std_nuclear_err": n_err_std,
                "norm_ridge": est_r_norm,
                "norm_nuclear": est_n_norm,
                "norm_ridge_err": r_err_norm,
                "norm_nuclear_err": n_err_norm,
            }
        )

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Inactive regularizer detection (should be ~0)
    inactive_ridge = df[df["gt_ridge"] == 0]
    inactive_nuclear = df[df["gt_nuclear"] == 0]

    print("\nInactive regularizer detection (avg absolute estimate when true=0):")
    print(
        f"  Ridge inactive:    Standard={inactive_ridge['std_ridge'].abs().mean():.4f}, Normalized={inactive_ridge['norm_ridge'].abs().mean():.4f}"
    )
    print(
        f"  Nuclear inactive:  Standard={inactive_nuclear['std_nuclear'].abs().mean():.4f}, Normalized={inactive_nuclear['norm_nuclear'].abs().mean():.4f}"
    )

    # Active regularizer recovery
    active_ridge = df[df["gt_ridge"] > 0]
    active_nuclear = df[df["gt_nuclear"] > 0]

    print("\nActive regularizer recovery (avg % error when true>0):")
    print(
        f"  Ridge active:      Standard={active_ridge['std_ridge_err'].mean():.0f}%, Normalized={active_ridge['norm_ridge_err'].mean():.0f}%"
    )
    print(
        f"  Nuclear active:    Standard={active_nuclear['std_nuclear_err'].mean():.0f}%, Normalized={active_nuclear['norm_nuclear_err'].mean():.0f}%"
    )

    # Both active
    both_active = df[(df["gt_ridge"] > 0) & (df["gt_nuclear"] > 0)]
    if len(both_active) > 0:
        print("\nWhen both active:")
        print(
            f"  Ridge:   Standard={both_active['std_ridge_err'].mean():.0f}%, Normalized={both_active['norm_ridge_err'].mean():.0f}%"
        )
        print(
            f"  Nuclear: Standard={both_active['std_nuclear_err'].mean():.0f}%, Normalized={both_active['norm_nuclear_err'].mean():.0f}%"
        )

    # Geometry summary
    print("\nGeometry summary (when regularization active):")
    reg_active = df[(df["gt_ridge"] > 0) | (df["gt_nuclear"] > 0)]
    print(f"  avg ||target||:    {reg_active['target_norm'].mean():.4f}")
    print(f"  avg cos(t,R):      {reg_active['cos_target_r'].mean():.3f}")
    print(f"  avg cos(t,N):      {reg_active['cos_target_nuc'].mean():.3f}")

    # Save results
    df.to_csv("ridge_nuclear_recovery_mse_results.csv", index=False)
    print("\nResults saved to ridge_nuclear_recovery_mse_results.csv")


if __name__ == "__main__":
    main()
