#!/usr/bin/env python3
"""
Ridge + RowNormVariance recovery test with multiple ground truth combinations.
Tests both standard and normalized estimators.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
import pandas as pd
from itertools import product

from core.bias import RidgeBias, RowNormVarianceBias, JointBias
from core.estimators import BiasWithCrossEntropyNormalized, BiasWithCrossEntropyScheduled, vector_to_parameter_views

torch.set_float32_matmul_precision('medium')

def compute_row_norm_variance_penalty(model):
    penalty = 0.0
    count = 0
    for p in model.parameters():
        if p.ndim >= 2:
            matrix = p if p.ndim == 2 else p.reshape(p.shape[0], -1)
            row_norms_sq = (matrix**2).sum(dim=1)
            mean_norm_sq = row_norms_sq.mean() + 1e-8
            normalized_norms = row_norms_sq / mean_norm_sq
            variance = torch.var(normalized_norms)
            penalty += variance
            count += 1
    return penalty / count if count > 0 else 0

def train_network(gt_ridge, gt_rnv, loader, device, seed=42):
    """Train network with ground truth regularization."""
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(30):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(net(X_b), y_b)
            
            if gt_ridge > 0:
                l2 = sum(p.pow(2).sum() for p in net.parameters())
                loss = loss + gt_ridge * l2
            
            if gt_rnv > 0:
                rnv = compute_row_norm_variance_penalty(net)
                loss = loss + gt_rnv * rnv
            
            loss.backward()
            optimizer.step()
    
    return net

def compute_geometry(net, X, y, device):
    """Compute geometric properties at trained network."""
    X, y = X.to(device), y.to(device)
    
    net.zero_grad()
    logits = net(X)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    grad_L = torch.cat([p.grad.flatten() for p in net.parameters()])
    target = -grad_L
    
    flat = torch.cat([p.flatten() for p in net.parameters()]).detach().requires_grad_(True)
    struct = vector_to_parameter_views(flat, net)
    
    rb = RidgeBias(enforce_positive=True)
    rb.beta.data = torch.tensor(0.0, device=device)
    rb = rb.to(device)
    r_val = rb(flat, struct)
    grad_r = torch.autograd.grad(r_val, flat)[0]
    
    flat2 = torch.cat([p.flatten() for p in net.parameters()]).detach().requires_grad_(True)
    struct2 = vector_to_parameter_views(flat2, net)
    rnvb = RowNormVarianceBias(enforce_positive=True)
    rnvb.beta.data = torch.tensor(0.0, device=device)
    rnvb = rnvb.to(device)
    rnv_val = rnvb(flat2, struct2)
    grad_rnv = torch.autograd.grad(rnv_val, flat2)[0]
    
    cos_r = torch.nn.functional.cosine_similarity(target.unsqueeze(0), grad_r.unsqueeze(0)).item()
    cos_rnv = torch.nn.functional.cosine_similarity(target.unsqueeze(0), grad_rnv.unsqueeze(0)).item()
    cos_r_rnv = torch.nn.functional.cosine_similarity(grad_r.unsqueeze(0), grad_rnv.unsqueeze(0)).item()
    
    return {
        'target_norm': target.norm().item(),
        'grad_r_norm': grad_r.norm().item(),
        'grad_rnv_norm': grad_rnv.norm().item(),
        'cos_target_r': cos_r,
        'cos_target_rnv': cos_rnv,
        'cos_r_rnv': cos_r_rnv,
    }

def estimate_bias(net, loader, method='standard', device='cpu'):
    """Run bias estimation."""
    ridge = RidgeBias(enforce_positive=True, init_value=0.01)
    rnv = RowNormVarianceBias(enforce_positive=True, init_value=0.01)
    joint = JointBias([ridge, rnv])
    
    if method == 'standard':
        estimator = BiasWithCrossEntropyScheduled(
            predictive_model=net, bias_model=joint,
            grad_match_loss_fn=nn.functional.mse_loss, bias_lr=0.1,
        )
    else:
        estimator = BiasWithCrossEntropyNormalized(
            predictive_model=net, bias_model=joint,
            grad_match_loss_fn=nn.functional.mse_loss, bias_lr=0.1,
        )
    
    accelerator = 'gpu' if device != 'cpu' and torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs=500, accelerator=accelerator, devices=1,
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        callbacks=[EarlyStopping(monitor='train_bias/loss', patience=75)]
    )
    trainer.fit(estimator, loader)
    
    if method == 'standard':
        params = joint.get_bias_params()
        return params['ridge/scale'], params['row_norm_variance/scale']
    else:
        lambdas = estimator.get_estimated_lambdas()
        return lambdas['ridge'], lambdas['row_norm_variance']

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    torch.manual_seed(42)
    X = torch.randn(2000, 784)
    y = torch.randint(0, 10, (2000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    # Ground truth combinations to test
    ridge_values = [0.0, 0.01, 0.05, 0.1]
    rnv_values = [0.0, 0.05, 0.1, 0.2]
    
    results = []
    
    for gt_ridge, gt_rnv in product(ridge_values, rnv_values):
        print(f"\n{'='*60}")
        print(f"Ground truth: Ridge={gt_ridge}, RNV={gt_rnv}")
        print('='*60)
        
        # Train network
        net = train_network(gt_ridge, gt_rnv, loader, device)
        
        # Compute geometry
        geom = compute_geometry(net, X[:256], y[:256], device)
        print(f"Geometry: cos(t,R)={geom['cos_target_r']:.3f}, cos(t,RNV)={geom['cos_target_rnv']:.3f}, cos(R,RNV)={geom['cos_r_rnv']:.3f}")
        
        # Standard estimator
        est_r_std, est_rnv_std = estimate_bias(net, loader, method='standard', device=device)
        r_err_std = abs(est_r_std - gt_ridge) / max(gt_ridge, 0.001) * 100 if gt_ridge > 0 else abs(est_r_std) * 100
        rnv_err_std = abs(est_rnv_std - gt_rnv) / max(gt_rnv, 0.001) * 100 if gt_rnv > 0 else abs(est_rnv_std) * 100
        print(f"Standard:   Ridge={est_r_std:.4f} (err {r_err_std:.0f}%), RNV={est_rnv_std:.4f} (err {rnv_err_std:.0f}%)")
        
        # Normalized estimator
        est_r_norm, est_rnv_norm = estimate_bias(net, loader, method='normalized', device=device)
        r_err_norm = abs(est_r_norm - gt_ridge) / max(gt_ridge, 0.001) * 100 if gt_ridge > 0 else abs(est_r_norm) * 100
        rnv_err_norm = abs(est_rnv_norm - gt_rnv) / max(gt_rnv, 0.001) * 100 if gt_rnv > 0 else abs(est_rnv_norm) * 100
        print(f"Normalized: Ridge={est_r_norm:.4f} (err {r_err_norm:.0f}%), RNV={est_rnv_norm:.4f} (err {rnv_err_norm:.0f}%)")
        
        results.append({
            'gt_ridge': gt_ridge,
            'gt_rnv': gt_rnv,
            **geom,
            'std_ridge': est_r_std,
            'std_rnv': est_rnv_std,
            'std_ridge_err': r_err_std,
            'std_rnv_err': rnv_err_std,
            'norm_ridge': est_r_norm,
            'norm_rnv': est_rnv_norm,
            'norm_ridge_err': r_err_norm,
            'norm_rnv_err': rnv_err_norm,
        })
    
    # Summary
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Inactive regularizer detection (should be ~0)
    inactive_ridge = df[df['gt_ridge'] == 0]
    inactive_rnv = df[df['gt_rnv'] == 0]
    
    print("\nInactive regularizer detection (avg absolute estimate when true=0):")
    print(f"  Ridge inactive:  Standard={inactive_ridge['std_ridge'].abs().mean():.4f}, Normalized={inactive_ridge['norm_ridge'].abs().mean():.4f}")
    print(f"  RNV inactive:    Standard={inactive_rnv['std_rnv'].abs().mean():.4f}, Normalized={inactive_rnv['norm_rnv'].abs().mean():.4f}")
    
    # Active regularizer recovery
    active_ridge = df[df['gt_ridge'] > 0]
    active_rnv = df[df['gt_rnv'] > 0]
    
    print("\nActive regularizer recovery (avg % error when true>0):")
    print(f"  Ridge active:    Standard={active_ridge['std_ridge_err'].mean():.0f}%, Normalized={active_ridge['norm_ridge_err'].mean():.0f}%")
    print(f"  RNV active:      Standard={active_rnv['std_rnv_err'].mean():.0f}%, Normalized={active_rnv['norm_rnv_err'].mean():.0f}%")
    
    # Both active
    both_active = df[(df['gt_ridge'] > 0) & (df['gt_rnv'] > 0)]
    print("\nWhen both active:")
    print(f"  Ridge:  Standard={both_active['std_ridge_err'].mean():.0f}%, Normalized={both_active['norm_ridge_err'].mean():.0f}%")
    print(f"  RNV:    Standard={both_active['std_rnv_err'].mean():.0f}%, Normalized={both_active['norm_rnv_err'].mean():.0f}%")
    
    # Save results
    df.to_csv('ridge_rnv_recovery_results.csv', index=False)
    print("\nResults saved to ridge_rnv_recovery_results.csv")

if __name__ == "__main__":
    main()
