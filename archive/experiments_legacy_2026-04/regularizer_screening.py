#!/usr/bin/env python3
"""
Screen all defined regularizers for their alignment with the loss gradient.
Regularizers with |cos(∇L, ∇R)| > threshold are candidates for gradient matching.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.bias import (
    RidgeBias, LassoBias, SmoothLassoBias, NuclearNormBias,
    OrthogonalBias, StableRankBias, SpectralEntropyBias,
    WeightCoherenceBias, RowNormVarianceBias, SpectralGapBias,
    LayerNormProductBias, LayerNormBalanceBias, EffectiveDepthBias,
    ElasticNet,
)
from core.estimators import vector_to_parameter_views

torch.set_float32_matmul_precision('medium')

def compute_regularizer_gradient(bias_class, net, device):
    """Compute the gradient of a regularizer w.r.t. flattened parameters."""
    # Create bias instance with scale=1
    if bias_class == ElasticNet:
        bias = ElasticNet(lambda_1_init=1.0, lambda_2_init=1.0)
    elif bias_class in [LassoBias]:
        bias = bias_class()
    else:
        bias = bias_class(enforce_positive=True, init_value=0.0)  # exp(0)=1
    
    bias = bias.to(device)
    
    # Flatten params with gradients
    flat = torch.cat([p.flatten() for p in net.parameters()]).detach().requires_grad_(True)
    struct = vector_to_parameter_views(flat, net)
    
    try:
        penalty = bias(flat, struct)
        grad = torch.autograd.grad(penalty, flat, retain_graph=False)[0]
        return grad, penalty.item()
    except Exception as e:
        print(f"  Error computing gradient for {bias_class.__name__}: {e}")
        return None, None

def screen_regularizers(net, X, y, device):
    """Screen all regularizers for alignment with loss gradient."""
    
    # Compute loss gradient
    net.zero_grad()
    logits = net(X.to(device))
    loss = nn.functional.cross_entropy(logits, y.to(device))
    loss.backward()
    grad_L = torch.cat([p.grad.flatten() for p in net.parameters()])
    target = -grad_L  # This is what regularizer gradients should match
    
    # List of regularizers to test
    regularizers = [
        # Non-scale-invariant (expected to have alignment)
        ("RidgeBias (L2)", RidgeBias),
        ("SmoothLassoBias", SmoothLassoBias),
        ("NuclearNormBias", NuclearNormBias),
        ("OrthogonalBias", OrthogonalBias),
        ("LayerNormProductBias", LayerNormProductBias),
        ("ElasticNet", ElasticNet),
        
        # Scale-invariant (expected to be orthogonal)
        ("StableRankBias", StableRankBias),
        ("SpectralEntropyBias", SpectralEntropyBias),
        ("WeightCoherenceBias", WeightCoherenceBias),
        ("RowNormVarianceBias", RowNormVarianceBias),
        ("SpectralGapBias", SpectralGapBias),
        ("LayerNormBalanceBias", LayerNormBalanceBias),
        ("EffectiveDepthBias", EffectiveDepthBias),
    ]
    
    results = []
    
    print(f"\n{'Regularizer':<25} {'|cos(∇L,∇R)|':>12} {'||∇R||':>12} {'penalty':>12} {'Scale-Inv?':>12}")
    print("=" * 75)
    
    for name, bias_class in regularizers:
        grad_R, penalty = compute_regularizer_gradient(bias_class, net, device)
        
        if grad_R is None:
            print(f"{name:<25} {'ERROR':>12}")
            continue
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            target.unsqueeze(0), grad_R.unsqueeze(0)
        ).item()
        
        grad_norm = grad_R.norm().item()
        
        # Check if scale-invariant (gradient orthogonal to W)
        flat_W = torch.cat([p.flatten() for p in net.parameters()])
        cos_W = torch.nn.functional.cosine_similarity(
            grad_R.unsqueeze(0), flat_W.unsqueeze(0)
        ).item()
        is_scale_inv = abs(cos_W) < 0.01
        
        results.append({
            'name': name,
            'cos_loss': abs(cos_sim),
            'grad_norm': grad_norm,
            'penalty': penalty,
            'scale_invariant': is_scale_inv,
            'cos_W': cos_W,
        })
        
        scale_inv_str = "Yes" if is_scale_inv else "No"
        print(f"{name:<25} {abs(cos_sim):>12.4f} {grad_norm:>12.2f} {penalty:>12.4f} {scale_inv_str:>12}")
    
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    torch.manual_seed(42)
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    
    # Test with different training conditions
    conditions = [
        ("Untrained (random init)", None, None),
        ("Trained with L2 (λ=0.01)", 0.01, None),
        ("Trained with L2 (λ=0.1)", 0.1, None),
    ]
    
    for condition_name, l2_lambda, other in conditions:
        print(f"\n{'='*75}")
        print(f"CONDITION: {condition_name}")
        print('='*75)
        
        # Create network
        torch.manual_seed(42)
        net = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        ).to(device)
        
        # Train if needed
        if l2_lambda is not None:
            loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            for epoch in range(30):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = nn.functional.cross_entropy(net(X_b), y_b)
                    l2 = sum(p.pow(2).sum() for p in net.parameters())
                    (loss + l2_lambda * l2).backward()
                    optimizer.step()
        
        # Screen regularizers
        results = screen_regularizers(net, X[:256], y[:256], device)
        
        # Summary
        print("\n--- CANDIDATES for gradient matching (|cos| > 0.05): ---")
        candidates = [r for r in results if r['cos_loss'] > 0.05]
        candidates.sort(key=lambda x: x['cos_loss'], reverse=True)
        for r in candidates:
            print(f"  {r['name']}: |cos|={r['cos_loss']:.4f}")
        
        print("\n--- UNLIKELY to work (|cos| < 0.01): ---")
        unlikely = [r for r in results if r['cos_loss'] < 0.01]
        for r in unlikely:
            print(f"  {r['name']}: |cos|={r['cos_loss']:.6f}")

if __name__ == "__main__":
    main()
