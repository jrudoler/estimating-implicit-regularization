import torch
from torch import Tensor
from typing import Optional, Literal
import warnings


def compute_Q_matrix(X: Tensor, k: int, eps: float, jit=None) -> Tensor:
    r"""
    Compute the Q matrix used for regularization based on the data matrix `X`,
    an iteration parameter `k`, and a step size `eps`.

    This constructs Q via eigendecomposition of the normalized covariance matrix:

        A = (1/n) XᵀX

    Then defines:

        D = S @ [(I - εS)^(-k) - I]⁻¹

    where S is the diagonal matrix of eigenvalues and the final Q is:

        Q = V D Vᵀ

    Args:
        X (torch.Tensor): Data matrix of shape (n, p).
        k (int): Number of gradient steps (must be ≥ 1).
        eps (float): Learning rate or step size. Must be < 1 / λ_max.

    Returns:
        torch.Tensor: Symmetric positive-definite Q matrix of shape (p, p).
    """
    n = X.shape[0]
    A = X.T @ X / n
    # if necessary, add jitter to ensure numerical stability
    if jit is not None:
        A += jit * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

    eigenvals, eigenvecs = torch.linalg.eigh(A)

    max_lr = 1 / eigenvals.max()
    assert eps < max_lr, f"eps {eps} is larger than max lr {max_lr}"

    S = torch.diag(eigenvals)
    I = torch.eye(len(eigenvals), dtype=S.dtype, device=S.device)

    D = S @ torch.linalg.inv(torch.matrix_power(I - eps * S, -k) - I)

    if not torch.allclose(D, torch.diag(torch.diag(D))):
        print("Warning: D is not diagonal")

    Q = eigenvecs @ D @ eigenvecs.T
    return Q


def compute_beta_closed_form(X: Tensor, y: Tensor, Q: Tensor) -> Tensor:
    r"""
    Compute closed-form solution minimizing (1/n) * ||y - Xβ||² + βᵀ Q β.


    .. math::
        \frac{1}{n}\,\|y - X\beta\|_2^2 \;+\; \beta^T\,Q\,\beta

    Solution:

    .. math::
        \beta_{\star} \;=\; \bigl(X^\top X + n\,Q\bigr)^{-1} \;X^\top\,y

    Args:
        X (torch.Tensor): Design matrix of shape (n, p)
        y (torch.Tensor): Target vector of shape (n,)
        Q (torch.Tensor): Penalty matrix of shape (p, p)

    Returns:
        torch.Tensor: Solution vector β* of shape (p,)
    """
    # Get the number of samples
    n = X.shape[0]

    # Form the matrix X^T X + nQ
    A = X.T @ X + n * Q  # shape: (p, p)

    # Right-hand side: X^T y
    b = X.T @ y  # shape: (p,)

    # Solve the linear system for beta
    beta_star = torch.linalg.solve(A, b)  # shape: (p,)

    return beta_star


def is_psd(matrix: Tensor, tol: float = 1e-5) -> bool:
    """Check if a symmetric matrix is positive semidefinite (PSD)."""
    if not torch.allclose(matrix, matrix.T, atol=tol):
        print("Matrix is not symmetric.")
        return False
    else:
        print("Matrix is symmetric.")
    try:
        # Eigenvalues should be >= -tol for numerical stability
        eigvals = torch.linalg.eigvalsh(matrix)
        return torch.all(eigvals >= -tol).item()
    except RuntimeError:
        return False


def rbf_kernel_torch(
    X: Tensor,
    Y: Optional[Tensor] = None,
    gamma: Optional[float] = None,
    enforce_psd: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute the RBF (Gaussian) kernel between X and Y:
      K[i,j] = exp(-gamma * ||X[i] - Y[j]||^2)

    If Y is None, uses Y = X.
    If gamma is None, defaults to 1.0 / n_features.
    """
    if Y is None:
        Y = X
    # default gamma = 1/n_features
    n_features = X.size(1)
    if gamma is None:
        gamma = 1.0 / float(n_features)
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X_norm = (X**2).sum(dim=1, keepdim=True)  # (n_X, 1)
    Y_norm = (Y**2).sum(dim=1, keepdim=True).t()  # (1, n_Y)
    sq_dists = X_norm + Y_norm - 2.0 * X @ Y.t()  # (n_X, n_Y)
    # sq_dists = torch.cdist(X, Y, p=2) ** 2  # (n_X, n_Y)
    K = torch.exp(-gamma * sq_dists)

    if enforce_psd and (Y is X):
        warnings.warn(
            f"Using the same X and Y, enforcing symmetry and positive semidefiniteness by adding {eps} on the diagonal.",
            UserWarning,
        )
        # enforce perfect symmetry
        K = 0.5 * (K + K.t())
        # add a tiny diagonal bump so all eigenvalues ≥ 0
        K = K + eps * torch.eye(K.size(0), device=K.device, dtype=K.dtype)
    return K


def orthogonal_rows(n: int, d: int, device: str = "cpu") -> Tensor:
    """
    Returns X ∈ R^{n×d} with orthonormal rows if n ≤ d,
    otherwise n near-orthogonal rows by block-stacking.
    """
    if n <= d:
        G = torch.randn(d, n, device=device)  # QR on (d×n)
        Q, _ = torch.linalg.qr(G, mode="reduced")  # Q: (d×n), orthonormal columns
        return Q.T  # (n×d), orthonormal rows
    # n > d: stack multiple near-orthogonal blocks and renormalize
    blocks = []
    remaining = n
    while remaining > 0:
        k = min(remaining, d)
        B = orthogonal_rows(k, d, device)
        blocks.append(B)
        remaining -= k
    X = torch.vstack(blocks)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    return X
