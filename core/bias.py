from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import abstractmethod
from torch.func import functional_call, vjp, hessian
from torch.optim import Optimizer
from typing import Dict, Callable, Tuple, Any, Optional


class GradientSquaredPenaltyScale(nn.Module):
    """Learns a scalar multiplier for squared gradient penalties."""

    def __init__(self, lambda_init: float = 1.0, enforce_positive: bool = True) -> None:
        super().__init__()
        initial_value = float(lambda_init)
        if enforce_positive:
            safe_value = max(initial_value, 1e-8)
            # Store log(lambda) so the forward pass always returns a positive value.
            self.lambda_param = nn.Parameter(torch.log(torch.tensor(safe_value)))
        else:
            self.lambda_param = nn.Parameter(torch.tensor(initial_value))
        self.enforce_positive = enforce_positive

    def forward(self, grad_vector: Optional[Tensor] = None) -> Tensor:
        scale = (
            torch.exp(self.lambda_param) if self.enforce_positive else self.lambda_param
        )
        if grad_vector is None:
            return scale
        penalty = torch.sum(grad_vector.pow(2))
        return scale * penalty


# GOAL: implement a class of models that represent a parametrization of the inductive
# bias of the model. This class should be able to be used with any pretrained model with
# known activations / weights.


class RidgeBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([1.0]))  # Initialize parameter

    def forward(self, flattened_params: torch.Tensor, **kwargs):
        return self.beta * torch.sum(flattened_params**2)


class LassoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([1.0]))  # Initialize parameters

    def forward(self, flattened_params, **kwargs):
        return self.alpha * torch.sum(torch.abs(flattened_params))


class SmoothLassoBias(nn.Module):
    def __init__(self, alpha_init: float = 1.0, smooth: float = 0.1) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))  # Initialize parameters
        self.smooth = smooth  # smoothness parameter, not learnable

    def forward(self, flattened_params: torch.Tensor, **kwargs):
        smooth_loss = torch.nn.functional.smooth_l1_loss(
            flattened_params,
            torch.zeros_like(flattened_params),
            beta=self.smooth,
            reduction="sum",
        )
        return self.alpha * smooth_loss


class MatrixRidgeBias(nn.Module):
    def __init__(self, dim: int = 10, Q_init: torch.Tensor = None):
        super().__init__()
        self.dim = dim
        if Q_init is not None:
            if Q_init.shape != (dim, dim):
                raise ValueError(
                    f"Q_init should be of shape ({dim}, {dim}), but got {Q_init.shape}"
                )
        self.Q = (
            nn.Parameter(Q_init, requires_grad=True)
            if Q_init is not None
            else nn.Parameter(torch.eye(dim))
        )  # Initialize parameter

    def forward(self, flattened_params: torch.Tensor, **kwargs):
        """
        Compute the quadratic form x^T Q x, where x is the flattened parameters.
        This is equivalent to the L2 regularization term with a matrix Q.
        """
        # check that flattened_params is of shape (dim,)
        if flattened_params.shape != (self.dim,):
            raise ValueError(
                f"flattened_params should be of shape ({self.dim},), but got {flattened_params.shape}"
            )
        # Compute the quadratic form
        loss = flattened_params @ self.Q @ flattened_params
        return loss


class DiagMatrixRidgeBias(nn.Module):
    def __init__(self, dim: int = 10, Q_init: torch.Tensor = None):
        super().__init__()
        self.dim = dim
        if Q_init is not None:
            if Q_init.shape != (dim,):
                raise ValueError(
                    f"Q_init should be of shape ({dim},), but got {Q_init.shape}"
                )
        self.Q = (
            nn.Parameter(Q_init, requires_grad=True)
            if Q_init is not None
            else nn.Parameter(torch.zeros(dim))
        )  # Initialize parameter

    def forward(self, flattened_params: torch.Tensor, **kwargs):
        """
        Compute the quadratic form x^T Q x, where x is the flattened parameters.
        This is equivalent to the L2 regularization term with a matrix Q.
        """
        # check that flattened_params is of shape (dim,)
        if flattened_params.shape != (self.dim,):
            raise ValueError(
                f"flattened_params should be of shape ({self.dim},), but got {flattened_params.shape}"
            )
        # Compute the quadratic form
        Q = torch.diag(self.Q)
        loss = flattened_params @ Q @ flattened_params
        return loss


class DiagPSDMatrixRidgeBias(nn.Module):
    """
    A numerically stable, efficient, and guaranteed-PSD bias model.

    - Efficient: Avoids creating a large diagonal matrix in memory.
    - PSD: Enforces positivity on the diagonal of Q via re-parameterization.
    """

    def __init__(self, dim: int, Q_init_diag: torch.Tensor = None):
        super().__init__()
        self.dim = dim

        if Q_init_diag is not None:
            # If initializing with a Q, store its log.
            # Add a small epsilon for numerical stability if some values are zero.
            initial_values = torch.log(Q_init_diag + 1e-8)
        else:
            # Start with a default of zeros (which corresponds to Q_diag of 1s).
            initial_values = torch.zeros(dim)

        # The learnable parameter is the LOG of the diagonal of Q.
        self.log_Q_diag = nn.Parameter(initial_values)

    def forward(self, flattened_params: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Efficiently computes the quadratic form βᵀQβ.
        """
        if flattened_params.shape != (self.dim,):
            raise ValueError(
                f"Expected flattened_params of shape ({self.dim},), but got {flattened_params.shape}"
            )

        # 1. Get the positive diagonal of Q by exponentiating the learned log-diagonal.
        # This is the re-parameterization trick that guarantees Q is positive definite.
        Q_diag = torch.exp(self.log_Q_diag)

        # 2. Compute the loss efficiently.
        # This is equivalent to `βᵀ * diag(Q_diag) * β` but avoids creating a huge matrix.
        loss = torch.sum(Q_diag * flattened_params.pow(2))

        return loss

    # """
    # Learns a PSD matrix Q of shape (dim, dim) by
    # parameterizing a lower-triangular matrix L and returning
    # Q = L @ L.T.
    # """

    # def __init__(self, dim: int):
    #     super().__init__()
    #     self.dim = dim
    #     # We'll store the unconstrained lower-triangular entries
    #     # Initialize with small random values
    #     L_init = 0.01 * torch.randn(dim, dim)
    #     # We'll force it to be lower-triangular in the forward pass
    #     self.L_unconstrained = nn.Parameter(L_init, requires_grad=True)

    # def forward(self, flattened_params: torch.Tensor, **kwargs):
    #     if flattened_params.shape != (self.dim,):
    #         raise ValueError(
    #             f"flattened_params should be of shape ({self.dim},), "
    #             f"but got {flattened_params.shape}"
    #         )
    #     # Create L as strictly lower-triangular or lower-triangular with diagonal
    #     L = torch.tril(self.L_unconstrained)
    #     # Then Q = L L^T is guaranteed symmetric PSD
    #     Q = L @ L.T
    #     loss = flattened_params @ Q @ flattened_params
    #     return loss


class ElasticNet(nn.Module):
    def __init__(
        self,
        lambda_1_init: float = 1.0,
        lambda_2_init: float = 1.0,
        smooth: float = 0.1,
        eps: float = 0.0,  # optional floor to avoid exact 0 on logs/plots
    ) -> None:
        super().__init__()
        # Trainable unconstrained parameters (log-space)
        self.theta_1 = nn.Parameter(torch.tensor(math.log(max(lambda_1_init, 1e-12))))
        self.theta_2 = nn.Parameter(torch.tensor(math.log(max(lambda_2_init, 1e-12))))
        self.smooth = float(smooth)
        self.eps = float(eps)

    # Positive lambdas exposed as properties
    @property
    def lambda_1(self) -> Tensor:
        return torch.exp(self.theta_1) + self.eps

    @property
    def lambda_2(self) -> Tensor:
        return torch.exp(self.theta_2) + self.eps

    def lambdas(self) -> tuple[Tensor, Tensor]:
        return self.lambda_1, self.lambda_2

    def forward(self, flattened_params: Tensor, **kwargs) -> Tensor:
        lam1, lam2 = self.lambdas()  # strictly > 0
        l1_penalty = lam1 * F.smooth_l1_loss(
            flattened_params,
            torch.zeros_like(flattened_params),
            beta=self.smooth,
            reduction="sum",
        )
        l2_penalty = lam2 * torch.sum(flattened_params**2)
        return l1_penalty + l2_penalty


# class ElasticNet(nn.Module):
#     def __init__(
#         self,
#         lambda_1_init: float = 1.0,
#         lambda_2_init: float = 1.0,
#         smooth: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.lambda_1 = nn.Parameter(torch.tensor([0.5]))  # Initialize parameters
#         self.lambda_2 = nn.Parameter(torch.tensor([0.5]))  # Initialize parameters
#         self.smooth = smooth  # smoothness parameter, not learnable

#     def forward(self, flattened_params, **kwargs):
#         # flattened_params = torch.cat([p.view(-1) for p in params.values()])
#         l1_penalty = self.lambda_1 * torch.nn.functional.smooth_l1_loss(
#             flattened_params,
#             torch.zeros_like(flattened_params),
#             beta=self.smooth,
#             reduction="sum",
#         )
#         l2_penalty = self.lambda_2 * torch.sum(flattened_params**2)
#         return l1_penalty + l2_penalty


# class DropoutBias(nn.Module):
#     def __init__(self, q: float = 0.5) -> None:
#         super().__init__()
#         self.q = q
#         self.implicit = nn.Parameter(torch.tensor([0.0]))  # Initialize parameters
#         self.explicit = nn.Parameter(torch.tensor([0.0]))  # Initialize parameters

#     def compute_explicit_term(self, model):
#         def F(params, inputs):
#             return functional_call(model, params, (inputs,))


#         pass

#     def compute_implicit_term(self, model):
#         pass

#     def forward(self, flattened_params: torch.Tensor) -> torch.Tensor:
#         # Apply dropout to the flattened parameters
#         mask = torch.bernoulli(
#             torch.full(flattened_params.shape, 1 - self.dropout_rate)
#         ).to(flattened_params.device)
#         return torch.sum(mask * flattened_params)


# class ImplicitBiasKRR(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         lambda_: float = 0.0,
#         Q_t_init: torch.Tensor = None,
#         omega_t_init: torch.Tensor = None,
#     ):
#         """
#         Initializes the implicit bias module for Kernel Ridge Regression.

#         Args:
#             lambda_ (float): Ridge regularization parameter for KRR.
#             dim (int): The dimension of the alpha vector (typically n, the number of samples).
#             Q_t_init (torch.Tensor, optional): Initial value for the Q_t matrix.
#                                                Should be of shape (dim, dim).
#                                                Defaults to an identity matrix if None.
#             omega_t_init (torch.Tensor, optional): Initial value for the omega_t vector.
#                                                    Should be of shape (dim,).
#                                                    Defaults to a zero vector if None.
#         """
#         super().__init__()
#         self.dim = dim
#         self.lambda_ = lambda_

#         if Q_t_init is not None:
#             if Q_t_init.shape != (self.dim, self.dim):
#                 raise ValueError(
#                     f"Q_t_init should be of shape ({self.dim}, {self.dim}), but got {Q_t_init.shape}"
#                 )
#             self.Q_t = nn.Parameter(Q_t_init.clone(), requires_grad=True)
#         else:
#             # Default initialization for Q_t (e.g., identity matrix)
#             self.Q_t = nn.Parameter(torch.eye(self.dim), requires_grad=True)

#         if omega_t_init is not None:
#             if omega_t_init.shape != (self.dim,):
#                 raise ValueError(
#                     f"omega_t_init should be of shape ({self.dim},), but got {omega_t_init.shape}"
#                 )
#             self.omega_t = nn.Parameter(omega_t_init.clone(), requires_grad=True)
#         else:
#             # Default initialization for omega_t (e.g., zero vector)
#             self.omega_t = nn.Parameter(torch.zeros(self.dim), requires_grad=True)

#     def forward(self, alpha: torch.Tensor, **kwargs) -> torch.Tensor:
#         """
#         Compute the implicit bias term: alpha^T Q_t alpha - 2 * omega_t^T alpha.

#         Args:
#             alpha (torch.Tensor): The parameter vector alpha, shape (dim,).

#         Returns:
#             torch.Tensor: The scalar value of the implicit bias term.
#         """
#         if alpha.shape != (self.dim,):
#             # If alpha is (dim, 1) or (1, dim), try to reshape.
#             if alpha.numel() == self.dim:
#                 alpha = alpha.view(self.dim)
#             else:
#                 raise ValueError(
#                     f"alpha should be of shape ({self.dim},) or be reshapeable to it, but got {alpha.shape}"
#                 )

#         # Quadratic term: alpha^T Q_t alpha
#         quadratic_term = alpha @ self.Q_t @ alpha

#         # Linear term: -2 * omega_t^T alpha
#         linear_term = -2 * self.omega_t @ alpha

#         return quadratic_term + linear_term


class DiagMatrixBiasKRR(nn.Module):
    def __init__(
        self,
        Q_t_init: torch.Tensor = None,
        dim: Optional[int] = None,
        t: int = 0,
        K: torch.Tensor = None,
        alpha_init: torch.Tensor = None,
        eta: float = 1e-2,
        lambda_: float = 0.0,
    ):
        """
        Initializes the implicit bias module for Kernel Ridge Regression,
        with a diagonal Q_t matrix. Computes the linear term based on the full
        theoretical Q_t matrix, and we learn the diagonal entries of Q_t for the quadratic term.


        Args:
            Q_t_init (torch.Tensor, optional): Initial value for the Q_t vector.
                                               Should be of shape (dim,).
                                               Defaults to a zero vector if None.
            dim (int, optional): The dimension of the alpha vector (typically n, the number of samples).
                                 If Q_t_init is provided, this can be None.
            t (int): The time step for the implicit bias computation.
            K (torch.Tensor): The kernel matrix of shape (n, n).
                              Must be provided and square.
            alpha_init (torch.Tensor): Initial value for the alpha vector.
            `eta` (float): Learning rate for the implicit bias computation.
            `lambda_` (float): Ridge regularization parameter for KRR
        """
        super().__init__()
        if K is None:
            raise ValueError("Must provide kernel matrix K")
        self.register_buffer("K", K)
        self.register_buffer("alpha_init", alpha_init)
        self.t = t
        self.eta = eta
        self.lambda_ = lambda_

        if Q_t_init is not None:
            if Q_t_init.ndim != 1:
                raise ValueError(
                    f"Q_t_init should be a 1D tensor of shape ({self.dim},), but got {Q_t_init.shape}"
                )
            if dim is not None and Q_t_init.shape[0] != dim:
                raise ValueError(
                    f"dim ({dim}) does not match Q_t_init shape ({Q_t_init.shape[0]})."
                )
            self.dim = Q_t_init.shape[0]
        else:
            if dim is None:
                raise ValueError("dim must be specified if Q_t_init is not provided.")
            self.dim = dim

        assert K.shape[0] == K.shape[1], "K must be a square matrix"
        assert K.shape[0] == self.dim, "K must match the dimension of Q_t_init"

        # Initialize the Q_t matrix diagonal
        self.Q_t = nn.Parameter(
            Q_t_init
            if Q_t_init is not None
            else torch.zeros(self.dim, device=self.K.device),
            requires_grad=True,
        )

    def forward(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the implicit bias term: alpha^T Q_t alpha - 2 * omega_t^T alpha.

        Args:
            alpha (torch.Tensor): The parameter vector alpha, shape (dim,).

        Returns:
            torch.Tensor: The scalar value of the implicit bias term.
        """
        device = self.Q_t.device
        if alpha.shape != (self.dim,):
            # If alpha is (dim, 1) or (1, dim), try to reshape.
            if alpha.numel() == self.dim:
                alpha = alpha.view(self.dim)
            else:
                raise ValueError(
                    f"alpha should be of shape ({self.dim},) or be reshapeable to it, but got {alpha.shape}"
                )
        I = torch.eye(self.dim, device=device)
        C = (1.0 / self.dim) * (self.K @ self.K) + self.lambda_ * self.K
        A = I - 2 * self.eta * C
        A_t = torch.linalg.matrix_power(A, self.t)
        Q_t_exact = C @ (torch.linalg.inv(I - A_t) - I)
        # omega_t = (C + torch.diag(self.Q_t)) @ A_t @ self.alpha_init
        omega_t_exact = (C + Q_t_exact) @ A_t @ self.alpha_init

        # Linear term: -2 * omega_t^T alpha
        linear_term = -2 * omega_t_exact @ alpha
        # Quadratic term: alpha^T Q_t alpha
        # quadratic_term = torch.sum(alpha**2 * self.Q_t)
        quadratic_term = alpha @ torch.diag(self.Q_t) @ alpha
        # Ridge term: lambda_ * alpha^T K alpha
        if self.lambda_ > 0:
            ridge_term = self.lambda_ * (alpha @ self.K @ alpha)
            quadratic_term += ridge_term

        # check if alpha_init is all zeros
        # if torch.all(self.alpha_init == 0):
        #     print("linear_term (should be zero):", linear_term)
        return linear_term + quadratic_term


class DiagMatrixBiasNTKRR(nn.Module):
    def __init__(
        self,
        Q_t_init: Optional[torch.Tensor] = None,
        dim: Optional[int] = None,
        t: int = 0,
        K: torch.Tensor = None,
        alpha_init: torch.Tensor = None,
        eta: float = 1e-2,
        lambda_: float = 0.0,
    ):
        """
        Initializes the implicit bias module for NTK Ridge Regression,
        with a diagonal Q_t matrix. Computes the linear term based on the full
        theoretical Q_t matrix, and we learn the diagonal entries of Q_t for the quadratic term.

        Args:
            Q_t_init (torch.Tensor, optional): Initial value for the Q_t vector.
                                               Should be of shape (dim,).
                                               Defaults to a zero vector if None.
            dim (int, optional): The dimension of the alpha vector (typically n, the number of samples).
                                 If Q_t_init is provided, this can be None.
            t (int): The time step for the implicit bias computation.
            K (torch.Tensor): The kernel matrix of shape (n, n).
                              Must be provided and square.
            alpha_init (torch.Tensor): Initial value for the alpha vector.
            `eta` (float): Learning rate for the implicit bias computation.
            `lambda_` (float): Ridge regularization parameter for KRR
        """
        super().__init__()
        if K is None:
            raise ValueError("Must provide kernel matrix K")
        self.register_buffer("K", K)
        self.register_buffer("alpha_init", alpha_init)
        self.t = t
        self.eta = eta
        self.lambda_ = lambda_

        if Q_t_init is not None:
            if Q_t_init.ndim != 1:
                raise ValueError(
                    f"Q_t_init should be a 1D tensor of shape ({self.dim},), but got {Q_t_init.shape}"
                )
            if dim is not None and Q_t_init.shape[0] != dim:
                raise ValueError(
                    f"dim ({dim}) does not match Q_t_init shape ({Q_t_init.shape[0]})."
                )
            self.dim = Q_t_init.shape[0]
        else:
            if dim is None:
                raise ValueError("dim must be specified if Q_t_init is not provided.")
            self.dim = dim
        self.Q_t = nn.Parameter(
            Q_t_init
            if Q_t_init is not None
            else torch.zeros(self.dim, device=self.K.device),
            requires_grad=True,
        )

    def forward(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = self.K.device

        if alpha.shape != (self.dim,):
            # If alpha is (dim, 1) or (1, dim), try to reshape.
            if alpha.numel() == self.dim:
                alpha = alpha.view(self.dim)
            else:
                raise ValueError(
                    f"alpha should be of shape ({self.dim},) or be reshapeable to it, but got {alpha.shape}"
                )
        I = torch.eye(self.dim, device=device)
        D = (1.0 / self.dim) * self.K + self.lambda_ * I
        A = I - 2 * self.eta * D
        A_t = torch.linalg.matrix_power(A, self.t)
        Q_t_exact = D @ (torch.linalg.inv(I - A_t) - I)
        omega_t_exact = (D + Q_t_exact) @ A_t @ self.alpha_init

        # Linear term: -2 * omega_t^T alpha
        linear_term = -2 * omega_t_exact @ alpha
        # Quadratic term: alpha^T Q_t alpha
        quadratic_term = alpha @ torch.diag(self.Q_t) @ alpha
        # Ridge term: lambda_ * alpha^T K alpha
        if self.lambda_ > 0:
            ridge_term = self.lambda_ * (alpha @ self.K @ alpha)
            quadratic_term += ridge_term
        return linear_term + quadratic_term
