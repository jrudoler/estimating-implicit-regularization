import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import abstractmethod
from torch.func import functional_call, vjp, hessian
from torch.optim import Optimizer
from typing import Dict, Callable, Tuple, Any


# GOAL: implement a class of models that represent a parametrization of the inductive
# bias of the model. This class should be able to be used with any pretrained model with
# known activations / weights.


class RidgeBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([1.0]))  # Initialize parameter

    def forward(self, flattened_params: torch.Tensor):
        return self.beta * torch.sum(flattened_params**2)


class LassoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([1.0]))  # Initialize parameters

    def forward(self, flattened_params):
        return self.alpha * torch.sum(torch.abs(flattened_params))


class SmoothLassoBias(nn.Module):
    def __init__(self, alpha_init: float = 1.0, smooth: float = 0.1) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))  # Initialize parameters
        self.smooth = smooth  # smoothness parameter, not learnable

    def forward(self, flattened_params: torch.Tensor) -> torch.Tensor:
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

    def forward(self, flattened_params: torch.Tensor):
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

    def forward(self, flattened_params: torch.Tensor):
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


class ElasticNet(nn.Module):
    def __init__(
        self,
        lambda_1_init: float = 1.0,
        lambda_2_init: float = 1.0,
        smooth: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_1 = nn.Parameter(torch.tensor([0.5]))  # Initialize parameters
        self.lambda_2 = nn.Parameter(torch.tensor([0.5]))  # Initialize parameters
        self.smooth = smooth  # smoothness parameter, not learnable

    def forward(self, flattened_params):
        # flattened_params = torch.cat([p.view(-1) for p in params.values()])
        l1_penalty = self.lambda_1 * torch.nn.functional.smooth_l1_loss(
            flattened_params,
            torch.zeros_like(flattened_params),
            beta=self.smooth,
            reduction="sum",
        )
        l2_penalty = self.lambda_2 * torch.sum(flattened_params**2)
        return l1_penalty + l2_penalty


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
