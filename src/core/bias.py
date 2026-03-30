from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, Tuple, Any, Optional, Iterable, List
from abc import abstractmethod


class Bias(nn.Module):
    @property
    @abstractmethod
    def bias_name(self) -> str:
        pass

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, flattened_params, structured_params, **kwargs):
        pass

    @abstractmethod
    def get_bias_params(self) -> Dict[str, float]:
        pass


class GradientSquaredPenaltyScale(Bias):
    """Learns a scalar multiplier for squared gradient penalties."""

    bias_name = "grad_squared"

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

    def get_bias_params(self) -> Dict[str, float]:
        val = (
            torch.exp(self.lambda_param).item()
            if self.enforce_positive
            else self.lambda_param.item()
        )
        return {"scale": val}


# GOAL: implement a class of models that represent a parametrization of the inductive
# bias of the model. This class should be able to be used with any pretrained model with
# known activations / weights.


class ScalarBias(Bias):
    bias_name = "scalar"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__()
        self.beta = nn.Parameter(
            torch.tensor(init_value) if init_value is not None else torch.randn(1) * 0.1
        )  # Initialize parameter
        self.enforce_positive = enforce_positive

    @abstractmethod
    def forward(self, flattened_params, structured_params, **kwargs):
        pass

    def get_bias_params(self) -> Dict[str, float]:
        if self.enforce_positive:
            return {"scale": torch.exp(self.beta).item()}
        else:
            return {"scale": self.beta.item()}


class SmoothedPowerBias(Bias):
    """
    Smoothed power-family regularizer:

        R_p(θ) = λ Σ_j (θ_j² + ε)^(p / 2)

    The scale λ and exponent p can be fixed or trainable. When p is trainable,
    it is constrained to [p_min, p_max] via a sigmoid reparameterization.
    """

    bias_name = "smoothed_power"

    def __init__(
        self,
        scale_init: float = 1.0,
        exponent_init: float = 2.0,
        epsilon: float = 1e-8,
        *,
        trainable_scale: bool = True,
        trainable_exponent: bool = False,
        p_min: float = 0.25,
        p_max: float = 4.0,
    ) -> None:
        super().__init__()
        if scale_init <= 0:
            raise ValueError("scale_init must be positive.")
        if exponent_init <= 0:
            raise ValueError("exponent_init must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if p_min <= 0 or p_max <= 0 or p_min >= p_max:
            raise ValueError("Require 0 < p_min < p_max for trainable exponents.")

        self.epsilon = float(epsilon)
        self.trainable_scale = bool(trainable_scale)
        self.trainable_exponent = bool(trainable_exponent)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        scale_tensor = torch.tensor(float(scale_init), dtype=torch.float32)
        if self.trainable_scale:
            self.log_scale = nn.Parameter(torch.log(scale_tensor))
        else:
            self.register_buffer("fixed_scale", scale_tensor)

        exponent_tensor = torch.tensor(float(exponent_init), dtype=torch.float32)
        if self.trainable_exponent:
            clipped = exponent_tensor.clamp(min=self.p_min + 1e-6, max=self.p_max - 1e-6)
            normalized = (clipped - self.p_min) / (self.p_max - self.p_min)
            self.raw_exponent = nn.Parameter(torch.logit(normalized))
        else:
            self.register_buffer("fixed_exponent", exponent_tensor)

    @property
    def scale(self) -> Tensor:
        if self.trainable_scale:
            return torch.exp(self.log_scale)
        return self.fixed_scale

    @property
    def exponent(self) -> Tensor:
        if self.trainable_exponent:
            return self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self.raw_exponent)
        return self.fixed_exponent

    def elementwise_penalty(self, flattened_params: Tensor) -> Tensor:
        exponent = self.exponent.to(device=flattened_params.device, dtype=flattened_params.dtype)
        epsilon = torch.tensor(self.epsilon, device=flattened_params.device, dtype=flattened_params.dtype)
        return (flattened_params.pow(2) + epsilon).pow(exponent / 2.0)

    def transformed_parameters(self, flattened_params: Tensor) -> Tensor:
        exponent = self.exponent.to(device=flattened_params.device, dtype=flattened_params.dtype)
        epsilon = torch.tensor(self.epsilon, device=flattened_params.device, dtype=flattened_params.dtype)
        return (flattened_params.pow(2) + epsilon).pow(exponent / 4.0)

    def forward(self, flattened_params: Tensor, structured_params=None, **kwargs) -> Tensor:
        scale = self.scale.to(device=flattened_params.device, dtype=flattened_params.dtype)
        return scale * self.elementwise_penalty(flattened_params).sum()

    def penalty_gradient(self, flattened_params: Tensor) -> Tensor:
        exponent = self.exponent.to(device=flattened_params.device, dtype=flattened_params.dtype)
        scale = self.scale.to(device=flattened_params.device, dtype=flattened_params.dtype)
        epsilon = torch.tensor(self.epsilon, device=flattened_params.device, dtype=flattened_params.dtype)
        base = flattened_params.pow(2) + epsilon
        return scale * exponent * flattened_params * base.pow(exponent / 2.0 - 1.0)

    def get_bias_params(self) -> Dict[str, float]:
        return {
            "scale": float(self.scale.detach().item()),
            "exponent": float(self.exponent.detach().item()),
        }


class SmoothedSchattenBias(Bias):
    """
    Smoothed Schatten power-family regularizer over weight matrices:

        R_p(W) = λ Σ_layers Σ_i (σ_i(W)^2 + ε)^(p / 2)

    where σ_i(W) are the singular values of each matrix-shaped parameter tensor.
    """

    bias_name = "smoothed_schatten"

    def __init__(
        self,
        scale_init: float = 1.0,
        exponent_init: float = 1.0,
        epsilon: float = 1e-8,
        *,
        trainable_scale: bool = True,
        trainable_exponent: bool = False,
        p_min: float = 0.25,
        p_max: float = 4.0,
    ) -> None:
        super().__init__()
        if scale_init <= 0:
            raise ValueError("scale_init must be positive.")
        if exponent_init <= 0:
            raise ValueError("exponent_init must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if p_min <= 0 or p_max <= 0 or p_min >= p_max:
            raise ValueError("Require 0 < p_min < p_max for trainable exponents.")

        self.epsilon = float(epsilon)
        self.trainable_scale = bool(trainable_scale)
        self.trainable_exponent = bool(trainable_exponent)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        scale_tensor = torch.tensor(float(scale_init), dtype=torch.float32)
        if self.trainable_scale:
            self.log_scale = nn.Parameter(torch.log(scale_tensor))
        else:
            self.register_buffer("fixed_scale", scale_tensor)

        exponent_tensor = torch.tensor(float(exponent_init), dtype=torch.float32)
        if self.trainable_exponent:
            clipped = exponent_tensor.clamp(min=self.p_min + 1e-6, max=self.p_max - 1e-6)
            normalized = (clipped - self.p_min) / (self.p_max - self.p_min)
            self.raw_exponent = nn.Parameter(torch.logit(normalized))
        else:
            self.register_buffer("fixed_exponent", exponent_tensor)

    @property
    def scale(self) -> Tensor:
        if self.trainable_scale:
            return torch.exp(self.log_scale)
        return self.fixed_scale

    @property
    def exponent(self) -> Tensor:
        if self.trainable_exponent:
            return self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self.raw_exponent)
        return self.fixed_exponent

    def singular_value_penalty_terms(self, singular_values: Tensor) -> Tensor:
        exponent = self.exponent.to(device=singular_values.device, dtype=singular_values.dtype)
        epsilon = torch.tensor(self.epsilon, device=singular_values.device, dtype=singular_values.dtype)
        return (singular_values.pow(2) + epsilon).pow(exponent / 2.0)

    def transformed_singular_values(self, singular_values: Tensor) -> Tensor:
        exponent = self.exponent.to(device=singular_values.device, dtype=singular_values.dtype)
        epsilon = torch.tensor(self.epsilon, device=singular_values.device, dtype=singular_values.dtype)
        return (singular_values.pow(2) + epsilon).pow(exponent / 4.0)

    def matrix_penalty(self, matrix: Tensor) -> Tensor:
        singular_values = torch.linalg.svdvals(matrix)
        return self.singular_value_penalty_terms(singular_values).sum()

    def forward(
        self,
        flattened_params: Tensor,
        structured_params: List[Tensor],
        **kwargs,
    ) -> Tensor:
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        for parameter in structured_params:
            if parameter.ndim < 2:
                continue
            matrix = parameter if parameter.ndim == 2 else parameter.reshape(parameter.shape[0], -1)
            penalty = penalty + self.matrix_penalty(matrix)
        scale = self.scale.to(device=flattened_params.device, dtype=flattened_params.dtype)
        return scale * penalty

    def penalty_gradient(
        self,
        flattened_params: Tensor,
        structured_params: List[Tensor],
    ) -> Tensor:
        exponent = self.exponent.to(device=flattened_params.device, dtype=flattened_params.dtype)
        scale = self.scale.to(device=flattened_params.device, dtype=flattened_params.dtype)
        epsilon = torch.tensor(self.epsilon, device=flattened_params.device, dtype=flattened_params.dtype)

        gradients: List[Tensor] = []
        for parameter in structured_params:
            if parameter.ndim < 2:
                gradients.append(torch.zeros_like(parameter).reshape(-1))
                continue

            matrix = parameter if parameter.ndim == 2 else parameter.reshape(parameter.shape[0], -1)
            left, singular_values, right_t = torch.linalg.svd(matrix, full_matrices=False)
            weights = exponent * singular_values * (singular_values.pow(2) + epsilon).pow(exponent / 2.0 - 1.0)
            grad_matrix = scale * ((left * weights.unsqueeze(0)) @ right_t)
            if parameter.ndim == 2:
                gradients.append(grad_matrix.reshape(-1))
            else:
                gradients.append(grad_matrix.reshape_as(parameter).reshape(-1))

        return torch.cat(gradients)

    def get_bias_params(self) -> Dict[str, float]:
        return {
            "scale": float(self.scale.detach().item()),
            "exponent": float(self.exponent.detach().item()),
        }


class OrthogonalBias(ScalarBias):
    bias_name = "orthogonal"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        penalty = torch.tensor(0.0, device=flattened_params.device)

        for w in structured_params:
            if w.ndim == 2:
                # Rows: features, Cols: dim
                # Check if we want row or col orthogonality.
                # Usually row orthogonality for Linear layers (neurons are orthogonal)
                # If W is (out_features, in_features):

                rows, cols = w.shape
                if rows > cols:
                    # Cannot be orthogonal if rows > cols
                    continue

                gram = torch.mm(w, w.t())  # (out, out)
                eye = torch.eye(rows, device=w.device)
                penalty = penalty + torch.norm(gram - eye, p="fro") ** 2

        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale * penalty


class NuclearNormBias(Bias):
    bias_name = "nuclear_norm"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__()
        self.beta = nn.Parameter(
            torch.tensor(init_value) if init_value is not None else torch.randn(1) * 0.1
        )  # Initialize parameter
        self.enforce_positive = enforce_positive

    def forward(
        self,
        flattened_params: torch.Tensor,
        structured_params: List[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        for weight in structured_params:
            if weight.ndim >= 2:
                matrix = (
                    weight if weight.ndim == 2 else weight.reshape(weight.shape[0], -1)
                )
                penalty = penalty + torch.linalg.matrix_norm(matrix, ord="nuc")
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty

    def get_bias_params(self) -> Dict[str, float]:
        if self.enforce_positive:
            return {"scale": torch.exp(self.beta).item()}
        else:
            return {"scale": self.beta.item()}


# =============================================================================
# Scale-Invariant Bias Regularizers (Orthogonal to L2)
# =============================================================================
# These regularizers have gradients orthogonal to W (and thus to L2 gradient = 2W)
# because they are scale-invariant: f(αW) = f(W) for all α > 0.
# This property follows from: ⟨∇f(W), W⟩ = d/dα f(αW)|_{α=1} = 0


class StableRankBias(ScalarBias):
    """
    Stable rank regularizer: srank(W) = ||W||_F² / ||W||_2²

    - Range: [1, min(rows, cols)]
    - Scale-invariant: gradient is orthogonal to W
    - Interpretation: Effective number of significant singular values
    - Literature: Used in matrix concentration inequalities (Vershynin, 2018)
    """

    bias_name = "stable_rank"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        count = 0
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                fro_sq = torch.sum(matrix**2)
                spectral_sq = torch.linalg.matrix_norm(matrix, ord=2) ** 2
                # Stable rank = ||W||_F² / ||W||_2²
                penalty = penalty + fro_sq / (spectral_sq + 1e-8)
                count += 1
        # Average across matrices for consistency
        if count > 0:
            penalty = penalty / count
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty


class SpectralEntropyBias(ScalarBias):
    """
    Spectral entropy regularizer: H(W) = -Σ pᵢ log(pᵢ)
    where pᵢ = σᵢ² / ||W||_F² (normalized squared singular values)

    - Scale-invariant: gradient is orthogonal to W
    - Range: [0, log(min(rows, cols))]
    - Interpretation: How spread out is the spectral energy?
    - Low entropy = energy concentrated in few singular values (low rank)
    - High entropy = energy spread across many singular values
    """

    bias_name = "spectral_entropy"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        count = 0
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                # Get singular values (efficient, no full SVD)
                sigmas = torch.linalg.svdvals(matrix)
                # Compute normalized squared singular values (probabilities)
                sigma_sq = sigmas**2
                total = sigma_sq.sum() + 1e-8
                probs = sigma_sq / total
                # Clamp to avoid log(0)
                probs = torch.clamp(probs, min=1e-10)
                # Entropy: -Σ pᵢ log(pᵢ)
                entropy = -torch.sum(probs * torch.log(probs))
                penalty = penalty + entropy
                count += 1
        if count > 0:
            penalty = penalty / count
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty


class WeightCoherenceBias(ScalarBias):
    """
    Weight coherence regularizer: coh(W) = ||W^T W - diag(W^T W)||_F² / ||W||_F⁴

    - Scale-invariant: gradient is orthogonal to W
    - Measures correlation between weight columns (off-diagonal of Gram matrix)
    - Interpretation: Penalizes redundant/correlated features
    - Literature: Related to coherence in compressed sensing
    """

    bias_name = "weight_coherence"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        count = 0
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                # Gram matrix: W^T W
                gram = matrix.T @ matrix
                # Off-diagonal part: Gram - diag(Gram)
                diag_gram = torch.diag(torch.diag(gram))
                off_diag = gram - diag_gram
                # Normalized coherence
                fro_sq = torch.sum(matrix**2)
                coherence = torch.sum(off_diag**2) / (fro_sq**2 + 1e-8)
                penalty = penalty + coherence
                count += 1
        if count > 0:
            penalty = penalty / count
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty


class RowNormVarianceBias(ScalarBias):
    """
    Row norm variance regularizer: Var(||w_i||² / mean(||w_j||²))

    - Scale-invariant: gradient is orthogonal to W
    - Interpretation: Prefers balanced weight magnitudes across neurons/rows
    - Low variance = all rows have similar norms (balanced network)
    - High variance = some rows dominate (imbalanced)
    """

    bias_name = "row_norm_variance"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        count = 0
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                # Compute squared row norms
                row_norms_sq = torch.sum(matrix**2, dim=1)
                # Normalize by mean to make scale-invariant
                mean_norm_sq = row_norms_sq.mean() + 1e-8
                normalized_norms = row_norms_sq / mean_norm_sq
                # Variance of normalized norms
                variance = torch.var(normalized_norms)
                penalty = penalty + variance
                count += 1
        if count > 0:
            penalty = penalty / count
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty


class SpectralGapBias(ScalarBias):
    """
    Spectral gap regularizer: log(σ₁ / σ₂)

    - Scale-invariant: gradient is orthogonal to W
    - Interpretation: How dominant is the principal component?
    - Large gap = low effective rank, first singular value dominates
    - Small gap = more distributed spectrum
    """

    bias_name = "spectral_gap"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        penalty = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        count = 0
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                # Need at least 2 singular values
                min_dim = min(matrix.shape)
                if min_dim < 2:
                    continue
                # Get top 2 singular values
                sigmas = torch.linalg.svdvals(matrix)
                sigma_1 = sigmas[0]
                sigma_2 = sigmas[1] + 1e-8  # Avoid log(0)
                # Log ratio is scale-invariant
                gap = torch.log(sigma_1 / sigma_2 + 1e-8)
                penalty = penalty + gap
                count += 1
        if count > 0:
            penalty = penalty / count
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty


# =============================================================================
# Gradient-Based Bias Regularizers
# =============================================================================
# These regularizers measure properties of the loss gradient or use gradients
# to capture training dynamics. They require gradient information via kwargs.


class GradientEntropyBias(ScalarBias):
    """
    Gradient entropy regularizer: H(|∇|) = -Σ pᵢ log(pᵢ)
    where pᵢ = |∇ᵢ|² / ||∇||²

    - Measures how spread out the gradient energy is across parameters
    - Low entropy = gradient concentrated in few parameters (sparse updates)
    - High entropy = gradient spread across many parameters (dense updates)
    - Scale-invariant w.r.t. gradient magnitude

    Requires: kwargs["grad_vector"] - flattened gradient tensor
    """

    bias_name = "gradient_entropy"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        grad_vector = kwargs.get("grad_vector")
        if grad_vector is None:
            # Return zero penalty if no gradient provided
            return torch.zeros(
                (), device=flattened_params.device, dtype=flattened_params.dtype
            )

        # Compute normalized squared gradient magnitudes (probabilities)
        grad_sq = grad_vector**2
        total = grad_sq.sum() + 1e-8
        probs = grad_sq / total
        # Clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-10)
        # Entropy: -Σ pᵢ log(pᵢ)
        entropy = -torch.sum(probs * torch.log(probs))

        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * entropy


class WeightGradientAlignmentBias(ScalarBias):
    """
    Weight-gradient alignment regularizer: cos(W, ∇) = <W, ∇> / (||W|| ||∇||)

    - Measures alignment between weights and their gradients
    - Positive alignment: weights moving in direction of gradient
    - Negative alignment: weights moving against gradient
    - At a minimum, gradient should be ~0, so this captures trajectory info
    - Scale-invariant (cosine similarity)

    Requires: kwargs["grad_vector"] - flattened gradient tensor
    """

    bias_name = "weight_gradient_alignment"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        grad_vector = kwargs.get("grad_vector")
        if grad_vector is None:
            return torch.zeros(
                (), device=flattened_params.device, dtype=flattened_params.dtype
            )

        # Cosine similarity between weights and gradients
        w_norm = torch.norm(flattened_params) + 1e-8
        g_norm = torch.norm(grad_vector) + 1e-8
        alignment = torch.dot(flattened_params, grad_vector) / (w_norm * g_norm)

        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * alignment


class GradientSparsityBias(ScalarBias):
    """
    Gradient sparsity regularizer: ||∇||₁ / ||∇||₂

    - Ratio of L1 to L2 norm of gradients
    - Range: [1, sqrt(d)] where d is number of parameters
    - Low value = sparse gradient (few large components)
    - High value = dense gradient (many similar-magnitude components)
    - Scale-invariant

    Requires: kwargs["grad_vector"] - flattened gradient tensor
    """

    bias_name = "gradient_sparsity"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        grad_vector = kwargs.get("grad_vector")
        if grad_vector is None:
            return torch.zeros(
                (), device=flattened_params.device, dtype=flattened_params.dtype
            )

        l1_norm = torch.norm(grad_vector, p=1)
        l2_norm = torch.norm(grad_vector, p=2) + 1e-8
        # L1/L2 ratio (scale-invariant)
        sparsity_ratio = l1_norm / l2_norm

        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * sparsity_ratio


# =============================================================================
# Gradient-Flow Proxy Regularizers (Weight-Based)
# =============================================================================
# These measure properties related to gradient flow through the network
# but only require access to the weight matrices.


class LayerNormProductBias(ScalarBias):
    """
    Layer norm product regularizer: Π ||Wₗ||₂ (product of spectral norms)

    - Upper bound on the Lipschitz constant of the network
    - Related to gradient explosion: large product = gradients can explode
    - Literature: Spectral normalization (Miyato et al., 2018)
    - NOT scale-invariant (intentionally measures overall magnitude)

    Note: Uses log for numerical stability: log(Π σₘₐₓ) = Σ log(σₘₐₓ)
    """

    bias_name = "layer_norm_product"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        log_product = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        count = 0
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                spectral_norm = torch.linalg.matrix_norm(matrix, ord=2)
                log_product = log_product + torch.log(spectral_norm + 1e-8)
                count += 1

        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * log_product


class LayerNormBalanceBias(ScalarBias):
    """
    Layer norm balance regularizer: Var(log ||Wₗ||₂)

    - Measures how balanced the layer norms are
    - Low variance = balanced norms = healthy gradient flow
    - High variance = imbalanced norms = potential gradient issues
    - Scale-invariant (variance of log is invariant to global scaling)
    - Literature: Related to "balanced networks" (Du et al., 2018)
    """

    bias_name = "layer_norm_balance"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        log_norms = []
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                spectral_norm = torch.linalg.matrix_norm(matrix, ord=2)
                log_norms.append(torch.log(spectral_norm + 1e-8))

        if len(log_norms) < 2:
            return torch.zeros(
                (), device=flattened_params.device, dtype=flattened_params.dtype
            )

        log_norms_tensor = torch.stack(log_norms)
        variance = torch.var(log_norms_tensor)

        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * variance


class EffectiveDepthBias(ScalarBias):
    """
    Effective depth regularizer based on gradient flow.

    Measures: mean(σₘᵢₙ(Wₗ)) / mean(σₘₐₓ(Wₗ))

    - Ratio of average minimum to maximum singular values
    - High ratio = well-conditioned layers = gradients flow through all layers
    - Low ratio = poorly-conditioned = some layers block gradient flow
    - Scale-invariant
    """

    bias_name = "effective_depth"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__(enforce_positive=enforce_positive, init_value=init_value)

    def forward(self, flattened_params, structured_params, **kwargs):
        sigma_mins = []
        sigma_maxs = []
        for w in structured_params:
            if w.ndim >= 2:
                matrix = w if w.ndim == 2 else w.reshape(w.shape[0], -1)
                sigmas = torch.linalg.svdvals(matrix)
                sigma_maxs.append(sigmas[0])
                # Min singular value (could be 0 for rank-deficient)
                sigma_mins.append(sigmas[-1])

        if len(sigma_mins) == 0:
            return torch.zeros(
                (), device=flattened_params.device, dtype=flattened_params.dtype
            )

        mean_min = torch.stack(sigma_mins).mean()
        mean_max = torch.stack(sigma_maxs).mean() + 1e-8
        ratio = mean_min / mean_max

        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * ratio


class LowRankBias(Bias):
    bias_name = "low_rank"

    def __init__(self, enforce_positive: bool = True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([1.0]))  # Initialize parameter
        self.enforce_positive = enforce_positive

    def compute_rank(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        rank = torch.linalg.matrix_rank(weight_matrix)
        return rank.to(weight_matrix.dtype)

    def compute_average_rank(
        self,
        structured_params: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        total_rank = torch.zeros((), device=device, dtype=dtype)
        count = 0
        for weight in structured_params:
            matrix = weight if weight.ndim == 2 else weight.reshape(weight.shape[0], -1)
            total_rank = total_rank + self.compute_rank(matrix)
            count += 1
        if count == 0:
            return total_rank
        return total_rank / float(count)

    def forward(
        self,
        flattened_params: torch.Tensor,
        structured_params: List[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        average_rank = self.compute_average_rank(
            structured_params,
            device=flattened_params.device,
            dtype=flattened_params.dtype,
        )
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * average_rank

    def get_bias_params(self) -> Dict[str, float]:
        if self.enforce_positive:
            return {"scale": torch.exp(self.beta).item()}
        else:
            return {"scale": self.beta.item()}


def iter_weight_matrices(
    structured_params: List[torch.Tensor],
) -> Iterable[Tuple[str, torch.Tensor]]:
    for parameter in structured_params:
        if parameter.ndim >= 2:
            yield parameter


class JointBias(nn.Module):
    """Combines multiple bias models, summing their losses together."""

    def __init__(self, bias_models: Iterable[nn.Module]):
        super().__init__()
        self.bias_models = nn.ModuleList(list(bias_models))

    def forward(
        self,
        flattened_params: torch.Tensor,
        structured_params: List[torch.Tensor],
        return_components: bool = False,
        **kwargs,
    ):
        total = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        components = {}

        for idx, bias_model in enumerate(self.bias_models):
            key = getattr(bias_model, "bias_name", bias_model.__class__.__name__)
            if key in components:
                key = f"{key}_{idx}"

            val = bias_model(flattened_params, structured_params, **kwargs)
            total = total + val

            # only store component if requested
            if return_components:
                components[key] = val

        if return_components:
            return total, components
        else:
            return total

    def get_bias_params(self) -> Dict[str, float]:
        bias_params = {}
        used_prefixes = set()

        for idx, bias_model in enumerate(self.bias_models):
            base_name = bias_model.bias_name
            prefix = base_name
            if prefix in used_prefixes:
                prefix = f"{base_name}_{idx}"
            used_prefixes.add(prefix)

            if hasattr(bias_model, "get_bias_params"):
                params = bias_model.get_bias_params()
                if isinstance(params, dict):
                    for k, v in params.items():
                        bias_params[f"{prefix}/{k}"] = v
                else:
                    bias_params[prefix] = params
        return bias_params


class RidgeBias(Bias):
    bias_name = "ridge"

    def __init__(
        self, enforce_positive: bool = True, init_value: Optional[float] = None
    ):
        super().__init__()
        self.beta = nn.Parameter(
            torch.tensor(init_value) if init_value is not None else torch.randn(1) * 0.1
        )  # Initialize parameter
        self.enforce_positive = enforce_positive
        self.init_value = init_value

    def get_bias_params(self) -> Dict[str, float]:
        if self.enforce_positive:
            return {"scale": torch.exp(self.beta).item()}
        else:
            return {"scale": self.beta.item()}

    def forward(self, flattened_params: torch.Tensor, structured_params=None, **kwargs):
        penalty = torch.sum(flattened_params.pow(2))
        scale = torch.exp(self.beta) if self.enforce_positive else self.beta
        return scale.squeeze() * penalty


class LassoBias(Bias):
    bias_name = "lasso"

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([1.0]))  # Initialize parameters

    def forward(self, flattened_params, structured_params=None, **kwargs):
        return self.alpha * torch.sum(torch.abs(flattened_params))

    def get_bias_params(self) -> Dict[str, float]:
        return {"alpha": self.alpha.item()}


class SmoothLassoBias(Bias):
    bias_name = "smooth_lasso"

    def __init__(
        self,
        enforce_positive: bool = True,
        init_value: Optional[float] = None,
        smooth: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor([init_value]) if init_value is not None else None
        )  # Initialize parameters
        self.smooth = smooth  # smoothness parameter, not learnable
        self.enforce_positive = enforce_positive
        self.init_value = init_value

    def get_bias_params(self) -> Dict[str, float]:
        if self.enforce_positive:
            return {"scale": torch.exp(self.alpha).item()}
        else:
            return {"scale": self.alpha.item()}

    def forward(self, flattened_params: torch.Tensor, structured_params=None, **kwargs):
        smooth_loss = torch.nn.functional.smooth_l1_loss(
            flattened_params,
            torch.zeros_like(flattened_params),
            beta=self.smooth,
            reduction="sum",
        )
        scale = torch.exp(self.alpha) if self.enforce_positive else self.alpha
        return scale.squeeze() * smooth_loss


class MatrixRidgeBias(Bias):
    bias_name = "matrix_ridge"

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

    def forward(self, flattened_params: torch.Tensor, structured_params=None, **kwargs):
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

    def get_bias_params(self) -> Dict[str, float]:
        # Return something informative? Or just ignore for now since it's a big matrix?
        # User wants "name" logic fixed primarily.
        # For matrices, returning the whole thing is too much. Maybe just norm?
        return {"norm": self.Q.norm().item()}


class DiagMatrixRidgeBias(Bias):
    bias_name = "diag_matrix_ridge"

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

    def forward(self, flattened_params: torch.Tensor, structured_params=None, **kwargs):
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

    def get_bias_params(self) -> Dict[str, float]:
        return {"norm": self.Q.norm().item()}


class DiagPSDMatrixRidgeBias(Bias):
    """
    A numerically stable, efficient, and guaranteed-PSD bias model.

    - Efficient: Avoids creating a large diagonal matrix in memory.
    - PSD: Enforces positivity on the diagonal of Q via re-parameterization.
    """

    bias_name = "diag_psd_matrix_ridge"

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

    def forward(
        self, flattened_params: torch.Tensor, structured_params=None, **kwargs
    ) -> torch.Tensor:
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

    def get_bias_params(self) -> Dict[str, float]:
        return {"norm": torch.exp(self.log_Q_diag).norm().item()}

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


class ElasticNet(Bias):
    bias_name = "elastic_net"

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

    def get_bias_params(self) -> Dict[str, float]:
        l1, l2 = self.lambdas()
        return {"lambda_1": l1.item(), "lambda_2": l2.item()}

    def forward(
        self, flattened_params: Tensor, structured_params=None, **kwargs
    ) -> Tensor:
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


class DiagMatrixBiasKRR(Bias):
    bias_name = "diag_matrix_bias_krr"

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

    def get_bias_params(self) -> Dict[str, float]:
        return {"norm": self.Q_t.norm().item()}


class DiagMatrixBiasNTKRR(Bias):
    bias_name = "diag_matrix_bias_ntkrr"

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

    def get_bias_params(self) -> Dict[str, float]:
        return {"norm": self.Q_t.norm().item()}
