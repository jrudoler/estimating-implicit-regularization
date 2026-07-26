import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import abstractmethod
from torch.func import functional_call, grad, vjp
from torch.optim import Optimizer
from typing import Dict, Callable, Tuple, Any
from collections import OrderedDict

from .bias import GradientSquaredPenaltyScale, DiagMatrixRidgeBias


class InductiveBiasEstimator(pl.LightningModule):
    """
    Base class for inductive bias estimation. Takes in a predictive model (f) and a bias model (R).
    The bias model (R) is trained to match the gradient of the loss function with respect to the
    predictive model parameters (f) using the bias model parameters (theta).
    """

    def __init__(
        self,
        predictive_model: nn.Module,
        bias_model: nn.Module,
        grad_match_loss_fn: Callable[..., torch.Tensor] = nn.functional.mse_loss,
        optimizer_cls: Callable[..., Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        # bias_model_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self.predictive_model = predictive_model
        self.predictive_model.eval()
        # self.predictive_model.requires_grad_(False)
        self.bias_model = bias_model
        self.grad_match_loss_fn = grad_match_loss_fn
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.save_hyperparameters(ignore=["predictive_model", "bias_model"])

    @abstractmethod
    def predictive_loss_grad(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the gradient of the loss on predictions with respect to predictive model params.

        Override or modify this method to change the loss gradient.
        """
        pass

    def setup(self, stage: str) -> None:
        self.flattened_params = (
            torch.nn.utils.parameters_to_vector(self.predictive_model.parameters())
            .detach()
            .requires_grad_()
            .to(self.device)
        )
        self.structured_params = vector_to_parameter_views(
            self.flattened_params, self.predictive_model
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = batch
        if y.ndim == 1:
            y = y.view(-1, 1)

        # Get current parameters from the predictive model.
        params: Dict[str, torch.Tensor] = dict(self.predictive_model.named_parameters())

        def model_output(p: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.predictive_model, p, (x,))

        # Compute predictions and the vector-Jacobian product function.
        predictions, vjp_func = vjp(model_output, params, X)
        # Compute the loss gradient using the dedicated method.
        # flip the sign because you set the gradient of (loss + bias) equal to zero
        # and solve for the bias gradient
        loss_gradient = -self.predictive_loss_grad(predictions, y)
        vjp_result = vjp_func(loss_gradient)[0]
        true_grad = torch.cat([v.view(-1) for v in vjp_result.values()]) / X.size(0)
        # have to normalize the gradient by the batch size because
        # vjp returns the sum of the gradients over the batch

        # Compute the bias output and its gradient.
        R_val = self.bias_model(self.flattened_params, self.structured_params)
        gradients = torch.autograd.grad(
            R_val, self.flattened_params, create_graph=True
        )[0]

        loss = self.grad_match_loss_fn(gradients, true_grad, reduction="mean")

        self.log("train_bias/loss", loss, prog_bar=False)

        # Scale-free adequacy diagnostics: how much of the residual loss gradient
        # does this regularizer family actually explain?  `true_grad` is -grad L and
        # `gradients` is grad R, so the gradient-matching residual is grad R + grad L
        # and residual_ratio in [0, 1] is the unexplained fraction of ||grad L||
        # (0 = fully explained, ->1 = out-of-family).  cosine is the alignment of
        # grad R with -grad L; for a one-parameter family at its optimum
        # residual_ratio = sqrt(1 - cosine^2).
        with torch.no_grad():
            grad_r = gradients.detach()
            neg_grad_l = true_grad.detach()
            loss_grad_norm = neg_grad_l.norm()
            if loss_grad_norm > 0:
                residual_ratio = (grad_r - neg_grad_l).norm() / loss_grad_norm
                cosine = torch.nn.functional.cosine_similarity(
                    grad_r, neg_grad_l, dim=0
                )
                self.log("train_bias/residual_ratio", residual_ratio, prog_bar=False)
                self.log("train_bias/grad_cosine", cosine, prog_bar=False)
                self.log("train_bias/projection_r2", cosine**2, prog_bar=False)
                self.log("train_bias/loss_grad_norm", loss_grad_norm, prog_bar=False)

        # Log parameters from the bias model
        if hasattr(self.bias_model, "get_bias_params"):
            bias_params = self.bias_model.get_bias_params()
            if isinstance(bias_params, dict):
                for name, value in bias_params.items():
                    self.log(f"train_bias/{name}", value, prog_bar=False)
            else:
                self.log("train_bias/param", bias_params, prog_bar=False)

        # Log gradients of the raw parameters
        for name, param in self.bias_model.named_parameters():
            if param.grad is not None:
                self.log(f"train_bias/{name}_grad", param.grad.norm(), prog_bar=False)

        return loss

    def configure_optimizers(self) -> Any:
        # Optimize only the bias model parameters.
        return self.optimizer_cls(self.bias_model.parameters(), lr=self.lr)

    # def _bias_model(
    #     self, params: torch.Tensor, extra_kwargs: Dict[str, Any]
    # ) -> torch.Tensor:
    #     """
    #     Forward through bias_model while discarding kwargs that its `forward`
    #     doesn't declare.  Works with **any** third-party module.
    #     """
    #     sig = inspect.signature(self.bias_model.forward)
    #     accepted = {
    #         k: v
    #         for k, v in extra_kwargs.items()
    #         if k in sig.parameters
    #         and sig.parameters[k].kind
    #         in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    #     }
    #     return self.bias_model(params, **accepted)


def vector_to_parameter_views(
    vector: torch.Tensor, model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Creates a dictionary of views into 'vector' that match the structure of 'model.named_parameters()'.
    The returned views are part of the same computational graph as 'vector'.
    """
    pointer = 0
    views = []
    for param in model.parameters():
        numel = param.numel()
        views.append(vector[pointer : pointer + numel].view_as(param))
        pointer += numel
    return views


class BiasWithMSE(InductiveBiasEstimator):
    def predictive_loss_grad(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if targets.ndim == 1:
            targets = targets.view(-1, 1)
        per_example_elements = max(int(predictions[0].numel()), 1)
        return -2 * (targets - predictions) / per_example_elements


class BiasWithMSETrajectory(BiasWithMSE):
    """
    MSE-based bias estimator for precomputed trajectory observations.

    Expects each training batch to be:
      - theta_batch: trajectory parameter vectors, shape (m, p)
      - target_batch: target gradients, shape (m, p)

    For DiagMatrixRidgeBias with diagonal q, predicted gradients are:
      grad_factor * (q * theta_batch)
    """

    def __init__(
        self,
        bias_model: DiagMatrixRidgeBias,
        *,
        grad_factor: float = 2.0,
        grad_match_loss_fn: Callable[..., torch.Tensor] = F.mse_loss,
        optimizer_cls: Callable[..., Optimizer] = torch.optim.Adam,
        lr: float = 1e-2,
    ) -> None:
        if grad_factor == 0:
            raise ValueError("grad_factor must be non-zero.")
        if not isinstance(bias_model, DiagMatrixRidgeBias):
            raise TypeError(
                "BiasWithMSETrajectory expects a DiagMatrixRidgeBias, "
                f"got {type(bias_model)}."
            )
        if bias_model.Q.ndim != 1:
            raise ValueError(
                "BiasWithMSETrajectory expects bias_model.Q to be 1D; "
                f"got shape {tuple(bias_model.Q.shape)}."
            )

        super().__init__(
            predictive_model=nn.Identity(),
            bias_model=bias_model,
            grad_match_loss_fn=grad_match_loss_fn,
            optimizer_cls=optimizer_cls,
            lr=lr,
        )
        # Avoid Lightning warnings about modules left in eval mode.
        self.predictive_model.train()
        self.grad_factor = float(grad_factor)

    def setup(self, stage: str) -> None:
        # Trajectory fitting does not depend on predictive-model parameters.
        return

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        theta_batch, target_batch = batch
        if theta_batch.ndim == 1:
            theta_batch = theta_batch.unsqueeze(0)
        if target_batch.ndim == 1:
            target_batch = target_batch.unsqueeze(0)
        if theta_batch.shape != target_batch.shape:
            raise ValueError(
                "theta_batch and target_batch must have matching shapes; "
                f"got {tuple(theta_batch.shape)} vs {tuple(target_batch.shape)}."
            )
        if theta_batch.shape[-1] != self.bias_model.Q.numel():
            raise ValueError(
                "Feature dimension must match DiagMatrixRidgeBias dimension; "
                f"got {theta_batch.shape[-1]} vs {self.bias_model.Q.numel()}."
            )

        q = self.bias_model.Q.view(1, -1)
        predicted_grad = self.grad_factor * q * theta_batch
        loss = self.grad_match_loss_fn(predicted_grad, target_batch, reduction="mean")

        self.log("train_bias/loss", loss, prog_bar=False)
        self.log("train_bias/q_norm", self.bias_model.Q.detach().norm(), prog_bar=False)
        return loss


class BiasWithBCE(InductiveBiasEstimator):
    def predictive_loss_grad(self, predictions, targets):
        expected_grad = F.sigmoid(predictions) - targets
        return expected_grad


class BiasWithCrossEntropy(InductiveBiasEstimator):
    def predictive_loss_grad(self, predictions, targets):
        # print("Predictions shape:", predictions.shape)
        # targets should be class indices, 1d tensor
        if targets.ndim == 2:
            targets = targets.squeeze()
        # predictions should be logits, 2d tensor
        # print("Targets shape:", targets.shape)
        one_hot_targets = F.one_hot(targets, num_classes=predictions.shape[1]).float()
        # print("One-hot targets shape:", one_hot_targets.shape)
        expected_grad = F.softmax(predictions, dim=1) - one_hot_targets
        # print("Expected gradient:", expected_grad.shape)
        return expected_grad


class BiasWithCrossEntropyScheduled(BiasWithCrossEntropy):
    """Bias estimator with adaptive LR scheduling."""

    def __init__(
        self,
        *args,
        bias_lr: float = 0.1,
        monitor_lr: str = "train_bias/loss",
        patience_lr: int = 10,
        factor_lr: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bias_lr = bias_lr
        self.monitor_lr = monitor_lr
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor_lr,
            },
        }


class BiasWithCrossEntropyNormalized(BiasWithCrossEntropy):
    """
    Bias estimator with normalized gradient matching (preconditioned regression).
    
    Addresses the gradient magnitude imbalance problem. The key insight is that
    when we have: target ≈ λ₁∇R₁ + λ₂∇R₂
    
    If ||∇R₁|| >> ||∇R₂||, the optimizer focuses on λ₁ (larger gradient contribution).
    
    Solution: Normalize the basis gradients (precondition the regression):
        ĝ₁ = ∇R₁ / ||∇R₁||,  ĝ₂ = ∇R₂ / ||∇R₂||
        
    Then fit: target ≈ a·ĝ₁ + b·ĝ₂
    
    Recover true λ values: λ₁ = a/||∇R₁||, λ₂ = b/||∇R₂||
    
    This ensures all regularizers contribute equally to the loss landscape.
    We learn 'a' and 'b' directly (not exp(beta)), then recover λ.
    
    Requires bias_model to be a JointBias containing multiple bias models.
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
        
        # Verify bias_model is a JointBias or has bias_models attribute
        if not hasattr(bias_model, 'bias_models'):
            raise ValueError(
                "BiasWithCrossEntropyNormalized requires a JointBias model "
                "with multiple bias models. Got: {}".format(type(bias_model))
            )
        
        # Create separate learnable coefficients for normalized space
        # These are 'a' and 'b' in the formulation above
        n_biases = len(bias_model.bias_models)
        # Initialize near zero since we expect small coefficients
        self.normalized_coefs = nn.Parameter(torch.zeros(n_biases))
        
        self.save_hyperparameters(ignore=["predictive_model", "bias_model", "grad_match_loss_fn"])

    def _compute_penalty_gradient(self, bias_model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient of the raw penalty (without learned coefficient).
        
        Returns (gradient, norm).
        """
        flat_params = (
            torch.nn.utils.parameters_to_vector(self.predictive_model.parameters())
            .detach()
            .requires_grad_()
            .to(self.device)
        )
        struct_params = vector_to_parameter_views(flat_params, self.predictive_model)
        
        # We need the gradient of the penalty BEFORE the coefficient is applied
        # For ScalarBias types: R = scale * penalty, so ∇R = scale * ∇penalty
        # We want just ∇penalty
        
        # Temporarily set coefficient to 1 to get pure penalty gradient
        if hasattr(bias_model, 'beta'):
            original_beta = bias_model.beta.data.clone()
            # Set beta such that the scale = 1
            # If enforce_positive: scale = exp(beta), so set beta = 0 -> scale = 1
            # If not enforce_positive: scale = beta, so set beta = 1 -> scale = 1
            if hasattr(bias_model, 'enforce_positive') and bias_model.enforce_positive:
                bias_model.beta.data = torch.zeros_like(bias_model.beta.data)  # exp(0) = 1
            else:
                bias_model.beta.data = torch.ones_like(bias_model.beta.data)  # beta = 1
        elif hasattr(bias_model, 'alpha'):
            original_alpha = bias_model.alpha.data.clone()
            bias_model.alpha.data = torch.ones_like(bias_model.alpha.data)
        
        R_i = bias_model(flat_params, struct_params)
        grad_i = torch.autograd.grad(R_i, flat_params, create_graph=True)[0]
        
        # Restore original coefficient
        if hasattr(bias_model, 'beta'):
            bias_model.beta.data = original_beta
        elif hasattr(bias_model, 'alpha'):
            bias_model.alpha.data = original_alpha
        
        norm_i = grad_i.norm() + self.eps
        return grad_i, norm_i

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = batch
        if y.ndim == 1:
            y = y.view(-1, 1)

        # Compute the target gradient (loss gradient w.r.t. model params)
        params: Dict[str, torch.Tensor] = dict(self.predictive_model.named_parameters())

        def model_output(p: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.predictive_model, p, (x,))

        predictions, vjp_func = vjp(model_output, params, X)
        loss_gradient = -self.predictive_loss_grad(predictions, y)
        vjp_result = vjp_func(loss_gradient)[0]
        true_grad = torch.cat([v.view(-1) for v in vjp_result.values()]) / X.size(0)

        # Compute each bias model's penalty gradient and normalize
        normalized_grads = []
        penalty_grad_norms = []
        bias_names = []
        
        for bias_model in self.bias_model.bias_models:
            name = getattr(bias_model, 'bias_name', bias_model.__class__.__name__)
            bias_names.append(name)
            
            grad_i, norm_i = self._compute_penalty_gradient(bias_model)
            normalized_grads.append(grad_i / norm_i)
            penalty_grad_norms.append(norm_i)

        # Combine normalized gradients with our learned coefficients
        # predicted = Σ coef_i * normalized_grad_i
        # where coef_i = self.normalized_coefs[i]
        predicted_grad = torch.zeros_like(true_grad)
        for i, norm_grad in enumerate(normalized_grads):
            predicted_grad = predicted_grad + self.normalized_coefs[i] * norm_grad

        # Gradient matching loss
        loss = self.grad_match_loss_fn(predicted_grad, true_grad, reduction="mean")

        # Logging
        self.log("train_bias/loss", loss, prog_bar=False)
        
        # Log cosine similarity
        cos_sim = F.cosine_similarity(
            predicted_grad.unsqueeze(0), true_grad.unsqueeze(0)
        ).item()
        self.log("train_bias/grad_cosine_sim", cos_sim, prog_bar=False)

        # Log per-regularizer info
        for i, (name, norm) in enumerate(zip(bias_names, penalty_grad_norms)):
            coef_val = self.normalized_coefs[i].item()
            norm_val = norm.item()
            
            # Coefficient in normalized space (a, b)
            self.log(f"train_bias/{name}/coef_normalized", coef_val, prog_bar=False)
            # Penalty gradient norm
            self.log(f"train_bias/{name}/penalty_grad_norm", norm_val, prog_bar=False)
            # True lambda = coef / ||∇penalty||
            true_lambda = coef_val / norm_val
            self.log(f"train_bias/{name}/lambda", true_lambda, prog_bar=False)

        return loss

    def get_estimated_lambdas(self) -> Dict[str, float]:
        """
        Get the estimated true λ values.
        
        λ_i = normalized_coef_i / ||∇penalty_i||
        
        Call this after training to get the final estimates.
        """
        lambdas = {}
        
        for i, bias_model in enumerate(self.bias_model.bias_models):
            name = getattr(bias_model, 'bias_name', bias_model.__class__.__name__)
            
            # Compute penalty gradient norm (without coefficient)
            _, norm_i = self._compute_penalty_gradient(bias_model)
            norm_val = norm_i.item()
            
            # Get our learned normalized coefficient
            coef_val = self.normalized_coefs[i].item()
            
            # True lambda = coef / norm
            lambdas[name] = coef_val / norm_val
        
        return lambdas

    def configure_optimizers(self):
        # Optimize ONLY the normalized coefficients, not the bias model params
        optimizer = torch.optim.Adam([self.normalized_coefs], lr=self.bias_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor_lr,
            patience=self.patience_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor_lr,
            },
        }


class BiasWithAutodiffLoss(InductiveBiasEstimator):
    def __init__(
        self,
        predictive_model: nn.Module,
        bias_model: nn.Module,
        predictive_loss_fn: Callable[..., torch.Tensor],
        grad_match_loss_fn: Callable[..., torch.Tensor] = nn.functional.mse_loss,
        optimizer_cls: Callable[..., Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        bias_model_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__(
            predictive_model,
            bias_model,
            grad_match_loss_fn,
            optimizer_cls,
            lr,
        )
        self.loss_fn = predictive_loss_fn

    def predictive_loss_grad(self, predictions: torch.Tensor, targets: torch.Tensor):
        # Compute per-sample gradient of the loss function with respect to the model params
        loss = self.loss_fn(predictions, targets)
        # Compute the gradient of the loss with respect to the model predictions (dL/dy_hat)
        # because we're using chain rule.
        per_sample_grad = torch.autograd.grad(
            loss, predictions, retain_graph=True, create_graph=True
        )[0]
        return per_sample_grad


class GradientSquaredPenaltyEstimator(InductiveBiasEstimator):
    def __init__(
        self,
        predictive_model: nn.Module,
        predictive_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        bias_model: nn.Module = None,
        grad_match_loss_fn: Callable[..., torch.Tensor] = nn.functional.mse_loss,
        optimizer_cls: Callable[..., Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        gd_step_size: float = 0.05,
    ) -> None:
        bias_module = bias_model or GradientSquaredPenaltyScale()
        super().__init__(
            predictive_model=predictive_model,
            # predictive_loss_fn=predictive_loss_fn,
            bias_model=bias_module,
            grad_match_loss_fn=grad_match_loss_fn,
            optimizer_cls=optimizer_cls,
            lr=lr,
        )
        self.predictive_loss_fn = predictive_loss_fn
        self._parameter_layout = self._build_parameter_layout()
        self._num_params = sum(p.numel() for p in self.predictive_model.parameters())
        self.predictive_model.eval()
        self.predictive_model.requires_grad_(False)
        self.gd_step_size = gd_step_size

    def _build_parameter_layout(self) -> Tuple[Tuple[str, torch.Size, slice], ...]:
        layout = []
        offset = 0
        for name, param in self.predictive_model.named_parameters():
            numel = param.numel()
            layout.append((name, param.shape, slice(offset, offset + numel)))
            offset += numel
        return tuple(layout)

    def _vector_to_parameters(self, vector: torch.Tensor) -> OrderedDict:
        params = OrderedDict()
        for name, shape, sl in self._parameter_layout:
            params[name] = vector[sl].view(shape)
        return params

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = batch
        if y.dtype.is_floating_point:
            y_for_loss = y.view(-1, 1) if y.ndim == 1 else y
        else:
            y_for_loss = y.view(-1)

        device = self.device
        flat_params = torch.cat(
            [p.reshape(-1) for _, p in self.predictive_model.named_parameters()]
        ).to(device)
        flat_params = flat_params.detach().clone()

        def loss_with_flat(params_vector: torch.Tensor) -> torch.Tensor:
            param_dict = self._vector_to_parameters(params_vector)
            preds = functional_call(self.predictive_model, param_dict, (X,))
            return self.predictive_loss_fn(preds, y_for_loss)

        jacobian_loss_fn = grad(loss_with_flat)
        loss_gradient_vector = jacobian_loss_fn(flat_params)

        # compute a weight update step
        flat_params_delta = -self.gd_step_size * loss_gradient_vector

        _, hvp_fn = vjp(jacobian_loss_fn, flat_params)
        hessian_times_grad = hvp_fn(loss_gradient_vector)[0]

        # derivative of lambda/m sum( g_i^2 ) w.r.t. params is 2*lambda/m * H * g
        # we learn lambda/m directly and multiply by num_params to get lambda
        # this way we maintain scale invariance w.r.t. model size
        # otherwise, for large models, the gradients wrt lambda are suppressed by the 1/m factor
        lambda_per_param = self.bias_model()
        predicted_grad = (2.0 * lambda_per_param) * hessian_times_grad

        lambda_ = lambda_per_param * self._num_params

        # Residual from a single GD step: (-Δw / h) - g ≈ (h / 2) * H g
        residual_target = 0.5 * self.gd_step_size * hessian_times_grad
        # residual_target = - loss_gradient_vector

        loss_value = self.grad_match_loss_fn(
            predicted_grad, residual_target, reduction="mean"
        )

        self.log("train_bias/loss", loss_value, prog_bar=True)
        self.log("train_bias/lambda", lambda_.detach(), prog_bar=True)
        self.log("stats/grad_norm", loss_gradient_vector.detach().norm(), prog_bar=True)
        self.log("stats/hvp_norm", hessian_times_grad.detach().norm(), prog_bar=True)
        self.log("stats/residual_norm", residual_target.detach().norm(), prog_bar=True)

        return loss_value
