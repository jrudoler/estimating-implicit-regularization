import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import abstractmethod
from torch.func import functional_call, grad, jvp, vjp
from torch.optim import Optimizer
from typing import Dict, Callable, Tuple, Any, Set
from collections import deque, OrderedDict
import inspect

from .bias import GradientSquaredPenaltyScale


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
        # self.predictive_model.eval()
        # self.predictive_model.requires_grad_(False)
        self.bias_model = bias_model
        self.grad_match_loss_fn = grad_match_loss_fn
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        # self.bias_model_kwargs = bias_model_kwargs or {}
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

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = batch
        if y.ndim == 1:
            y = y.view(-1, 1)

        # Get current parameters from the predictive model.
        params: Dict[str, torch.Tensor] = dict(self.predictive_model.named_parameters())
        flattened_params = (
            torch.cat([p.view(-1) for p in params.values()]).detach().requires_grad_()
        )
        flattened_params = flattened_params.to(self.device)

        offset = 0
        named_param_views: Dict[str, torch.Tensor] = {}
        for name, param in params.items():
            numel = param.numel()
            view = flattened_params[offset : offset + numel].view_as(param)
            named_param_views[name] = view
            offset += numel

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
        R_val = self.bias_model(flattened_params, named_param_views)
        # flattened_params,
        # extra_kwargs=self.bias_model_kwargs
        # | {
        #     "epoch": self.current_epoch,
        #     "global_step": self.global_step,
        # },
        # )
        gradients = torch.autograd.grad(R_val, flattened_params, create_graph=True)[0]

        loss = self.grad_match_loss_fn(gradients, true_grad, reduction="mean")

        self.log("train_bias/loss", loss, prog_bar=False)

        # Log all parameters from the bias model.
        for name, param in self.bias_model.named_parameters():
            # Ensure the parameter is logged as a scalar if it is a single value.
            if param.numel() == 1:
                self.log(f"train_bias/{name}", param.detach().item(), prog_bar=False)
                if param.grad is not None:
                    self.log(
                        f"train_bias/{name}_grad", param.grad.norm(), prog_bar=False
                    )
                continue
            # otherwise, if the parameter is a tensor, log its norm.
            else:
                self.log(f"train_bias/{name}", param.norm(), prog_bar=False)
                # Log the gradient norm as well.
                if param.grad is not None:
                    self.log(
                        f"train_bias/{name}_grad", param.grad.norm(), prog_bar=False
                    )
                    continue

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


class BiasWithMSE(InductiveBiasEstimator):
    def predictive_loss_grad(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return -2 * (targets - predictions)


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
            bias_model_kwargs,
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
