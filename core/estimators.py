import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import abstractmethod
from torch.func import functional_call, vjp
from torch.optim import Optimizer
from typing import Dict, Callable, Tuple, Any, Set
from collections import deque
import inspect


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
        bias_model_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self.predictive_model = predictive_model
        # self.predictive_model.eval()
        # self.predictive_model.requires_grad_(False)
        self.bias_model = bias_model
        self.grad_match_loss_fn = grad_match_loss_fn
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.bias_model_kwargs = bias_model_kwargs or {}
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
        R_val = self.bias_model(flattened_params)
        # flattened_params,
        # extra_kwargs=self.bias_model_kwargs
        # | {
        #     "epoch": self.current_epoch,
        #     "global_step": self.global_step,
        # },
        # )
        gradients = torch.autograd.grad(R_val, flattened_params, create_graph=True)[0]

        loss = self.grad_match_loss_fn(gradients, true_grad, reduction="mean")

        self.log("train/loss", loss, prog_bar=False)

        # Log all parameters from the bias model.
        for name, param in self.bias_model.named_parameters():
            # Ensure the parameter is logged as a scalar if it is a single value.
            if param.numel() == 1:
                self.log(f"bias/{name}", param.detach().item(), prog_bar=False)
                continue
            # otherwise, if the parameter is a tensor, log its norm.
            else:
                self.log(f"bias/{name}", param.norm(), prog_bar=False)
                # Log the gradient norm as well.
                if param.grad is not None:
                    self.log(f"bias/{name}_grad", param.grad.norm(), prog_bar=False)
                    continue

        return loss

    def configure_optimizers(self) -> Any:
        # Optimize only the bias model parameters.
        return self.optimizer_cls(self.bias_model.parameters(), lr=self.lr)

    # def on_before_backward(self, loss: torch.Tensor) -> None:
    #     """Fail fast if any tensor in the backward graph is on a CPU
    #     while others are on a GPU (or vice-versa)."""
    #     print(
    #         f"--- Debugging devices in on_before_backward (Epoch {self.current_epoch}, Global Step {self.global_step}) ---"
    #     )

    #     expected_device = self.device
    #     print(f"Expected device (self.device): {expected_device}")

    #     # Check loss tensor's device
    #     if loss.device != expected_device:
    #         print(
    #             f"WARNING: Loss tensor is on device {loss.device}, but expected {expected_device}."
    #         )
    #     else:
    #         print(f"Loss tensor device: {loss.device} (Matches expected)")

    #     # Check model parameters' devices
    #     for name, param in self.named_parameters():
    #         if param.device != expected_device:
    #             print(
    #                 f"WARNING: Parameter '{name}' is on device {param.device}, but expected {expected_device}."
    #             )

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
