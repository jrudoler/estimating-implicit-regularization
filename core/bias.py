import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import abstractmethod
from torch.func import functional_call, vjp
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
    ) -> None:
        super().__init__()
        self.predictive_model = predictive_model
        self.bias_model = bias_model
        self.grad_match_loss_fn = grad_match_loss_fn
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.save_hyperparameters()

    @abstractmethod
    def predictive_loss_grad(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the gradient of the loss on predictions with respect to predictive model params.

        Default implementation for MSE loss gradient: 2*(predictions - targets).
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

        # Compute the bias output and its gradient.
        R_val = self.bias_model(flattened_params)
        gradients = torch.autograd.grad(R_val, flattened_params, create_graph=True)[0]

        loss = self.grad_match_loss_fn(gradients, true_grad, reduction="mean")
        self.log("train/loss", loss, prog_bar=False)

        # Log all parameters from the bias model.
        for name, param in self.bias_model.named_parameters():
            # Ensure the parameter is logged as a scalar.
            self.log(f"bias/{name}", param.detach().item(), prog_bar=False)

        return loss

    def configure_optimizers(self) -> Any:
        # Optimize only the bias model parameters.
        return self.optimizer_cls(self.bias_model.parameters(), lr=self.lr)


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
    ) -> None:
        super().__init__(
            predictive_model,
            bias_model,
            grad_match_loss_fn,
            optimizer_cls,
            lr,
        )
        self.loss_fn = predictive_loss_fn
        self.save_hyperparameters()

    def predictive_loss_grad(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # Compute per-sample gradient of the loss function with respect to the model params
        loss = self.loss_fn(predictions, targets)
        # Compute the gradient of the loss with respect to the model predictions (dL/dy_hat)
        # because we're using chain rule.
        per_sample_grad = torch.autograd.grad(
            loss, predictions, retain_graph=True, create_graph=True
        )[0]
        return per_sample_grad
