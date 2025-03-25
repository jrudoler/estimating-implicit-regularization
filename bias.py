import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import ABC, abstractmethod
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


# class LassoBias(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor([1.0]))  # Initialize parameters

#     def forward(self, flattened_params):
#         return self.alpha * torch.sum(torch.abs(flattened_params))


# class EpsilonSmoothLassoBias(nn.Module):
#     def __init__(self, beta_init: float = 1.0, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.beta = nn.Parameter(torch.tensor([beta_init]))
#         self.eps = eps

#     def forward(self, flattened_params: torch.Tensor) -> torch.Tensor:
#         # Smooth L1: sqrt(x^2 + eps) approximates |x|
#         return self.beta * torch.sum(torch.sqrt(flattened_params**2 + self.eps))


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


class BiasLightningModule(pl.LightningModule):
    def __init__(
        self,
        predictive_model: nn.Module,
        bias_model: nn.Module,
        loss_fn: Callable[..., torch.Tensor] = nn.functional.mse_loss,
        optimizer_cls: Callable[..., Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.predictive_model = predictive_model
        self.bias_model = bias_model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.save_hyperparameters()

    @abstractmethod
    def compute_loss_gradient(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the gradient of the loss with respect to the predictions.

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
        loss_gradient = self.compute_loss_gradient(predictions, y)
        vjp_result = vjp_func(loss_gradient)[0]
        true_grad = torch.cat([v.view(-1) for v in vjp_result.values()]) / X.size(0)

        # Compute the bias output and its gradient.
        R_val = self.bias_model(flattened_params)
        gradients = torch.autograd.grad(R_val, flattened_params, create_graph=True)[0]

        loss = self.loss_fn(gradients, true_grad, reduction="mean")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        # Optimize only the bias model parameters.
        return self.optimizer_cls(self.bias_model.parameters(), lr=self.lr)


class BiasWithMSE(BiasLightningModule):
    def compute_loss_gradient(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return 2 * (targets - predictions)


class BiasWithCrossEntropy(BiasLightningModule):
    def compute_loss_gradient(self, predictions, targets):
        return predictions - targets


# Example usage:
if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from torch.utils.data import DataLoader, TensorDataset

    # Define your predictive and bias models.
    predictive_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    bias_model = RidgeBias()

    # Create a dummy dataset.
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate the Lightning module.
    module = BiasLightningModule(
        predictive_model=predictive_model,
        bias_model=bias_model,
        loss_fn=nn.functional.mse_loss,  # Change as needed.
        optimizer_cls=torch.optim.Adam,
        lr=1e-3,
    )

    # Train using the Trainer interface.
    trainer = Trainer(max_epochs=10)
    trainer.fit(module, dataloader)
