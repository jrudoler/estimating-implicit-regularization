import torch
import torch.nn as nn
from torch import Tensor

import lightning as pl
from lightning import LightningModule
import wandb
from typing import Union, Optional, Any, Type, Callable, Tuple


def load_model_from_artifact(
    model_class: Type, artifact_ref: str, checkpoint_name: str = "model.ckpt"
) -> Any:
    """
    Loads a model checkpoint from a wandb artifact.

    Args:
        model_class: The model class used to load the checkpoint (e.g., NoisyMLP).
        artifact_ref: The full artifact reference (e.g., 'jhrudoler-penn/inductive-bias/model-n7awb9ov:v0').
        checkpoint_name: The name of the checkpoint file within the artifact.

    Returns:
        The loaded model.
    """
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    checkpoint_path = artifact.download()
    checkpoint_path = f"{checkpoint_path}/{checkpoint_name}"
    # Load and return the model using the provided model class
    return model_class.load_from_checkpoint(checkpoint_path)


class NoisyMLP(LightningModule):
    def __init__(
        self,
        in_features=784,
        out_features=10,
        hidden_features=128,
        dropout_rate=0.2,
        num_hidden_layers=4,
        loss_func=None,
        lr=1e-2,
    ):
        super().__init__()

        self.loss_func = loss_func or (
            nn.BCEWithLogitsLoss() if out_features == 1 else nn.CrossEntropyLoss()
        )

        # Build sequential layers dynamically
        layers = []

        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        # layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(
            num_hidden_layers - 1
        ):  # -1 because we already added one hidden layer
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))

        self.layers = nn.Sequential(*layers)
        self.save_hyperparameters(ignore=["loss_func"])

    def forward(self, x):
        # Flatten input if it's not already flattened
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # make sure y is the right shape
        if len(y.shape) == 1:
            y = y.view(-1, 1).float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # make sure y is the right shape
        if len(y.shape) == 1:
            y = y.view(-1, 1).float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        # return torch.optim.Adam(self.parameters(), lr=1e-3)


class LinearNetwork(pl.LightningModule):
    """Linear network with one hidden layer."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        l1_lambda=0.0,
        l2_lambda=0.0,
        l1_smooth=0.1,
        lr=1e-2,
    ):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            l1_lambda: L1 regularization strength
            l2_lambda: L2 regularization strength
            l1_smooth: L1 regularization smoothness
        """
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        return self.linear(x)
        # Flatten input if it's not already flattened

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        # Add L1 and L2 regularization
        loss += self.weight_regularization()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        # Add L1 and L2 regularization
        loss += self.weight_regularization()
        # Log the loss
        self.log("val/loss", loss)
        return loss

    def weight_regularization(self):
        """
        Compute the L1 and L2 regularization terms.
        """
        l1_reg = 0.0
        l2_reg = 0.0
        # copy and flatten the parameters
        flattened_params = torch.cat([param.view(-1) for param in self.parameters()])
        # L1 and L2 regularization
        # smooth the L1 regularization
        l1_reg = torch.nn.functional.smooth_l1_loss(
            flattened_params,
            torch.zeros_like(flattened_params),
            beta=self.hparams.l1_smooth,
            reduction="sum",
        )
        l2_reg = torch.sum(flattened_params**2)
        return self.hparams.l1_lambda * l1_reg + self.hparams.l2_lambda * l2_reg

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class NonLinearNetwork(pl.LightningModule):
    """Same as LinearNetwork but with ReLU activation"""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        l1_lambda=0.0,
        l2_lambda=0.0,
        l1_smooth=0.1,
        lr=1e-2,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            hidden_features: Hidden layer dimension
            l1_lambda: L1 regularization strength
            l2_lambda: L2 regularization strength
            l1_smooth: L1 regularization smoothness
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        # Add L1 and L2 regularization
        loss += self.weight_regularization()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        # Add L1 and L2 regularization
        loss += self.weight_regularization()
        # Log the loss
        self.log("val/loss", loss)
        return loss

    def weight_regularization(self):
        """
        Compute the L1 and L2 regularization terms.
        """
        l1_reg = 0.0
        l2_reg = 0.0
        # copy and flatten the parameters
        flattened_params = torch.cat([param.view(-1) for param in self.parameters()])
        # L1 and L2 regularization
        # smooth the L1 regularization
        l1_reg = torch.nn.functional.smooth_l1_loss(
            flattened_params,
            torch.zeros_like(flattened_params),
            beta=self.hparams.l1_smooth,
            reduction="sum",
        )
        l2_reg = torch.sum(flattened_params**2)
        return self.hparams.l1_lambda * l1_reg + self.hparams.l2_lambda * l2_reg

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LinearRegression(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        fit_intercept=False,
        lr: float = 1e-2,
        init_zeros=True,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=fit_intercept)
        if init_zeros:
            # Initialize weights and bias to zero
            nn.init.zeros_(self.linear.weight)
            if fit_intercept:
                nn.init.zeros_(self.linear.bias)
        self.loss_func = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # make sure y is the right shape
        if len(y.shape) == 1:
            y = y.view(-1, 1).float()
        assert y.shape == y_hat.shape, f"y shape: {y.shape}, y_hat shape: {y_hat.shape}"
        loss = self.loss_func(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


class KernelRegression(pl.LightningModule):
    def __init__(
        self,
        kernel: Union[Callable[[Tensor, Tensor], Tensor], Tensor],
        n_train_samples: int,
        fit_intercept: bool = False,
        ridge_lambda: Optional[float] = None,
        init_zeros: bool = True,
        init_weights: Optional[Tensor] = None,
        lr: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kernel_linear = nn.Linear(n_train_samples, 1, bias=fit_intercept)
        # save initial weights
        if init_zeros:
            nn.init.zeros_(self.kernel_linear.weight)
            if fit_intercept:
                nn.init.zeros_(self.kernel_linear.bias)
        elif init_weights is not None:
            # Initialize weights with provided initial weights
            if init_weights.shape != self.kernel_linear.weight.shape:
                raise ValueError(
                    f"Initial weights shape {init_weights.shape} does not match model weights shape {self.kernel_linear.weight.shape}."
                )
            self.kernel_linear.weight.data.copy_(init_weights)
            if fit_intercept and self.kernel_linear.bias is not None:
                nn.init.zeros_(self.kernel_linear.bias)
        self.init_weights = self.kernel_linear.weight.detach().clone()
        self.init_bias = (
            self.kernel_linear.bias.detach().clone() if fit_intercept else None
        )

        self.kernel = kernel
        if isinstance(self.kernel, Tensor):
            # Ensure kernel is a square matrix
            assert self.kernel.shape[0] == self.kernel.shape[1], (
                "Kernel must be square."
            )
            # warning
            import warnings

            warnings.warn("Using a precomputed kernel. batch inputs will be ignored.")
        self.loss_func = KernelRidgeMSELoss(
            ridge_lambda=self.hparams.ridge_lambda or 0.0
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_K: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        If return_K is False (default), returns y_hat.
        If True, returns (y_hat, K).
        """
        if isinstance(self.kernel, Tensor):
            # Ensure kernel is on the same device as the model
            self.kernel = self.kernel.to(self.kernel_linear.weight.device)
            y_hat = self.kernel_linear(self.kernel)
            if return_K:
                return y_hat, self.kernel
        elif callable(self.kernel):
            K = self.kernel(x, x)
            # ensure K is on the same device as the model
            K = K.to(self.kernel_linear.weight.device)
            y_hat = self.kernel_linear(K)
            if return_K:
                return y_hat, K
        return y_hat

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        if y.dim() == 1:
            y = y.view(-1, 1).float()

        # get both y_hat and K
        y_hat, K = self(x, return_K=True)
        alpha = self.kernel_linear.weight.squeeze(0)
        loss = self.loss_func(y_hat, y, K, alpha)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        if y.dim() == 1:
            y = y.view(-1, 1).float()

        y_hat, K = self(x, return_K=True)
        alpha = self.kernel_linear.weight.squeeze(0)
        loss = self.loss_func(y_hat, y, K, alpha)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


class KernelRidgeMSELoss(nn.MSELoss):
    def __init__(self, ridge_lambda: float = 0.0, **mse_kwargs) -> None:
        super().__init__(**mse_kwargs)
        self.ridge_lambda = ridge_lambda

    def forward(self, y_hat: Tensor, y: Tensor, K: Tensor, alpha: Tensor) -> Tensor:
        # MSELoss already handles reduction for you
        base = super().forward(y_hat, y)
        if self.ridge_lambda != 0.0:
            reg = alpha @ K @ alpha
            return base + self.ridge_lambda * reg
        return base
