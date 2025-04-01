import torch
import torch.nn as nn

import lightning as pl
from lightning import LightningModule
import wandb
from typing import Union, Tuple, Optional, Any, Type


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
    ):
        super().__init__()

        self.loss_func = (
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
        self.save_hyperparameters()

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
        return torch.optim.SGD(self.parameters(), lr=1e-2)
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
