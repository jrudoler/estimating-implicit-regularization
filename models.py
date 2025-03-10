import torch
import torch.nn as nn

from lightning import LightningModule


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

        # Build sequential layers dynamically
        layers = []

        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU())
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

    def forward(self, x):
        # Flatten input if it's not already flattened
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
