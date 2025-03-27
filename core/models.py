import torch
import torch.nn as nn

import lightning as pl
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback
import wandb
import scipy
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


class WandBCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        wandb.finish()


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
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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


# Callback for logging activations
class ActivationLogger(Callback):
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        module: Union[torch.nn.Module, Tuple[torch.nn.Module]] = torch.nn.ReLU,
    ):
        """
        Args:
            log_every_n_epochs: Only log activations every N validation epochs
                                to avoid huge logs.
            module: The module type to hook (e.g., nn.ReLU) or tuple of types
                   to hook (e.g., (nn.Linear, nn.Sigmoid)).
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.module = module  # The module type to hook (e.g., nn.ReLU)
        self.handles = []  # Store hook handles so we can remove them later
        self.activations = {}  # Will hold the latest outputs from each hooked layer

    def on_fit_start(self, trainer, pl_module):
        # Register hooks on each layer of interest (e.g., nn.ReLU activations) in pl_module.layers
        idx = 0
        for layer in pl_module.layers:
            if isinstance(layer, torch.nn.ReLU):
                handle = layer.register_forward_hook(self._make_hook(f"linear_{idx}"))
                self.handles.append(handle)
                idx += 1

    def _make_hook(self, layer_name):
        """Create a hook function that captures the forward output."""

        def hook(module, input, output):
            # Store the output in a dict, detach so we don't keep gradients
            self.activations[layer_name] = output.detach()

        return hook

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log activations to wandb after each validation epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return  # Skip logging if it's not the right epoch

        # Grab one batch from the validation dataloader
        sample_batch = next(iter(trainer.val_dataloaders))
        x, _ = sample_batch

        # Move input to the correct device
        x = x.to(pl_module.device)

        # Optionally set the model to eval mode so dropout is inactive
        pl_module.eval()
        with torch.no_grad():
            _ = pl_module(x)  # Forward pass populates self.activations via the hooks

        # Now log the stored activations to wandb
        for layer_name, act in self.activations.items():
            # Flatten for histogram logging
            act_data = act.view(-1).cpu().numpy()
            trainer.logger.experiment.log(
                {
                    f"activations/{layer_name}": wandb.Histogram(act_data),
                    f"activations/{layer_name}_mean": act_data.mean(),
                    f"activations/{layer_name}_std": act_data.std(),
                    f"activations/{layer_name}_entropy": scipy.stats.entropy(
                        scipy.special.softmax(act_data)
                    ),
                    "global_step": trainer.global_step,
                    "epoch": trainer.current_epoch,
                }
            )

        # Clear the activations dict
        self.activations.clear()

        # Return model to train mode if desired
        pl_module.train()

    def on_fit_end(self, trainer, pl_module):
        # Remove the hooks
        for h in self.handles:
            h.remove()
        self.handles.clear()
