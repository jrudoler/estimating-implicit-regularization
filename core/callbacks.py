from typing import Union, Tuple
import torch
from lightning.pytorch.callbacks import Callback
import wandb
import scipy


class WandBCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        wandb.finish()


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
