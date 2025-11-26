from typing import Union, Tuple, List, Dict
import torch
from lightning.pytorch.callbacks import Callback
import wandb
import scipy
import itertools


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


def r2_subset(Xs: List[torch.Tensor], y: torch.Tensor, subset):
    if len(subset) == 0:
        return 0.0
    X = torch.stack([Xs[i] for i in subset], dim=1)  # d × k
    beta = torch.linalg.pinv(X) @ y
    y_hat = X @ beta
    return float((y_hat @ y_hat) / (y @ y + 1e-12))


def shapley_R2(Xs: List[torch.Tensor], y: torch.Tensor, nsamples=None):
    m = len(Xs)
    players = list(range(m))
    phi = torch.zeros(m)

    if nsamples is None:
        perms = list(itertools.permutations(players))
    else:
        perms = [torch.randperm(m).tolist() for _ in range(nsamples)]

    for perm in perms:
        S = []
        r2_S = 0.0
        for p in perm:
            r2_before = r2_S
            r2_after = r2_subset(Xs, y, S + [p])
            phi[p] += r2_after - r2_before
            S.append(p)
            r2_S = r2_after

    return phi / len(perms)


class ShapleyBiasCallback(Callback):
    """
    Computes Shapley-R^2 attribution for each component of JointBias.
    Requires JointBias.forward(..., return_components=True).
    """

    def __init__(self, every_n_epochs=1, nsamples=None):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.nsamples = nsamples

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        device = pl_module.device

        # ---- Get one batch ----
        X, y = next(iter(trainer.datamodule.val_dataloader()))
        X, y = X.to(device), y.to(device)
        if y.ndim == 1:
            y = y.view(-1, 1)

        # ---- Rebuild flat params ----
        params = dict(pl_module.predictive_model.named_parameters())
        flat = torch.cat([p.reshape(-1) for p in params.values()]).detach().clone()
        flat = flat.requires_grad_(True)

        named_views = {}
        offset = 0
        for name, p in params.items():
            n = p.numel()
            named_views[name] = flat[offset : offset + n].view_as(p)
            offset += n

        # ---- True gradient y_vec ----
        def model_output(p, x):
            return torch.func.functional_call(pl_module.predictive_model, p, (x,))

        preds, vjp_fn = torch.func.vjp(model_output, params, X)
        loss_grad_pred = -pl_module.predictive_loss_grad(preds, y)
        vjp_res = vjp_fn(loss_grad_pred)[0]
        true_grad = torch.cat([g.reshape(-1) for g in vjp_res.values()]) / X.size(0)
        y_vec = -true_grad

        # ---- Compute each component's gradient ----
        total_R, comp_dict = pl_module.bias_model(
            flat, named_views, return_components=True
        )

        Xs = []
        names = list(comp_dict.keys())
        for name in names:
            grad_i = torch.autograd.grad(comp_dict[name], flat, retain_graph=True)[
                0
            ].detach()
            Xs.append(grad_i)

        # ---- Shapley R^2 ----
        phi = shapley_R2(Xs, y_vec, nsamples=self.nsamples)

        # ---- Log ----
        for k, v in zip(names, phi):
            trainer.logger.experiment.add_scalar(f"shapley/{k}_R2", float(v), epoch)
        trainer.logger.experiment.add_scalar(
            "shapley/total_R2", float(phi.sum()), epoch
        )
