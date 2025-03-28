import os
import glob
import torch
import wandb
import argparse
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from core.data import MNISTDataModule
from core.models import NoisyMLP
from core.bias import RidgeBias, BiasWithCrossEntropy
from accelerate.test_utils.testing import get_backend

torch.set_float32_matmul_precision("high")


def train_bias(model_ckpt, run_name, max_epochs=10, lr=1e-3):
    device, n_devices, _ = get_backend()
    # Instantiate predictive model (architecture must match the saved one)
    model = NoisyMLP(out_features=1)
    state_dict = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Set up the bias training module
    module = BiasWithCrossEntropy(
        predictive_model=model,
        bias_model=RidgeBias(),
        grad_match_loss_fn=torch.nn.functional.mse_loss,
        optimizer_cls=torch.optim.Adam,
        lr=lr,
    )
    wandb_logger = WandbLogger(project="inductive-bias", name=run_name)
    mnist = MNISTDataModule(batch_size=32)
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[EarlyStopping(monitor="train/loss")],
        accelerator=device,
        devices=n_devices,
    )
    trainer.fit(module, mnist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bias training.")
    parser.add_argument(
        "--model_run",
        type=str,
        required=True,
        help="Model run name used to locate the model checkpoint. For example: 'noisy-mlp'",
    )
    parser.add_argument(
        "--bias_run",
        type=str,
        required=True,
        help="Bias run name for the Wandb logger.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    args = parser.parse_args()

    save_dir = "./saved_models"
    pattern = os.path.join(save_dir, f"{args.model_run}.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No saved model found for pattern {pattern}. Run train_networks.py first."
        )
    model_ckpt = files[0]
    train_bias(model_ckpt, args.bias_run, max_epochs=args.max_epochs, lr=args.lr)
