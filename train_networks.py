import os
import argparse
import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from core.data import MNISTDataModule
from core.models import NoisyMLP
from accelerate.test_utils.testing import get_backend

torch.set_float32_matmul_precision("high")


def train_model(
    model, run_name, dropout_rate, max_epochs=25, early_stopping=False, patience=3
):
    device, n_devices, _ = get_backend()
    # Setup wandb logger for this run
    print("Initializing wandb logger...")
    wandb_logger = WandbLogger(project="inductive-bias", name=run_name, log_model=True)
    print("Initializing data module...")
    mnist = MNISTDataModule(batch_size=32)
    print("Initializing trainer...")
    callbacks = []
    if early_stopping:
        print("Early stopping enabled.")
        callbacks.append(EarlyStopping(monitor="train/loss", patience=patience))
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=device,
        devices=n_devices,
    )
    print(f"Training model with dropout rate: {dropout_rate}")
    trainer.fit(model, mnist)

    # Save model checkpoint locally
    os.makedirs("./saved_models", exist_ok=True)
    save_path = f"./saved_models/{run_name}.pt"
    torch.save(model.state_dict(), save_path)

    # Log the saved file as a wandb artifact
    artifact = wandb.Artifact(run_name, type="model")
    artifact.add_file(save_path)
    wandb_logger.experiment.log_artifact(artifact)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dropout_rate", type=float, required=True)
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument(
        "--early_stopping", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Patience for early stopping"
    )
    args = parser.parse_args()

    print("Initializing model...")
    model = NoisyMLP(dropout_rate=args.dropout_rate, out_features=1)
    train_model(
        model,
        args.run_name,
        args.dropout_rate,
        args.max_epochs,
        args.early_stopping,
        args.patience,
    )
