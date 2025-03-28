import os
import glob
import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from core.data import MNISTDataModule
from core.models import NoisyMLP
from core.bias import RidgeBias, BiasWithCrossEntropy

def train_bias(model_ckpt, run_name, max_epochs=10, lr=1e-3):
    # Instantiate predictive model (architecture must match the saved one)
    model = NoisyMLP(dropout_rate=0.0, out_features=1)
    state_dict = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Set up the bias training module
    module = BiasWithCrossEntropy(
        predictive_model=model,
        bias_model=RidgeBias(),
        loss_fn=torch.nn.functional.mse_loss,
        optimizer_cls=torch.optim.Adam,
        lr=lr
    )
    wandb_logger = WandbLogger(project="inductive-bias", name=run_name)
    mnist = MNISTDataModule(batch_size=32)
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[EarlyStopping(monitor="train/loss")]
    )
    trainer.fit(module, mnist)

def main():
    # Automatically retrieve the saved noisy model (or change the pattern as needed)
    save_dir = "./saved_models"
    files = glob.glob(os.path.join(save_dir, "noisy-mlp.pt"))
    if not files:
        print("No saved model found. Run train_networks.py first.")
        return
    model_ckpt = files[0]
    train_bias(model_ckpt, "ridge-bias-noisy")

if __name__ == "__main__":
    wandb.init(project="inductive-bias", reinit=True)
    main()
    wandb.finish()
