import torch
import torch.nn as nn
import wandb
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from core.bias import ElasticNet
from core.models import LinearNetwork, NonLinearNetwork
from core.estimators import BiasWithMSE


# ----- SETTING THE SEED -----
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ----- INIT & CONFIG -----
run = wandb.init(project="inductive-bias", job_type="sweep")
cfg = wandb.config
l1 = cfg.get("l1", 0.1)
l2 = cfg.get("l2", 0.01)
smooth = cfg.get("smooth", 0.01)

# ----- SYNTHETIC DATA -----
input_dim, output_dim, hidden_dim = 10, 1, 200
N = 5000
X = torch.randn(N, input_dim)
true_betas = 5 * torch.randn(input_dim)
y = (X @ true_betas).view(-1, 1) + 0.5 * torch.randn(N, 1)

dataset = TensorDataset(X, y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

# ----- STAGE 1: TRAIN LINEAR NETWORK -----
model = NonLinearNetwork(
    input_dim,
    output_dim,
    hidden_dim,
    l1_lambda=l1,
    l1_smooth=smooth,
    l2_lambda=l2,
    lr=1e-3,
)

logger = WandbLogger(
    project="inductive-bias",
    name=f"trained-l1-{l1}_l2-{l2}",
    experiment=run,
    save_dir=run.dir,
)
trainer = Trainer(
    max_epochs=200,
    logger=logger,
    callbacks=[EarlyStopping(monitor="train/loss", patience=15)],
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# ----- STAGE 2: GRADIENT-MATCHING BIAS ESTIMATION -----
# Reconstruct functional model with frozen weights

bias_model = ElasticNet(smooth=smooth)

bias_estimator = BiasWithMSE(
    predictive_model=model,  # this is your already-trained model
    bias_model=bias_model,
    grad_match_loss_fn=torch.nn.functional.mse_loss,
    lr=1e-2,
    optimizer_cls=torch.optim.Adam,
)

bias_logger = WandbLogger(
    project="inductive-bias", name=f"bias-recovery-l1-{l1}_l2-{l2}", log_model=False
)

bias_trainer = Trainer(
    max_epochs=1000,
    logger=bias_logger,
    callbacks=[
        EarlyStopping(monitor="train/loss", patience=50, mode="min"),
    ],
    accelerator="gpu",
    devices=1,
)

# Create a DataLoader for the full dataset
bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)
