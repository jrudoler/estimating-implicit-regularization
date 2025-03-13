# MNIST dataset, lightning datamodule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = datasets.MNIST(
            root=".", train=True, download=True, transform=self.transform
        )
        self.val_dataset = datasets.MNIST(
            root=".", train=False, download=True, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
