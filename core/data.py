# MNIST dataset, lightning datamodule
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from typing import Optional, Tuple, Any, Callable


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.target_transform = lambda y: 1 if y == 3 else 0
        self.train_dataset = datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.val_dataset = datasets.MNIST(
            root=".",
            train=False,
            download=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class WikiTextDataModule(LightningDataModule):
    def __init__(
        self,
        wiki_path: str = "wikitext-2-v1",
        batch_size: int = 32,
        max_length: int = 512,
        tokenizer: Optional[Callable] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer or (
            lambda x: x
        )  # Default to identity function if no tokenizer is provided
        self.dataset = load_dataset("Salesforce/wikitext", wiki_path)
        if tokenizer is not None:
            self.dataset = self.dataset.map(
                lambda x: {"input_ids": self.tokenizer(x["text"])},
                batched=True,
                remove_columns=["text"],
            )
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["validation"]
        self.test_dataset = self.dataset["test"]
        self.collate_fn = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )


class FullBatchDataModule(LightningDataModule):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.dataset = TensorDataset(X, y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=len(self.dataset))
