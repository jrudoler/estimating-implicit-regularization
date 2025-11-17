# MNIST dataset, lightning datamodule
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from typing import Optional, Tuple, Any, Callable
from pathlib import Path

# class MNISTDataModule(LightningDataModule):
#     def __init__(self, batch_size=32):
#         super().__init__()
#         self.batch_size = batch_size
#         self.transform = transforms.Compose([transforms.ToTensor()])
#         self.target_transform = lambda y: 1 if y == 3 else 0
#         self.train_dataset = datasets.MNIST(
#             root=".",
#             train=True,
#             download=True,
#             transform=self.transform,
#             target_transform=self.target_transform,
#         )
#         self.val_dataset = datasets.MNIST(
#             root=".",
#             train=False,
#             download=True,
#             transform=self.transform,
#             target_transform=self.target_transform,
#         )

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size)


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
    def __init__(
        self, X: torch.Tensor, y: torch.Tensor, num_workers: Optional[int] = None
    ):
        super().__init__()
        self.dataset = TensorDataset(X, y)
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=len(self.dataset), num_workers=self.num_workers
        )


class MNISTLightningDataModule(LightningDataModule):
    def __init__(
        self, root: Path, batch_size: int, num_workers: int, val_fraction: float = 0.1
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.transform = transforms.ToTensor()
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def prepare_data(self) -> None:  # type: ignore[override]
        datasets.MNIST(root=self.root, train=True, download=True)
        datasets.MNIST(root=self.root, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:  # type: ignore[override]
        if stage == "fit" or stage is None:
            full_train = datasets.MNIST(
                root=self.root,
                train=True,
                download=False,
                transform=self.transform,
            )
            val_size = int(len(full_train) * self.val_fraction)
            train_size = len(full_train) - val_size
            self._train_dataset, self._val_dataset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        if stage == "test" or stage is None:
            self._test_dataset = datasets.MNIST(
                root=self.root,
                train=False,
                download=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
