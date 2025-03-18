import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

# GOAL: implement a class of models that represent a parametrization of the inductive
# bias of the model. This class should be able to be used with any pretrained model with
# known activations / weights.


class BiasModel(pl.LightningModule):
    """
    Base class for bias models. This class should be subclassed to implement specific
    bias models. The model's parameters should be defined upon initialization.
    The penalty function should be implemented in the subclass.
    The forward method should be used to compute the penalty based on the activations.
    """

    def __init__(
        self, learning_rate: float = 1e-3, weight_decay: float = 0.0, **kwargs
    ):
        """
        Initialize the bias model.

        Args:
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for regularization
            **kwargs: Additional arguments for specific bias model implementations
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @abstractmethod
    def penalty(self, act: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the penalty for the model based on the activations.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        return self.penalty(x)

    def configure_optimizers(self):
        """
        Configure optimizers for the bias model.
        """
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


class L2BiasModel(BiasModel):
    """
    L2 Bias Model. This model computes the L2 penalty based on the activations.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        l2_lambda: float = 1e-4,
        **kwargs,
    ):
        """
        Initialize the L2 bias model.

        Args:
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for regularization
            l2_lambda: Lambda for L2 penalty
            **kwargs: Additional arguments for specific bias model implementations
        """
        super().__init__(learning_rate, weight_decay, **kwargs)
        self.l2_lambda = l2_lambda
