"""The per-layer activation-weighted quadratic penalty, made injectable.

    P(theta) = sum_l c_l * ( tr(W_l diag(m_l) W_l^T) + ||b_l||^2 )

``m_l`` holds the second moments of layer ``l``'s inputs, measured in eval mode on
the training data at a fixed endpoint.  This is the same functional whose gradient
``analysis/dropout_structured_regularizers`` fits by exact NNLS; the fit selects
``c`` with ``m`` held constant, so the faithful injection freezes ``m`` too.

Design constraints inherited from the existing retrain harness
-------------------------------------------------------------
1. ``src/core`` is not modified.
2. ``c`` and ``m`` are **buffers**, never parameters.  ``model.parameters()`` --
   and therefore the flattened parameter vector, its ordering, and the SGD
   parameter group -- is byte-for-byte identical to the unpenalized baseline,
   which is what makes the endpoint re-estimation comparable across conditions.
   Buffers also allow a negative ``c_l`` (an anti-penalty), which the marginal
   protocol needs.
3. Only the **training** objective is penalized.  ``val/loss`` stays pure
   cross-entropy so EarlyStopping fires on the same rule in every condition.

Layers are addressed by ORDINAL position, never by name: inserting ``nn.Dropout``
shifts the ``nn.Sequential`` indices, so the second Linear layer is ``network.4``
in the dropout=0.3 model and ``network.3`` in the dropout=0 model.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from reuse import (  # noqa: E402  (loaded by path; see reuse.py)
    PenalizedDeepReLUClassifier,
    linear_layers,
    loss_gradient_and_moments,
)


class ActivationQuadraticPenalty(nn.Module):
    """``sum_l c_l * (tr(W_l diag(m_l) W_l^T) + ||b_l||^2)`` with ``c``, ``m`` fixed."""

    def __init__(self, coefficients: Sequence[float], moments: Sequence[Tensor]) -> None:
        super().__init__()
        if len(coefficients) != len(moments):
            raise ValueError("one coefficient per layer is required")
        self.register_buffer(
            "coefficients", torch.tensor([float(c) for c in coefficients])
        )
        for index, moment in enumerate(moments):
            self.register_buffer(f"moment_{index}", moment.detach().clone().float())
        if list(self.parameters()):
            raise AssertionError("the penalty must contribute no parameters")

    @property
    def n_layers(self) -> int:
        return int(self.coefficients.numel())

    def moment(self, index: int) -> Tensor:
        return getattr(self, f"moment_{index}")

    def forward(self, layers: Sequence[nn.Linear]) -> Tensor:
        if len(layers) != self.n_layers:
            raise ValueError(
                f"penalty was built for {self.n_layers} layers, got {len(layers)}"
            )
        total = torch.zeros((), device=self.coefficients.device)
        for index, layer in enumerate(layers):
            moment = self.moment(index)
            if moment.numel() != layer.weight.shape[1]:
                raise ValueError(
                    f"layer {index}: {moment.numel()} moments for "
                    f"{layer.weight.shape[1]} input features"
                )
            block = (layer.weight.pow(2) * moment.unsqueeze(0)).sum()
            if layer.bias is not None:
                # The augmented-feature formula: a constant input with second moment 1.
                block = block + layer.bias.pow(2).sum()
            total = total + self.coefficients[index] * block
        return total

    @torch.no_grad()
    def set_moments(self, moments: Sequence[Tensor]) -> None:
        for index, moment in enumerate(moments):
            self.moment(index).copy_(moment.to(self.moment(index)))


class ActQuadPenalizedClassifier(PenalizedDeepReLUClassifier):
    """``DeepReLUClassifier`` + an activation-quadratic term on the training loss.

    Subclasses the existing explicit-penalty model so the scalar-family path,
    the unpenalized validation objective, and the parameter layout are inherited
    verbatim.  ``penalty_family`` is left at ``"none"`` in every condition run
    here; only the activation-quadratic term is added.
    """

    def __init__(self, *args, act_quad: Optional[ActivationQuadraticPenalty] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act_quad = act_quad
        if act_quad is not None and len(self.linear_modules()) != act_quad.n_layers:
            raise ValueError("penalty layer count does not match the model")

    def linear_modules(self) -> List[nn.Linear]:
        return [module for _name, module in linear_layers(self)]

    def act_quad_penalty(self) -> Tensor:
        if self.act_quad is None:
            return torch.zeros((), device=self.device)
        return self.act_quad(self.linear_modules())

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        base = super().training_step(batch, batch_idx)
        if self.act_quad is None:
            return base
        penalty = self.act_quad_penalty()
        self.log("train/actquad_penalty", penalty, on_step=False, on_epoch=True)
        self.log("train/objective", base + penalty, on_step=False, on_epoch=True)
        return base + penalty


class MomentRefresh(Callback):
    """Re-measure ``E[h^2]`` on the training data every ``interval`` epochs.

    The secondary ("refreshed") variant.  The primary variant freezes the moments
    at the values measured at the dropout endpoint, because that is the fixed
    quadratic form the NNLS fit selected; refreshing tracks the moving activations
    instead, at the cost of no longer being a stationary penalty.
    """

    def __init__(self, loader_factory: Callable[[], Iterable], interval: int) -> None:
        if interval < 1:
            raise ValueError("refresh interval must be >= 1")
        self.loader_factory = loader_factory
        self.interval = interval
        self.refreshes = 0

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if pl_module.act_quad is None:
            return
        if (trainer.current_epoch + 1) % self.interval:
            return
        was_training = pl_module.training
        summary = loss_gradient_and_moments(pl_module, self.loader_factory())
        pl_module.act_quad.set_moments(ordinal_moments(pl_module, summary.moments))
        pl_module.zero_grad(set_to_none=True)
        pl_module.train(was_training)
        self.refreshes += 1


def ordinal_moments(model: nn.Module, moments: Dict[str, Tensor]) -> List[Tensor]:
    """Reorder a name-keyed moment dict into forward-order Linear-layer position."""
    return [moments[name] for name, _module in linear_layers(model)]


def moments_by_name(model: nn.Module, moments: Sequence[Tensor]) -> Dict[str, Tensor]:
    """Inverse of :func:`ordinal_moments`: key ordinal moments by this model's names."""
    layers = linear_layers(model)
    if len(layers) != len(moments):
        raise ValueError("moment count does not match the model's Linear layers")
    return {name: moment for (name, _module), moment in zip(layers, moments)}


def moment_summary(moments: Sequence[Tensor]) -> List[Dict[str, float]]:
    """Per-layer descriptors of ``E[h^2]``.

    ``cv`` (coefficient of variation across input units) says how much of the
    penalty's structure is *within*-layer anisotropy.  When ``cv`` is small the
    fixed quadratic form is close to a rescaled layerwise ridge, and the fact that
    hidden-unit indices are only loosely comparable between two independently
    trained networks matters correspondingly less.
    """
    out = []
    for moment in moments:
        values = moment.detach().double()
        mean = float(values.mean())
        out.append(
            {
                "dim": int(values.numel()),
                "mean": mean,
                "std": float(values.std(unbiased=False)),
                "cv": float(values.std(unbiased=False) / mean) if mean > 0 else float("nan"),
                "min": float(values.min()),
                "max": float(values.max()),
                "sum": float(values.sum()),
            }
        )
    return out
