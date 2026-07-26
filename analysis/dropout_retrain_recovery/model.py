"""A ``DeepReLUClassifier`` that can carry an ARBITRARY explicit penalty.

``core.models.DeepReLUClassifier`` only supports an ``l2_lambda`` penalty (and it
applies it as ``0.5 * l2_lambda * sum(theta^2)``, which is *not* the ``P(theta)``
whose coefficient the closed-form estimator reports).  The retraining experiment
needs to add ``s * P(theta)`` to the training objective for an arbitrary
``core.bias`` family, with ``P`` defined **exactly** as the estimator defines it,
including the possibility of a negative ``s``.

Design notes
------------
1. ``src/core`` is not modified: this is a subclass that wraps ``training_step``.
2. The ``core.bias`` module is reused verbatim for ``P``, but its learnable
   ``beta`` is replaced by a **buffer** of value 0.  With ``enforce_positive=True``
   the module's internal ``scale = exp(beta) = 1`` exactly, so ``module(...)``
   returns ``P(theta)`` and nothing else.  Making ``beta`` a buffer rather than a
   ``Parameter`` matters for two reasons:
     * the penalty module can then be registered as a normal submodule (so
       ``.to(device)`` works) while contributing **zero** entries to
       ``model.parameters()``.  The flattened parameter vector, its ordering, and
       the SGD parameter group are therefore byte-for-byte identical to the
       unpenalized baseline, which is required for the closed-form
       re-estimation to be comparable across conditions.
     * the coefficient is applied by us as a plain Python float, so a *negative*
       coefficient (an anti-penalty) is expressible; ``exp(beta)`` could not.
3. The explicit penalty is added to the **training** objective only.  ``val/loss``
   stays pure cross-entropy so that the early-stopping rule is byte-identical
   across every condition (target, ablated baseline, and all retrained
   conditions).  Otherwise the penalty term would silently change the stopping
   time and confound the comparison with a second, uncontrolled difference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.bias import (  # noqa: E402
    NuclearNormBias,
    RidgeBias,
    SpectralEntropyBias,
    SpectralGapBias,
    StableRankBias,
)
from core.estimators import vector_to_parameter_views  # noqa: E402
from core.models import DeepReLUClassifier  # noqa: E402

FAMILIES = {
    "ridge": RidgeBias,
    "nuclear_norm": NuclearNormBias,
    "stable_rank": StableRankBias,
    "spectral_entropy": SpectralEntropyBias,
    "spectral_gap": SpectralGapBias,
}


def unit_scale_penalty_module(family: str) -> torch.nn.Module:
    """A ``core.bias`` family instance that evaluates ``P(theta)`` with scale == 1.

    The learnable ``beta`` is swapped for a zero buffer, so ``exp(beta) == 1`` and
    the module holds no parameters (see module docstring).
    """
    module = FAMILIES[family](enforce_positive=True, init_value=0.0)
    beta = module.beta.detach().clone().zero_()
    del module.beta  # drop the nn.Parameter from module._parameters
    module.register_buffer("beta", beta)
    assert list(module.parameters()) == [], "penalty module must hold no parameters"
    return module


class PenalizedDeepReLUClassifier(DeepReLUClassifier):
    """``DeepReLUClassifier`` + ``penalty_coef * P(theta)`` on the training loss."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        depth: int,
        width: int,
        dropout: float,
        batchnorm: bool,
        l2_lambda: float,
        lr: float,
        momentum: float = 0.9,
        linear_layer_bias: bool = True,
        penalty_family: Optional[str] = None,
        penalty_coef: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            depth=depth,
            width=width,
            dropout=dropout,
            batchnorm=batchnorm,
            l2_lambda=l2_lambda,
            lr=lr,
            momentum=momentum,
            linear_layer_bias=linear_layer_bias,
        )
        self.penalty_family = penalty_family
        self.penalty_coef = float(penalty_coef)
        self.penalty_module = (
            unit_scale_penalty_module(penalty_family)
            if penalty_family is not None and penalty_family != "none"
            else None
        )

    # -- penalty ----------------------------------------------------------
    def penalty_value(self) -> Tensor:
        """``P(theta)``, differentiable w.r.t. the model parameters."""
        if self.penalty_module is None:
            return torch.zeros((), device=self.device)
        flat = torch.nn.utils.parameters_to_vector(self.parameters())
        structured = vector_to_parameter_views(flat, self)
        return self.penalty_module(flat, structured)

    def explicit_penalty(self) -> Tensor:
        if self.penalty_module is None or self.penalty_coef == 0.0:
            return torch.zeros((), device=self.device)
        return self.penalty_coef * self.penalty_value()

    # -- steps ------------------------------------------------------------
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        base = super().training_step(batch, batch_idx)  # logs train/loss (pure CE)
        if self.penalty_module is None or self.penalty_coef == 0.0:
            return base
        penalty = self.explicit_penalty()
        self.log("train/penalty_P", penalty / self.penalty_coef, on_step=False, on_epoch=True)
        self.log("train/penalty_term", penalty, on_step=False, on_epoch=True)
        self.log("train/objective", base + penalty, on_step=False, on_epoch=True)
        return base + penalty

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # Deliberately NOT penalized: val/loss must be the same functional across
        # conditions so EarlyStopping fires on the same rule everywhere.
        return super().validation_step(batch, batch_idx)

    def hyperparameters(self) -> Dict[str, Any]:
        out = dict(self.hparams)
        out["penalty_family"] = self.penalty_family
        out["penalty_coef"] = self.penalty_coef
        return out
