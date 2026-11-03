"""Training-loss-only plateau stopping, independent of gradient estimates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass
class LossPlateau:
    """Check nonoverlapping mean-loss blocks; reset patience after LR changes.

    Patience is measured in training epochs, not checks. The stopping decision
    never uses validation data, a fitted coefficient, or a gradient diagnostic.
    """

    window: int = 100
    relative_improvement: float = .001
    absolute_improvement: float = 0.
    patience: int = 1000
    lr_patience: int = 500
    factor: float = .1
    min_lr: float = .0005
    losses: list[float] = field(default_factory=list)
    best: float | None = None
    bad_epochs: int = 0
    reductions: int = 0
    epochs: int = 0

    def __post_init__(self) -> None:
        if self.window < 1 or self.lr_patience < self.window or self.patience <= self.lr_patience:
            raise ValueError("Require window <= LR patience < stopping patience")
        if self.patience % self.window or self.lr_patience % self.window:
            raise ValueError("Patiences must be multiples of the averaging window")
        if not 0 <= self.relative_improvement < 1 or not 0 < self.factor < 1 or self.min_lr <= 0:
            raise ValueError("Invalid tolerance or learning-rate settings")
        if not math.isfinite(self.absolute_improvement) or self.absolute_improvement < 0:
            raise ValueError("Absolute improvement must be finite and nonnegative")

    def update(self, loss: float, lr: float) -> dict[str, Any] | None:
        if not math.isfinite(loss) or loss < 0 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Expected finite nonnegative loss and positive LR")
        self.epochs += 1
        self.losses.append(loss)
        if len(self.losses) < self.window:
            return None
        mean = math.fsum(self.losses) / len(self.losses)
        self.losses.clear()
        improved = self.best is None or mean < min(
            self.best * (1 - self.relative_improvement), self.best - self.absolute_improvement)
        if improved:
            self.best, self.bad_epochs = mean, 0
        else:
            self.bad_epochs += self.window
        record = {"epoch": self.epochs, "mean_train_loss": mean,
                  "best_mean_loss": self.best, "bad_epochs": self.bad_epochs,
                  "lr": lr, "next_lr": lr, "action": "continue"}
        if self.bad_epochs >= self.lr_patience and lr > self.min_lr * (1 + 1e-12):
            record.update(action="reduce_lr", next_lr=max(self.min_lr, lr * self.factor))
            self.best, self.bad_epochs = None, 0
            self.reductions += 1
        elif self.bad_epochs >= self.patience:
            record["action"] = "stop"
        return record

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


def coefficient_stability(before: float, after: float, *, absolute: float = 1e-7,
                          relative: float = .05) -> dict[str, Any]:
    """Practical near-zero band OR a relative check; retain both raw changes."""
    if not all(math.isfinite(x) for x in (before, after)):
        raise ValueError("Nonfinite fitted coefficient")
    change = abs(after - before)
    near_zero = max(abs(before), abs(after)) <= absolute
    relative_change = change / abs(before) if before else None
    return {"before": before, "after": after, "absolute_change": change,
            "relative_change": relative_change, "both_practically_zero": near_zero,
            "absolute_zero_band": absolute, "relative_tolerance": relative,
            "passes": near_zero or (relative_change is not None and relative_change <= relative)}
