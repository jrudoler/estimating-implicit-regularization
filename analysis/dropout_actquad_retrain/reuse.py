"""Load the two upstream harnesses by file path, so nothing is copied or edited.

``analysis/dropout_retrain_recovery/run.py`` supplies the endpoint measurement of
the existing retrain experiment (full-batch cross-entropy gradient, the scalar
closed-form estimator, spectral summaries, penalty values, and the
explicit-penalty ``LightningModule``).  ``analysis/dropout_structured_regularizers/run.py``
supplies the activation-quadratic basis and the exact-NNLS fit.

Loading both by path -- the same trick the retrain harness uses on the closed-form
sweep -- guarantees that the coefficients injected here, the coefficients that were
published as the better-fitting family, and the coefficients re-estimated at the
retrained endpoints all come from one implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]

_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load(name: str, relative: str) -> ModuleType:
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


retrain = _load("_upstream_retrain", "analysis/dropout_retrain_recovery/run.py")
structured = _load(
    "_upstream_structured", "analysis/dropout_structured_regularizers/run.py"
)

# Names used elsewhere, listed here so the provenance of each is explicit.
full_batch_loss_grad = retrain.full_batch_loss_grad
closed_form = retrain.closed_form
spectral_summaries = retrain.spectral_summaries
penalty_values = retrain.penalty_values
SCALAR_FAMILIES = retrain.FAMILIES
PenalizedDeepReLUClassifier = retrain.PenalizedDeepReLUClassifier

linear_layers = structured.linear_layers
loss_gradient_and_moments = structured.loss_gradient_and_moments
combine_summaries = structured.combine_summaries
make_split_loaders = structured.make_split_loaders
fit_all_candidates = structured.fit_all_candidates
activation_quadratic_bases = structured.activation_quadratic_bases
exact_nnls = structured.exact_nnls
