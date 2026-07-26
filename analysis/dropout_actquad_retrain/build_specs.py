"""Turn the target/ablated endpoints into one injection spec per seed.

Why per seed rather than one median coefficient vector for all seeds (which is what
the scalar retrain experiment did): the quadratic form is defined by ``E[h^2]``, and
for every layer past the first those moments are indexed by *hidden unit*.  Hidden
units of two independently trained networks are only comparable up to permutation,
so moment vectors cannot be averaged across seeds -- the average would be
meaningless.  They can only be paired.  Pairing is well defined here because
``torch.manual_seed(seed)`` gives the dropout=0.3 and dropout=0.0 models the same
initialization (``nn.Dropout`` holds no parameters and consumes no RNG), which
``analyze.py`` verifies from the recorded init digests.

Coefficient vectors written per seed:

``coef_total``     ``c*(q=0.3)``                    -- protocol T
``coef_marginal``  ``c*(q=0.3) - c*(q=0.0)``        -- protocol M
``coef_10x``       ``10 * c*(q=0.3)``               -- overshoot control

Each ``c*`` is the exact-NNLS fit produced at its own endpoint by the
structured-regularizer harness, i.e. each uses its own endpoint's moments.  The
injected quadratic *form* is always the q=0.3 one, since that is the form whose fit
is being tested.  ``coef_marginal_same_basis`` re-fits the q=0.0 endpoint in the
q=0.3 form and is reported as a check that this mixing of forms does not drive the
marginal vector; it is not injected.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reuse import REPO, activation_quadratic_bases, exact_nnls  # noqa: E402
from penalty import ActQuadPenalizedClassifier, moments_by_name  # noqa: E402

STEPS_PER_EPOCH = math.ceil(54000 / 2048)  # MNIST train split after the 10% val cut
LR_MILESTONES = (60, 100, 200)
LR_GAMMA = 0.1


def ordinal_coefficients(record: Dict, family: str = "activation_quadratic") -> List[float]:
    fitted = record["structured"][family]["full"]["coefficients"]
    return [float(fitted[name]) for name in record["config"]["layer_names"]]


def summed_effective_lr(epochs: int, lr: float, momentum: float) -> float:
    """Total step size accumulated over training under SGD+momentum and MultiStepLR.

    Momentum contributes an asymptotic factor ``1/(1-beta)``.  Used only to bound
    the multiplicative growth a negative coefficient can produce.
    """
    total = 0.0
    for epoch in range(epochs):
        decay = LR_GAMMA ** sum(epoch >= milestone for milestone in LR_MILESTONES)
        total += STEPS_PER_EPOCH * lr * decay / (1.0 - momentum)
    return total


def growth_bound(
    coefficients: List[float], moments: List[torch.Tensor], epochs: int, lr: float, momentum: float
) -> Dict[str, float]:
    """Worst-case multiplicative growth of a weight under a negative coefficient.

    A negative ``c_l`` makes the penalty an anti-penalty: the update becomes
    ``w <- w * (1 + 2|c_l| m_j eta)``.  Reporting the bound is how the scalar
    experiment justified running its negative marginal coefficient rather than
    clamping it.
    """
    horizon = summed_effective_lr(epochs, lr, momentum)
    out = {}
    worst = 1.0
    for index, (coefficient, moment) in enumerate(zip(coefficients, moments)):
        if coefficient >= 0:
            continue
        factor = math.exp(2.0 * abs(coefficient) * float(moment.max()) * horizon)
        out[f"layer{index + 1}"] = factor
        worst = max(worst, factor)
    out["worst_case"] = worst
    out["epochs"] = epochs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    root = REPO / "data" / "generated" / "dropout_actquad_retrain"
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--artifacts", type=Path, default=root / "artifacts")
    ap.add_argument("--out", type=Path, default=root / "specs")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument(
        "--growth-epochs",
        type=int,
        default=0,
        help="horizon for the anti-penalty growth bound; 0 = median target epochs",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    targets = {}
    ablateds = {}
    for path in sorted(args.results.glob("*.json")):
        record = json.loads(path.read_text())
        condition = record["config"]["condition"]
        if condition == "target":
            targets[record["config"]["seed"]] = record
        elif condition == "ablated":
            ablateds[record["config"]["seed"]] = record
    seeds = sorted(set(targets) & set(ablateds))
    if not seeds:
        raise SystemExit("no seed has both a target and an ablated run yet")
    missing = sorted(set(targets) ^ set(ablateds))
    if missing:
        print(f"WARNING: seeds without both endpoints are skipped: {missing}")

    target_epochs = [targets[s]["stationarity"]["epochs_run"] for s in seeds]
    matched_budget = int(round(st.median(target_epochs)))
    horizon_epochs = args.growth_epochs or matched_budget

    per_seed: Dict[int, Dict] = {}
    for seed in seeds:
        target, ablated = targets[seed], ablateds[seed]
        target_art = torch.load(
            args.artifacts / f"{target['config']['tag']}.pt", weights_only=True
        )
        ablated_art = torch.load(
            args.artifacts / f"{ablated['config']['tag']}.pt", weights_only=True
        )
        moments = [m.float() for m in target_art["moments"]]

        coef_total = ordinal_coefficients(target)
        coef_ablated = ordinal_coefficients(ablated)
        coef_marginal = [t - a for t, a in zip(coef_total, coef_ablated)]

        # Same-basis check: re-fit the ablated endpoint in the target's quadratic form.
        reference = ActQuadPenalizedClassifier(
            input_dim=784,
            num_classes=10,
            depth=target["config"]["depth"],
            width=target["config"]["width"],
            dropout=0.0,
            batchnorm=False,
            l2_lambda=0.0,
            lr=args.lr,
            momentum=args.momentum,
            penalty_family="none",
            penalty_coef=0.0,
        )
        torch.nn.utils.vector_to_parameters(
            ablated_art["param_vector"], reference.parameters()
        )
        same_basis_fit = exact_nnls(
            -ablated_art["grad_vector"],
            activation_quadratic_bases(reference, moments_by_name(reference, moments)),
        )
        coef_ablated_same_basis = [
            float(same_basis_fit["coefficients"][name])
            for name in ablated["config"]["layer_names"]
        ]
        coef_marginal_same_basis = [
            t - a for t, a in zip(coef_total, coef_ablated_same_basis)
        ]

        moment_file = f"seed{seed}_moments.pt"
        torch.save(moments, args.out / moment_file)
        spec = {
            "seed": seed,
            "moment_file": moment_file,
            "target_tag": target["config"]["tag"],
            "ablated_tag": ablated["config"]["tag"],
            "target_layer_names": target["config"]["layer_names"],
            "ablated_layer_names": ablated["config"]["layer_names"],
            "coef_total": coef_total,
            "coef_marginal": coef_marginal,
            "coef_10x": [10.0 * c for c in coef_total],
            "coef_ablated_own_basis": coef_ablated,
            "coef_ablated_same_basis": coef_ablated_same_basis,
            "coef_marginal_same_basis": coef_marginal_same_basis,
            "target_epochs_run": target["stationarity"]["epochs_run"],
            "negative_components": {
                "coef_marginal": [
                    i + 1 for i, c in enumerate(coef_marginal) if c < 0
                ],
                "coef_marginal_same_basis": [
                    i + 1 for i, c in enumerate(coef_marginal_same_basis) if c < 0
                ],
            },
            "growth_bound_marginal": growth_bound(
                coef_marginal, moments, horizon_epochs, args.lr, args.momentum
            ),
        }
        (args.out / f"seed{seed}.json").write_text(json.dumps(spec, indent=2))
        per_seed[seed] = spec

    def medians(key: str) -> List[float]:
        vectors = [per_seed[s][key] for s in seeds]
        return [st.median([v[i] for v in vectors]) for i in range(len(vectors[0]))]

    summary = {
        "seeds": seeds,
        "matched_budget_epochs": matched_budget,
        "target_epochs_run": {
            "values": target_epochs,
            "median": st.median(target_epochs),
            "min": min(target_epochs),
            "max": max(target_epochs),
        },
        "ablated_epochs_run": [ablateds[s]["stationarity"]["epochs_run"] for s in seeds],
        "median_coefficients": {
            key: medians(key)
            for key in (
                "coef_total",
                "coef_marginal",
                "coef_10x",
                "coef_ablated_own_basis",
                "coef_ablated_same_basis",
                "coef_marginal_same_basis",
            )
        },
        "coefficient_ranges": {
            key: [
                [min(per_seed[s][key][i] for s in seeds), max(per_seed[s][key][i] for s in seeds)]
                for i in range(len(per_seed[seeds[0]][key]))
            ]
            for key in ("coef_total", "coef_marginal")
        },
        "worst_case_growth_marginal": max(
            per_seed[s]["growth_bound_marginal"]["worst_case"] for s in seeds
        ),
        "growth_horizon_epochs": horizon_epochs,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
