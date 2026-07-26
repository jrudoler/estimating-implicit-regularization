#!/usr/bin/env python3
"""Aggregate the A3 (MLP vs CNN) effective-regularizer sweep into a profile table.

Reads every JSON record written by ``run.py`` under a results directory and
reports, per (arch, dataset, family), the median across seeds of

* ``closed_form_scale``          -- the EXACT optimum of the 1-D gradient-matching
                                    problem.  This is the reportable coefficient;
                                    the iterative fit is known not to converge
                                    (see data/generated/verify_estimator_convergence).
* ``closed_form_grad_cosine``    -- cos(grad P, -grad L); its SIGN says whether any
                                    positive coefficient helps at all.
* ``closed_form_residual_ratio`` -- unexplained fraction of ||grad L|| at the
                                    optimum.  1.0 is the "no regularizer" baseline.
* ``relative_grad_norm``         -- ||grad L|| / ||theta|| at the endpoint; the
                                    stationarity precondition for gradient matching.
* test accuracy

Nothing here is a validation of the estimator: an architecture class is not an
ablatable module, so there is no reference coefficient to recover.  This is a
descriptive comparison of two profiles measured with the same instrument.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ARCHES = ("mlp", "cnn")
DATASETS = ("mnist", "cifar10")
FAMILIES = (
    "ridge",
    "nuclear_norm",
    "stable_rank",
    "spectral_entropy",
    "spectral_gap",
)


def median(values: Sequence[float]) -> float:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def iqr(values: Sequence[float]) -> Tuple[float, float]:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return float("nan"), float("nan")
    return clean[0], clean[-1]


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U with a normal approximation (ties corrected).

    n=5 per group, so the p-value is coarse; the minimum attainable two-sided p
    for 5-vs-5 with no ties is 2/252 = 0.0079.  Reported alongside a rank-biserial
    effect size, which is the more informative number at this sample size.
    """
    a = [v for v in a if not math.isnan(v)]
    b = [v for v in b if not math.isnan(v)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks: List[float] = [0.0] * len(combined)
    i = 0
    tie_correction = 0.0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        t = j - i + 1
        tie_correction += t**3 - t
        i = j + 1
    rank_a = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    u_a = rank_a - na * (na + 1) / 2.0
    u = min(u_a, na * nb - u_a)
    mu = na * nb / 2.0
    n = na + nb
    var = (na * nb / 12.0) * ((n + 1) - tie_correction / (n * (n - 1)))
    if var <= 0:
        return u, float("nan")
    z = (u - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return u, min(1.0, p)


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Common-language effect size: 2*P(a>b) - 1, ties at 0.5.  In [-1, 1]."""
    a = [v for v in a if not math.isnan(v)]
    b = [v for v in b if not math.isnan(v)]
    if not a or not b:
        return float("nan")
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return 2.0 * wins / (len(a) * len(b)) - 1.0


def cliffs_delta_label(delta: float) -> str:
    d = abs(delta)
    if math.isnan(d):
        return "n/a"
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def _test_accuracy(metrics: Dict[str, Any]) -> float:
    for key in ("test/acc", "test/accuracy", "test_acc", "test/acc_epoch"):
        if key in metrics:
            return float(metrics[key])
    for key, value in metrics.items():
        if key.startswith("test") and "acc" in key:
            return float(value)
    return float("nan")


def load(results_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! unparseable: {path.name}")
            continue
        cfg = record.get("config", {})
        cf = record.get("closed_form", {}) or {}
        st = record.get("stationarity", {}) or {}
        adq = record.get("adequacy", {}) or {}
        pm = record.get("predictive_metrics", {}) or {}
        fs = record.get("fit_status", {}) or {}
        bias_params = record.get("bias_params", {}) or {}
        records.append(
            {
                "path": path,
                "arch": cfg.get("arch"),
                "dataset": cfg.get("dataset"),
                "family": cfg.get("family"),
                "seed": cfg.get("seed"),
                "param_count": cfg.get("param_count"),
                "cf_scale": cf.get("closed_form_scale", float("nan")),
                "cf_cosine": cf.get("closed_form_grad_cosine", float("nan")),
                "cf_resid": cf.get("closed_form_residual_ratio", float("nan")),
                "cf_resid_clamped": cf.get(
                    "closed_form_residual_ratio_clamped", float("nan")
                ),
                "cf_r2": cf.get("closed_form_projection_r2", float("nan")),
                "penalty_grad_norm": cf.get("penalty_grad_norm", float("nan")),
                "rel_grad_norm": st.get("relative_grad_norm", float("nan")),
                "grad_norm": st.get("full_batch_grad_norm", float("nan")),
                "train_ce": st.get("train_ce_loss", float("nan")),
                "param_norm": st.get("param_norm", float("nan")),
                "iter_scale": next(iter(bias_params.values()), float("nan")),
                "iter_resid": adq.get("train_bias/residual_ratio", float("nan")),
                "iter_cosine": adq.get("train_bias/grad_cosine", float("nan")),
                "test_acc": _test_accuracy(pm),
                "epochs": fs.get("model_epochs_run"),
                "runtime": fs.get("runtime_seconds"),
            }
        )
    return records


def fmt(value: float, spec: str = ".3g") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/generated/arch_effective_regularizer/results"),
    )
    args = parser.parse_args()

    records = load(args.results_dir.expanduser())
    print(f"Loaded {len(records)} records from {args.results_dir}\n")

    cells: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        cells[(r["arch"], r["dataset"], r["family"])].append(r)

    # ------------------------------------------------------------ completeness
    missing = [
        f"{a}/{d}/{f}:{5 - len(cells[(a, d, f)])}"
        for a in ARCHES
        for d in DATASETS
        for f in FAMILIES
        if len(cells[(a, d, f)]) < 5
    ]
    print(f"Incomplete cells (want 5 seeds): {missing if missing else 'none'}\n")

    # -------------------------------------------------------------- main table
    print("## Effective-regularizer profile (median over seeds)\n")
    header = (
        f"| {'dataset':8} | {'arch':4} | {'family':16} | {'n':>1} | "
        f"{'cf coeff':>10} | {'cf coeff range':>21} | {'cosine':>7} | "
        f"{'resid':>7} | {'resid_clamped':>13} | {'rel_gradnorm':>12} | {'test acc':>8} |"
    )
    print(header)
    print("|" + "|".join("-" * len(c) for c in header.split("|")[1:-1]) + "|")
    for dataset in DATASETS:
        for arch in ARCHES:
            for family in FAMILIES:
                rs = cells[(arch, dataset, family)]
                if not rs:
                    continue
                lo, hi = iqr([r["cf_scale"] for r in rs])
                print(
                    f"| {dataset:8} | {arch:4} | {family:16} | {len(rs):>1} | "
                    f"{fmt(median([r['cf_scale'] for r in rs])):>10} | "
                    f"{fmt(lo) + ' .. ' + fmt(hi):>21} | "
                    f"{fmt(median([r['cf_cosine'] for r in rs]), '+.4f'):>7} | "
                    f"{fmt(median([r['cf_resid'] for r in rs]), '.4f'):>7} | "
                    f"{fmt(median([r['cf_resid_clamped'] for r in rs]), '.4f'):>13} | "
                    f"{fmt(median([r['rel_grad_norm'] for r in rs]), '.3e'):>12} | "
                    f"{fmt(median([r['test_acc'] for r in rs]), '.4f'):>8} |"
                )
        print()

    # ------------------------------------------------------- negative optima
    print("## Negative closed-form optima (no positive coefficient beats s=0)\n")
    neg_by_arch: Dict[str, int] = defaultdict(int)
    tot_by_arch: Dict[str, int] = defaultdict(int)
    print(f"| {'dataset':8} | {'arch':4} | {'family':16} | neg/n | median cosine |")
    print("|----------|------|------------------|-------|---------------|")
    for dataset in DATASETS:
        for arch in ARCHES:
            for family in FAMILIES:
                rs = cells[(arch, dataset, family)]
                if not rs:
                    continue
                neg = sum(1 for r in rs if r["cf_scale"] < 0)
                tot_by_arch[arch] += len(rs)
                neg_by_arch[arch] += neg
                if neg:
                    print(
                        f"| {dataset:8} | {arch:4} | {family:16} | {neg:>2}/{len(rs)} | "
                        f"{fmt(median([r['cf_cosine'] for r in rs]), '+.4f'):>13} |"
                    )
    print()
    for arch in ARCHES:
        print(
            f"  {arch}: {neg_by_arch[arch]}/{tot_by_arch[arch]} runs with a negative optimum"
        )
    print()

    # ------------------------------------------------------------ stationarity
    print("## Stationarity by (arch, dataset), pooled over families and seeds\n")
    print(
        f"| {'dataset':8} | {'arch':4} | {'n':>3} | {'med rel_gradnorm':>16} | "
        f"{'min':>10} | {'max':>10} | {'med gradnorm':>13} | {'med train CE':>12} | {'med epochs':>10} |"
    )
    print(
        "|----------|------|-----|------------------|------------|------------|"
        "---------------|--------------|------------|"
    )
    stat_pool: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for dataset in DATASETS:
        for arch in ARCHES:
            rs = [r for r in records if r["arch"] == arch and r["dataset"] == dataset]
            if not rs:
                continue
            vals = [r["rel_grad_norm"] for r in rs]
            stat_pool[(arch, dataset)] = vals
            lo, hi = iqr(vals)
            print(
                f"| {dataset:8} | {arch:4} | {len(rs):>3} | {fmt(median(vals), '.4e'):>16} | "
                f"{fmt(lo, '.3e'):>10} | {fmt(hi, '.3e'):>10} | "
                f"{fmt(median([r['grad_norm'] for r in rs]), '.3e'):>13} | "
                f"{fmt(median([r['train_ce'] for r in rs]), '.3e'):>12} | "
                f"{fmt(median([float(r['epochs']) for r in rs if r['epochs'] is not None]), '.0f'):>10} |"
            )
    print()
    for dataset in DATASETS:
        a = stat_pool.get(("mlp", dataset), [])
        b = stat_pool.get(("cnn", dataset), [])
        if a and b:
            u, p = mann_whitney_u(a, b)
            d = rank_biserial(a, b)
            ratio = median(b) / median(a) if median(a) else float("nan")
            print(
                f"  {dataset}: rel_gradnorm cnn/mlp median ratio = {fmt(ratio, '.2f')}x, "
                f"U={fmt(u, '.0f')} p={fmt(p, '.2g')} rank-biserial={fmt(d, '+.2f')} "
                f"({cliffs_delta_label(d)})"
            )
    print()

    # ------------------------------------------- per-family arch contrast tests
    print("## MLP vs CNN per family: closed-form residual_ratio (adequacy)\n")
    print(
        f"| {'dataset':8} | {'family':16} | {'mlp med':>8} | {'cnn med':>8} | "
        f"{'better':>6} | {'U':>4} | {'p':>7} | {'rank-bis':>8} | {'size':>10} |"
    )
    print(
        "|----------|------------------|----------|----------|--------|------|---------|----------|------------|"
    )
    contrasts = []
    for dataset in DATASETS:
        for family in FAMILIES:
            a = [r["cf_resid"] for r in cells[("mlp", dataset, family)]]
            b = [r["cf_resid"] for r in cells[("cnn", dataset, family)]]
            if not a or not b:
                continue
            u, p = mann_whitney_u(a, b)
            d = rank_biserial(a, b)
            ma, mb = median(a), median(b)
            better = "mlp" if ma < mb else ("cnn" if mb < ma else "tie")
            contrasts.append((dataset, family, ma, mb, u, p, d))
            print(
                f"| {dataset:8} | {family:16} | {fmt(ma, '.4f'):>8} | {fmt(mb, '.4f'):>8} | "
                f"{better:>6} | {fmt(u, '.0f'):>4} | {fmt(p, '.3g'):>7} | "
                f"{fmt(d, '+.2f'):>8} | {cliffs_delta_label(d):>10} |"
            )
    print()

    print("## MLP vs CNN per family: closed-form grad_cosine\n")
    print(
        f"| {'dataset':8} | {'family':16} | {'mlp med':>9} | {'cnn med':>9} | "
        f"{'U':>4} | {'p':>7} | {'rank-bis':>8} |"
    )
    print(
        "|----------|------------------|-----------|-----------|------|---------|----------|"
    )
    for dataset in DATASETS:
        for family in FAMILIES:
            a = [r["cf_cosine"] for r in cells[("mlp", dataset, family)]]
            b = [r["cf_cosine"] for r in cells[("cnn", dataset, family)]]
            if not a or not b:
                continue
            u, p = mann_whitney_u(a, b)
            d = rank_biserial(a, b)
            print(
                f"| {dataset:8} | {family:16} | {fmt(median(a), '+.4f'):>9} | "
                f"{fmt(median(b), '+.4f'):>9} | {fmt(u, '.0f'):>4} | "
                f"{fmt(p, '.3g'):>7} | {fmt(d, '+.2f'):>8} |"
            )
    print()

    # ------------------------------------------------------------- best family
    print("## Best-describing family per (arch, dataset), by median residual_ratio\n")
    for dataset in DATASETS:
        for arch in ARCHES:
            ranked = sorted(
                (
                    (median([r["cf_resid"] for r in cells[(arch, dataset, f)]]), f)
                    for f in FAMILIES
                    if cells[(arch, dataset, f)]
                ),
            )
            order = ", ".join(f"{f}={fmt(v, '.4f')}" for v, f in ranked)
            print(f"  {dataset:8} {arch:4}: {order}")
    print()

    # ------------------------------------------------------- seed-level spread
    print("## Seed-level closed-form coefficients (reproducibility)\n")
    for dataset in DATASETS:
        for family in FAMILIES:
            for arch in ARCHES:
                rs = sorted(cells[(arch, dataset, family)], key=lambda r: r["seed"])
                if not rs:
                    continue
                vals = ", ".join(f"{r['seed']}:{fmt(r['cf_scale'], '.3e')}" for r in rs)
                print(f"  {dataset:8} {arch:4} {family:16} {vals}")
        print()

    # ---------------------------------------------- iterative-fit sanity check
    print("## Iterative fit vs closed form (known non-converged; sanity only)\n")
    for dataset in DATASETS:
        for arch in ARCHES:
            rs = [r for r in records if r["arch"] == arch and r["dataset"] == dataset]
            ratios = [
                r["iter_scale"] / r["cf_scale"]
                for r in rs
                if r["cf_scale"] and r["cf_scale"] > 0 and not math.isnan(r["iter_scale"])
            ]
            print(
                f"  {dataset:8} {arch:4}: median iter/closed_form = "
                f"{fmt(median(ratios), '.3g')} over n={len(ratios)} runs with s*>0"
            )
    print()

    print("## Parameter counts\n")
    for dataset in DATASETS:
        for arch in ARCHES:
            counts = {
                r["param_count"] for r in records
                if r["arch"] == arch and r["dataset"] == dataset
            }
            print(f"  {dataset:8} {arch:4}: {sorted(counts)}")


if __name__ == "__main__":
    main()
