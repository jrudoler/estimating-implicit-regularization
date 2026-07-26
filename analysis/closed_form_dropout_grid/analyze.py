"""Per-facet analysis of the FULL Figure-5 grid re-estimated by closed form.

Figure 5 reports that the estimated ridge coefficient rises monotonically with the
dropout rate over depth {3,5} x width {128,256,512}.  Those estimates came from an
iterative gradient-matching fit that does not converge (overshoots the exact 1-D
least-squares optimum by 38-61x; see verify_estimator_convergence/).  A single-cell
closed-form re-estimation (depth=3 width=256) found the coefficient FLAT.  This
script asks whether "flat" holds in all six facets, or only that one.

Everything is per facet (depth, width):
  1. Spearman(dropout, closed-form ridge s*) + median s* per dropout rate, printed
     next to the ORIGINAL iterative Spearman recomputed from
     data/generated/dropout_bias_estimation/runs.parquet (column estimated_ridge).
  2. The verdict: how many facets show a positive closed-form trend.
  3. Same table for ridge grad_cosine, stable_rank s* and cosine.
  4. Scale check: (original iterative s*) / (closed-form s*) per facet vs the
     predicted Adam-underflow freeze scale sqrt(eps * p / (8 ||theta||^2)).
  5. relative_grad_norm vs dropout per facet.

Usage:
  PYTHONPATH=src uv run python analysis/closed_form_dropout_grid/analyze.py \
      --results-dir data/generated/closed_form_dropout_grid/results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FAMILIES = ("ridge", "nuclear_norm", "stable_rank", "spectral_entropy", "spectral_gap")
FACETS = [(3, 128), (3, 256), (3, 512), (5, 128), (5, 256), (5, 512)]
DROPOUTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
ADAM_EPS = 1e-8


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: unreadable {path.name}")
            continue
        cfg, st = d["config"], d["stationarity"]
        row = {
            "file": path.name,
            "depth": cfg["depth"],
            "width": cfg["width"],
            "dropout": float(cfg["dropout"]),
            "seed": cfg["seed"],
            "param_count": cfg["param_count"],
            "epochs_run": st["epochs_run"],
            "full_batch_grad_norm": st["full_batch_grad_norm"],
            "param_norm": st["param_norm"],
            "relative_grad_norm": st["relative_grad_norm"],
            "test_acc": d["test"].get("test/acc"),
            "test_loss": d["test"].get("test/loss"),
        }
        for fam in FAMILIES:
            cf = d["closed_form"][fam]
            row[f"{fam}_s"] = cf.get("scale_star")
            row[f"{fam}_cos"] = cf.get("grad_cosine")
            row[f"{fam}_resid"] = cf.get("residual_ratio")
            row[f"{fam}_gradp"] = cf.get("grad_p_norm")
        rows.append(row)
    return pd.DataFrame(rows)


def sig(p: float) -> str:
    if p != p:
        return "  n/a"
    if p < 1e-3:
        return "***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return "  *"
    return " ns"


def rho_line(g: pd.DataFrame, col: str) -> tuple[float, float, int]:
    sub = g.dropna(subset=[col])
    if sub[col].nunique() < 2 or len(sub) < 4:
        return float("nan"), float("nan"), len(sub)
    r, p = spearmanr(sub["dropout"], sub[col])
    return float(r), float(p), len(sub)


def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/generated/closed_form_dropout_grid/results"),
    )
    ap.add_argument(
        "--original-parquet",
        type=Path,
        default=Path("data/generated/dropout_bias_estimation/runs.parquet"),
    )
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    cf = load_results(args.results_dir)
    if cf.empty:
        raise SystemExit(f"no results under {args.results_dir}")

    # ---------------------------------------------------------------- coverage
    section("0. COVERAGE")
    print(f"closed-form runs loaded: {len(cf)} / 480 expected")
    cov = cf.groupby(["depth", "width"]).size().rename("n").reset_index()
    print(cov.to_string(index=False))
    missing = []
    for d, w in FACETS:
        for do in DROPOUTS:
            for s in (123, 42, 666, 314, 17, 111, 325, 643, 432, 51):
                m = cf[
                    (cf.depth == d)
                    & (cf.width == w)
                    & (np.isclose(cf.dropout, do))
                    & (cf.seed == s)
                ]
                if m.empty:
                    missing.append((d, w, do, s))
    print(f"missing cells: {len(missing)}")
    if missing:
        print("  " + ", ".join(f"d{d}w{w}do{do}s{s}" for d, w, do, s in missing[:30]))
    per_cell = (
        cf.groupby(["depth", "width", "dropout"]).size().rename("n_seeds").reset_index()
    )
    short = per_cell[per_cell.n_seeds < 10]
    if not short.empty:
        print("cells with <10 seeds:")
        print(short.to_string(index=False))

    # ------------------------------------------------- original iterative data
    orig = pd.read_parquet(args.original_parquet)
    orig = orig[orig.depth.isin([3, 5])].copy()

    # =========================================================== 1. ridge s*
    section("1. RIDGE s*: closed form vs ORIGINAL iterative, per facet")
    print("(original Spearman recomputed from the parquet, not taken on trust)")
    print()
    hdr = (
        f"{'facet':>12} {'n':>4} | {'CLOSED FORM rho':>15} {'p':>10} {'':>3} | "
        f"{'ITERATIVE rho':>13} {'p':>10} {'':>3}"
    )
    print(hdr)
    print("-" * len(hdr))
    facet_rows = []
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        o = orig[(orig.depth == d) & (orig.width == w)]
        rc, pc, nc = rho_line(g, "ridge_s")
        ro, po = spearmanr(o.dropout, o.estimated_ridge)
        print(
            f"{f'd{d} w{w}':>12} {nc:>4} | {rc:>+15.3f} {pc:>10.2e} {sig(pc)} | "
            f"{ro:>+13.3f} {po:>10.2e} {sig(po)}"
        )
        facet_rows.append(
            dict(depth=d, width=w, n_cf=nc, rho_cf=rc, p_cf=pc, rho_iter=float(ro), p_iter=float(po))
        )

    print()
    print("Median closed-form ridge s* at each dropout rate (rows = facets):")
    med = (
        cf.pivot_table(
            index=["depth", "width"], columns="dropout", values="ridge_s", aggfunc="median"
        )
    )
    print(med.applymap(lambda v: f"{v:.3e}" if pd.notna(v) else "  --").to_string())
    print()
    print("Same, ORIGINAL iterative estimated_ridge:")
    medo = orig.pivot_table(
        index=["depth", "width"], columns="dropout", values="estimated_ridge", aggfunc="median"
    )
    print(medo.applymap(lambda v: f"{v:.3e}" if pd.notna(v) else "  --").to_string())
    print()
    print("Closed-form ridge s*: between-seed IQR per facet (median over dropout rates),")
    print("and the max-vs-min spread of the per-rate medians, in IQR units:")
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        if g.empty:
            continue
        iqrs = g.groupby("dropout")["ridge_s"].agg(lambda s: s.quantile(0.75) - s.quantile(0.25))
        m = g.groupby("dropout")["ridge_s"].median()
        spread = m.max() - m.min()
        print(
            f"  d{d} w{w}: median IQR={iqrs.median():.3e}  median-range={spread:.3e}"
            f"  = {spread / iqrs.median():.2f} IQR   (argmin d={m.idxmin()}, argmax d={m.idxmax()})"
        )

    # ---------------------------------------------------------- 2. the verdict
    section("2. VERDICT: does the closed-form ridge coefficient increase with dropout?")
    pos_sig = [r for r in facet_rows if r["rho_cf"] > 0 and r["p_cf"] < 0.05]
    neg_sig = [r for r in facet_rows if r["rho_cf"] < 0 and r["p_cf"] < 0.05]
    ns = [r for r in facet_rows if not (r["p_cf"] < 0.05)]
    print(f"facets with SIGNIFICANT POSITIVE trend (rho>0, p<0.05): {len(pos_sig)} / 6")
    for r in pos_sig:
        print(f"    d{r['depth']} w{r['width']}: rho={r['rho_cf']:+.3f} p={r['p_cf']:.2e}"
              f"  (iterative was {r['rho_iter']:+.3f})")
    print(f"facets with SIGNIFICANT NEGATIVE trend: {len(neg_sig)} / 6")
    for r in neg_sig:
        print(f"    d{r['depth']} w{r['width']}: rho={r['rho_cf']:+.3f} p={r['p_cf']:.2e}"
              f"  (iterative was {r['rho_iter']:+.3f})")
    print(f"facets with NO significant trend: {len(ns)} / 6")
    for r in ns:
        print(f"    d{r['depth']} w{r['width']}: rho={r['rho_cf']:+.3f} p={r['p_cf']:.2e}"
              f"  (iterative was {r['rho_iter']:+.3f})")
    print()
    print("Every facet's ITERATIVE trend, for contrast:")
    print("  all 6 positive and p<1e-5" if all(
        r["rho_iter"] > 0 and r["p_iter"] < 1e-5 for r in facet_rows
    ) else "  (mixed)")
    # excluding dropout=0 (where R == 0 by construction)
    print()
    print("Repeat excluding dropout=0 (R == 0 there by construction, so its ridge")
    print("signal is pure non-stationarity):")
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w) & (cf.dropout > 0)]
        o = orig[(orig.depth == d) & (orig.width == w) & (orig.dropout > 0)]
        rc, pc, nc = rho_line(g, "ridge_s")
        ro, po = spearmanr(o.dropout, o.estimated_ridge)
        print(f"  d{d} w{w}: closed form rho={rc:+.3f} p={pc:.2e}{sig(pc)}"
              f"   |  iterative rho={ro:+.3f} p={po:.2e}")

    # ------------------------------------------- 3. adequacy / other families
    section("3. ADEQUACY AND stable_rank: per-facet Spearman(dropout, .)")
    cols = [
        ("ridge_s", "ridge s*"),
        ("ridge_cos", "ridge |cos| (signed)"),
        ("stable_rank_s", "stable_rank s*"),
        ("stable_rank_cos", "stable_rank cos"),
        ("relative_grad_norm", "||grad L||/||theta||"),
    ]
    hdr = f"{'facet':>10} " + " ".join(f"{lbl:>22}" for _, lbl in cols)
    print(hdr)
    print("-" * len(hdr))
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        cells = []
        for col, _ in cols:
            r, p, _n = rho_line(g, col)
            cells.append(f"{r:+.3f}{sig(p)}({p:.0e})".rjust(22))
        print(f"{f'd{d} w{w}':>10} " + " ".join(cells))
    print()
    print("*** p<1e-3, ** p<0.01, * p<0.05, ns not significant")

    print()
    print("Endpoint values (median at dropout 0.0 -> 0.5):")
    hdr = f"{'facet':>10} {'ridge cos':>22} {'ridge s*':>24} {'rel_grad_norm':>24} {'stable_rank cos':>22}"
    print(hdr)
    print("-" * len(hdr))
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        if g.empty:
            continue
        out = []
        for col, fmt in (
            ("ridge_cos", "{:.3f}"),
            ("ridge_s", "{:.3e}"),
            ("relative_grad_norm", "{:.2e}"),
            ("stable_rank_cos", "{:+.3f}"),
        ):
            m = g.groupby("dropout")[col].median()
            lo = m.get(0.0, float("nan"))
            hi = m.get(0.5, float("nan"))
            out.append(f"{fmt.format(lo)} -> {fmt.format(hi)}")
        print(f"{f'd{d} w{w}':>10} {out[0]:>22} {out[1]:>24} {out[2]:>24} {out[3]:>22}")

    print()
    print("Ridge-aligned PROJECTION <2 theta, -grad L>/||theta||  =  s* * ||grad P||^2/||theta||")
    print("(the 'fixed L2 part' quantity from closed_form_dropout_matched), median per facet:")
    cf["ridge_proj"] = cf["ridge_s"] * cf["ridge_gradp"] ** 2 / cf["param_norm"]
    hdr = f"{'facet':>10} {'rho(d, proj)':>16} {'p':>10} {'proj @0 -> @0.5':>28} {'fold':>7}"
    print(hdr)
    print("-" * len(hdr))
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        if g.empty:
            continue
        r, p, _ = rho_line(g, "ridge_proj")
        m = g.groupby("dropout")["ridge_proj"].median()
        lo, hi = m.get(0.0, float("nan")), m.get(0.5, float("nan"))
        gm = g.groupby("dropout")["relative_grad_norm"].median()
        print(f"{f'd{d} w{w}':>10} {r:>+16.3f} {p:>10.2e} "
              f"{f'{lo:.3e} -> {hi:.3e}':>28} {hi / lo:>7.2f}x"
              f"   [rel_grad_norm {gm.get(0.5) / gm.get(0.0):.2f}x]")

    # ------------------------------------------------------- 4. scale check
    section("4. SCALE CHECK: iterative / closed-form ratio vs the predicted freeze scale")
    print("Predicted Adam-underflow freeze scale for ridge with mse_loss(reduction='mean'):")
    print("   |d loss/d beta| = (2/p) s^2 ||grad P||^2 = eps  ->  s_freeze = sqrt(eps p / (8 ||theta||^2))")
    print("   (grad P = 2 theta, so ||grad P||^2 = 4 ||theta||^2; eps = 1e-8)")
    print()
    hdr = (
        f"{'facet':>10} {'p':>9} {'||theta||':>10} {'iter med s*':>13} {'cf med s*':>12} "
        f"{'ratio':>8} {'s_freeze pred':>14} {'iter/pred':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    scale_rows = []
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        o = orig[(orig.depth == d) & (orig.width == w)]
        if g.empty:
            continue
        p_count = int(g.param_count.median())
        theta = float(g.param_norm.median())
        it = float(o.estimated_ridge.median())
        cfm = float(g.ridge_s.median())
        ratio = it / cfm
        pred = math.sqrt(ADAM_EPS * p_count / (8.0 * theta**2))
        print(
            f"{f'd{d} w{w}':>10} {p_count:>9d} {theta:>10.2f} {it:>13.3e} {cfm:>12.3e} "
            f"{ratio:>8.1f}x {pred:>14.3e} {it / pred:>10.2f}"
        )
        scale_rows.append(
            dict(depth=d, width=w, param_count=p_count, param_norm=theta,
                 iter_med=it, cf_med=cfm, ratio=ratio, s_freeze_pred=pred)
        )
    sdf = pd.DataFrame(scale_rows)
    print()
    if len(sdf) >= 4:
        for a, b in (
            ("param_count", "ratio"),
            ("param_count", "iter_med"),
            ("s_freeze_pred", "iter_med"),
            ("param_norm", "ratio"),
        ):
            r, p = spearmanr(sdf[a], sdf[b])
            print(f"  Spearman({a}, {b}) over the 6 facets = {r:+.3f} (p={p:.3f})")
        lr = np.corrcoef(np.log(sdf.s_freeze_pred), np.log(sdf.iter_med))[0, 1]
        print(f"  Pearson(log s_freeze_pred, log iterative median s*) = {lr:+.3f}")
        print(f"  iterative/predicted ratio: min={sdf.eval('iter_med/s_freeze_pred').min():.2f} "
              f"max={sdf.eval('iter_med/s_freeze_pred').max():.2f} "
              f"(a constant ratio across facets is the prediction)")

    print()
    print("Matched run-for-run ratio (same depth/width/dropout/seed in both datasets):")
    m = cf.merge(
        orig[["depth", "width", "dropout", "seed", "estimated_ridge"]],
        on=["depth", "width", "dropout", "seed"], how="inner",
    )
    m["ratio"] = m.estimated_ridge / m.ridge_s
    print(f"  matched pairs: {len(m)}")
    print(
        m.groupby(["depth", "width"])["ratio"]
        .agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
        .to_string(float_format=lambda v: f"{v:.1f}")
    )

    # ------------------------------------------------ 5. relative grad norm
    section("5. relative_grad_norm vs dropout, per facet (should rise everywhere)")
    hdr = f"{'facet':>10} {'rho':>8} {'p':>10} {'':>3} {'median @0.0':>13} {'median @0.5':>13} {'fold':>7}"
    print(hdr)
    print("-" * len(hdr))
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        if g.empty:
            continue
        r, p, _ = rho_line(g, "relative_grad_norm")
        m2 = g.groupby("dropout")["relative_grad_norm"].median()
        lo, hi = m2.get(0.0, float("nan")), m2.get(0.5, float("nan"))
        print(f"{f'd{d} w{w}':>10} {r:>+8.3f} {p:>10.2e} {sig(p)} {lo:>13.2e} {hi:>13.2e} {hi / lo:>6.2f}x")

    # ---------------------------------------------------------------- extras
    section("6. Other families, per-facet Spearman(dropout, s*) and (dropout, cos)")
    hdr = f"{'facet':>10} " + " ".join(f"{f:>18}" for f in FAMILIES)
    print("s*:")
    print(hdr)
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        cells = []
        for fam in FAMILIES:
            r, p, _ = rho_line(g, f"{fam}_s")
            cells.append(f"{r:+.3f}{sig(p)}".rjust(18))
        print(f"{f'd{d} w{w}':>10} " + " ".join(cells))
    print()
    print("cosine:")
    print(hdr)
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w)]
        cells = []
        for fam in FAMILIES:
            r, p, _ = rho_line(g, f"{fam}_cos")
            cells.append(f"{r:+.3f}{sig(p)}".rjust(18))
        print(f"{f'd{d} w{w}':>10} " + " ".join(cells))
    print()
    print("Median cosine at dropout=0.5 per facet (which family is best aligned?):")
    hdr = f"{'facet':>10} " + " ".join(f"{f:>18}" for f in FAMILIES)
    print(hdr)
    for d, w in FACETS:
        g = cf[(cf.depth == d) & (cf.width == w) & np.isclose(cf.dropout, 0.5)]
        if g.empty:
            continue
        cells = [f"{g[f'{fam}_cos'].median():+.4f}".rjust(18) for fam in FAMILIES]
        print(f"{f'd{d} w{w}':>10} " + " ".join(cells))

    section("7. Sanity: test accuracy, epochs, residual_ratio")
    print(
        cf.groupby(["depth", "width"])[
            ["test_acc", "epochs_run", "ridge_resid", "param_norm", "full_batch_grad_norm"]
        ]
        .median()
        .to_string(float_format=lambda v: f"{v:.5g}")
    )
    bad = cf[cf.ridge_resid > 1.0]
    print(f"runs with ridge residual_ratio > 1 (should be none at the optimum): {len(bad)}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "n_runs": int(len(cf)),
                    "facet_ridge": facet_rows,
                    "scale_check": scale_rows,
                    "missing": [list(x) for x in missing],
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
