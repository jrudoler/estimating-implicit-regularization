"""Aggregate the retrain-recovery runs and score each condition against the TARGET.

Different seeds give different weights, so raw weight vectors are never compared.
Instead each condition is summarized by the DISTRIBUTION of endpoint statistics
across seeds, and the question asked of every retrained condition is:

    does it move from the ABLATED BASELINE (dropout=0, no penalty) toward the
    TARGET (dropout=0.3, no penalty)?

Reported per statistic:
  * median and [min, max] / IQR across seeds
  * recovery fraction  (median_cond - median_ablated) / (median_target - median_ablated)
      1.0  = exactly reproduced the target
      0.0  = did nothing (still at baseline)
      >1   = overshot
      <0   = moved away from the target
  * Mann-Whitney U p-values and rank-biserial effect sizes vs TARGET and vs ABLATED
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from scipy.stats import mannwhitneyu

ORDER = [
    "target",
    "ablated",
    "ridge_T",
    "ridge_M",
    "ridge_T10x",
    "srank_T",
    "srank_M",
    "srank_T10x",
]

LABELS = {
    "target": "TARGET dropout=0.3",
    "ablated": "ABLATED dropout=0.0",
    "ridge_T": "ridge protocol T (s=+3.557e-05)",
    "ridge_M": "ridge protocol M (s=-5.684e-06, anti-penalty)",
    "ridge_T10x": "ridge 10x control (s=+3.557e-04)",
    "srank_T": "stable_rank protocol T (s=+6.309e-05)",
    "srank_M": "stable_rank protocol M (s=+9.755e-05)",
    "srank_T10x": "stable_rank 10x control (s=+6.309e-04)",
}


def metrics(d: Dict) -> Dict[str, float]:
    cf = d["closed_form"]
    return {
        "test_acc": d["test"]["test/acc"],
        "test_loss": d["test"]["test/loss"],
        "param_norm": d["stationarity"]["param_norm"],
        "mean_stable_rank": d["spectral"]["mean_stable_rank"],
        "mean_spectral_entropy": d["spectral"]["mean_spectral_entropy"],
        "relative_grad_norm": d["stationarity"]["relative_grad_norm"],
        "epochs_run": d["stationarity"]["epochs_run"],
        "s_ridge": cf["ridge"]["scale_star"],
        "s_stable_rank": cf["stable_rank"]["scale_star"],
        "s_nuclear_norm": cf["nuclear_norm"]["scale_star"],
        "s_spectral_entropy": cf["spectral_entropy"]["scale_star"],
        "cos_ridge": abs(cf["ridge"]["grad_cosine"]),
        "cos_stable_rank": abs(cf["stable_rank"]["grad_cosine"]),
        "resid_ridge": cf["ridge"]["residual_ratio"],
        "resid_stable_rank": cf["stable_rank"]["residual_ratio"],
        "P_ridge": d["penalty_values"]["ridge"],
        "P_stable_rank": d["penalty_values"]["stable_rank"],
    }


HEADLINE = [
    "test_acc",
    "param_norm",
    "mean_stable_rank",
    "mean_spectral_entropy",
    "s_ridge",
    "s_stable_rank",
    "relative_grad_norm",
]


def rank_biserial(a: List[float], b: List[float]) -> float:
    """2*U/(n1*n2) - 1 in [-1, 1]; +1 means every a exceeds every b."""
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def fmt(v: float, key: str) -> str:
    if key.startswith("s_") or key == "relative_grad_norm":
        return f"{v:+.4e}" if key.startswith("s_") else f"{v:.3e}"
    if key == "test_acc":
        return f"{v:.4f}"
    return f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("data/generated/dropout_retrain_recovery/results"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for path in sorted(glob.glob(str(args.results / "*.json"))):
        d = json.load(open(path))
        by_cond[d["config"]["condition"]].append(d)

    vals: Dict[str, Dict[str, List[float]]] = {}
    for cond, runs in by_cond.items():
        m = [metrics(d) for d in runs]
        vals[cond] = {k: [x[k] for x in m] for k in m[0]}

    conds = [c for c in ORDER if c in vals] + [c for c in sorted(vals) if c not in ORDER]
    print("runs per condition: " + ", ".join(f"{c}={len(by_cond[c])}" for c in conds))
    print()

    # ---- headline table -------------------------------------------------
    header = f"{'condition':<44}" + "".join(f"{k:>22}" for k in HEADLINE)
    print(header)
    print("-" * len(header))
    for c in conds:
        row = f"{LABELS.get(c, c):<44}"
        for k in HEADLINE:
            v = vals[c][k]
            row += f"{fmt(st.median(v), k):>22}"
        print(row)
    print()
    print("spread (min .. max across seeds)")
    print("-" * len(header))
    for c in conds:
        row = f"{LABELS.get(c, c):<44}"
        for k in HEADLINE:
            v = vals[c][k]
            row += f"{fmt(min(v), k) + '..' + fmt(max(v), k):>22}"
        print(row)
    print()

    # ---- recovery fractions + tests ------------------------------------
    if "target" in vals and "ablated" in vals:
        tgt, abl = vals["target"], vals["ablated"]
        print("recovery fraction  (med_cond - med_ablated) / (med_target - med_ablated)")
        print("  1.0 = reproduced target | 0.0 = still at baseline | >1 overshoot | <0 wrong direction")
        hdr = f"{'condition':<44}" + "".join(f"{k:>22}" for k in HEADLINE)
        print(hdr)
        print("-" * len(hdr))
        for c in conds:
            if c in ("target", "ablated"):
                continue
            row = f"{LABELS.get(c, c):<44}"
            for k in HEADLINE:
                den = st.median(tgt[k]) - st.median(abl[k])
                num = st.median(vals[c][k]) - st.median(abl[k])
                row += f"{(num / den if den != 0 else float('nan')):>+22.3f}"
            print(row)
        print()

        print("Mann-Whitney two-sided p / rank-biserial effect size")
        for k in HEADLINE:
            print(f"\n  [{k}]")
            p_ta = mannwhitneyu(tgt[k], abl[k], alternative="two-sided").pvalue
            print(
                f"    {'TARGET vs ABLATED':<46} p={p_ta:.2e}  rb={rank_biserial(tgt[k], abl[k]):+.3f}"
                f"   (target={fmt(st.median(tgt[k]), k)}, ablated={fmt(st.median(abl[k]), k)})"
            )
            for c in conds:
                if c in ("target", "ablated"):
                    continue
                p_t = mannwhitneyu(vals[c][k], tgt[k], alternative="two-sided").pvalue
                p_a = mannwhitneyu(vals[c][k], abl[k], alternative="two-sided").pvalue
                print(
                    f"    {c:<46} vs TARGET p={p_t:.2e} rb={rank_biserial(vals[c][k], tgt[k]):+.3f}"
                    f" | vs ABLATED p={p_a:.2e} rb={rank_biserial(vals[c][k], abl[k]):+.3f}"
                )

    # ---- full family profile (the sharpest reproduction test) -----------
    # If retraining with the explicit regularizer really reproduced the dropout
    # endpoint, then the WHOLE re-estimated profile (every family's coefficient and
    # every family's goodness of fit) should match the target's, not the baseline's.
    prof = [
        "s_ridge",
        "cos_ridge",
        "resid_ridge",
        "s_stable_rank",
        "cos_stable_rank",
        "resid_stable_rank",
        "s_nuclear_norm",
        "s_spectral_entropy",
    ]
    print("\nre-estimated closed-form family profile (medians)")
    hdr = f"{'condition':<44}" + "".join(f"{k:>20}" for k in prof)
    print(hdr)
    print("-" * len(hdr))
    for c in conds:
        row = f"{LABELS.get(c, c):<44}"
        for k in prof:
            v = st.median(vals[c][k])
            row += f"{(f'{v:+.4e}' if k.startswith('s_') else f'{v:.4f}'):>20}"
        print(row)

    print("\npenalty values P(theta) at endpoint (medians) and injected coefficient")
    print(f"{'condition':<44}{'coef injected':>16}{'P_ridge':>14}{'P_stable_rank':>16}{'coef*P':>14}")
    for c in conds:
        coef = by_cond[c][0]["config"]["penalty_coef"]
        fam = by_cond[c][0]["config"]["penalty_family"]
        pr = st.median(vals[c]["P_ridge"])
        ps = st.median(vals[c]["P_stable_rank"])
        term = coef * (pr if fam == "ridge" else ps if fam == "stable_rank" else 0.0)
        print(f"{LABELS.get(c, c):<44}{coef:>+16.4e}{pr:>14.2f}{ps:>16.3f}{term:>+14.5f}")

    # ---- additivity of the injected coefficient -------------------------
    # If the estimator's s* were simply additive under explicit injection, then
    #   s*(retrained) = s*(ablated) + s_injected.
    # It is not, because adding the penalty moves the endpoint, which changes the
    # implicit part.  The pass-through fraction quantifies how much of the injected
    # coefficient survives as measured effective regularization.
    print("\nadditivity of the injected coefficient (same family as injected)")
    print(
        f"{'condition':<44}{'injected':>15}{'s*_ablated':>15}{'s*_measured':>15}"
        f"{'predicted':>15}{'pass-through':>14}"
    )
    for c in conds:
        fam = by_cond[c][0]["config"]["penalty_family"]
        if fam not in ("ridge", "stable_rank"):
            continue
        key = "s_ridge" if fam == "ridge" else "s_stable_rank"
        inj = by_cond[c][0]["config"]["penalty_coef"]
        s_abl = st.median(vals["ablated"][key])
        s_meas = st.median(vals[c][key])
        print(
            f"{LABELS.get(c, c):<44}{inj:>+15.4e}{s_abl:>+15.4e}{s_meas:>+15.4e}"
            f"{s_abl + inj:>+15.4e}{(s_meas - s_abl) / inj:>13.2f}x"
        )

    # ---- per-layer spectral detail --------------------------------------
    # Compared by ORDINAL position, not by parameter name: inserting Dropout
    # modules shifts the nn.Sequential indices, so network.4.weight in the
    # dropout=0.3 model is network.3.weight in the dropout=0 model.
    def ordered_layers(d):
        return list(d["spectral"]["per_layer"].values())

    n_layers = len(ordered_layers(by_cond[conds[0]][0]))
    for stat in ("stable_rank", "spectral_entropy", "fro_norm"):
        print(f"\nper-layer {stat} by ordinal position (median across seeds)")
        hdr = f"{'condition':<44}" + "".join(f"{'W' + str(i + 1):>14}" for i in range(n_layers))
        print(hdr)
        print("-" * len(hdr))
        for c in conds:
            row = f"{LABELS.get(c, c):<44}"
            for i in range(n_layers):
                row += f"{st.median([ordered_layers(d)[i][stat] for d in by_cond[c]]):>14.3f}"
            print(row)

    if args.json_out:
        summary = {
            c: {
                "n": len(by_cond[c]),
                "median": {k: st.median(v) for k, v in vals[c].items()},
                "min": {k: min(v) for k, v in vals[c].items()},
                "max": {k: max(v) for k, v in vals[c].items()},
            }
            for c in conds
        }
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
