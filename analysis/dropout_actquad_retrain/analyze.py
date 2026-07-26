"""Aggregate actquad retrain-recovery runs and score each condition vs TARGET.

Same recovery / Mann-Whitney / composite-distance logic as
``analysis/dropout_retrain_recovery/analyze.py``, plus the measurements that
experiment lacked:

* function-space agreement with the seed-matched TARGET (top-1 agreement,
  mean symmetric KL on test softmax);
* parameter-space L2 distance to the seed-matched TARGET;
* re-estimated per-layer activation-quadratic coefficients.

Artifacts (``.pt``) are required for the function- and parameter-space rows;
JSON-only metrics still work if artifacts are missing.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

ORDER = [
    "target",
    "ablated",
    "actquad_T",
    "actquad_M",
    "actquad_M_matched",
    "actquad_10x",
]

LABELS = {
    "target": "TARGET dropout=0.3",
    "ablated": "ABLATED dropout=0.0",
    "actquad_T": "actquad protocol T (c*(0.3))",
    "actquad_M": "actquad protocol M (c*(0.3)-c*(0))",
    "actquad_M_matched": "actquad M matched-budget",
    "actquad_10x": "actquad 10x control (10*c*(0.3))",
}

HEADLINE = [
    "test_acc",
    "param_norm",
    "mean_stable_rank",
    "mean_spectral_entropy",
    "s_ridge",
    "actquad_r2",
    "top1_agree_vs_target",
    "symkl_vs_target",
    "param_l2_vs_target",
    "relative_grad_norm",
    "epochs_run",
]

COMPOSITE_KEYS = [
    "test_acc",
    "param_norm",
    "mean_stable_rank",
    "mean_spectral_entropy",
    "s_ridge",
    "s_stable_rank",
    "actquad_r2",
    "relative_grad_norm",
]


def rank_biserial(a: List[float], b: List[float]) -> float:
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def fmt(v: float, key: str) -> str:
    if key.startswith("s_") or key in ("relative_grad_norm", "symkl_vs_target"):
        if key.startswith("s_"):
            return f"{v:+.4e}"
        return f"{v:.3e}"
    if key in ("test_acc", "top1_agree_vs_target", "actquad_r2"):
        return f"{v:.4f}"
    if key == "param_l2_vs_target":
        return f"{v:.3f}"
    if key == "epochs_run":
        return f"{v:.1f}"
    return f"{v:.3f}"


def actquad_coefficients(record: Dict[str, Any]) -> List[float]:
    fitted = record["structured"]["activation_quadratic"]["full"]["coefficients"]
    return [float(fitted[name]) for name in record["config"]["layer_names"]]


def metrics(record: Dict[str, Any]) -> Dict[str, float]:
    cf = record["closed_form"]
    aq = record["structured"]["activation_quadratic"]["full"]
    coefs = actquad_coefficients(record)
    out = {
        "test_acc": float(record["test"]["test/acc"]),
        "test_loss": float(record["test"]["test/loss"]),
        "param_norm": float(record["stationarity"]["param_norm"]),
        "mean_stable_rank": float(record["spectral"]["mean_stable_rank"]),
        "mean_spectral_entropy": float(record["spectral"]["mean_spectral_entropy"]),
        "relative_grad_norm": float(record["stationarity"]["relative_grad_norm"]),
        "epochs_run": float(record["stationarity"]["epochs_run"]),
        "s_ridge": float(cf["ridge"]["scale_star"]),
        "s_stable_rank": float(cf["stable_rank"]["scale_star"]),
        "cos_ridge": abs(float(cf["ridge"]["grad_cosine"])),
        "resid_ridge": float(cf["ridge"]["residual_ratio"]),
        "actquad_r2": float(aq["projection_r2"]),
        "actquad_cos": abs(float(aq["cosine"])),
        "actquad_resid": float(aq["residual_ratio"]),
        "P_ridge": float(record["penalty_values"]["ridge"]),
        "P_stable_rank": float(record["penalty_values"]["stable_rank"]),
    }
    for i, c in enumerate(coefs):
        out[f"c_actquad_L{i + 1}"] = c
    return out


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    kl_pq = (p * (p.log() - q.log())).sum(dim=1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=1)
    return 0.5 * (kl_pq + kl_qp)


def pairwise_vs_target(
    artifact: Dict[str, Any], target_art: Dict[str, Any]
) -> Dict[str, float]:
    p = artifact["test_probs"].float()
    q = target_art["test_probs"].float()
    if p.shape != q.shape:
        raise ValueError("test probability shapes disagree")
    agree = float((p.argmax(1) == q.argmax(1)).float().mean())
    symkl = float(_symmetric_kl(p, q).mean())
    theta = artifact["param_vector"].float().reshape(-1)
    theta_t = target_art["param_vector"].float().reshape(-1)
    if theta.numel() != theta_t.numel():
        raise ValueError("parameter vector lengths disagree")
    return {
        "top1_agree_vs_target": agree,
        "symkl_vs_target": symkl,
        "param_l2_vs_target": float((theta - theta_t).norm()),
        "param_cos_vs_target": float(
            F.cosine_similarity(theta.unsqueeze(0), theta_t.unsqueeze(0)).item()
        ),
    }


def load_runs(
    results_dir: Path, artifacts_dir: Path
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, List[float]]]]:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for path in sorted(glob.glob(str(results_dir / "*.json"))):
        record = json.loads(Path(path).read_text())
        by_cond[record["config"]["condition"]].append(record)

    target_arts = {}
    if "target" in by_cond:
        for record in by_cond["target"]:
            art_path = artifacts_dir / f"{record['config']['tag']}.pt"
            if art_path.exists():
                target_arts[record["config"]["seed"]] = torch.load(
                    art_path, weights_only=True
                )

    vals: Dict[str, Dict[str, List[float]]] = {}
    for cond, runs in by_cond.items():
        rows = []
        for record in runs:
            row = metrics(record)
            art_path = artifacts_dir / f"{record['config']['tag']}.pt"
            seed = record["config"]["seed"]
            if art_path.exists() and seed in target_arts:
                row.update(pairwise_vs_target(torch.load(art_path, weights_only=True), target_arts[seed]))
            else:
                row.setdefault("top1_agree_vs_target", float("nan"))
                row.setdefault("symkl_vs_target", float("nan"))
                row.setdefault("param_l2_vs_target", float("nan"))
                row.setdefault("param_cos_vs_target", float("nan"))
            rows.append(row)
        keys = sorted({k for row in rows for k in row})
        vals[cond] = {k: [row.get(k, float("nan")) for row in rows] for k in keys}
    return by_cond, vals


def median_finite(xs: List[float]) -> float:
    finite = [x for x in xs if x == x]
    if not finite:
        return float("nan")
    return float(st.median(finite))


def composite_distance(
    vals: Dict[str, Dict[str, List[float]]], keys: List[str]
) -> Dict[str, float]:
    """Mean absolute median gap to TARGET, in TARGET between-seed sd units."""
    if "target" not in vals:
        return {}
    tgt = vals["target"]
    out = {}
    for cond, series in vals.items():
        if cond == "target":
            continue
        gaps = []
        for key in keys:
            if key not in tgt or key not in series:
                continue
            sd = st.pstdev(tgt[key]) if len(tgt[key]) > 1 else 0.0
            scale = sd if sd > 0 else 1.0
            gaps.append(abs(median_finite(series[key]) - median_finite(tgt[key])) / scale)
        out[cond] = float(st.mean(gaps)) if gaps else float("nan")
    return out


def write_markdown(
    path: Path,
    by_cond: Dict[str, List[Dict]],
    vals: Dict[str, Dict[str, List[float]]],
    composite: Dict[str, float],
) -> None:
    conds = [c for c in ORDER if c in vals] + [c for c in sorted(vals) if c not in ORDER]
    lines: List[str] = []
    lines.append("# Activation-quadratic retrain recovery (NeurIPS rebuttal)")
    lines.append("")
    lines.append(
        "Ablate dropout and retrain with the per-layer activation-weighted quadratic "
        "that `dropout_structured_regularizers` fitted better than ridge/stable-rank. "
        "Primary arm freezes `E[h²]` at the q=0.3 endpoint (fixed-moment)."
    )
    lines.append("")
    lines.append(
        "Runs: "
        + ", ".join(f"{c}={len(by_cond[c])}" for c in conds)
        + "."
    )
    lines.append("")
    lines.append("## Headline table (medians)")
    lines.append("")
    cols = [
        "condition",
        "test_acc",
        "||θ||",
        "stable_rank",
        "s*_ridge",
        "actquad R²",
        "top1↔TARGET",
        "symKL↔TARGET",
        "||Δθ||↔TARGET",
        "epochs",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for c in conds:
        v = vals[c]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABELS.get(c, c),
                    fmt(median_finite(v["test_acc"]), "test_acc"),
                    fmt(median_finite(v["param_norm"]), "param_norm"),
                    fmt(median_finite(v["mean_stable_rank"]), "mean_stable_rank"),
                    fmt(median_finite(v["s_ridge"]), "s_ridge"),
                    fmt(median_finite(v["actquad_r2"]), "actquad_r2"),
                    fmt(median_finite(v["top1_agree_vs_target"]), "top1_agree_vs_target"),
                    fmt(median_finite(v["symkl_vs_target"]), "symkl_vs_target"),
                    fmt(median_finite(v["param_l2_vs_target"]), "param_l2_vs_target"),
                    fmt(median_finite(v["epochs_run"]), "epochs_run"),
                ]
            )
            + " |"
        )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    abl_agree = median_finite(vals.get("ablated", {}).get("top1_agree_vs_target", [float("nan")]))
    best_cond = None
    best_agree = -1.0
    for c in conds:
        if c in ("target", "ablated"):
            continue
        agree = median_finite(vals[c]["top1_agree_vs_target"])
        if agree == agree and agree > best_agree:
            best_agree = agree
            best_cond = c
    tgt_acc = median_finite(vals["target"]["test_acc"]) if "target" in vals else float("nan")
    abl_acc = median_finite(vals["ablated"]["test_acc"]) if "ablated" in vals else float("nan")

    reinstated = False
    notes = []
    if best_cond is not None and "ablated" in vals and "target" in vals:
        # Reinstate if closer to TARGET than ABLATED on function space AND
        # accuracy moves toward TARGET with non-significant or favorable MW vs TARGET.
        agree_gain = best_agree - abl_agree
        acc = median_finite(vals[best_cond]["test_acc"])
        p_acc_t = mannwhitneyu(
            vals[best_cond]["test_acc"], vals["target"]["test_acc"], alternative="two-sided"
        ).pvalue
        p_agree_a = mannwhitneyu(
            vals[best_cond]["top1_agree_vs_target"],
            vals["ablated"]["top1_agree_vs_target"],
            alternative="two-sided",
        ).pvalue
        # Strict: must beat ablated on agreement and not remain fully separated from target on acc.
        reinstated = agree_gain > 0 and p_agree_a < 0.05 and p_acc_t > 0.05
        notes.append(
            f"Best non-baseline on top-1 agreement with TARGET: **{best_cond}** "
            f"(median agree={best_agree:.4f}; ABLATED={abl_agree:.4f}; "
            f"Δ={agree_gain:+.4f})."
        )
        notes.append(
            f"Accuracy: TARGET={tgt_acc:.4f}, ABLATED={abl_acc:.4f}, "
            f"{best_cond}={acc:.4f} (MW vs TARGET p={p_acc_t:.2e})."
        )
    if reinstated:
        lines.append(
            f"**Positive:** `{best_cond}` moves toward the dropout endpoint on "
            "function-space agreement and is not fully separated from TARGET on test accuracy."
        )
    else:
        lines.append(
            "**Negative / incomplete reinstate:** injecting the activation-quadratic "
            "family does **not** clearly reinstate the dropout=0.3 endpoint relative to "
            "the ablated baseline on the primary function-space and accuracy criteria."
        )
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    # Composite ranking
    lines.append("## Composite distance to TARGET")
    lines.append("")
    lines.append(
        "Mean of absolute median gaps on "
        + ", ".join(COMPOSITE_KEYS)
        + ", each divided by TARGET between-seed sd (lower = closer)."
    )
    lines.append("")
    ranked = sorted(composite.items(), key=lambda kv: kv[1])
    lines.append("| rank | condition | composite distance |")
    lines.append("|---:|---|---:|")
    for i, (c, d) in enumerate(ranked, 1):
        lines.append(f"| {i} | {LABELS.get(c, c)} | {d:.2f} |")
    lines.append("")

    # Mann-Whitney
    lines.append("## Mann-Whitney vs TARGET / ABLATED")
    lines.append("")
    if "target" in vals and "ablated" in vals:
        for key in [
            "test_acc",
            "top1_agree_vs_target",
            "symkl_vs_target",
            "param_l2_vs_target",
            "param_norm",
            "actquad_r2",
            "s_ridge",
        ]:
            lines.append(f"### `{key}`")
            lines.append("")
            p_ta = mannwhitneyu(
                vals["target"][key], vals["ablated"][key], alternative="two-sided"
            ).pvalue
            lines.append(
                f"- TARGET vs ABLATED: p={p_ta:.2e}, "
                f"rb={rank_biserial(vals['target'][key], vals['ablated'][key]):+.3f} "
                f"(medians {fmt(median_finite(vals['target'][key]), key)} / "
                f"{fmt(median_finite(vals['ablated'][key]), key)})"
            )
            for c in conds:
                if c in ("target", "ablated"):
                    continue
                a = vals[c][key]
                if all(math.isnan(x) for x in a):
                    continue
                p_t = mannwhitneyu(a, vals["target"][key], alternative="two-sided").pvalue
                p_a = mannwhitneyu(a, vals["ablated"][key], alternative="two-sided").pvalue
                lines.append(
                    f"- {c}: vs TARGET p={p_t:.2e} rb={rank_biserial(a, vals['target'][key]):+.3f} "
                    f"| vs ABLATED p={p_a:.2e} rb={rank_biserial(a, vals['ablated'][key]):+.3f} "
                    f"(median {fmt(median_finite(a), key)})"
                )
            lines.append("")

    # Matched budget
    lines.append("## Matched-budget effect (actquad_M)")
    lines.append("")
    if "actquad_M" in vals and "actquad_M_matched" in vals:
        for key in ["test_acc", "top1_agree_vs_target", "symkl_vs_target", "epochs_run", "param_norm"]:
            early = median_finite(vals["actquad_M"][key])
            matched = median_finite(vals["actquad_M_matched"][key])
            lines.append(
                f"- {key}: early-stop median={fmt(early, key)}, "
                f"matched-budget median={fmt(matched, key)}"
            )
        p = mannwhitneyu(
            vals["actquad_M"]["top1_agree_vs_target"],
            vals["actquad_M_matched"]["top1_agree_vs_target"],
            alternative="two-sided",
        ).pvalue
        lines.append(f"- MW top1_agree early-stop vs matched: p={p:.2e}")
    else:
        lines.append("Matched-budget arm not present in results.")
    lines.append("")

    # Re-estimated coefficients
    lines.append("## Re-estimated actquad coefficients (medians)")
    lines.append("")
    coef_keys = [k for k in next(iter(vals.values())) if k.startswith("c_actquad_L")]
    if coef_keys:
        lines.append("| condition | " + " | ".join(coef_keys) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(coef_keys)) + "|")
        for c in conds:
            cells = [f"{median_finite(vals[c][k]):+.4e}" for k in coef_keys]
            lines.append(f"| {LABELS.get(c, c)} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Fixed-moment injection freezes `E[h²]` at the TARGET endpoint; moments of the "
        "retrained network drift, so the injected form is not the network's own activation "
        "quadratic at the new endpoint."
    )
    lines.append(
        "- Protocol T injects total effective regularization (including SGD/early-stopping "
        "bias also present at dropout=0); protocol M is the dropout-marginal contrast."
    )
    lines.append(
        "- TARGET trains longer under early stopping; matched-budget M isolates that confound "
        "for one arm only."
    )
    lines.append(
        "- One architecture (DeepReLU d3/w256), one dataset (MNIST), one dropout rate (0.3)."
    )
    lines.append(
        "- Truth over positive results: a clean negative is informative for the rebuttal "
        "(reviewer request for a case where the retrained model still deviates)."
    )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path("data/generated/dropout_actquad_retrain")
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--artifacts", type=Path, default=root / "artifacts")
    ap.add_argument("--json-out", type=Path, default=root / "summary.json")
    ap.add_argument("--md-out", type=Path, default=root / "FINDINGS.md")
    args = ap.parse_args()

    by_cond, vals = load_runs(args.results, args.artifacts)
    conds = [c for c in ORDER if c in vals] + [c for c in sorted(vals) if c not in ORDER]
    print("runs per condition: " + ", ".join(f"{c}={len(by_cond[c])}" for c in conds))
    print()

    header = f"{'condition':<40}" + "".join(f"{k:>18}" for k in HEADLINE)
    print(header)
    print("-" * len(header))
    for c in conds:
        row = f"{LABELS.get(c, c):<40}"
        for k in HEADLINE:
            row += f"{fmt(median_finite(vals[c][k]), k):>18}"
        print(row)
    print()

    composite = composite_distance(vals, COMPOSITE_KEYS)
    print("composite distance to TARGET (lower=closer)")
    for c, d in sorted(composite.items(), key=lambda kv: kv[1]):
        print(f"  {c:<32} {d:.3f}")
    print()

    if "target" in vals and "ablated" in vals:
        print("Mann-Whitney (selected keys)")
        for k in ["test_acc", "top1_agree_vs_target", "symkl_vs_target", "param_l2_vs_target"]:
            print(f"  [{k}]")
            for c in conds:
                if c in ("target", "ablated"):
                    continue
                a = vals[c][k]
                if all(math.isnan(x) for x in a):
                    continue
                p_t = mannwhitneyu(a, vals["target"][k], alternative="two-sided").pvalue
                p_a = mannwhitneyu(a, vals["ablated"][k], alternative="two-sided").pvalue
                print(
                    f"    {c:<28} vsT p={p_t:.2e} rb={rank_biserial(a, vals['target'][k]):+.3f}"
                    f" | vsA p={p_a:.2e} rb={rank_biserial(a, vals['ablated'][k]):+.3f}"
                )

    summary = {
        "n_per_condition": {c: len(by_cond[c]) for c in conds},
        "median": {c: {k: median_finite(v) for k, v in vals[c].items()} for c in conds},
        "composite_distance_to_target": composite,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2))
    write_markdown(args.md_out, by_cond, vals, composite)
    print(f"\nwrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
