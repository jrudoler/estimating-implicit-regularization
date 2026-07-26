"""Analysis of the matched-stationarity dropout sweep.

Three endpoint definitions are compared, all from the same trajectories:

  final   -- the end of the long fixed budget (no early stopping)
  matched -- the FIRST probe with relative_grad_norm <= a common target
             (matched-target design)
  band    -- for every run, the probe whose relative_grad_norm is closest to a
             single common value g* (matched-band design; g* defaults to the
             lowest level every run actually reaches, i.e. max over runs of the
             per-run minimum relative_grad_norm)

For each endpoint we report
  * Spearman(dropout, relative_grad_norm)   <- the confound check; must be ~0
  * per-dropout median relative_grad_norm
  * per family: Spearman(dropout, scale_star) and Spearman(dropout, grad_cosine)

Usage:
  PYTHONPATH=src uv run python analysis/closed_form_dropout_matched/analyze.py \
      --results-dir data/generated/closed_form_dropout_matched/results
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

FAMILIES = ["ridge", "nuclear_norm", "stable_rank", "spectral_entropy", "spectral_gap"]


def load(results_dir: Path):
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"[warn] unreadable {path}")
    return runs


def spear(x, y):
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def partial_spear(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-then-residualise)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]

    r, p = stats.pearsonr(resid(rx, rz), resid(ry, rz))
    return float(r), float(p)


def pick_final(run):
    return run["final"]


def pick_matched(run):
    return run.get("matched")


def make_band_picker(g_star, key="relative_grad_norm"):
    """Pick, per run, the probe whose `key` is closest (in log) to a common g*."""

    def pick(run):
        cand = [r for r in run["trajectory"] if r.get(key, 0.0) > 0]
        if not cand:
            return None
        return min(cand, key=lambda r: abs(np.log(r[key]) - np.log(g_star)))

    return pick


def common_window(runs, key="relative_grad_norm"):
    """[highest per-run minimum, lowest per-run maximum] of `key` across runs."""
    mins, maxs = [], []
    for r in runs:
        vals = [rec[key] for rec in r["trajectory"] if rec.get(key, 0.0) > 0]
        mins.append(min(vals))
        maxs.append(max(vals))
    return max(mins), min(maxs), mins, maxs


def summarize(runs, picker, label, extra_lines=()):
    rows = []
    for run in runs:
        rec = picker(run)
        if rec is None:
            continue
        rows.append((run["config"]["dropout"], run["config"]["seed"], rec))
    if not rows:
        print(f"\n### {label}: no usable runs\n")
        return None

    drop = [r[0] for r in rows]
    rgn = [r[2]["relative_grad_norm"] for r in rows]
    print(f"\n{'=' * 78}\n### {label}   (n={len(rows)} of {len(runs)} runs)")
    for line in extra_lines:
        print(line)
    r, p = spear(drop, rgn)
    print(f"CONFOUND CHECK  Spearman(dropout, relative_grad_norm)       = {r:+.3f}  (p={p:.3g})")
    tg = [r_[2].get("relative_train_grad_norm") for r_ in rows]
    if all(v is not None for v in tg):
        rt, pt = spear(drop, tg)
        print(f"                Spearman(dropout, relative_TRAIN_grad_norm) = {rt:+.3f}  (p={pt:.3g})")
        cr = [r_[2]["contamination_ratio"] for r_ in rows]
        rcr, pcr = spear(drop, cr)
        print(f"                Spearman(dropout, contamination_ratio)      = {rcr:+.3f}  (p={pcr:.3g})")

    print(f"\n{'dropout':>8} {'n':>3} {'med rel_grad':>13} {'med rel_train_grad':>19} "
          f"{'med contam':>11} {'med epoch':>10} {'med train_loss':>15}")
    for d in sorted(set(drop)):
        sub = [r_[2] for r_ in rows if r_[0] == d]

        def med(key):
            vals = [s.get(key) for s in sub]
            vals = [v for v in vals if v is not None]
            return statistics.median(vals) if vals else float("nan")

        print(
            f"{d:>8.2f} {len(sub):>3d} {statistics.median(s['relative_grad_norm'] for s in sub):>13.4e} "
            f"{med('relative_train_grad_norm'):>19.4e} {med('contamination_ratio'):>11.4f} "
            f"{statistics.median(s['epoch'] for s in sub):>10.0f} "
            f"{statistics.median(s['train_loss'] for s in sub):>15.4e}"
        )

    print(f"\n{'family':>17} {'rho(d,s*)':>11} {'p':>10} {'rho(d,cos)':>11} {'p':>10} "
          f"{'med s* @dmin':>13} {'med s* @dmax':>13} {'med cos @dmin':>14} {'med cos @dmax':>14}")
    dmin, dmax = min(drop), max(drop)
    out = {}
    for fam in FAMILIES:
        s = [r_[2]["closed_form"][fam]["scale_star"] for r_ in rows]
        c = [r_[2]["closed_form"][fam]["grad_cosine"] for r_ in rows]
        rs, ps = spear(drop, s)
        rc, pc = spear(drop, c)
        s_lo = statistics.median([v for v, d in zip(s, drop) if d == dmin])
        s_hi = statistics.median([v for v, d in zip(s, drop) if d == dmax])
        c_lo = statistics.median([v for v, d in zip(c, drop) if d == dmin])
        c_hi = statistics.median([v for v, d in zip(c, drop) if d == dmax])
        print(f"{fam:>17} {rs:>+11.3f} {ps:>10.2e} {rc:>+11.3f} {pc:>10.2e} "
              f"{s_lo:>13.4e} {s_hi:>13.4e} {c_lo:>+14.4f} {c_hi:>+14.4f}")
        out[fam] = {
            "rho_scale": rs, "p_scale": ps, "rho_cosine": rc, "p_cosine": pc,
            "median_scale_lo": s_lo, "median_scale_hi": s_hi,
            "median_cosine_lo": c_lo, "median_cosine_hi": c_hi,
        }

    # Decomposition: the projection of the residual gradient onto each family
    # direction, per unit ||theta||.  grad_cosine divides by ||grad L||, so a rising
    # ||grad L|| deflates it mechanically; this projection does not, which separates
    # "the family explains less" from "there is more gradient to explain".
    print(f"\n{'family':>17} {'rho(d, proj)':>13} {'p':>10} {'med proj @dmin':>15} {'med proj @dmax':>15}")
    for fam in FAMILIES:
        proj = [r_[2]["closed_form"][fam]["grad_cosine"] * r_[2]["relative_grad_norm"]
                for r_ in rows]
        rp, pp_ = spear(drop, proj)
        p_lo = statistics.median([v for v, d in zip(proj, drop) if d == dmin])
        p_hi = statistics.median([v for v, d in zip(proj, drop) if d == dmax])
        print(f"{fam:>17} {rp:>+13.3f} {pp_:>10.2e} {p_lo:>+15.4e} {p_hi:>+15.4e}")
        out[fam].update({"rho_proj": rp, "p_proj": pp_,
                         "median_proj_lo": p_lo, "median_proj_hi": p_hi})

    # partials, for comparison with the unmatched sweep
    print(f"\n{'family':>17} {'partial rho(d,cos | rel_grad)':>31} {'p':>10}")
    for fam in FAMILIES:
        c = [r_[2]["closed_form"][fam]["grad_cosine"] for r_ in rows]
        if len(set(np.round(rgn, 12))) > 2:
            pr, pp = partial_spear(drop, c, rgn)
            print(f"{fam:>17} {pr:>+31.3f} {pp:>10.2e}")
            out[fam]["partial_rho_cosine"] = pr
            out[fam]["partial_p_cosine"] = pp

    # dropout > 0 only: at dropout = 0 the implicit regulariser is identically zero
    # (R := E_mask[L_drop] - L = 0), so that cell has no signal to explain and its
    # entire residual gradient is non-stationarity (contamination_ratio == 1 exactly).
    pos = [r_ for r_ in rows if r_[0] > 0]
    if len({r_[0] for r_ in pos}) > 2:
        dp = [r_[0] for r_ in pos]
        print(f"\ndropout > 0 only (n={len(pos)}):")
        print(f"{'family':>17} {'rho(d,s*)':>11} {'p':>10} {'rho(d,cos)':>11} {'p':>10}")
        for fam in FAMILIES:
            s = [r_[2]["closed_form"][fam]["scale_star"] for r_ in pos]
            c = [r_[2]["closed_form"][fam]["grad_cosine"] for r_ in pos]
            rs, ps = spear(dp, s)
            rc, pc = spear(dp, c)
            print(f"{fam:>17} {rs:>+11.3f} {ps:>10.2e} {rc:>+11.3f} {pc:>10.2e}")
            out[fam]["rho_scale_pos"] = rs
            out[fam]["p_scale_pos"] = ps
            out[fam]["rho_cosine_pos"] = rc
            out[fam]["p_cosine_pos"] = pc
        rr, pr = spear(dp, [r_[2]["relative_grad_norm"] for r_ in pos])
        print(f"{'(rel_grad_norm)':>17} {rr:>+11.3f} {pr:>10.2e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--band", type=float, default=None,
                    help="common relative_grad_norm level g* for the matched-band design "
                         "(default: the lowest level reached by every run)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    runs = load(args.results_dir)
    if not runs:
        raise SystemExit(f"no results in {args.results_dir}")

    print(f"loaded {len(runs)} runs")
    n_target = sum(1 for r in runs if r["run"]["target_met"])
    print(f"target_met: {n_target}/{len(runs)}   "
          f"target={runs[0]['config']['target_rel_grad']:g}")
    stops = {}
    for r in runs:
        stops[r["run"]["stopped_reason"]] = stops.get(r["run"]["stopped_reason"], 0) + 1
    print(f"stopped_reason: {stops}")
    print(f"wall_minutes: median {statistics.median(r['run']['wall_minutes'] for r in runs):.1f}, "
          f"max {max(r['run']['wall_minutes'] for r in runs):.1f}")
    print(f"epochs_run: median {statistics.median(r['run']['epochs_run'] for r in runs):.0f}, "
          f"min {min(r['run']['epochs_run'] for r in runs)}, "
          f"max {max(r['run']['epochs_run'] for r in runs)}")
    if not runs[0]["run"]["target_met"] or True:
        by_drop = {}
        for r in runs:
            by_drop.setdefault(r["config"]["dropout"], []).append(r["run"]["target_met"])
        print("target_met by dropout: " + ", ".join(
            f"{d:g}:{sum(v)}/{len(v)}" for d, v in sorted(by_drop.items())))

    results = {}
    results["final"] = summarize(runs, pick_final, "ENDPOINT = final (fixed long budget)")
    results["matched"] = summarize(
        runs, pick_matched,
        f"ENDPOINT = matched target (first probe with rel_grad_norm <= "
        f"{runs[0]['config']['target_rel_grad']:g})")

    # --- (B) matched band on the eval-mode gradient (the literal confound axis) ---
    lo, hi, mins, maxs = common_window(runs, "relative_grad_norm")
    g_star = args.band if args.band is not None else lo
    note = (f"g* = {g_star:.4e}   (commonly reachable window [{lo:.3e}, {hi:.3e}]; "
            f"per-run min rel_grad_norm ranges {min(mins):.3e}-{max(mins):.3e})")
    results["band"] = summarize(runs, make_band_picker(g_star),
                                "ENDPOINT = matched band on relative_grad_norm",
                                extra_lines=(note,))
    for mult in (2.0, 4.0):
        if g_star * mult <= hi:
            results[f"band_x{mult:g}"] = summarize(
                runs, make_band_picker(g_star * mult),
                f"ENDPOINT = matched band on relative_grad_norm, "
                f"g* = {g_star * mult:.4e} (robustness)")

    # --- (C) matched band on the TRAIN-objective gradient (the honest convergence axis) ---
    tlo, thi, tmins, tmaxs = common_window(runs, "relative_train_grad_norm")
    t_star = tlo
    tnote = (f"g* = {t_star:.4e} on relative_train_grad_norm (= ||grad L + grad R||/||theta||, "
             f"the non-stationarity residual). Commonly reachable window "
             f"[{tlo:.3e}, {thi:.3e}]; per-run minima {min(tmins):.3e}-{max(tmins):.3e}")
    results["band_train"] = summarize(
        runs, make_band_picker(t_star, "relative_train_grad_norm"),
        "ENDPOINT = matched band on relative_TRAIN_grad_norm", extra_lines=(tnote,))
    for mult in (2.0, 4.0):
        if t_star * mult <= thi:
            results[f"band_train_x{mult:g}"] = summarize(
                runs, make_band_picker(t_star * mult, "relative_train_grad_norm"),
                f"ENDPOINT = matched band on relative_TRAIN_grad_norm, "
                f"g* = {t_star * mult:.4e} (robustness)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
