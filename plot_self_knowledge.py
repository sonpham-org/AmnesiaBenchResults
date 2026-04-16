"""Three figures:
  1. cost vs composite_success_prediction_score   (did the model correctly
     decide whether to attempt?)
  2. cost vs composite_window_prediction_score    (was n_predicted close to
     n_reliable?)   [regenerated for easy comparison]
  3. cost vs composite_self_knowledge_score       (combined; geometric mean
     of the two per-problem scores, so both must be good)

Per-problem:
  win_pred  — amnesia_kaggle/scoring.py:
                ratio = n_reliable / n_predicted
                ratio <=1 : score = ratio
                ratio  >1 : score = (1/ratio)^2
                unsolved or no prediction : score = 0
  succ_pred — confusion-matrix scoring:
                TP (attempt=T, solved)   = 1.0
                FP (attempt=T, unsolved) = 0.0
                FN (attempt=F, solved)   = 0.8
                TN (attempt=F, unsolved) = 1.0
  self_knowledge = sqrt(win_pred * succ_pred)
       in [0, 1]; requires both to be decent simultaneously.

Panels:
  A  "solved-only" — keep only problems the model solved.
  B  "including unsolved" — every problem counts; cost for unsolved is
     cost_unbounded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_plot import MODELS, MODEL_COLORS, ROOT, compute_n_reliable, entry_cost_nanodollars


def _is_reused(entry):
    trials = entry.get("trials", []) or []
    return bool(trials) and all(t.get("finish_reason") == "reused_unbounded" for t in trials)


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def win_pred(n_reliable, n_predicted):
    """Abstention-aware window prediction.

    Returns None when the model abstained (no finite n_predicted) — the
    caller should drop those problems from the composite so we don't
    penalize the model for honestly declining to predict.

    Returns 0.0 when the model committed a number but failed to solve.
    Returns the scoring ratio otherwise.
    """
    if not _finite(n_predicted) or n_predicted <= 0:
        return None
    if not _finite(n_reliable):
        return 0.0
    r = n_reliable / n_predicted
    return r if r <= 1.0 else (1.0 / r) ** 2


def succ_pred(attempt, solved):
    if attempt and solved:      return 1.0
    if attempt and not solved:  return 0.0
    if not attempt and solved:  return 0.8
    return 1.0


def load_all():
    by_model = {}
    for m in MODELS:
        res = json.loads((ROOT / m / "results_scott25.json").read_text())
        trs = json.loads((ROOT / m / "traces_scott25.json").read_text())
        tbp = {t["problem_id"]: t for t in trs}
        per_pid = {}
        for r in res:
            pid = r["problem_id"]
            trace = tbp.get(pid)
            nr, entry, _ = compute_n_reliable(trace) if trace else (math.inf, None, None)
            cost_unb = int(r.get("phase_breakdown", {}).get("unbounded", {}).get("cost_nanodollars") or 0)
            if entry is not None:
                raw = entry_cost_nanodollars(entry)
                cost_nr = cost_unb if (raw == 0 and _is_reused(entry)) else int(raw)
            else:
                cost_nr = None
            per_pid[pid] = {
                "n_reliable": nr if math.isfinite(nr) else None,
                "n_predicted": r.get("n_predicted"),
                "attempt": bool(r.get("attempt", True)),
                "cost_nr_run": cost_nr,
                "cost_unb": cost_unb,
            }
        by_model[m] = per_pid
    return by_model


def build_points(by_model, score_fn, include_unsolved: bool):
    """score_fn(d) -> float | None. None means 'drop this problem'."""
    out = {}
    for m, per_pid in by_model.items():
        pts = []
        for pid, d in per_pid.items():
            solved = d["n_reliable"] is not None
            if not solved and not include_unsolved:
                continue
            score = score_fn(d)
            if score is None:
                continue  # abstention → drop from composite
            if solved:
                cost = d["cost_nr_run"] if d["cost_nr_run"] is not None else d["cost_unb"]
            else:
                cost = d["cost_unb"]
            if cost is None or cost <= 0:
                cost = 1
            pts.append((cost, score))
        out[m] = pts
    return out


def draw(ax, pts_by_model, title, ylabel):
    rng = np.random.default_rng(0)
    rows = []
    for i, m in enumerate(MODELS):
        pts = pts_by_model[m]
        if not pts:
            continue
        c = MODEL_COLORS[m]
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        log_x = np.log10(xs) + rng.uniform(-0.04, 0.04, size=len(xs))
        ax.scatter(10 ** log_x, ys, color=c, alpha=0.55, s=40,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{m} (n={len(pts)})")
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], s=180, color=c,
                   edgecolors="black", linewidths=1.5, zorder=6)
        rows.append((m, len(pts), mx, my))
    ax.set_xscale("log")
    ax.set_xlabel("nanodollars  (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    return rows


METRICS = [
    ("success_pred",
     lambda d: succ_pred(d["attempt"], d["n_reliable"] is not None),
     "success_prediction per problem",
     "cost_vs_success_pred.png",
     "composite_success_prediction_score",
    ),
    ("window_pred",
     lambda d: win_pred(d["n_reliable"], d["n_predicted"]),
     "window_prediction per problem",
     "cost_vs_window_pred_v2.png",
     "composite_window_prediction_score",
    ),
    ("self_knowledge",
     lambda d: (
         None if (wp := win_pred(d["n_reliable"], d["n_predicted"])) is None
         else math.sqrt(succ_pred(d["attempt"], d["n_reliable"] is not None) * wp)
     ),
     "self_knowledge per problem  =  sqrt(succ · win)",
     "cost_vs_self_knowledge.png",
     "composite_self_knowledge_score",
    ),
]


def main():
    by_model = load_all()
    for key, fn, ylabel, outfile, composite_name in METRICS:
        pA = build_points(by_model, fn, include_unsolved=False)
        pB = build_points(by_model, fn, include_unsolved=True)
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 14), sharex=True)
        sA = draw(ax_a, pA, "A. Solved-only", ylabel)
        sB = draw(ax_b, pB, "B. Including unsolved (cost=cost_unbounded)", ylabel)
        fig.suptitle(f"AmnesiaBench Scott-25 — cost vs {composite_name}")
        fig.tight_layout()
        out = ROOT / outfile
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved: {out}")

        def _pr(title, rows):
            print(f"  {title}")
            print(f"    {'model':<18} {'n':>3} {'mean_cost_nd':>14} {'composite':>10}")
            for m, n, mx, my in sorted(rows, key=lambda r: -r[3]):
                print(f"    {m:<18} {n:>3} {mx:>14,.0f} {my:>10.4f}")

        _pr("A. Solved-only", sA)
        _pr("B. Including unsolved", sB)
        print()


if __name__ == "__main__":
    main()
