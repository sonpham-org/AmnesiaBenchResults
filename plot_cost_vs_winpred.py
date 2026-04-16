"""Cost vs window_prediction score — ABSTENTION-AWARE variant.

Revised rule (per user request):
  Only score the window prediction on problems where the model actually
  committed to a number of tokens (`n_predicted` is finite and > 0).
  If the model abstained (no prediction / predicted ∞), exclude that
  problem entirely — do NOT penalize the model for being honest about
  not knowing.

Per-problem window_prediction (same ratio logic as before, applied only
to the eligible subset):
    ratio = n_reliable / n_predicted
    score = ratio            if ratio <= 1 (under-predicted, good)
          = (1/ratio)**2     if ratio > 1  (overconfident, soft penalty)
          = 0                if unsolved (model committed but failed)

Composite = mean over problems where the model made a prediction.

Two views (both include unsolved-but-predicted):
  A. x-axis = cost of the n_reliable run (reused → cost_unbounded);
     if the problem was unsolved, cost = cost_unbounded.
  B. x-axis = cost-per-token (per-model rate,
     sum(cost_nanodollars) / sum(input+output tokens) across all 25).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_plot import MODELS, MODEL_COLORS, ROOT, compute_n_reliable, entry_cost_nanodollars


def _is_reused(entry: dict) -> bool:
    trials = entry.get("trials", []) or []
    return bool(trials) and all(
        t.get("finish_reason") == "reused_unbounded" for t in trials
    )


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def predicted_a_number(n_predicted) -> bool:
    return _finite(n_predicted) and n_predicted > 0


def window_pred_score(n_reliable, n_predicted) -> float:
    """Assumes the caller has already filtered to 'predicted a number'."""
    if not _finite(n_reliable):
        return 0.0                 # committed + failed → 0
    ratio = n_reliable / n_predicted
    return ratio if ratio <= 1.0 else (1.0 / ratio) ** 2


def load_all():
    """Return (by_model, cost_per_token):
         by_model[m][pid] = {n_reliable, n_predicted, cost_nr_run, cost_unb}
         cost_per_token[m] = sum(cost_nanodollars) / sum(tokens) for that model.
    """
    by_model = {}
    cpt = {}
    for m in MODELS:
        res = json.loads((ROOT / m / "results_scott25.json").read_text())
        trs = json.loads((ROOT / m / "traces_scott25.json").read_text())
        tbp = {t["problem_id"]: t for t in trs}

        total_cost = 0
        total_tok = 0
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

            total_cost += int(r.get("cost_nanodollars", 0) or 0)
            total_tok += int(r.get("input_tokens", 0) or 0) + int(r.get("output_tokens", 0) or 0)

            per_pid[pid] = {
                "n_reliable": nr if math.isfinite(nr) else None,
                "n_predicted": r.get("n_predicted"),
                "cost_nr_run": cost_nr,
                "cost_unb": cost_unb,
            }
        by_model[m] = per_pid
        cpt[m] = (total_cost / total_tok) if total_tok > 0 else 0.0
    return by_model, cpt


def build_points_per_problem(by_model):
    """Return dict[m] -> list[(cost_nd, score, pid)].

    Only problems where the model actually predicted a number are included.
    Unsolved (n_reliable = ∞) but predicted → score=0, cost=cost_unb.
    Solved + predicted → score = win-pred ratio, cost = cost of n_reliable run.
    """
    out = {}
    for m, per_pid in by_model.items():
        pts = []
        for pid, d in per_pid.items():
            if not predicted_a_number(d["n_predicted"]):
                continue                       # abstained → skip
            score = window_pred_score(d["n_reliable"], d["n_predicted"])
            solved = d["n_reliable"] is not None
            if solved:
                cost = d["cost_nr_run"] if d["cost_nr_run"] is not None else d["cost_unb"]
            else:
                cost = d["cost_unb"]
            if cost is None or cost <= 0:
                cost = 1
            pts.append((cost, score, pid))
        out[m] = pts
    return out


def draw_scatter_cost(ax, pts_by_model, title, xlabel):
    rng = np.random.default_rng(0)
    rows = []
    for m in MODELS:
        pts = pts_by_model[m]
        if not pts:
            continue
        c = MODEL_COLORS[m]
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        log_x = np.log10(np.clip(xs, 1e-9, None)) + rng.uniform(-0.04, 0.04, size=len(xs))
        ax.scatter(10 ** log_x, ys, color=c, alpha=0.55, s=40,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{m} (n={len(pts)})")
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], s=180, color=c,
                   edgecolors="black", linewidths=1.5, zorder=6)
        rows.append((m, len(pts), mx, my))
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("window_prediction per problem")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    return rows


def draw_scatter_cpt(ax, pts_by_model, cpt, title):
    rng = np.random.default_rng(1)
    rows = []
    for m in MODELS:
        pts = pts_by_model[m]
        if not pts:
            continue
        c = MODEL_COLORS[m]
        x0 = cpt[m]
        ys = np.array([p[1] for p in pts], dtype=float)
        log_x = np.log10(np.clip(np.full(len(ys), x0, dtype=float), 1e-9, None)) \
                + rng.uniform(-0.04, 0.04, size=len(ys))
        ax.scatter(10 ** log_x, ys, color=c, alpha=0.55, s=40,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{m} (cpt={x0:,.0f}, n={len(pts)})")
        my = float(np.mean(ys))
        ax.scatter([x0], [my], s=180, color=c,
                   edgecolors="black", linewidths=1.5, zorder=6)
        rows.append((m, len(pts), x0, my))
    ax.set_xscale("log")
    ax.set_xlabel("average cost per token  (nanodollars/token, log scale)")
    ax.set_ylabel("window_prediction per problem")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    return rows


def main():
    by_model, cpt = load_all()
    pts = build_points_per_problem(by_model)

    # Top: vs cost of the n_reliable run.
    # Bottom: vs cost-per-token (per-model rate).
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 14))
    sA = draw_scatter_cost(
        ax_a, pts,
        "A. cost of the n_reliable run   (abstentions excluded)",
        "nanodollars of the n_reliable run  (log scale)",
    )
    sB = draw_scatter_cpt(
        ax_b, pts, cpt,
        "B. cost per token   (abstentions excluded)",
    )
    fig.suptitle("AmnesiaBench Scott-25 — window_prediction "
                 "(abstention-aware: problems where the model declined to "
                 "predict are excluded)")
    fig.tight_layout()
    out = ROOT / "cost_vs_winpred.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    # Also count abstentions per model so we know what's being excluded.
    def _pr(title, rows):
        print(f"\n{title}")
        print(f"  {'model':<18} {'predicted':>9} {'mean_cost':>14} {'composite_winpred':>18}")
        for m, n, mx, my in sorted(rows, key=lambda r: -r[3]):
            print(f"  {m:<18} {n:>9} {mx:>14,.1f} {my:>18.4f}")

    _pr("A. vs cost of n_reliable run", sA)
    _pr("B. vs cost per token", sB)

    print(f"\nAbstentions (n_predicted is None / ∞ / 0):")
    for m in MODELS:
        predicted = sum(1 for d in by_model[m].values() if predicted_a_number(d["n_predicted"]))
        abstained = 25 - predicted
        print(f"  {m:<18} predicted={predicted:>2}/25   abstained={abstained:>2}")


if __name__ == "__main__":
    main()
