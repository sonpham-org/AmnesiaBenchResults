"""Cost vs. context-efficiency, two versions:

  A. Solved-only                — skip unsolved problems entirely.
  B. Including-unsolved          — unsolved → score=0, cost=cost_unbounded.

For each (model, problem):
  solved:
    score = baseline_n_reliable / n_reliable                  (∈ (0, 1])
    cost  = nanodollars of the run that defined n_reliable
            (if that run was 'reused_unbounded', fall back to cost_unbounded)
  unsolved:
    score = 0
    cost  = cost_unbounded   (what the model spent trying and failing)

Baseline_n_reliable[pid] = min n_reliable across the 8 observed models.

Plot: scatter of individual points + per-model centroid with
±1 SD error bars on BOTH axes. X-axis is log.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_plot import MODELS, MODEL_COLORS, ROOT, compute_n_reliable, entry_cost_nanodollars


def _is_reused(entry: dict) -> bool:
    trials = entry.get("trials", []) or []
    if not trials:
        return False
    return all(t.get("finish_reason") == "reused_unbounded" for t in trials)


def load_all():
    """Returns (by_model, baseline) where
         by_model[model][pid] = {'n_reliable', 'cost_unb', 'cost_nr_run'}
         baseline[pid] = min finite n_reliable across models.
    """
    by_model = {}
    for m in MODELS:
        res = json.loads((ROOT / m / "results_scott25.json").read_text())
        trs = json.loads((ROOT / m / "traces_scott25.json").read_text())
        tbp = {t["problem_id"]: t for t in trs}
        entries = {}
        for r in res:
            pid = r["problem_id"]
            trace = tbp.get(pid)
            nr, entry, _ = compute_n_reliable(trace) if trace else (math.inf, None, None)
            cost_unb = r.get("phase_breakdown", {}).get("unbounded", {}).get("cost_nanodollars")
            cost_unb = int(cost_unb) if cost_unb is not None else 0
            if entry is not None:
                raw = entry_cost_nanodollars(entry)
                if raw == 0 and _is_reused(entry):
                    cost_nr = cost_unb
                else:
                    cost_nr = int(raw)
            else:
                cost_nr = None
            entries[pid] = {
                "n_reliable": nr if math.isfinite(nr) else None,
                "cost_unb": cost_unb,
                "cost_nr_run": cost_nr,
            }
        by_model[m] = entries

    pids = sorted({p for pm in by_model.values() for p in pm})
    baseline = {}
    for pid in pids:
        nrs = [by_model[m][pid]["n_reliable"] for m in MODELS
               if pid in by_model[m] and by_model[m][pid]["n_reliable"] is not None]
        baseline[pid] = min(nrs) if nrs else None
    return by_model, baseline, pids


def build_points(by_model, baseline, pids, include_unsolved: bool):
    """Yield one list per model: [(cost_nd, score), ...]."""
    out = {}
    for m in MODELS:
        pts = []
        for pid in pids:
            e = by_model[m].get(pid)
            if e is None:
                continue
            nr = e["n_reliable"]
            base = baseline.get(pid)
            if nr is not None and base is not None and nr > 0:
                # Solved
                score = base / nr
                cost = e["cost_nr_run"] if e["cost_nr_run"] is not None else e["cost_unb"]
                pts.append((cost, score))
            else:
                # Unsolved
                if include_unsolved:
                    pts.append((e["cost_unb"], 0.0))
        out[m] = pts
    return out


def draw_panel(ax, pts_by_model, title):
    rng = np.random.default_rng(0)

    summary = []
    for i, m in enumerate(MODELS):
        pts = pts_by_model[m]
        if not pts:
            continue
        color = MODEL_COLORS[m]
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        # jitter x in log space so overlapping dots (same cost value) separate
        log_x = np.log10(np.clip(xs, 1, None))
        log_x += rng.uniform(-0.04, 0.04, size=len(log_x))
        xs_plot = 10 ** log_x
        ax.scatter(xs_plot, ys, color=color, alpha=0.55, s=40,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{m} (n={len(pts)})")

        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))
        sd_x = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        sd_y = float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0
        ax.scatter([mean_x], [mean_y], s=180, color=color,
                   edgecolors="black", linewidths=1.5, zorder=6, marker="o")
        summary.append((m, len(pts), mean_x, sd_x, mean_y, sd_y))

    ax.set_xscale("log")
    ax.set_xlabel("nanodollars  (log scale)")
    ax.set_ylabel("context_efficiency per problem   =   baseline_n_reliable / n_reliable")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return summary


def main():
    by_model, baseline, pids = load_all()

    fig, ax = plt.subplots(figsize=(13, 8))
    pts_all = build_points(by_model, baseline, pids, include_unsolved=True)
    sB = draw_panel(ax, pts_all,
                    "AmnesiaBench Scott-25 — cost vs context_efficiency  "
                    "(including unsolved: score=0, cost=cost_unbounded)")
    fig.tight_layout()
    out = ROOT / "cost_vs_ctxeff.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    print(f"\n  {'model':<18} {'n':>3} {'mean_cost':>14} {'sd_cost':>14} "
          f"{'mean_ctx_eff':>13} {'sd':>7}")
    for m, n, mx, sx, my, sy in sB:
        print(f"  {m:<18} {n:>3} {mx:>14,.0f} {sx:>14,.0f} {my:>13.4f} {sy:>7.4f}")


if __name__ == "__main__":
    main()
