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


def load_cost_per_token():
    """Per-model cost_per_token matching amnesia_kaggle/scoring.py:
       sum(cost_nanodollars) / sum(input+output tokens) across all 25."""
    cpt = {}
    for m in MODELS:
        res = json.loads((ROOT / m / "results_scott25.json").read_text())
        tc = tt = 0
        for r in res:
            tc += int(r.get("cost_nanodollars", 0) or 0)
            tt += int(r.get("input_tokens", 0) or 0) + int(r.get("output_tokens", 0) or 0)
        cpt[m] = (tc / tt) if tt > 0 else 0.0
    return cpt


def draw_cpt_panel(ax, pts_by_model, cpt, title):
    """Same Y as draw_panel but X is the per-model cost_per_token scalar."""
    rng = np.random.default_rng(1)
    summary = []
    for m in MODELS:
        pts = pts_by_model[m]
        if not pts:
            continue
        c = MODEL_COLORS[m]
        ys = np.array([p[1] for p in pts], dtype=float)
        x0 = cpt[m]
        log_x = np.log10(np.clip(np.full(len(ys), x0, dtype=float), 1e-9, None)) \
                + rng.uniform(-0.04, 0.04, size=len(ys))
        ax.scatter(10 ** log_x, ys, color=c, alpha=0.55, s=40,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{m} (cpt={x0:,.0f}, n={len(pts)})")
        my = float(np.mean(ys))
        ax.scatter([x0], [my], s=180, color=c,
                   edgecolors="black", linewidths=1.5, zorder=6)
        summary.append((m, len(pts), x0, my))
    ax.set_xscale("log")
    ax.set_xlabel("average cost per token  (nanodollars/token, log scale)")
    ax.set_ylabel("context_efficiency per problem")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85),
              fontsize=8, framealpha=0.9)
    return summary


def main():
    by_model, baseline, pids = load_all()
    cpt = load_cost_per_token()
    pts_all = build_points(by_model, baseline, pids, include_unsolved=True)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 14))
    sA = draw_panel(ax_a, pts_all,
                    "A. cost of the n_reliable run  (unsolved: cost_unbounded)")
    sB = draw_cpt_panel(ax_b, pts_all, cpt,
                        "B. cost per token  (per-model rate)")
    fig.suptitle("AmnesiaBench Scott-25 — context_efficiency vs cost  "
                 "(including unsolved: score=0)")
    fig.tight_layout()
    out = ROOT / "cost_vs_ctxeff.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    def _pr(title, rows, col_label):
        print(f"\n{title}")
        print(f"  {'model':<18} {'n':>3} {col_label:>14} {'mean_ctx_eff':>13}")
        for m, n, mx, my in sorted(rows, key=lambda r: -r[3]):
            print(f"  {m:<18} {n:>3} {mx:>14,.1f} {my:>13.4f}")

    # draw_panel returns 6-tuples; draw_cpt_panel returns 4-tuples. Normalize.
    sA_short = [(m, n, mx, my) for (m, n, mx, _sx, my, _sy) in sA]
    _pr("A. cost of n_reliable run (nd)", sA_short, "mean_cost")
    _pr("B. cost per token (nd/token)", sB, "cost/tok")


if __name__ == "__main__":
    main()
