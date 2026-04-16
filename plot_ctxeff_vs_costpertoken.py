"""context_efficiency vs  nanodollars/token.

Per-model cost-per-token (matches `amnesia_kaggle/scoring.py`):
    cost_per_token = sum(cost_nanodollars) / sum(input_tokens + output_tokens)
    summed over all 25 problems.

Per-problem context_efficiency:
    ctx_eff = baseline_n_reliable / n_reliable     (solved)
            = 0                                     (unsolved)
    baseline = min n_reliable across the 8 observed models per problem.

Because cost_per_token is a single scalar per model, every dot for a given
model shares an x-coordinate. Small dots are individual problems (jittered
slightly in log-x for visual separation); big dot is the model mean =
composite_context_efficiency_score.

Two panels as before:
  A. Solved-only (unsolved excluded)
  B. Including unsolved (score=0)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_plot import MODELS, MODEL_COLORS, ROOT, compute_n_reliable


def load_all():
    by_model = {}          # model -> {pid: {n_reliable}}
    cost_per_token = {}    # model -> float nd/token
    for m in MODELS:
        res = json.loads((ROOT / m / "results_scott25.json").read_text())
        trs = json.loads((ROOT / m / "traces_scott25.json").read_text())
        tbp = {t["problem_id"]: t for t in trs}

        total_cost = 0
        total_tokens = 0
        per_pid = {}
        for r in res:
            pid = r["problem_id"]
            trace = tbp.get(pid)
            nr, _, _ = compute_n_reliable(trace) if trace else (math.inf, None, None)
            per_pid[pid] = {"n_reliable": nr if math.isfinite(nr) else None}
            total_cost += int(r.get("cost_nanodollars", 0) or 0)
            total_tokens += int(r.get("input_tokens", 0) or 0) + int(r.get("output_tokens", 0) or 0)
        by_model[m] = per_pid
        cost_per_token[m] = (total_cost / total_tokens) if total_tokens > 0 else 0.0

    pids = sorted({p for pm in by_model.values() for p in pm})
    baseline = {}
    for pid in pids:
        nrs = [by_model[m][pid]["n_reliable"] for m in MODELS
               if pid in by_model[m] and by_model[m][pid]["n_reliable"] is not None]
        baseline[pid] = min(nrs) if nrs else None
    return by_model, baseline, pids, cost_per_token


def build_scores(by_model, baseline, pids, include_unsolved: bool):
    out = {}
    for m, per_pid in by_model.items():
        scores = []
        for pid in pids:
            e = per_pid.get(pid)
            if e is None:
                continue
            nr = e["n_reliable"]
            base = baseline.get(pid)
            if nr is not None and base is not None and nr > 0:
                scores.append(base / nr)
            elif include_unsolved:
                scores.append(0.0)
        out[m] = scores
    return out


def draw(ax, scores_by_model, cost_per_token, title):
    rng = np.random.default_rng(0)
    rows = []
    for i, m in enumerate(MODELS):
        scores = scores_by_model[m]
        if not scores:
            continue
        c = MODEL_COLORS[m]
        cpt = cost_per_token[m]
        # small x-jitter so individual dots don't all stack on one vertical line
        xs = np.full(len(scores), cpt, dtype=float)
        log_x = np.log10(np.clip(xs, 1e-9, None)) + rng.uniform(-0.04, 0.04, size=len(xs))
        xs_plot = 10 ** log_x
        ys = np.array(scores, dtype=float)
        ax.scatter(xs_plot, ys, color=c, alpha=0.55, s=40,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{m} (cpt={cpt:,.0f}, n={len(scores)})")
        my = float(np.mean(ys))
        ax.scatter([cpt], [my], s=180, color=c,
                   edgecolors="black", linewidths=1.5, zorder=6)
        rows.append((m, len(scores), cpt, my))
    ax.set_xscale("log")
    ax.set_xlabel("average cost per token  (nanodollars/token, log scale)")
    ax.set_ylabel("context_efficiency per problem  =  baseline_n_reliable / n_reliable")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    return rows


def main():
    by_model, baseline, pids, cpt = load_all()
    pA = build_scores(by_model, baseline, pids, include_unsolved=False)
    pB = build_scores(by_model, baseline, pids, include_unsolved=True)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 14), sharex=True)
    sA = draw(ax_a, pA, cpt, "A. Solved-only (unsolved excluded)")
    sB = draw(ax_b, pB, cpt, "B. Including unsolved (score=0)")
    fig.suptitle("AmnesiaBench Scott-25 — context_efficiency vs cost-per-token  "
                 "(small dot = one problem; big dot = per-model mean)")
    fig.tight_layout()
    out = ROOT / "ctxeff_vs_costpertoken.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    def _pr(title, rows):
        print(f"\n{title}")
        print(f"  {'model':<18} {'n':>3} {'cost/token_nd':>14} {'composite_ctx_eff':>18}")
        for m, n, cpt_, my in sorted(rows, key=lambda r: r[2]):
            print(f"  {m:<18} {n:>3} {cpt_:>14,.1f} {my:>18.4f}")

    _pr("A. Solved-only", sA)
    _pr("B. Including unsolved", sB)


if __name__ == "__main__":
    main()
