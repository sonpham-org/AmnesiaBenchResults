"""Cost vs compaction_ratio  (n_reliable / n_while_unbounded).

Solved-only: compaction_ratio is only defined when both n_reliable and
n_while_unbounded are finite (per amnesia_kaggle/scoring.py:
"Problems where either unbounded or compact failed are excluded").

x = cost of the run that defined n_reliable (reused → cost_unbounded)
y = compaction_ratio     (lower = better; < 1 means compaction beat
                          the model's natural unbounded usage)

Small dot = one (problem, model); big dot = per-model mean. Log-x.
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


def model_points(model: str) -> list[tuple[float, float]]:
    res = json.loads((ROOT / model / "results_scott25.json").read_text())
    trs = json.loads((ROOT / model / "traces_scott25.json").read_text())
    tbp = {t["problem_id"]: t for t in trs}

    pts = []
    for r in res:
        pid = r["problem_id"]
        trace = tbp.get(pid)
        if not trace:
            continue
        nr, entry, _ = compute_n_reliable(trace)
        nwu = r.get("n_while_unbounded")
        if not (isinstance(nr, (int, float)) and math.isfinite(nr)):
            continue
        if not (isinstance(nwu, (int, float)) and math.isfinite(nwu)) or nwu <= 0:
            continue
        cost_unb = r.get("phase_breakdown", {}).get("unbounded", {}).get("cost_nanodollars") or 0
        if entry is not None:
            raw = entry_cost_nanodollars(entry)
            cost = int(cost_unb) if (raw == 0 and _is_reused(entry)) else int(raw)
        else:
            cost = 0
        if cost <= 0:
            continue  # degenerate, can't plot on log x
        pts.append((cost, nr / nwu))
    return pts


def main():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    rng = np.random.default_rng(0)

    summary = []
    for i, m in enumerate(MODELS):
        pts = model_points(m)
        if not pts:
            continue
        color = MODEL_COLORS[m]
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        log_x = np.log10(xs) + rng.uniform(-0.04, 0.04, size=len(xs))
        ax.scatter(10 ** log_x, ys, color=color, alpha=0.55, s=40,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{m} (n={len(pts)})")
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], s=180, color=color,
                   edgecolors="black", linewidths=1.5, zorder=6)
        summary.append((m, len(pts), mx, my))

    ax.set_xscale("log")
    ax.set_xlabel("nanodollars of the n_reliable run  (log scale)")
    ax.set_ylabel("compaction_ratio  =  n_reliable / n_while_unbounded   (lower = better)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("AmnesiaBench Scott-25 — cost vs compaction_ratio  (solved problems only)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    out = ROOT / "cost_vs_compaction_ratio.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    print(f"\n{'model':<18} {'n':>3} {'mean_cost':>14} {'mean_comp':>10}")
    for m, n, mx, my in sorted(summary, key=lambda r: r[3]):
        print(f"  {m:<18} {n:>3} {mx:>14,.0f} {my:>10.3f}")


if __name__ == "__main__":
    main()
