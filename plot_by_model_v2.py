"""Three strip-plots, one per metric, same layout as compression_by_model.png:

  1. compression_ratio   = n_reliable / n_while_unbounded   (lower = better)
  2. cost_nr_run (nd)    = nanodollars of the defining-N trial set; if those
                           were all `reused_unbounded`, fall back to the
                           unbounded-phase cost (user rule).
  3. ctx_eff_per_problem = baseline_n_reliable / n_reliable (higher = better).
     The per-model MEAN of this column equals the
     composite_context_efficiency_score defined in
     ~/GitHub/autoresearch-arena/amnesia_bench/score.py:26-28 and in the
     scott25 scoring module: mean(baseline/n_reliable) over solved problems,
     where baseline = min n_reliable across all models for that problem.

Each dot = one solved problem. Black bar = mean. Whisker = ±1 SD.
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


def per_problem_metrics(model: str) -> dict[str, dict]:
    """Returns {pid: {n_reliable, n_while_unbounded, cost_nr_run}}."""
    res = json.loads((ROOT / model / "results_scott25.json").read_text())
    trs = json.loads((ROOT / model / "traces_scott25.json").read_text())
    tbp = {t["problem_id"]: t for t in trs}

    out = {}
    for r in res:
        pid = r["problem_id"]
        trace = tbp.get(pid)
        if not trace:
            continue
        nr, entry, _ = compute_n_reliable(trace)
        nwu = r.get("n_while_unbounded")
        cost_unb = r.get("phase_breakdown", {}).get("unbounded", {}).get("cost_nanodollars")

        if entry is not None:
            raw = entry_cost_nanodollars(entry)
            if raw == 0 and _is_reused(entry) and cost_unb is not None:
                cost_nr = int(cost_unb)
            else:
                cost_nr = int(raw)
        else:
            cost_nr = None

        out[pid] = {
            "n_reliable": nr if math.isfinite(nr) else None,
            "n_while_unbounded": nwu if isinstance(nwu, (int, float)) and math.isfinite(nwu) else None,
            "cost_nr_run": cost_nr,
        }
    return out


def build_series():
    """Returns dict[model] -> {pid: (compression, cost_nr, ctx_eff)}."""
    by_model = {m: per_problem_metrics(m) for m in MODELS}

    # Baseline = min n_reliable across all observed models per problem.
    pids = sorted({pid for pm in by_model.values() for pid in pm})
    baseline = {}
    for pid in pids:
        nrs = [by_model[m][pid]["n_reliable"]
               for m in MODELS
               if pid in by_model[m] and by_model[m][pid]["n_reliable"] is not None]
        baseline[pid] = min(nrs) if nrs else None

    series = {}
    for m, pm in by_model.items():
        entries = {}
        for pid, d in pm.items():
            nr = d["n_reliable"]; nwu = d["n_while_unbounded"]; cost = d["cost_nr_run"]
            base = baseline.get(pid)
            comp = (nr / nwu) if (nr is not None and nwu not in (None, 0)) else None
            ctx_eff = (base / nr) if (nr is not None and base is not None and nr > 0) else None
            entries[pid] = (comp, cost, ctx_eff)
        series[m] = entries
    return series, baseline


def strip_plot(ax, rows, *, ylabel, title, yscale="linear",
               hline=None, hline_label=None, annotate_fmt="{mean:.2f}±{sd:.2f}"):
    rng = np.random.default_rng(0)
    xpos = np.arange(len(rows))

    for i, (m, vals) in enumerate(rows):
        color = MODEL_COLORS[m]
        if not vals:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, alpha=0.55, s=45,
                   edgecolors="white", linewidths=0.4, zorder=3)
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ax.hlines(mean, i - 0.30, i + 0.30, colors="black", linewidth=2, zorder=5)
        ax.errorbar(i, mean, yerr=sd, color="black",
                    capsize=6, capthick=1.5, elinewidth=1.5, fmt="none", zorder=4)
        ax.text(i + 0.05, mean,
                " " + annotate_fmt.format(mean=mean, sd=sd) + f"\n  n={len(vals)}",
                fontsize=8.5, va="center", ha="left", color="#222")

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        if hline_label:
            ax.text(len(rows) - 0.5, hline, hline_label,
                    fontsize=9, color="gray", ha="right", va="bottom")

    ax.set_xticks(xpos)
    ax.set_xticklabels([m for m, _ in rows], rotation=20, ha="right")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.grid(True, axis="y", which="both", alpha=0.25)


def main():
    series, baseline = build_series()

    def col(idx):
        return [(m, [v[idx] for v in series[m].values() if v[idx] is not None])
                for m in MODELS]

    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    strip_plot(axes[0], col(0),
               ylabel="compression_ratio  =  n_reliable / n_while_unbounded",
               title="compression_ratio per model   (lower = better)",
               hline=1.0, hline_label="1.0 = compact matched natural usage")

    strip_plot(axes[1], col(1),
               ylabel="nanodollars of the n_reliable run (reused→unbounded)",
               title="cost of the n_reliable run per model   (lower = better)",
               yscale="log",
               annotate_fmt="μ={mean:,.0f}\n  σ={sd:,.0f}")

    strip_plot(axes[2], col(2),
               ylabel="baseline_n_reliable / n_reliable    (per problem)",
               title=("context_efficiency per problem — mean = composite_context_efficiency_score\n"
                      "(baseline = min n_reliable across the 8 observed models; higher = better)"),
               hline=1.0, hline_label="1.0 = this model set the per-problem baseline")

    fig.tight_layout()
    out = ROOT / "by_model_strip_3panel.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    print(f"\n{'model':<18} {'n':>3} {'mean_comp':>9} {'mean_cost_nd':>14} "
          f"{'composite_ctx_eff':>18}")
    for m in MODELS:
        vals = series[m]
        comps = [v[0] for v in vals.values() if v[0] is not None]
        costs = [v[1] for v in vals.values() if v[1] is not None]
        ctx = [v[2] for v in vals.values() if v[2] is not None]
        n = len(comps)
        mc = statistics.fmean(comps) if comps else float("nan")
        mcost = statistics.fmean(costs) if costs else float("nan")
        mce = statistics.fmean(ctx) if ctx else float("nan")
        print(f"{m:<18} {n:>3} {mc:>9.3f} {mcost:>14,.0f} {mce:>18.4f}")


if __name__ == "__main__":
    main()
