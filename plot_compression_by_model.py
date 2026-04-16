"""Per-model compression_ratio distribution.

For each solved problem per model, compute
    compression_ratio = n_reliable / n_while_unbounded
then plot one column per model: all individual points (jittered),
the mean as a wide horizontal bar, and ±1 SD as error bars.

n_reliable is recomputed from traces (make_plot.compute_n_reliable).
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_plot import MODELS, MODEL_COLORS, ROOT, compute_n_reliable


def model_compressions(model: str) -> list[tuple[str, float]]:
    res = json.loads((ROOT / model / "results_scott25.json").read_text())
    trs = json.loads((ROOT / model / "traces_scott25.json").read_text())
    tbp = {t["problem_id"]: t for t in trs}

    out: list[tuple[str, float]] = []
    for r in res:
        pid = r["problem_id"]
        trace = tbp.get(pid)
        if not trace:
            continue
        nr, _entry, _src = compute_n_reliable(trace)
        nwu = r.get("n_while_unbounded")
        if not (isinstance(nr, (int, float)) and math.isfinite(nr)):
            continue
        if not (isinstance(nwu, (int, float)) and math.isfinite(nwu)) or nwu <= 0:
            continue
        out.append((pid, nr / nwu))
    return out


def main():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(12, 7))
    rows = []
    for m in MODELS:
        pts = model_compressions(m)
        if pts:
            vals = [v for _, v in pts]
            mean = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        else:
            vals, mean, sd = [], float("nan"), float("nan")
        rows.append((m, vals, mean, sd))

    xs_label = [m for m, *_ in rows]
    xpos = np.arange(len(rows))

    for i, (m, vals, mean, sd) in enumerate(rows):
        color = MODEL_COLORS[m]
        if vals:
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.55, s=45,
                       edgecolors="white", linewidths=0.4, zorder=3)
            # mean bar
            ax.hlines(mean, i - 0.30, i + 0.30, colors="black", linewidth=2, zorder=5)
            # ±1 SD error bar
            ax.errorbar(i, mean, yerr=sd, color="black",
                        capsize=6, capthick=1.5, elinewidth=1.5,
                        fmt="none", zorder=4)
            # annotate count + mean±sd
            ax.text(i, mean, f"  μ={mean:.2f}±{sd:.2f}\n  n={len(vals)}",
                    fontsize=9, va="center", ha="left", color="#222")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(rows) - 0.5, 1.02, "1.0 = compact matched natural usage",
            fontsize=9, color="gray", ha="right")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xs_label, rotation=20, ha="right")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylabel("compression_ratio  =  n_reliable / n_while_unbounded   (lower = better)")
    ax.set_title("AmnesiaBench Scott-25 — compression ratio per model\n"
                 "(each dot = one solved problem; black bar = mean; whisker = ±1 SD)")
    ax.grid(True, axis="y", alpha=0.25)

    out = ROOT / "compression_by_model.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    print(f"\n{'model':<18} {'n':>3} {'mean':>7} {'sd':>7} {'min':>7} {'max':>7}")
    for m, vals, mean, sd in rows:
        if vals:
            print(f"{m:<18} {len(vals):>3} {mean:>7.3f} {sd:>7.3f} "
                  f"{min(vals):>7.3f} {max(vals):>7.3f}")
        else:
            print(f"{m:<18}   0    —       —       —       —")


if __name__ == "__main__":
    main()
