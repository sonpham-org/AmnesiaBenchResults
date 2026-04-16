"""Compare model n_reliable to the human baseline on op_char_count.

Only one problem has a documented human trace: op_char_count,
n_reliable_human = 200 tokens (prompt ~95 tokens + compaction trigger
at 100 tokens; human testing drove it to ~200).

Bar chart: one bar per model, y-axis log, with a horizontal reference
line at 200 tokens marking the human floor.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_plot import MODELS, MODEL_COLORS, ROOT, compute_n_reliable

HUMAN_BASELINE = 200
PROBLEM = "op_char_count"
PROMPT_TOKENS = 0    # set to 95 to subtract the prompt itself


def main():
    rows = []
    for m in MODELS:
        res = json.loads((ROOT / m / "results_scott25.json").read_text())
        trs = json.loads((ROOT / m / "traces_scott25.json").read_text())
        tbp = {t["problem_id"]: t for t in trs}
        for r in res:
            if r["problem_id"] != PROBLEM:
                continue
            nr, _, _ = compute_n_reliable(tbp[PROBLEM])
            nwu = r.get("n_while_unbounded")
            rows.append((m, nr, nwu))
            break

    fig, ax = plt.subplots(figsize=(12, 6.5))
    # Sort by n_reliable (inf at end)
    sort_key = lambda r: (math.inf if not math.isfinite(r[1]) else r[1])
    rows.sort(key=sort_key)

    human_floor = HUMAN_BASELINE - PROMPT_TOKENS  # = 105

    x = np.arange(len(rows))
    bar_vals = []
    unsolved_idx = []
    for i, (m, nr, nwu) in enumerate(rows):
        if math.isfinite(nr):
            bar_vals.append(max(nr - PROMPT_TOKENS, 0))
        else:
            bar_vals.append(human_floor)  # placeholder for hatched UNSOLVED bar
            unsolved_idx.append(i)

    bars = ax.bar(x, bar_vals, color=[MODEL_COLORS[m] for m, _, _ in rows],
                  edgecolor="black", linewidth=0.6)
    for i in unsolved_idx:
        bars[i].set_hatch("////")
        bars[i].set_facecolor("lightgrey")
        bars[i].set_edgecolor("grey")
    for i, (m, nr, nwu) in enumerate(rows):
        if math.isfinite(nr):
            nwu_disp = (max(int(nwu) - PROMPT_TOKENS, 0)
                        if isinstance(nwu, (int, float)) and math.isfinite(nwu) else None)
            label = f"{int(bar_vals[i]):,}"
            if nwu_disp is not None:
                label += f"\n(nwu={nwu_disp:,})"
        else:
            label = "UNSOLVED"
        ax.text(i, bar_vals[i] + (max(bar_vals) * 0.015),
                label, ha="center", va="bottom", fontsize=9, color="black")

    ax.axhline(human_floor, color="red", linestyle="--", linewidth=2,
               label=f"human baseline = {human_floor} tokens", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _, _ in rows], rotation=20, ha="right")
    ax.set_ylabel("n_reliable  (tokens)")
    ax.set_title(f"AmnesiaBench Scott-25 — {PROBLEM}: n_reliable per model vs human\n"
                 "grey hatched = UNSOLVED (n_reliable = ∞); nwu shown beneath")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, max(v for v in bar_vals if v < math.inf) * 1.15)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "char_count_vs_human.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    print(f"\n{PROBLEM}  (prompt = {PROMPT_TOKENS} tokens subtracted)\n")
    print(f"  human floor = {HUMAN_BASELINE} - {PROMPT_TOKENS} = {HUMAN_BASELINE - PROMPT_TOKENS}")
    print(f"  {'model':<18} {'extra_tokens':>13} {'× human':>8}")
    floor = HUMAN_BASELINE - PROMPT_TOKENS
    for m, nr, nwu in rows:
        if math.isfinite(nr):
            extra = int(nr) - PROMPT_TOKENS
            mult = f"{extra / floor:.2f}×"
            print(f"  {m:<18} {extra:>13,} {mult:>8}")
        else:
            print(f"  {m:<18} {'∞':>13} {'—':>8}")


if __name__ == "__main__":
    main()
