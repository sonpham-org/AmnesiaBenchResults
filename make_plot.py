"""Scatter: compression_ratio (n_reliable / n_while_unbounded) vs nanodollars
of the run that *defines* n_reliable — i.e., the sweep/binary_search entry
whose N equals n_reliable.

One point per (model, problem). Inf compression_ratio (unsolved) is skipped.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/tmp/scott25")
MODELS = [
    "claude-haiku",
    "claude-opus",
    "claude-sonnet",
    "deepseek-3.2",
    "glm-5",
    "gpt-5.4",
    "qwen-480b",
]

# Stable color per model — every plot uses this mapping so a given model
# is always the same color across figures.
_TAB10 = plt.get_cmap("tab10").colors
MODEL_COLORS = {
    "claude-haiku":    _TAB10[0],  # blue
    "claude-opus":     _TAB10[1],  # orange
    "claude-sonnet":   _TAB10[2],  # green
    "deepseek-3.2":    _TAB10[3],  # red
    "glm-5":           _TAB10[5],  # brown   (tab10[4] = purple, skip for visibility)
    "gpt-5.4":         _TAB10[6],  # pink
    "qwen-480b":       _TAB10[7],  # grey
}


def compute_n_reliable(trace_item: dict) -> tuple[float, dict | None, str | None]:
    """Recompute n_reliable from the raw search entries.

    Pass rule (matches amnesia_kaggle/log_search.py):
      - Inner search (binary_search phase): 3 trials per N, passes iff >= 2
        succeed. n_reliable = smallest N that passes.
      - Sweep phase: 1 trial per N. Used only if binary_search has no
        passing entry (e.g., the problem only ever hit sweep before giving up).

    Returns (n_reliable, defining_entry, source_phase). defining_entry is
    the exact {N, passed, trials} dict whose N equals n_reliable.
    """
    bs_entries = trace_item.get("phases", {}).get("binary_search", {}).get("entries", []) or []
    sw_entries = trace_item.get("phases", {}).get("sweep", {}).get("entries", []) or []

    def _passed(entry: dict) -> bool:
        """Pass rule used by log_search.py:
           - 1-trial entries (outer sweep): must succeed (1/1).
           - 3-trial entries (inner binary_search + 3-trial sweeps): >= 2/3."""
        trials = entry.get("trials", [])
        if not trials:
            return False
        succ = sum(1 for t in trials if t.get("success"))
        if len(trials) >= 3:
            return succ >= 2
        return succ == len(trials)

    candidates = []
    for e in bs_entries:
        if _passed(e):
            candidates.append((e, "binary_search"))
    for e in sw_entries:
        if _passed(e):
            candidates.append((e, "sweep"))

    if not candidates:
        return math.inf, None, None

    # n_reliable is the smallest passing N across both phases.
    best_entry, best_src = min(candidates, key=lambda x: x[0]["N"])
    return float(best_entry["N"]), best_entry, best_src


def entry_cost_nanodollars(entry: dict) -> int:
    """Sum cost_nanodollars across all trials of the single defining entry."""
    return sum(int(t.get("cost_nanodollars", 0) or 0) for t in entry.get("trials", []))


def load_points(model_dir: str):
    results = json.loads((ROOT / model_dir / "results_scott25.json").read_text())
    traces = json.loads((ROOT / model_dir / "traces_scott25.json").read_text())
    traces_by_pid = {t["problem_id"]: t for t in traces}

    pts = []
    mismatches = []
    for r in results:
        pid = r["problem_id"]
        trace = traces_by_pid.get(pid)
        if trace is None:
            continue

        nr_computed, defining_entry, src = compute_n_reliable(trace)
        nwu = r.get("n_while_unbounded")

        if not math.isfinite(nr_computed) or defining_entry is None:
            continue
        if not (isinstance(nwu, (int, float)) and math.isfinite(nwu)) or nwu <= 0:
            continue

        nr_results = r.get("n_reliable")
        if isinstance(nr_results, (int, float)) and math.isfinite(nr_results):
            if int(nr_results) != int(nr_computed):
                mismatches.append((pid, nr_results, nr_computed))

        comp = nr_computed / nwu
        cost = entry_cost_nanodollars(defining_entry)
        # cost == 0 can happen when the defining-N run reused the unbounded
        # trace (finish_reason='reused_unbounded'): no new tokens billed.
        pts.append((comp, max(cost, 1), pid, int(nr_computed), src))

    if mismatches:
        print(f"  [{model_dir}] {len(mismatches)} n_reliable mismatch(es) vs results file: {mismatches[:3]}")
    return pts


def main():
    fig, ax = plt.subplots(figsize=(11, 7))

    summary = []
    for m in MODELS:
        pts = load_points(m)
        if not pts:
            print(f"{m}: no solved points")
            continue
        color = MODEL_COLORS[m]
        fresh = [p for p in pts if p[1] > 1]
        reused = [p for p in pts if p[1] <= 1]
        if fresh:
            ax.scatter([p[0] for p in fresh], [p[1] for p in fresh],
                       label=f"{m} (n={len(pts)}, {len(reused)} reused)",
                       color=color, alpha=0.85, s=60,
                       edgecolors="white", linewidths=0.5)
        else:
            ax.scatter([], [], label=f"{m} (n={len(pts)}, {len(reused)} reused)",
                       color=color, s=60)
        if reused:
            ax.scatter([p[0] for p in reused], [p[1] for p in reused],
                       facecolors="none", edgecolors=color, alpha=0.9,
                       s=60, linewidths=1.3, marker="o")
        all_xs = [p[0] for p in pts]
        fresh_ys = [p[1] for p in fresh]
        med_c = sorted(all_xs)[len(all_xs) // 2]
        med_d = sorted(fresh_ys)[len(fresh_ys) // 2] if fresh_ys else None
        summary.append((m, len(pts), len(reused), med_c, med_d))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compression_ratio  =  n_reliable / n_while_unbounded   (lower = better)")
    ax.set_ylabel("nanodollars of the run that defined n_reliable  (log scale)")
    ax.set_title("AmnesiaBench Scott-25 — compression vs cost-of-defining-run\n"
                 "(n_reliable recomputed from traces; cost from that one run only)")
    ax.grid(True, which="both", alpha=0.25)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(bottom=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, title="model (hollow = reused_unbounded, 0 nd)")

    out = ROOT / "compression_vs_nanodollars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nSaved: {out}")

    print("\nPer-model summary:")
    print(f"  {'model':<18} {'n':>4} {'reused':>6} {'med_comp':>10} {'med_nd_fresh':>14}")
    for m, n, r, mc, md in summary:
        md_s = f"{md:,}" if md is not None else "—"
        print(f"  {m:<18} {n:>4} {r:>6} {mc:>10.4f} {md_s:>14}")


if __name__ == "__main__":
    main()
