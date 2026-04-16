"""Context window growth over time.

X = release date, Y = max context window (tokens, log scale).
Models colored by vendor family; the 7 AmnesiaBench Scott-25 cohort
models are highlighted with large rings and bold labels.

Dates and context windows were verified via web search on 2026-04-16.
Sources linked in the accompanying README.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path("/tmp/scott25")


# ── Vendor color palette (not the per-model palette from make_plot) ────────
FAMILY = {
    "OpenAI":    "#10a37f",
    "Anthropic": "#d97757",
    "Google":    "#4285f4",
    "Meta":      "#7e57c2",
    "DeepSeek":  "#d32f2f",
    "Alibaba":   "#8d6e63",
    "Zhipu":     "#c2185b",
}


# (label, release_date, context_window_tokens, family, is_cohort, label_offset)
MODELS: list[tuple[str, date, int, str, bool, tuple[int, int] | None]] = [
    ("GPT-3.5",             date(2022, 11, 30),     4_096,  "OpenAI",    False, (5, 8)),
    ("GPT-4",               date(2023,  3, 14),     8_192,  "OpenAI",    False, (5, 8)),
    ("Claude 2",            date(2023,  7, 11),   100_000,  "Anthropic", False, (5, 8)),
    ("GPT-4 Turbo",         date(2023, 11,  6),   128_000,  "OpenAI",    False, (-25, -18)),
    ("Claude 2.1",          date(2023, 11, 21),   200_000,  "Anthropic", False, (5, 8)),
    ("Gemini 1.0",          date(2023, 12,  6),    32_000,  "Google",    False, (5, 8)),
    ("Gemini 1.5 Pro",      date(2024,  2, 15), 1_000_000,  "Google",    False, (5, 8)),
    ("Claude 3 Opus",       date(2024,  3,  4),   200_000,  "Anthropic", False, (5, 10)),
    ("GPT-4o",              date(2024,  5, 13),   128_000,  "OpenAI",    False, (-35, -18)),
    ("Claude 3.5 Sonnet",   date(2024,  6, 20),   200_000,  "Anthropic", False, (5, 10)),
    ("Llama 3.1 405B",      date(2024,  7, 23),   128_000,  "Meta",      False, (-75, 10)),
    ("Qwen 2.5 72B",        date(2024,  9, 19),   128_000,  "Alibaba",   False, (5, -16)),
    ("Gemini 2.0 Flash",    date(2024, 12, 11), 1_000_000,  "Google",    False, (-85, 10)),
    ("DeepSeek V3",         date(2024, 12, 26),   128_000,  "DeepSeek",  False, (-60, 14)),
    ("DeepSeek R1",         date(2025,  1, 20),   128_000,  "DeepSeek",  False, (5, -18)),
    ("Claude 3.7 Sonnet",   date(2025,  2, 24),   200_000,  "Anthropic", False, (-5, 14)),
    ("Gemini 2.5 Pro",      date(2025,  3, 25), 1_000_000,  "Google",    False, (5, 10)),
    ("Llama 4 Scout",       date(2025,  4,  5),10_000_000,  "Meta",      False, (5, 8)),
    ("Claude Opus 4",       date(2025,  5, 22),   200_000,  "Anthropic", False, (5, -18)),
    # ── Cohort members (highlighted) ────────────────────────────────────
    ("Qwen3-Coder-480B",    date(2025,  7, 22),   256_000,  "Alibaba",   True,  (-115, -22)),
    ("GPT-5",               date(2025,  8,  7),   400_000,  "OpenAI",    False, (5, 8)),
    ("Claude Sonnet 4.5",   date(2025,  9, 29), 1_000_000,  "Anthropic", True,  (-140, -20)),
    ("DeepSeek V3.2",       date(2025,  9, 29),   128_000,  "DeepSeek",  True,  (8, -6)),
    ("Claude Haiku 4.5",    date(2025, 10, 15),   200_000,  "Anthropic", True,  (-105, 12)),
    ("Claude Opus 4.6",     date(2026,  2,  5), 1_000_000,  "Anthropic", True,  (-120, 12)),
    ("GLM-5",               date(2026,  2, 11),   200_000,  "Zhipu",     True,  (8, -6)),
    ("Claude Sonnet 4.6",   date(2026,  2, 17), 1_000_000,  "Anthropic", False, (10, -10)),
    ("GPT-5.4",             date(2026,  3,  5), 1_000_000,  "OpenAI",    True,  (-60, 14)),
]


def main():
    fig, ax = plt.subplots(figsize=(15, 8.5))

    # Uniform dots — no per-model labels, no cohort highlighting.
    for label, d, ctx, fam, is_cohort, offset in MODELS:
        color = FAMILY[fam]
        ax.scatter([d], [ctx], s=70, color=color, alpha=0.7,
                   edgecolors="white", linewidths=0.4, zorder=4)

    ax.set_yscale("log")
    ax.set_ylim(2_000, 20_000_000)
    ax.set_xlim(date(2022, 10, 1), date(2026, 5, 1))

    # Y-axis: only clean base-10 ticks — 1k, 10k, 100k, 1M, 10M.
    y_ticks = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    def _fmt(v, _pos):
        if v >= 1_000_000:
            return f"{v/1_000_000:g}M"
        if v >= 1_000:
            return f"{v/1_000:g}k"
        return str(int(v))
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_minor_locator(plt.NullLocator())  # no minor ticks

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    ax.set_xlabel("release date")
    ax.set_ylabel("max context window  (tokens)")
    ax.set_title("Frontier LLM context windows over time")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")

    # Custom legend with uniform marker size so every vendor gets the same
    # swatch — otherwise the first-appearing dot determines the size, which
    # made Zhipu look "bigger" just because GLM-5 is a cohort model.
    handles = [Line2D([0], [0], marker="o", linestyle="",
                      markersize=8, markerfacecolor=FAMILY[f],
                      markeredgecolor="white", markeredgewidth=0.5,
                      label=f) for f in FAMILY]
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              framealpha=0.92, title="vendor", title_fontsize=10)

    out = ROOT / "context_window_timeline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    cohort = [(m[0], m[1], m[2]) for m in MODELS if m[4]]
    print(f"\nCohort releases ({len(cohort)}):")
    for l, d, c in sorted(cohort, key=lambda r: r[1]):
        print(f"  {d.isoformat()}  {l:<24} {c:>10,} tokens")


if __name__ == "__main__":
    main()
