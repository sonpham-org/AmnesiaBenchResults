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

    # Non-cohort: small filled dots, light labels above.
    # Cohort: big ring + thicker outline + bold label below.
    drawn_legend_families = set()
    for label, d, ctx, fam, is_cohort, offset in MODELS:
        color = FAMILY[fam]
        legend_label = fam if fam not in drawn_legend_families else None
        drawn_legend_families.add(fam)
        off = offset if offset is not None else (5, 8)

        if is_cohort:
            ax.scatter([d], [ctx], s=260, color=color, alpha=0.9,
                       edgecolors="black", linewidths=1.8, zorder=6,
                       label=legend_label)
            ax.annotate(label, (d, ctx),
                        xytext=off, textcoords="offset points",
                        fontsize=9.5, fontweight="bold", color="black", zorder=8)
        else:
            ax.scatter([d], [ctx], s=70, color=color, alpha=0.55,
                       edgecolors="white", linewidths=0.4, zorder=4,
                       label=legend_label)
            ax.annotate(label, (d, ctx),
                        xytext=off, textcoords="offset points",
                        fontsize=8, color="#333", zorder=7)

    # Horizontal reference at 131k — the context the AmnesiaBench harness
    # exercised, regardless of each model's maximum capability.
    ax.axhline(131_072, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(date(2022, 12, 1), 131_072 * 1.1,
            "AmnesiaBench test context = 131,072 tokens",
            fontsize=9, color="gray")

    ax.set_yscale("log")
    ax.set_ylim(2_000, 20_000_000)
    ax.set_xlim(date(2022, 10, 1), date(2026, 5, 1))

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    ax.set_xlabel("release date")
    ax.set_ylabel("max context window  (tokens, log scale)")
    ax.set_title("Frontier LLM context windows over time  "
                 "(filled rings with black border = AmnesiaBench Scott-25 cohort)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92,
              title="vendor", title_fontsize=10)

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
