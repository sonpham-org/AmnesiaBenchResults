"""Parsers for model outputs in AmnesiaBench.

Extracts three things from free-text model output:
  - {final_answer: "..."} (Scott's plan §1 — detected in ALL assistant turns,
    including compaction turns, per plan.md General Notes)
  - <compact>...</compact> (plan §7 — the compaction summary tag)
  - {attempt: "...", N: "..."} (plan §6 — prediction response)

Port of amnesia_bench/evaluate.py:_extract_final_answer_from_content +
predict.py:_parse_prediction_response. We avoid importing from amnesia_bench/
so this package is self-contained for Kaggle upload.
"""

from __future__ import annotations

import math
import re
from typing import Optional


# ── Final answer ────────────────────────────────────────────────────────────
# Plan.md §1 says the format is exactly:  {final_answer: "YOUR_NUMBER_HERE"}
# We accept:
#   - with or without optional whitespace around the colon and inside braces
#   - quoted value (the canonical form) — any non-quote chars
#   - bare "final_answer: ..." without outer braces (defensive fallback)
# We do NOT accept \boxed{} — that's the old amnesia_bench v2 format.

_FINAL_ANSWER_PATTERNS = (
    re.compile(r'\{\s*final_answer\s*:\s*"([^"]+)"\s*\}'),
    re.compile(r'final_answer\s*:\s*"([^"]+)"'),
)


def extract_final_answer(text: Optional[str]) -> Optional[str]:
    """Return the raw string inside `{final_answer: "..."}`, or None.

    Matches even when the answer appears in a compaction turn or on a
    truncated output (plan.md General Notes). Strips surrounding whitespace
    from the captured value.
    """
    if not text:
        return None
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def check_answer(extracted: Optional[str], ground_truth) -> bool:
    """Compare extracted answer to ground truth — STRICT format, FLEXIBLE number.

    The model must output the answer as a pure number value inside
    `{final_answer: "..."}`. We reject anything that isn't a valid number
    representation (no "x = 3", no "\\boxed{3}", no "$3$", no prose).

    Flexible on number formatting:
      - Commas as thousand separators: "1,234" matches 1234
      - Int/float equivalence: "3" matches 3, "3.0" matches 3
      - Leading/trailing whitespace stripped
    """
    if extracted is None:
        return False
    a = str(extracted).strip()
    b = str(ground_truth).strip()

    # Remove commas (thousand separators) from both sides
    a_clean = a.replace(",", "")
    b_clean = b.replace(",", "")

    # Exact string match after comma stripping
    if a_clean == b_clean:
        return True

    # Numeric equivalence — only if BOTH sides parse as numbers.
    # This rejects anything with letters, operators, or prose.
    try:
        return float(a_clean) == float(b_clean)
    except (ValueError, TypeError):
        return False


# ── Compaction tag ──────────────────────────────────────────────────────────

_COMPACT_PATTERN = re.compile(r"<compact>(.*?)</compact>", re.DOTALL)


def extract_compact_tag(text: Optional[str]) -> Optional[str]:
    """Return the content inside `<compact>...</compact>`, or None if absent."""
    if not text:
        return None
    match = _COMPACT_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


# ── Prediction parser ───────────────────────────────────────────────────────
# Plan.md §4 says the format is:
#   {attempt: "True", N: "1024"}   (or "False" + N="0" / "inf" / any int)
# Per plan.md §6: "If the answers can't be extracted because of incorrect
# formatting in the model's response, set attempt = True and N = inf."

_ATTEMPT_PATTERN = re.compile(r'attempt\s*:\s*"?(True|False)"?', re.IGNORECASE)
_N_PATTERN = re.compile(r'\bN\s*:\s*"?(\d+|inf|infinity)"?', re.IGNORECASE)


def parse_prediction(raw: Optional[str]) -> tuple[bool, float]:
    """Parse `{attempt: "...", N: "..."}` into (attempt: bool, n_predicted: float).

    Returns:
        (attempt, n_predicted)
        - attempt: True if the model intends to try, False if it opts out.
        - n_predicted: positive int as float, or math.inf if:
            * attempt is False (model opted out),
            * N value is "0", "inf", or "infinity",
            * either field can't be parsed.

    Fallback on any parse failure: (True, inf), per plan.md §6 rule.
    """
    if not raw:
        return True, math.inf

    attempt_match = _ATTEMPT_PATTERN.search(raw)
    n_match = _N_PATTERN.search(raw)

    if not attempt_match or not n_match:
        return True, math.inf

    attempt = attempt_match.group(1).lower() == "true"
    n_raw = n_match.group(1).lower()

    if not attempt:
        return False, math.inf
    if n_raw in ("inf", "infinity", "0"):
        return attempt, math.inf
    try:
        n_val = int(n_raw)
        if n_val <= 0:
            return attempt, math.inf
        return attempt, float(n_val)
    except ValueError:
        return True, math.inf
