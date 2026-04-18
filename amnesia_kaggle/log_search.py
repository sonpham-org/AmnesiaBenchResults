"""Log-scale nested binary search for n_reliable.

Plan.md §7: "Due to the wide range of possible context window sizes, the
binary searches should use log-scale midpoints, not linear midpoints."

  mid = exp((ln(lo) + ln(hi)) / 2)   (geometric mean)

Two phases per sweep:

  Outer: checks_per_N = 1, stop when (hi - lo) < 0.05 * mid
  Inner: checks_per_N = 3, pass iff >= 2/3 succeed, stop when (hi - lo) <= 1
    - Inner range centered on outer transition zone, expanded ×3 to avoid
      missing the transition point.

Ports amnesia_bench/evaluate.py:_outer_binary_search + _inner_binary_search,
swapping linear midpoints for geometric ones and decoupling the trial
function (injected) so this module is LLM-agnostic and unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


# ── Search parameters (match plan.md §7) ────────────────────────────────────

OUTER_CHECKS_PER_N = 1
INNER_CHECKS_PER_N = 3
INNER_PASS_THRESHOLD = 2       # 2 / 3 = 66.7%
OUTER_STOP_RATIO = 0.05        # stop when step < 5% of current N
INNER_STOP_ABS = 1             # stop when hi - lo <= 1 token
INNER_EXPAND_FACTOR = 3        # inner range = 3× outer transition zone


# ── Types ───────────────────────────────────────────────────────────────────

@dataclass
class TrialOutcome:
    """Result of a single trial at a given N.

    success: True iff the model produced the correct final answer under budget.
    aborted: True if the trial was aborted (cost cap, context exceeded, etc.).
             An aborted trial is treated as a failure for search purposes.
    cost_nanodollars: Cumulative cost for this trial (for caller to aggregate).
    input_tokens: Input tokens used in this trial.
    output_tokens: Output tokens generated in this trial.
    finish_reason: Free-text reason ('solved', 'budget_exceeded', 'cost_cap',
                   'compaction_insufficient', 'no_answer_no_compact', etc.).
    traces: List of chat message dicts captured during the trial.
    """
    success: bool
    aborted: bool = False
    cost_nanodollars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    traces: list = field(default_factory=list)


@dataclass
class SearchLog:
    """Record of every N tested during a search."""
    entries: list[dict] = field(default_factory=list)

    def add(self, N: int, passed: bool, trials: Iterable[TrialOutcome]) -> None:
        self.entries.append({
            "N": N,
            "passed": passed,
            "trials": [
                {
                    "success": t.success,
                    "aborted": t.aborted,
                    "cost_nanodollars": t.cost_nanodollars,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "finish_reason": t.finish_reason,
                    "traces": t.traces,
                }
                for t in trials
            ],
        })
        self.entries.sort(key=lambda e: -e["N"])


# ── Geometric midpoint ──────────────────────────────────────────────────────

def log_mid(lo: int, hi: int) -> int:
    """Return the geometric-mean midpoint of [lo, hi], clipped to (lo, hi).

    For the binary search to make progress we need lo < mid < hi. If lo and
    hi are adjacent integers, this returns hi (caller will detect termination).
    """
    if lo <= 0:
        lo = 1
    if hi <= lo:
        return hi
    mid = int(round(math.exp((math.log(lo) + math.log(hi)) / 2.0)))
    if mid <= lo:
        mid = lo + 1
    if mid >= hi:
        mid = hi - 1 if hi - 1 > lo else hi
    return mid


# ── Outer binary search ─────────────────────────────────────────────────────

TrialFn = Callable[[int, int], list[TrialOutcome]]
"""Signature: (N, n_trials) -> list of n_trials TrialOutcomes at budget N."""


def outer_search(
    trial_fn: TrialFn,
    lo: int,
    hi: int,
    log: Optional[SearchLog] = None,
) -> tuple[int, int, SearchLog]:
    """Coarse search: 1 trial per N, stop when step < 5% of mid.

    Returns (transition_lo, transition_hi, log) — the bracket in which the
    fail→pass transition occurs. Use these to seed the inner search.

    If the lowest tested N already passes → transition_lo = lo, transition_hi = that N.
    If the highest tested N still fails → transition_lo = that N, transition_hi = hi.
    """
    if log is None:
        log = SearchLog()

    transition_lo = lo
    transition_hi = hi

    while True:
        mid = log_mid(lo, hi)
        if mid <= lo or mid >= hi:
            break
        step_size = hi - lo
        if step_size < OUTER_STOP_RATIO * mid:
            break

        trials = trial_fn(mid, OUTER_CHECKS_PER_N)
        passed = sum(1 for t in trials if t.success) >= 1
        log.add(mid, passed, trials)

        if passed:
            transition_hi = mid
            hi = mid
        else:
            transition_lo = mid
            lo = mid

    return transition_lo, transition_hi, log


# ── Inner binary search ─────────────────────────────────────────────────────

def inner_search(
    trial_fn: TrialFn,
    transition_lo: int,
    transition_hi: int,
    n_max: int,
    log: Optional[SearchLog] = None,
) -> tuple[float, SearchLog]:
    """Fine search: 3 trials per N, pass iff >= 2/3 succeed, stop at hi-lo<=1.

    Seeds the range by centering on the outer transition zone and expanding
    by ×3 (plan.md §7 example: "If N=610 fails and N=630 passes, center at
    620 and expand to [590, 650]").

    Returns (n_reliable, log) where n_reliable is the smallest passing N.
    Returns (math.inf, log) if no N in the expanded range achieves the
    pass threshold.
    """
    if log is None:
        log = SearchLog()

    # Center + expand ×3
    mid = (transition_lo + transition_hi) // 2
    half_width = max(1, (transition_hi - transition_lo) * INNER_EXPAND_FACTOR // 2)
    lo = max(1, mid - half_width)
    hi = min(n_max, mid + half_width)
    if hi < transition_hi:
        hi = transition_hi

    n_reliable: float = math.inf

    # Before the sweep, test whether the high end actually passes. If it
    # doesn't (because the outer transition was an artifact of noise), we
    # short-circuit to inf rather than spinning the binary search.
    high_trials = trial_fn(hi, INNER_CHECKS_PER_N)
    high_passed = sum(1 for t in high_trials if t.success) >= INNER_PASS_THRESHOLD
    log.add(hi, high_passed, high_trials)
    if not high_passed:
        return math.inf, log
    n_reliable = float(hi)

    # Now narrow from below
    while hi - lo > INNER_STOP_ABS:
        mid_n = log_mid(lo, hi)
        if mid_n <= lo or mid_n >= hi:
            break
        trials = trial_fn(mid_n, INNER_CHECKS_PER_N)
        passed = sum(1 for t in trials if t.success) >= INNER_PASS_THRESHOLD
        log.add(mid_n, passed, trials)
        if passed:
            n_reliable = float(mid_n)
            hi = mid_n
        else:
            lo = mid_n

    return n_reliable, log


# ── Top-level driver ────────────────────────────────────────────────────────

def find_n_reliable(
    trial_fn: TrialFn,
    n_min: int,
    n_max: int,
) -> tuple[float, SearchLog]:
    """Run the full nested search and return n_reliable (min N with >=66.7% pass).

    Returns (n_reliable, full_log). n_reliable is math.inf if unsolved.
    """
    log = SearchLog()
    transition_lo, transition_hi, log = outer_search(trial_fn, n_min, n_max, log)
    n_reliable, log = inner_search(trial_fn, transition_lo, transition_hi, n_max, log)
    return n_reliable, log
