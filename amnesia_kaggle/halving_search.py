"""Halving search for n_reliable — N/2^k descent + binary search refinement.

Phase 3 (Sweep): Test at N, N/2, N/4, N/8... Stop at first failure.
Phase 4 (Binary Search): Refine between last pass and first fail.

Optimization: unbounded trial results are reused as free data points.
Since the prompt is identical and the model doesn't know its budget
(max_tokens is API-level), an unbounded trial that used X tokens and
succeeded is equivalent to a PASS at any N >= X. This avoids redundant
API calls during the sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .log_search import TrialOutcome, SearchLog


PASS_THRESHOLD = 2  # 2/3 = 66.7%
TRIALS_PER_N = 3
MIN_N = 64  # Don't bother testing below this


TrialFn = Callable[[int, int], list[TrialOutcome]]


def _check_unbounded_at_N(
    unbounded_trials: list[dict],
    N: int,
    compact_mode: bool = False,
) -> tuple[int, int]:
    """Check how many unbounded trials can be reused at budget N.

    No-compact mode: trial is reusable if tokens_used <= N
      (model solves before output limit)
    Compact mode: trial is reusable if tokens_used <= N/2
      (model solves before compaction would have fired at 50%)

    Returns (n_pass, n_reusable) from the unbounded data.
    """
    n_pass = 0
    n_reusable = 0
    threshold = N * 0.5 if compact_mode else N
    for t in unbounded_trials:
        if not t.get("success"):
            continue
        tokens = t.get("tokens_used", 0)
        if tokens <= threshold:
            n_reusable += 1
            n_pass += 1
    return n_pass, n_reusable


def halving_search(
    trial_fn: TrialFn,
    context_window: int,
    n_while_unbounded: float = math.inf,
    unbounded_trials: Optional[list[dict]] = None,
    refine: bool = True,
    compact_mode: bool = False,
) -> tuple[float, SearchLog]:
    """Find n_reliable by halving the context window.

    Args:
        trial_fn: (N, n_trials) -> list[TrialOutcome]. Runs n_trials at budget N.
        context_window: Model's full context window size.
        n_while_unbounded: Tokens the model used naturally (unbounded).
        unbounded_trials: Trial logs from unbounded phase. Used as free
            data points to skip unnecessary API calls.
        refine: If True, binary-search between last_pass and first_fail.
        compact_mode: True if trial_fn runs compact trials (reuse threshold
            is N/2). False for no-compact trials (reuse threshold is N).

    Returns:
        (n_reliable, log). n_reliable = math.inf if no level passed.
    """
    log = SearchLog()
    if unbounded_trials is None:
        unbounded_trials = []

    # Generate halving levels: ctx, ctx/2, ctx/4, ...
    levels = []
    n = context_window
    while n >= MIN_N:
        levels.append(n)
        n //= 2

    last_pass: Optional[int] = None
    first_fail: Optional[int] = None

    for level in levels:
        passed, trials = _test_at_N(trial_fn, level, unbounded_trials, compact_mode)
        log.add(level, passed, trials)

        if passed:
            last_pass = level
        else:
            first_fail = level
            break

    if last_pass is None:
        return math.inf, log

    if first_fail is None:
        return float(last_pass), log

    if not refine or (last_pass - first_fail) <= 1:
        return float(last_pass), log

    # ── Binary search refinement between first_fail and last_pass ────────
    lo = first_fail
    hi = last_pass
    n_reliable = float(last_pass)

    while hi - lo > max(hi // 20, 1):  # stop at ~5% precision
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break

        passed, trials = _test_at_N(trial_fn, mid, unbounded_trials, compact_mode)
        log.add(mid, passed, trials)

        if passed:
            n_reliable = float(mid)
            hi = mid
        else:
            lo = mid

    return n_reliable, log


def _test_at_N(
    trial_fn: TrialFn,
    N: int,
    unbounded_trials: list[dict],
    compact_mode: bool = False,
) -> tuple[bool, list[TrialOutcome]]:
    """Test at budget N, reusing unbounded data where possible.

    Reuse threshold depends on mode:
      - No-compact: tokens <= N (model would solve before budget runs out)
      - Compact: tokens <= N/2 (model would solve before compaction fires)

    Strategy:
    1. Count reusable passes from unbounded
    2. If already >= PASS_THRESHOLD → PASS without API calls
    3. Otherwise, run additional trials to fill remaining slots
    """
    free_pass, free_reusable = _check_unbounded_at_N(unbounded_trials, N, compact_mode)
    threshold = N * 0.5 if compact_mode else N

    # Build TrialOutcome objects for reusable results only
    free_outcomes: list[TrialOutcome] = []
    for t in unbounded_trials:
        if not t.get("success"):
            continue
        tokens = t.get("tokens_used", 0)
        if tokens <= threshold:
            free_outcomes.append(TrialOutcome(
                success=True, finish_reason="reused_unbounded",
                input_tokens=0, output_tokens=tokens,
            ))

    # Already enough passes?
    if free_pass >= PASS_THRESHOLD:
        return True, free_outcomes

    # How many more trials could we run?
    trials_remaining = TRIALS_PER_N - len(free_outcomes)
    if trials_remaining <= 0:
        return free_pass >= PASS_THRESHOLD, free_outcomes

    # Can we even reach threshold with remaining trials?
    max_possible_pass = free_pass + trials_remaining
    if max_possible_pass < PASS_THRESHOLD:
        # Impossible — run the trials anyway for trace data but we know it's FAIL
        new_trials = trial_fn(N, trials_remaining)
        all_outcomes = free_outcomes + new_trials
        total_pass = free_pass + sum(1 for t in new_trials if t.success)
        return total_pass >= PASS_THRESHOLD, all_outcomes

    # Run additional trials
    new_trials = trial_fn(N, trials_remaining)
    all_outcomes = free_outcomes + new_trials
    total_pass = free_pass + sum(1 for t in new_trials if t.success)
    return total_pass >= PASS_THRESHOLD, all_outcomes
