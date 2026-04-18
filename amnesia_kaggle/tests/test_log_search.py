"""Unit tests for amnesia_kaggle/log_search.py.

Uses mock trial functions instead of real LLM calls.
"""

from __future__ import annotations

import math

import pytest

from amnesia_kaggle.log_search import (
    INNER_CHECKS_PER_N,
    OUTER_CHECKS_PER_N,
    TrialOutcome,
    find_n_reliable,
    inner_search,
    log_mid,
    outer_search,
)


# ── log_mid ─────────────────────────────────────────────────────────────────

class TestLogMid:
    def test_basic_geometric_mean(self):
        # sqrt(100 * 10000) = 1000
        assert log_mid(100, 10000) == 1000

    def test_small_range(self):
        # sqrt(1 * 100) = 10
        assert log_mid(1, 100) == 10

    def test_wide_range(self):
        # sqrt(1 * 1_000_000) = 1000
        assert log_mid(1, 1_000_000) == 1000

    def test_lo_greater_than_one(self):
        # sqrt(1024 * 65536) = 8192
        assert log_mid(1024, 65536) == 8192

    def test_adjacent_clips_to_hi(self):
        # [5, 6]: mid = 5.48 → 5, but 5 == lo → bumped to 6, then clipped to hi
        result = log_mid(5, 6)
        assert result == 6  # caller detects termination

    def test_returns_strictly_inside(self):
        """For any lo < hi - 1, the result should satisfy lo < mid < hi."""
        for lo, hi in [(1, 100), (10, 10000), (500, 8000), (1, 262144)]:
            mid = log_mid(lo, hi)
            assert lo < mid < hi, f"log_mid({lo}, {hi}) = {mid}"

    def test_lo_zero_coerced(self):
        # log(0) would crash; implementation coerces to 1
        result = log_mid(0, 100)
        assert 1 < result < 100

    def test_progress_on_wide_range(self):
        """Repeated halving of [lo, mid] should converge."""
        lo, hi = 1, 262144
        steps = 0
        while hi - lo > 1 and steps < 50:
            mid = log_mid(lo, hi)
            if mid <= lo or mid >= hi:
                break
            hi = mid
            steps += 1
        assert steps < 30  # log-scale ~= log2(262144) = 18 steps


# ── Synthetic trial functions ──────────────────────────────────────────────

def make_step_fn(threshold: int, deterministic: bool = True):
    """Return a trial_fn that passes iff N >= threshold.

    If deterministic=True: succeed iff N >= threshold (every trial).
    If deterministic=False: always fail at N=threshold-1, always succeed at
    N>=threshold, succeed 1/3 of the time at threshold-1 (models can be noisy).
    """
    def fn(N: int, n_trials: int) -> list[TrialOutcome]:
        results = []
        for i in range(n_trials):
            if deterministic:
                success = N >= threshold
            else:
                if N >= threshold:
                    success = True
                elif N >= threshold - 100:
                    success = (i == 0)  # fluky success on first trial
                else:
                    success = False
            results.append(TrialOutcome(
                success=success,
                cost_nanodollars=100,
                input_tokens=N,
                output_tokens=10,
                finish_reason="solved" if success else "wrong_answer",
            ))
        return results
    return fn


# ── outer_search ────────────────────────────────────────────────────────────

class TestOuterSearch:
    def test_deterministic_convergence(self):
        """Model passes iff N >= 1000. Outer should bracket [near_1000, near_1000]."""
        trial_fn = make_step_fn(threshold=1000)
        lo, hi, log = outer_search(trial_fn, lo=1, hi=8000)
        assert lo < 1000 < hi
        # Outer stop is at step < 5% of mid; near 1000, step ~= 50
        assert hi - lo < 0.10 * 1000

    def test_high_threshold(self):
        """Threshold well above the midpoint of the range."""
        trial_fn = make_step_fn(threshold=6000)
        lo, hi, log = outer_search(trial_fn, lo=1, hi=8000)
        assert lo < 6000 <= hi or lo <= 6000 < hi

    def test_always_passes(self):
        """If the model passes even at N=1, the transition_hi should be near lo."""
        trial_fn = make_step_fn(threshold=0)
        lo, hi, log = outer_search(trial_fn, lo=1, hi=10000)
        # High end is set early; search quickly bottoms out
        assert hi <= 100

    def test_never_passes(self):
        """If nothing passes, transition_lo ends near hi."""
        trial_fn = make_step_fn(threshold=1_000_000)
        lo, hi, log = outer_search(trial_fn, lo=1, hi=10000)
        assert hi == 10000  # never narrowed from the top


# ── inner_search ────────────────────────────────────────────────────────────

class TestInnerSearch:
    def test_deterministic_converges_to_threshold(self):
        """With 3 deterministic trials, n_reliable should equal the threshold."""
        trial_fn = make_step_fn(threshold=1000)
        # Simulate an outer result that bracketed around 1000
        n_reliable, log = inner_search(trial_fn, transition_lo=950, transition_hi=1050, n_max=10000)
        assert n_reliable <= 1001  # should find the exact threshold
        assert n_reliable >= 1000

    def test_reports_inf_if_high_end_fails(self):
        """If the upper bound of the inner range doesn't pass, return inf."""
        trial_fn = make_step_fn(threshold=5000)
        # Seed the inner search too low
        n_reliable, log = inner_search(trial_fn, transition_lo=100, transition_hi=200, n_max=500)
        assert n_reliable == math.inf

    def test_high_passes_immediately(self):
        """If the model passes at the seed's high end, short-circuit."""
        trial_fn = make_step_fn(threshold=10)
        n_reliable, log = inner_search(trial_fn, transition_lo=10, transition_hi=20, n_max=100)
        assert n_reliable <= 20


# ── find_n_reliable (full driver) ───────────────────────────────────────────

class TestFindNReliable:
    def test_convergence_wide_range(self):
        """A wide [1, 100k] range with threshold at 5000 should converge."""
        trial_fn = make_step_fn(threshold=5000)
        n_reliable, log = find_n_reliable(trial_fn, n_min=1, n_max=100_000)
        assert not math.isinf(n_reliable)
        # Tolerate slight slop from outer→inner handoff
        assert 4900 <= n_reliable <= 5100
        # Sanity: the search should not have run more than ~50 unique Ns
        assert len(log.entries) < 50

    def test_wide_range_low_threshold(self):
        trial_fn = make_step_fn(threshold=50)
        n_reliable, log = find_n_reliable(trial_fn, n_min=1, n_max=100_000)
        assert not math.isinf(n_reliable)
        assert 30 <= n_reliable <= 100

    def test_unsolvable(self):
        """A model that never passes → n_reliable = inf."""
        trial_fn = make_step_fn(threshold=1_000_000_000)
        n_reliable, log = find_n_reliable(trial_fn, n_min=1, n_max=10_000)
        assert n_reliable == math.inf
