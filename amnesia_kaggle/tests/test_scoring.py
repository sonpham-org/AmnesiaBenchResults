"""Unit tests for amnesia_kaggle/scoring.py."""

from __future__ import annotations

import math

import pytest

from amnesia_kaggle.scoring import compute_scores


BASELINES = {
    "prob_a": 1000,
    "prob_b": 2000,
    "prob_c": 4000,
}


def _per_problem(
    pid: str,
    n_unbounded: float,
    n_reliable: float,
    n_predicted: float,
    attempt: bool = True,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cost_nanodollars: int = 15000,
) -> dict:
    return {
        "problem_id": pid,
        "n_while_unbounded": n_unbounded,
        "n_reliable": n_reliable,
        "n_predicted": n_predicted,
        "attempt": attempt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_nanodollars": cost_nanodollars,
    }


class TestComputeScores:
    # ── Context efficiency ─────────────────────────────────────────────

    def test_ctx_eff_perfect_match_baseline(self):
        """Model matches baseline exactly on all problems → ctx_eff = 1.0."""
        results = [
            _per_problem("prob_a", 3000, 1000, 1000),
            _per_problem("prob_b", 3000, 2000, 2000),
            _per_problem("prob_c", 5000, 4000, 4000),
        ]
        scores = compute_scores(results, BASELINES, model_ctx_window=262144)
        assert scores["composite_context_efficiency_score"] == pytest.approx(1.0, abs=1e-9)

    def test_ctx_eff_half_as_efficient(self):
        """Model needs 2x baseline → ctx_eff = 0.5."""
        results = [
            _per_problem("prob_a", 3000, 2000, 2000),
            _per_problem("prob_b", 3000, 4000, 4000),
            _per_problem("prob_c", 5000, 8000, 8000),
        ]
        scores = compute_scores(results, BASELINES, model_ctx_window=262144)
        assert scores["composite_context_efficiency_score"] == pytest.approx(0.5, abs=1e-9)

    def test_ctx_eff_unsolved_contributes_zero(self):
        results = [
            _per_problem("prob_a", 3000, 1000, 1000),           # 1.0
            _per_problem("prob_b", math.inf, math.inf, math.inf),  # 0.0
        ]
        scores = compute_scores(results, {"prob_a": 1000, "prob_b": 2000}, model_ctx_window=262144)
        assert scores["composite_context_efficiency_score"] == pytest.approx(0.5, abs=1e-9)

    def test_ctx_eff_better_than_baseline(self):
        """Model beats baseline → ctx_eff > 1.0."""
        results = [_per_problem("prob_a", 3000, 500, 500)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_context_efficiency_score"] == pytest.approx(2.0, abs=1e-9)

    # ── Window prediction ──────────────────────────────────────────────

    def test_window_pred_perfect(self):
        results = [_per_problem("prob_a", 3000, 1000, 1000)]  # ratio = 1.0
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_window_prediction_score"] == pytest.approx(1.0)

    def test_window_pred_under_predicted(self):
        """n_reliable / n_predicted = 0.5 → score = 0.5."""
        results = [_per_problem("prob_a", 3000, 1000, 2000)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_window_prediction_score"] == pytest.approx(0.5)

    def test_window_pred_overconfident_soft_penalty(self):
        """Plan.md §8: overconfident gets (1/ratio)^2, not 0."""
        # ratio = 2.0 → (1/2)^2 = 0.25
        results = [_per_problem("prob_a", 3000, 2000, 1000)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_window_prediction_score"] == pytest.approx(0.25)

    def test_window_pred_heavily_overconfident(self):
        # ratio = 10 → (1/10)^2 = 0.01
        results = [_per_problem("prob_a", 3000, 10000, 1000)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_window_prediction_score"] == pytest.approx(0.01)

    def test_window_pred_skips_if_ctx_too_small(self):
        """If model_ctx_window < baseline, the problem isn't counted."""
        results = [
            _per_problem("prob_a", 3000, 1000, 1000),   # eligible
            _per_problem("prob_b", 3000, 2000, 2000),   # NOT eligible (ctx<baseline)
        ]
        scores = compute_scores(results, BASELINES, model_ctx_window=1500)
        # Only prob_a was eligible, ratio = 1.0 → score = 1.0
        assert scores["composite_window_prediction_score"] == pytest.approx(1.0)

    def test_window_pred_no_prediction_is_zero(self):
        results = [_per_problem("prob_a", 3000, 1000, math.inf)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_window_prediction_score"] == pytest.approx(0.0)

    # ── Success prediction ─────────────────────────────────────────────

    def test_suc_pred_tp(self):
        results = [_per_problem("prob_a", 3000, 1000, 1000, attempt=True)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_success_prediction_score"] == 1.0

    def test_suc_pred_fp(self):
        """Predicted True, but failed → 0 (very bad)."""
        results = [_per_problem("prob_a", math.inf, math.inf, 1000, attempt=True)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_success_prediction_score"] == 0.0

    def test_suc_pred_fn(self):
        """Predicted False, but succeeded → 0.8 (less bad)."""
        results = [_per_problem("prob_a", 3000, 1000, math.inf, attempt=False)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_success_prediction_score"] == 0.8

    def test_suc_pred_tn(self):
        results = [_per_problem("prob_a", math.inf, math.inf, math.inf, attempt=False)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["composite_success_prediction_score"] == 1.0

    # ── Accuracy ───────────────────────────────────────────────────────

    def test_accuracy_all_solved(self):
        results = [
            _per_problem("prob_a", 3000, 1000, 1000),
            _per_problem("prob_b", 3000, 2000, 2000),
            _per_problem("prob_c", 5000, 4000, 4000),
        ]
        scores = compute_scores(results, BASELINES, model_ctx_window=262144)
        assert scores["accuracy_score"] == 1.0

    def test_accuracy_partial(self):
        results = [
            _per_problem("prob_a", 3000, 1000, 1000),                # solved
            _per_problem("prob_b", math.inf, math.inf, math.inf),    # unsolved
        ]
        scores = compute_scores(results, {"prob_a": 1000, "prob_b": 2000}, model_ctx_window=262144)
        assert scores["accuracy_score"] == pytest.approx(0.5)

    def test_accuracy_unbounded_only_counts(self):
        """n_unbounded < inf but n_reliable = inf → still counts as solved (plan.md §8)."""
        results = [_per_problem("prob_a", 3000, math.inf, math.inf)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["accuracy_score"] == 1.0

    def test_accuracy_reliable_only_counts(self):
        """n_unbounded = inf but n_reliable < inf → also counts."""
        results = [_per_problem("prob_a", math.inf, 1500, 2000)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["accuracy_score"] == 1.0

    def test_accuracy_ineligible_problem_not_counted(self):
        """Problem with baseline > model ctx window is excluded from denom."""
        results = [_per_problem("prob_c", 3000, 4000, 4000)]  # baseline = 4000
        scores = compute_scores(results, BASELINES, model_ctx_window=3000)
        assert scores["accuracy_score"] == 0.0
        assert scores["n_problems_eligible"] == 0

    # ── Cost ───────────────────────────────────────────────────────────

    def test_avg_cost_per_token(self):
        results = [
            _per_problem("prob_a", 3000, 1000, 1000,
                         input_tokens=1000, output_tokens=500, cost_nanodollars=15000),
            _per_problem("prob_b", 3000, 2000, 2000,
                         input_tokens=2000, output_tokens=1000, cost_nanodollars=30000),
        ]
        scores = compute_scores(results, BASELINES, model_ctx_window=262144)
        # (15000 + 30000) / (1500 + 3000) = 45000 / 4500 = 10
        assert scores["average_cost_per_token_nanodollars"] == pytest.approx(10.0)

    def test_zero_tokens_safe(self):
        results = [_per_problem("prob_a", math.inf, math.inf, math.inf,
                                 input_tokens=0, output_tokens=0, cost_nanodollars=0)]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["average_cost_per_token_nanodollars"] == 0.0

    # ── Problems not in baselines are skipped ──────────────────────────

    def test_unknown_problem_skipped(self):
        results = [
            _per_problem("prob_a", 3000, 1000, 1000),
            _per_problem("unknown_prob", 3000, 1000, 1000),
        ]
        scores = compute_scores(results, {"prob_a": 1000}, model_ctx_window=262144)
        assert scores["n_problems_scored"] == 1
        assert scores["composite_context_efficiency_score"] == 1.0

    # ── Bookkeeping counters ───────────────────────────────────────────

    def test_counter_fields_populated(self):
        results = [
            _per_problem("prob_a", 3000, 1000, 1000,
                         input_tokens=500, output_tokens=100, cost_nanodollars=6000),
            _per_problem("prob_b", math.inf, math.inf, math.inf,
                         input_tokens=200, output_tokens=50, cost_nanodollars=2500),
        ]
        scores = compute_scores(results, {"prob_a": 1000, "prob_b": 2000}, model_ctx_window=262144)
        assert scores["n_problems_scored"] == 2
        assert scores["n_problems_eligible"] == 2
        assert scores["n_problems_solved"] == 1
        assert scores["total_input_tokens"] == 700
        assert scores["total_output_tokens"] == 150
        assert scores["total_cost_nanodollars"] == 8500
