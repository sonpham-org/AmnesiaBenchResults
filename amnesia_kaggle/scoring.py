"""Scoring for AmnesiaBench v1 — 5 metrics per Scott's plan.md §8.

Pure function. Takes per-problem results and frozen baselines, returns a
dict of the five component scores. No single "final_score" — per plan.md,
the leaderboard renders these as an ARC-Prize-style chart (x = cost/token,
y = composite_context_efficiency_score).

Formula reference (plan.md §8):

  baseline_n_reliable[pid] = n_reliable from the reference model (frozen)

  ctx_eff_per_problem = baseline_n_reliable / model_n_reliable
                      = 0 if model_n_reliable = inf
  composite_context_efficiency_score = mean(ctx_eff_per_problem)

  window_pred_ratio = n_reliable / n_predicted
  window_pred_per_problem =
      ratio             if ratio <= 1.0              (under-predicted, good)
      (1 / ratio) ** 2  if ratio > 1.0               (overconfident, penalized)
      0                 if n_reliable = inf or n_predicted = inf
    (only include problems where the model's context window >= baseline)
  composite_window_prediction_score = mean(window_pred_per_problem)

  suc_pred_per_problem:
      1.0 if attempt = True and actually_solved        (TP)
      0.0 if attempt = True and not actually_solved    (FP, very bad)
      0.8 if attempt = False and actually_solved       (FN, less bad)
      1.0 if attempt = False and not actually_solved   (TN)
  composite_success_prediction_score = mean(suc_pred_per_problem)

  accuracy_score = (# problems solved correctly) / (# eligible problems)
    solved correctly iff n_while_unbounded < inf OR n_reliable < inf
    eligible iff model_ctx_window >= baseline_n_reliable

  compaction_ratio = mean(n_reliable / n_while_unbounded)
    Measures compaction overhead relative to natural usage. < 1.0 means
    compaction let the model solve in less context than it naturally used;
    > 1.0 is the typical case (compaction has overhead). Problems where
    either unbounded or compact failed are excluded from the mean.

  average_cost_per_token_nanodollars =
    sum(cost_nanodollars) / sum(input_tokens + output_tokens)
"""

from __future__ import annotations

import math
from typing import Iterable


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_scores(
    per_problem: list[dict],
    baselines: dict[str, int],
    model_ctx_window: int,
) -> dict:
    """Compute the 5 AmnesiaBench v1 metrics.

    Args:
        per_problem: One dict per problem. Expected keys:
            - "problem_id": str
            - "n_while_unbounded": int | float('inf')
            - "n_reliable": int | float('inf')   (min of no-compact + compact sweeps)
            - "n_predicted": int | float('inf')
            - "attempt": bool
            - "input_tokens": int
            - "output_tokens": int
            - "cost_nanodollars": int
        baselines: {problem_id: baseline_n_reliable} from the reference model.
            Problems missing from baselines are skipped entirely.
        model_ctx_window: The model's context window size, for the
            "eligible" check in accuracy + window_prediction.

    Returns:
        {
            "composite_context_efficiency_score": float,
            "composite_window_prediction_score":  float,
            "composite_success_prediction_score": float,
            "accuracy_score":                     float,
            "average_cost_per_token_nanodollars": float,
            # extras for debugging/assertions:
            "n_problems_scored": int,
            "n_problems_eligible": int,
            "n_problems_solved": int,
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_cost_nanodollars": int,
        }
    """
    ctx_eff_scores: list[float] = []
    win_pred_scores: list[float] = []
    suc_pred_scores: list[float] = []
    compaction_ratios: list[float] = []
    n_eligible = 0
    n_solved_correctly = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_nanodollars = 0
    n_scored = 0

    for p in per_problem:
        pid = p.get("problem_id")
        base = baselines.get(pid)
        if base is None:
            # Problem not in frozen baselines — skip it.
            continue
        n_scored += 1

        n_unbounded = _as_float(p.get("n_while_unbounded"))
        n_reliable = _as_float(p.get("n_reliable"))
        n_predicted = _as_float(p.get("n_predicted"))
        attempt = bool(p.get("attempt", True))

        # Token/cost accumulation (all problems count toward the cost metric)
        total_input_tokens += int(p.get("input_tokens", 0) or 0)
        total_output_tokens += int(p.get("output_tokens", 0) or 0)
        total_cost_nanodollars += int(p.get("cost_nanodollars", 0) or 0)

        eligible = model_ctx_window >= base
        if eligible:
            n_eligible += 1

        # Accuracy: solved at any budget (unbounded OR n_reliable)
        solved_correctly = (n_unbounded < math.inf) or (n_reliable < math.inf)
        if eligible and solved_correctly:
            n_solved_correctly += 1

        # Context efficiency: always computed (even when attempt=False)
        if n_reliable < math.inf and base > 0:
            ctx_eff_scores.append(base / n_reliable)
        else:
            ctx_eff_scores.append(0.0)

        # Compaction ratio: n_reliable / n_while_unbounded. Measures overhead
        # of compaction relative to the model's natural token usage.
        #   < 1.0 → compaction lets the model solve in less context than it
        #           naturally used (rare; compaction focused the model)
        #   = 1.0 → compaction has no overhead
        #   > 1.0 → compaction overhead (expected case)
        #   inf   → model failed to solve under any compact budget
        if n_unbounded > 0 and n_unbounded < math.inf and n_reliable < math.inf:
            compaction_ratios.append(n_reliable / n_unbounded)
        else:
            # Skip when we can't compute a meaningful ratio (unbounded failed
            # or compact failed); don't contribute to the mean.
            pass

        # Window prediction: only for problems where model can reach baseline
        if eligible:
            if n_reliable < math.inf and n_predicted < math.inf and n_predicted > 0:
                ratio = n_reliable / n_predicted
                if ratio <= 1.0:
                    win_pred_scores.append(ratio)            # under-predicted = good
                else:
                    win_pred_scores.append((1.0 / ratio) ** 2)  # overconfident = penalized softly
            else:
                # Either unsolved or no prediction → zero contribution
                win_pred_scores.append(0.0)

        # Success prediction (TP / FP / FN / TN)
        actually_solved = n_reliable < math.inf
        if attempt and actually_solved:
            suc_pred_scores.append(1.0)  # TP
        elif attempt and not actually_solved:
            suc_pred_scores.append(0.0)  # FP — very bad
        elif not attempt and actually_solved:
            suc_pred_scores.append(0.8)  # FN — less bad
        else:
            suc_pred_scores.append(1.0)  # TN

    total_tokens = total_input_tokens + total_output_tokens
    avg_cost = (total_cost_nanodollars / total_tokens) if total_tokens > 0 else 0.0

    return {
        "composite_context_efficiency_score": _mean(ctx_eff_scores),
        "composite_window_prediction_score":  _mean(win_pred_scores),
        "composite_success_prediction_score": _mean(suc_pred_scores),
        "compaction_ratio":                   (_mean(compaction_ratios) if compaction_ratios else math.inf),
        "accuracy_score":                     (n_solved_correctly / n_eligible) if n_eligible > 0 else 0.0,
        "average_cost_per_token_nanodollars": avg_cost,
        "n_problems_scored":        n_scored,
        "n_problems_eligible":      n_eligible,
        "n_problems_solved":        n_solved_correctly,
        "total_input_tokens":       total_input_tokens,
        "total_output_tokens":      total_output_tokens,
        "total_cost_nanodollars":   total_cost_nanodollars,
    }


def _as_float(value) -> float:
    """Coerce int / None / 'inf' strings to float (with math.inf for missing)."""
    if value is None:
        return math.inf
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return math.inf
        return float(value)
    if isinstance(value, str):
        if value.lower() in ("inf", "infinity", ""):
            return math.inf
        try:
            return float(value)
        except ValueError:
            return math.inf
    return math.inf
