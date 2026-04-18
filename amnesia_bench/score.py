# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Scoring job for AmnesiaBench v3. Loads all prediction + evaluation result files,
#   computes all component scores per the v3 spec, and prints per-model table.
#   Integration points: called by cli.py; reads from results_dir; uses models.py for
#   cost-per-token rates. No clients, no API calls — pure offline analysis.
# SRP/DRY check: Pass — all scoring math is here and nowhere else. Result loading is
#   inline (no shared loader needed since only score.py reads both file types together).

import json
import math
from pathlib import Path
from typing import Optional

from .utils import prediction_filename, evaluation_filename, sanitize_model_name

_PACKAGE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = _PACKAGE_DIR.parent / "results"


def compute_scores(results_dir: Optional[Path] = None) -> None:
    """
    Load all prediction + evaluation files, compute component scores, print table.

    Scoring formulas:
      composite_context_efficiency_score =
          mean(baseline_n_reliable / n_reliable) over solved problems
          where baseline_n_reliable = min n_reliable across all models for that problem

      composite_efficiency_prediction_score =
          mean(score_per_problem) where:
            score = n_reliable / n_predicted  if <= 1.0 (under-predicted = good)
            score = 0.0                       if > 1.0  (overconfident = penalized)

      composite_success_prediction_score =
          mean(score_per_problem) where:
            score = 1.0  if correct success (predicted True, actually passed)
            score = 0.0  if false positive   (predicted True, actually failed)
            score = 0.8  if false negative   (predicted False, actually passed)
            score = 1.0  if correct failure  (predicted False, actually failed)

      accuracy_score =
          (problems solved at unbounded) / (problems where model context >= baseline_n_reliable)

      average_cost_per_token =
          (input_tokens * cost_per_input + output_tokens * cost_per_output)
          / max(1, input_tokens + output_tokens)

      final_score =
          (context_efficiency ^ 2.0)
          * (efficiency_prediction ^ 0.5)
          * (success_prediction ^ 1.5)
          * (accuracy ^ 1.0)
          / max(1e-12, cost_per_token ^ 1.0)
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    results_dir = Path(results_dir)

    # ── Load all evaluation files ─────────────────────────────────────────────
    eval_files = sorted(results_dir.glob("*_evaluation.json"))
    if not eval_files:
        print(f"No evaluation files found in {results_dir}")
        return

    # data[model_name][problem_id] = {prediction: dict, evaluation: dict}
    data: dict = {}

    for ef in eval_files:
        try:
            eval_result = json.loads(ef.read_text())
        except Exception as e:
            print(f"  [score] Cannot read {ef.name}: {e}")
            continue

        model_name = eval_result.get("model_name", "unknown")
        problem_id = eval_result.get("problem_id", ef.stem)

        # Try to load paired prediction file
        pred_path = prediction_filename(results_dir, model_name, problem_id)
        prediction = None
        if pred_path.exists():
            try:
                prediction = json.loads(pred_path.read_text())
            except Exception:
                pass

        data.setdefault(model_name, {})[problem_id] = {
            "evaluation": eval_result,
            "prediction": prediction,
        }

    if not data:
        print("No parseable evaluation files found.")
        return

    all_problems = sorted({pid for m in data.values() for pid in m})

    # ── Compute per-problem baselines ─────────────────────────────────────────
    # baseline_n_reliable[problem_id] = min n_reliable across all models that solved it
    baseline_n_reliable: dict = {}
    for pid in all_problems:
        values = []
        for model_data in data.values():
            if pid in model_data:
                nr = model_data[pid]["evaluation"].get("n_reliable")
                if nr is not None:
                    values.append(nr)
        baseline_n_reliable[pid] = min(values) if values else None

    # ── Per-model scoring ─────────────────────────────────────────────────────
    model_scores = {}

    for model_name in sorted(data.keys()):
        model_data = data[model_name]

        context_eff_scores = []
        eff_pred_scores = []
        success_pred_scores = []
        total_input_tokens = 0
        total_output_tokens = 0
        problems_solved_unbounded = 0
        problems_eligible = 0  # where baseline exists and model could reach it

        for pid in all_problems:
            if pid not in model_data:
                continue
            entry = model_data[pid]
            eval_r = entry["evaluation"]
            pred_r = entry["prediction"]

            n_reliable = eval_r.get("n_reliable")
            n_while_unbounded = eval_r.get("n_while_unbounded")
            baseline = baseline_n_reliable.get(pid)

            # Token accumulation
            total_input_tokens += eval_r.get("total_input_tokens", 0)
            total_output_tokens += eval_r.get("total_output_tokens", 0)
            if pred_r:
                total_input_tokens += pred_r.get("input_tokens", 0)
                total_output_tokens += pred_r.get("output_tokens", 0)

            # Accuracy: solved unbounded / eligible
            if baseline is not None:
                # Eligible = baseline exists (some model solved it)
                problems_eligible += 1
                if n_while_unbounded is not None:
                    problems_solved_unbounded += 1

            # Context efficiency: only for solved problems
            if n_reliable is not None and baseline is not None and baseline > 0:
                context_eff_scores.append(baseline / n_reliable)

            # Efficiency prediction score
            if pred_r is not None:
                n_predicted = pred_r.get("n_predicted")  # None = inf
                if n_reliable is not None and n_predicted is not None and n_predicted > 0:
                    ratio = n_reliable / n_predicted
                    eff_pred_scores.append(min(1.0, ratio))
                else:
                    # inf prediction or no n_reliable → score 0
                    eff_pred_scores.append(0.0)

            # Success prediction score
            predicted_attempt = pred_r.get("attempt", True) if pred_r else True
            actually_solved = n_reliable is not None

            if predicted_attempt and actually_solved:
                success_pred_scores.append(1.0)   # correct True positive
            elif predicted_attempt and not actually_solved:
                success_pred_scores.append(0.0)   # false positive
            elif not predicted_attempt and actually_solved:
                success_pred_scores.append(0.8)   # false negative
            else:
                success_pred_scores.append(1.0)   # correct True negative

        # Component means
        ctx_eff = _mean(context_eff_scores)
        eff_pred = _mean(eff_pred_scores)
        succ_pred = _mean(success_pred_scores)
        accuracy = problems_solved_unbounded / max(1, problems_eligible)

        # Average cost per token (requires cost rates from models.json — try to load)
        cost_per_token = _estimate_cost_per_token(model_name, total_input_tokens, total_output_tokens)

        # Final score
        final = (
            (ctx_eff ** 2.0)
            * (eff_pred ** 0.5)
            * (succ_pred ** 1.5)
            * (accuracy ** 1.0)
            / max(1e-12, cost_per_token ** 1.0)
        )

        model_scores[model_name] = {
            "context_efficiency": ctx_eff,
            "efficiency_prediction": eff_pred,
            "success_prediction": succ_pred,
            "accuracy": accuracy,
            "cost_per_token": cost_per_token,
            "final_score": final,
            "problems_solved": len(context_eff_scores),
            "problems_total": len(all_problems),
            "problems_eligible": problems_eligible,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }

    # ── Print table ───────────────────────────────────────────────────────────
    _print_table(model_scores, all_problems, baseline_n_reliable, data)


def _print_table(model_scores: dict, all_problems: list, baselines: dict, data: dict) -> None:
    W = 130

    # Per-problem detail
    print(f"\n{'=' * W}")
    print("  AmnesiaBench v3 — Per-Problem Results")
    print(f"{'=' * W}")
    hdr = (
        f"{'Model':<28} {'Problem':<30} {'n_reliable':>11} {'n_predicted':>12} "
        f"{'baseline':>9} {'ctx_eff':>8} {'eff_pred':>9}"
    )
    print(hdr)
    print("-" * W)

    for model_name in sorted(model_scores.keys()):
        for pid in all_problems:
            if pid not in data.get(model_name, {}):
                continue
            entry = data[model_name][pid]
            eval_r = entry["evaluation"]
            pred_r = entry["prediction"]

            n_reliable = eval_r.get("n_reliable")
            baseline = baselines.get(pid)
            n_predicted = pred_r.get("n_predicted") if pred_r else None

            ctx_eff_str = (
                f"{baseline / n_reliable:.3f}"
                if n_reliable and baseline else "—"
            )
            eff_pred_str = "—"
            if n_reliable and n_predicted and n_predicted > 0:
                r = n_reliable / n_predicted
                eff_pred_str = f"{min(1.0, r):.3f}"

            print(
                f"{model_name:<28} {pid:<30} "
                f"{str(n_reliable) if n_reliable else 'null':>11} "
                f"{str(n_predicted) if n_predicted else 'inf':>12} "
                f"{str(baseline) if baseline else '—':>9} "
                f"{ctx_eff_str:>8} {eff_pred_str:>9}"
            )

    # Per-model composite
    print(f"\n{'=' * W}")
    print("  AmnesiaBench v3 — Composite Scores")
    print(f"{'=' * W}")
    print(
        f"{'Model':<28} {'CtxEff':>8} {'EffPred':>8} {'SucPred':>8} "
        f"{'Accuracy':>9} {'CostPerTok':>12} {'FinalScore':>12} {'Solved':>8}"
    )
    print("-" * W)

    for model_name in sorted(model_scores.keys()):
        s = model_scores[model_name]
        solved_str = f"{s['problems_solved']}/{s['problems_total']}"
        print(
            f"{model_name:<28} "
            f"{s['context_efficiency']:>8.4f} "
            f"{s['efficiency_prediction']:>8.4f} "
            f"{s['success_prediction']:>8.4f} "
            f"{s['accuracy']:>9.4f} "
            f"{s['cost_per_token']:>12.6f} "
            f"{s['final_score']:>12.4f} "
            f"{solved_str:>8}"
        )

    print(f"{'=' * W}")
    print(
        "\nFormula: final = (ctx_eff^2.0) * (eff_pred^0.5) * (suc_pred^1.5) "
        "* (accuracy^1.0) / (cost_per_token^1.0)"
    )
    print(
        "  ctx_eff     = mean(baseline_n_reliable / n_reliable) over solved problems\n"
        "  eff_pred    = mean(min(1, n_reliable / n_predicted)) — overconfident → 0\n"
        "  suc_pred    = mean(1.0/0.0/0.8/1.0 for TP/FP/FN/TN)\n"
        "  accuracy    = solved_unbounded / eligible_problems\n"
        "  cost_per_tok= weighted avg of input/output token costs from models.json\n"
    )


def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _estimate_cost_per_token(
    model_name: str, input_tokens: int, output_tokens: int
) -> float:
    """
    Load cost rates from models.json and compute weighted average cost per token.
    Falls back to a tiny non-zero default if model not found or tokens = 0.
    """
    try:
        from .models import load_models_json
        models = load_models_json()
        for m in models:
            if m["name"] == model_name:
                cost_in = m.get("cost_per_input_token", 0.0)
                cost_out = m.get("cost_per_output_token", 0.0)
                total_toks = input_tokens + output_tokens
                if total_toks == 0:
                    return 1e-9  # avoid division-by-zero in final score
                return (
                    (input_tokens * cost_in + output_tokens * cost_out) / total_toks
                )
    except Exception:
        pass
    return 1e-9  # default: effectively free (local model)
