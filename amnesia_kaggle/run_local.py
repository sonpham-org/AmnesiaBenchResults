#!/usr/bin/env python3
"""Local smoke-test runner for amnesia_kaggle.

Usage:
    # Requires .env with MODEL_PROXY_API_KEY per kaggle-benchmarks/local_development.md

    # Single problem, unbounded phase only (~1 min)
    python run_local.py --problem digit_sum_ten --phase unbounded

    # Single problem, full pipeline (~15-20 min)
    python run_local.py --problem digit_sum_ten

    # All 10 problems (used for baseline generation)
    python run_local.py --all --out data/baselines_generated.json

    # Pick a specific model (defaults to LLM_DEFAULT in .env)
    python run_local.py --problem digit_sum_ten --model google/gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from amnesia_kaggle.harness import (  # noqa: E402
    find_n_reliable,
    run_prediction,
    run_unbounded,
)
from amnesia_kaggle.scoring import compute_scores  # noqa: E402


def load_problems() -> list[dict]:
    return json.loads((_HERE / "data" / "problems.json").read_text())


def load_baselines() -> dict:
    return json.loads((_HERE / "data" / "baselines.json").read_text())


def get_llm(model_name: str | None):
    import kaggle_benchmarks as kbench
    if model_name:
        return kbench.llms[model_name]
    return kbench.llm


def format_n(n: float) -> str:
    return "inf" if math.isinf(n) else str(int(n))


def run_one_problem(llm, problem: dict, temperature: float, ctx_window: int, phase: str):
    pid = problem["problem_id"]
    problem_text = problem["problem_text"]
    ground_truth = problem["ground_truth"]

    print(f"\n── {pid} ───────────────────────────────────")
    print(f"Problem: {problem_text[:120]}{'...' if len(problem_text) > 120 else ''}")
    print(f"Ground truth: {ground_truth}")

    t0 = time.time()

    # Phase 1: unbounded
    print(f"\n[unbounded] running 3 trials...")
    unb = run_unbounded(llm, problem_text, ground_truth, temperature=temperature)
    print(f"  solved={unb.solved}  n_while_unbounded={format_n(unb.n_while_unbounded)}"
          f"  cost={unb.cost_nanodollars} nanodollars  context_exceeded={unb.context_exceeded}")
    for log in unb.trial_logs:
        print(f"    trial {log['trial_idx']}: {log['finish_reason']}"
              f"  tokens={log['tokens_used']}  {log['wall_time_s']}s")

    if phase == "unbounded":
        return {
            "problem_id": pid,
            "n_while_unbounded": unb.n_while_unbounded,
            "cost_nanodollars": unb.cost_nanodollars,
            "phase": "unbounded",
        }

    # Phase 2: prediction
    print(f"\n[predict] asking model to self-assess...")
    pred = run_prediction(llm, problem_text, temperature=temperature)
    print(f"  attempt={pred.attempt}  n_predicted={format_n(pred.n_predicted)}")

    if phase == "predict":
        return {
            "problem_id": pid,
            "n_while_unbounded": unb.n_while_unbounded,
            "attempt": pred.attempt,
            "n_predicted": pred.n_predicted,
            "cost_nanodollars": unb.cost_nanodollars + pred.cost_nanodollars,
            "phase": "predict",
        }

    # Phase 3+4: sweep (no-compact and compact)
    should_run_sweep = (unb.n_while_unbounded < math.inf) or unb.context_exceeded
    if not should_run_sweep:
        print("\n[sweep] skipped — unbounded failed for non-context reasons")
        return {
            "problem_id": pid,
            "n_while_unbounded": unb.n_while_unbounded,
            "n_reliable_no_compact": math.inf,
            "n_reliable_compact": math.inf,
            "n_reliable": math.inf,
            "attempt": pred.attempt,
            "n_predicted": pred.n_predicted,
            "unbounded_solved": unb.solved,
            "input_tokens": unb.input_tokens + pred.input_tokens,
            "output_tokens": unb.output_tokens + pred.output_tokens,
            "cost_nanodollars": unb.cost_nanodollars + pred.cost_nanodollars,
            "wall_time_s": round(time.time() - t0, 2),
        }

    n_max = int(unb.n_while_unbounded) if unb.n_while_unbounded < math.inf else ctx_window
    cost_cap = 3 * max(1, unb.cost_nanodollars)
    token_cap = int(5 * unb.n_while_unbounded) if unb.n_while_unbounded < math.inf else None

    print(f"\n[sweep no-compact] n_max={n_max}  cost_cap={cost_cap}")
    n_reliable_nc, log_nc, cost_nc, inp_nc, out_nc = find_n_reliable(
        llm, problem_text, ground_truth, n_max=n_max,
        compaction_enabled=False, temperature=temperature,
        cost_cap_nanodollars=cost_cap,
    )
    print(f"  n_reliable_no_compact={format_n(n_reliable_nc)}"
          f"  trials={len(log_nc.entries)}  cost={cost_nc}")

    print(f"\n[sweep compact] n_max={n_max}  cost_cap={cost_cap}  token_cap={token_cap}")
    n_reliable_c, log_c, cost_c, inp_c, out_c = find_n_reliable(
        llm, problem_text, ground_truth, n_max=n_max,
        compaction_enabled=True, temperature=temperature,
        cost_cap_nanodollars=cost_cap,
        token_cap=token_cap,
    )
    print(f"  n_reliable_compact={format_n(n_reliable_c)}"
          f"  trials={len(log_c.entries)}  cost={cost_c}")

    n_reliable = min(n_reliable_nc, n_reliable_c)

    return {
        "problem_id": pid,
        "n_while_unbounded": unb.n_while_unbounded,
        "n_reliable_no_compact": n_reliable_nc,
        "n_reliable_compact": n_reliable_c,
        "n_reliable": n_reliable,
        "attempt": pred.attempt,
        "n_predicted": pred.n_predicted,
        "unbounded_solved": unb.solved,
        "input_tokens": (unb.input_tokens + pred.input_tokens + inp_nc + inp_c),
        "output_tokens": (unb.output_tokens + pred.output_tokens + out_nc + out_c),
        "cost_nanodollars": (unb.cost_nanodollars + pred.cost_nanodollars + cost_nc + cost_c),
        "wall_time_s": round(time.time() - t0, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", help="Run one problem by ID")
    parser.add_argument("--all", action="store_true", help="Run all 10 problems")
    parser.add_argument("--phase", choices=["unbounded", "predict", "all"], default="all",
                        help="Which phase(s) to run (default: all)")
    parser.add_argument("--model", help="Override LLM (e.g. google/gemini-2.5-flash)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ctx-window", type=int, default=262144)
    parser.add_argument("--out", help="Write per-problem results to JSON file")
    args = parser.parse_args()

    if not args.problem and not args.all:
        parser.error("--problem or --all is required")

    problems = load_problems()
    if args.problem:
        problems = [p for p in problems if args.problem in p["problem_id"]]
        if not problems:
            parser.error(f"No problem matched: {args.problem}")

    llm = get_llm(args.model)
    print(f"Model: {getattr(llm, 'name', str(llm))}")
    print(f"Temperature: {args.temperature}  Context window: {args.ctx_window}")
    print(f"Problems: {len(problems)}")

    results = []
    for problem in problems:
        try:
            r = run_one_problem(
                llm, problem,
                temperature=args.temperature,
                ctx_window=args.ctx_window,
                phase=args.phase,
            )
            results.append(r)
        except KeyboardInterrupt:
            print("\nInterrupted — saving partial results")
            break
        except Exception as e:
            print(f"\nERROR on {problem['problem_id']}: {e}")
            import traceback
            traceback.print_exc()

    if args.out and results:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=_json_default))
        print(f"\nWrote results to {out_path}")

    if args.phase == "all" and len(results) > 0:
        baselines = load_baselines()
        # Strip metadata keys starting with __
        baselines_clean = {k: v for k, v in baselines.items() if not k.startswith("__")}
        scores = compute_scores(results, baselines_clean, model_ctx_window=args.ctx_window)
        print("\n── Scores ──────────────────────────────────")
        for k, v in scores.items():
            print(f"  {k:40s} {v}")


def _json_default(obj):
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    raise TypeError(f"not serializable: {type(obj)}")


if __name__ == "__main__":
    main()
