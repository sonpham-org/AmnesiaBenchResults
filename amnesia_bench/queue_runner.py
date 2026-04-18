#!/usr/bin/env python3
"""
AmnesiaBench Queue Runner — Run full pipeline across multiple problems in parallel.

Each worker picks the next unfinished (model, problem) pair from the queue and runs it.
Multiple workers share the same Ollama endpoint but solve DIFFERENT problems concurrently.

Usage:
    # Run 3 workers in parallel on all AIMO problems
    python3 queue_runner.py --model qwen3.5:35b --problem-type aimo3 --workers 3

    # Run on Spark B via tunnel
    OLLAMA_HOST=http://localhost:11435 python3 queue_runner.py --model mistral-small3.2 --workers 4

    # Full pipeline (predict + unbounded + compact) with 2 workers
    python3 queue_runner.py --model qwen3.5:35b --problem-type aimo3 --workers 2 --full

Author: Claude Opus 4.6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Import from ollama_runner
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ollama_runner import (
    load_all_problems,
    run_prediction,
    run_unbounded,
    binary_search_window,
    compaction_sweep,
    RESULTS_DIR,
)

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Thread-safe print
_print_lock = threading.Lock()
def tprint(msg):
    with _print_lock:
        print(msg, flush=True)


def is_done(model_safe: str, problem_id: str, stage: str, variant: str = "vanilla") -> bool:
    """Check if a (model, problem, stage, variant) result already exists."""
    if stage == "prediction":
        return (RESULTS_DIR / f"{model_safe}_{problem_id}_prediction.json").exists()
    elif stage == "unbounded":
        return (RESULTS_DIR / f"{model_safe}_{problem_id}_Unbounded.json").exists()
    elif stage == "compact":
        return (RESULTS_DIR / f"{model_safe}_{problem_id}_NoTIR_Compact_{variant}.json").exists()
    elif stage == "sweep":
        return (RESULTS_DIR / f"{model_safe}_{problem_id}_Sweep_{variant}.json").exists()
    return False


def run_problem_pipeline(
    model: str,
    model_safe: str,
    problem: dict,
    stages: list[str],
    unbounded_runs: int = 3,
    trials_per_step: int = 3,
    threshold: float = 0.6,
    initial_window: int = 32768,
    context_max: int = 131072,
    prompt_variant: str = "vanilla",
) -> dict:
    """Run requested stages for one (model, problem) pair. Returns summary dict."""
    pid = problem["problem_id"]
    results = {"problem_id": pid, "model": model}

    # Prediction
    if "prediction" in stages:
        pred_path = RESULTS_DIR / f"{model_safe}_{pid}_prediction.json"
        if pred_path.exists():
            tprint(f"  [{pid}] predict: SKIP (exists)")
            with open(pred_path) as f:
                prediction = json.load(f)
        else:
            tprint(f"  [{pid}] predict: running...")
            prediction = run_prediction(model, problem, verbose=False)
            with open(pred_path, "w") as f:
                json.dump(prediction, f, indent=2)
            tprint(f"  [{pid}] predict: done (N={prediction.get('n_reliable_prediction')})")
        results["prediction"] = prediction
    else:
        prediction = {}

    # Unbounded
    if "unbounded" in stages:
        ub_path = RESULTS_DIR / f"{model_safe}_{pid}_Unbounded.json"
        if ub_path.exists():
            tprint(f"  [{pid}] unbounded: SKIP (exists)")
        else:
            tprint(f"  [{pid}] unbounded: {unbounded_runs} runs...")
            ub_result = run_unbounded(model, problem, n_runs=unbounded_runs,
                                      context_max=context_max, verbose=False,
                                      out_path=ub_path,
                                      prompt_variant=prompt_variant)
            rate = ub_result.get("solve_rate", 0)
            avg = ub_result.get("avg_tokens", 0)
            tprint(f"  [{pid}] unbounded: done (rate={rate:.0%}, avg={avg} tok)")
        results["unbounded"] = True

    # Compact binary search — only if unbounded succeeded, use actual tokens as upper bound
    if "compact" in stages:
        co_path = RESULTS_DIR / f"{model_safe}_{pid}_NoTIR_Compact_{prompt_variant}.json"
        if co_path.exists():
            tprint(f"  [{pid}] compact: SKIP (exists)")
        else:
            # Read unbounded result — skip compact if model never solved unbounded
            ub_path = RESULTS_DIR / f"{model_safe}_{pid}_Unbounded.json"
            compact_upper = initial_window
            should_run_compact = True
            if ub_path.exists():
                try:
                    with open(ub_path) as f:
                        ub_data = json.load(f)
                    solve_rate = ub_data.get("solve_rate", 0)
                    if solve_rate == 0:
                        tprint(f"  [{pid}] compact: SKIP (unbounded solve_rate=0, no hope)")
                        should_run_compact = False
                    else:
                        max_tok = ub_data.get("max_tokens") or ub_data.get("avg_tokens")
                        if max_tok and max_tok > 0:
                            # Start binary search at context_window // 2
                            # but cap to unbounded tokens if smaller
                            compact_upper = min(context_max // 2, max_tok)
                            compact_upper = max(compact_upper, 512)
                except Exception:
                    pass

            if should_run_compact:
                # Load unbounded runs for reuse in compaction (no regeneration needed)
                ub_runs = None
                if ub_path.exists():
                    try:
                        ub_runs = json.load(open(ub_path)).get("runs", [])
                    except Exception:
                        pass
                tprint(f"  [{pid}] compact: binary search [256, {compact_upper}] (reusing {len(ub_runs or [])} unbounded runs)...")
                co_result = binary_search_window(
                    model=model, problem=problem, compaction=True,
                    trials_per_step=trials_per_step, threshold=threshold,
                    initial_window=compact_upper, verbose=False,
                    out_path=co_path,
                    unbounded_runs=ub_runs,
                    prompt_variant=prompt_variant,
                )
                co_result["prediction"] = {
                    "success_prediction": prediction.get("success_prediction"),
                    "n_reliable_prediction": prediction.get("n_reliable_prediction"),
                    "raw_response": prediction.get("raw_response"),
                }
                with open(co_path, "w") as f:
                    json.dump(co_result, f, indent=2)
                mw = co_result.get("minimum_window")
                tprint(f"  [{pid}] compact: done (min_window={mw})")
        results["compact"] = True

    # Sweep: test compaction at multiple truncation points
    if "sweep" in stages:
        sw_path = RESULTS_DIR / f"{model_safe}_{pid}_Sweep_{prompt_variant}.json"
        if sw_path.exists():
            tprint(f"  [{pid}] sweep: SKIP (exists)")
        else:
            ub_path = RESULTS_DIR / f"{model_safe}_{pid}_Unbounded.json"
            ub_runs = None
            should_sweep = True
            if ub_path.exists():
                try:
                    ub_data = json.load(open(ub_path))
                    if ub_data.get("solve_rate", 0) == 0:
                        tprint(f"  [{pid}] sweep: SKIP (unbounded solve_rate=0)")
                        should_sweep = False
                    else:
                        ub_runs = ub_data.get("runs", [])
                except Exception:
                    pass

            if should_sweep and ub_runs:
                tprint(f"  [{pid}] sweep: testing compaction curve ({len(ub_runs)} unbounded runs)...")
                compaction_sweep(
                    model=model, problem=problem,
                    unbounded_runs=ub_runs,
                    trials_per_point=trials_per_step,
                    verbose=False,
                    out_path=sw_path,
                    prompt_variant=prompt_variant,
                    context_window=context_max,
                )
                tprint(f"  [{pid}] sweep: done")
        results["sweep"] = True

    return results


def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench Queue Runner")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--problem", help="Single problem ID")
    parser.add_argument("--problem-type", help="Problem type filter (aimo3, arc, math, number_theory, combinatorics)")
    parser.add_argument("--max-problems", type=int, default=999)
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--unbounded-runs", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3, help="Trials per binary search step")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--initial-window", type=int, default=32768)
    parser.add_argument("--context-max", type=int, default=131072)
    parser.add_argument("--full", action="store_true", help="Run all stages: predict + unbounded + compact")
    parser.add_argument("--unbounded-only", action="store_true", help="Only unbounded runs")
    parser.add_argument("--compact-only", action="store_true", help="Only compact binary search")
    parser.add_argument("--predict-only", action="store_true", help="Only prediction")
    parser.add_argument("--sweep-only", action="store_true", help="Only compaction sweep")
    parser.add_argument("--sweep", action="store_true", help="Include sweep in full pipeline")
    parser.add_argument("--prompt-variant", default="vanilla", help="Compaction prompt variant (vanilla, structured, surgical, notebook)")

    args = parser.parse_args()

    # Determine stages
    if args.full:
        stages = ["prediction", "unbounded", "compact"]
        if args.sweep:
            stages.append("sweep")
    elif args.unbounded_only:
        stages = ["unbounded"]
    elif args.compact_only:
        stages = ["compact"]
    elif args.predict_only:
        stages = ["prediction"]
    elif args.sweep_only:
        stages = ["sweep"]
    else:
        stages = ["prediction", "unbounded", "compact"]

    all_problems = load_all_problems()

    # Select problems
    problems = []
    if args.problem:
        if args.problem in all_problems:
            problems = [all_problems[args.problem]]
        else:
            print(f"Unknown problem: {args.problem}")
            sys.exit(1)
    elif args.problem_type:
        for pid, p in sorted(all_problems.items()):
            topic = p.get("topic", "")
            if topic == args.problem_type or pid.startswith(args.problem_type):
                problems.append(p)
        problems = problems[:args.max_problems]
    else:
        # Default: all AIMO problems
        for pid, p in sorted(all_problems.items()):
            if "aimo3" in pid or p.get("topic") in ("number_theory", "combinatorics"):
                problems.append(p)
        problems = problems[:args.max_problems]

    model_safe = args.model.replace("/", "_").replace(":", "_")
    RESULTS_DIR.mkdir(exist_ok=True)

    # Filter to only problems that have work to do
    todo = []
    for p in problems:
        needs_work = False
        for stage in stages:
            if not is_done(model_safe, p["problem_id"], stage, args.prompt_variant):
                needs_work = True
                break
        if needs_work:
            todo.append(p)

    total = len(problems)
    skip = total - len(todo)

    print(f"{'='*60}")
    print(f"  Queue Runner: {args.model}")
    print(f"  Problems: {total} ({skip} already done, {len(todo)} to run)")
    print(f"  Stages: {stages}")
    print(f"  Workers: {args.workers}")
    print(f"  Ollama: {OLLAMA_BASE}")
    print(f"{'='*60}")

    if not todo:
        print("Nothing to do!")
        return

    t_start = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for p in todo:
            fut = pool.submit(
                run_problem_pipeline,
                model=args.model,
                model_safe=model_safe,
                problem=p,
                stages=stages,
                unbounded_runs=args.unbounded_runs,
                trials_per_step=args.trials,
                threshold=args.threshold,
                initial_window=args.initial_window,
                context_max=args.context_max,
                prompt_variant=args.prompt_variant,
            )
            futures[fut] = p["problem_id"]

        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                result = fut.result()
                completed += 1
                tprint(f"\n  DONE [{completed}/{len(todo)}]: {pid}")
            except Exception as e:
                failed += 1
                tprint(f"\n  FAILED [{pid}]: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Finished: {completed} done, {failed} failed")
    print(f"  Wall time: {elapsed/60:.1f} min")
    print(f"  Results in: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
