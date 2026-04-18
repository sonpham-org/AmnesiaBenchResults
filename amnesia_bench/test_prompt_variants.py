#!/usr/bin/env python3
"""
A/B test compaction prompt variants on a single problem.

File naming: {model}_{pid}_t{trial}_{context}.json
  context = "Unbounded" | "w{window}_{variant}"

Usage:
    # Test all variants with mistral-small3.2 on scott_prime_sequence
    python3 amnesia_bench/test_prompt_variants.py --model mistral-small3.2 --problem scott_prime_sequence

    # Test on Spark B
    OLLAMA_HOST=http://192.168.100.11:11434 python3 amnesia_bench/test_prompt_variants.py --model mistral-small3.2

    # Specific variant only
    python3 amnesia_bench/test_prompt_variants.py --model mistral-small3.2 --variant structured

    # Multiple problems
    python3 amnesia_bench/test_prompt_variants.py --model mistral-small3.2 --problem-type scott
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ollama_runner import (
    load_all_problems,
    run_trial,
    run_unbounded,
    trial_context,
    trial_path,
    _model_safe,
    _cache_get,
    _cache_put,
    RESULTS_DIR,
)
from compaction_prompts import PROMPT_VARIANTS

RESULTS_DIR.mkdir(exist_ok=True)

# Prompt tuning results go in a separate directory
PROMPT_TUNING_DIR = RESULTS_DIR.parent / "results_prompt_tuning"
PROMPT_TUNING_DIR.mkdir(exist_ok=True)


def _pt_path(model, pid, trial_idx, ctx):
    """Path for a prompt tuning trial result."""
    return PROMPT_TUNING_DIR / f"{_model_safe(model)}_{pid}_t{trial_idx}_{ctx}.json"


def _pt_get(model, pid, trial_idx, ctx):
    """Load cached prompt tuning trial from disk."""
    p = _pt_path(model, pid, trial_idx, ctx)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _pt_put(model, pid, trial_idx, result, ctx):
    """Save prompt tuning trial to disk."""
    p = _pt_path(model, pid, trial_idx, ctx)
    with open(p, "w") as f:
        json.dump(result, f, indent=2)


def run_variant_trial(model, problem, variant, window, unbounded_run, trial_idx=0):
    """Run one compaction trial, checking file cache first."""
    pid = problem["problem_id"]
    ctx = trial_context(window, variant)

    # File cache check (prompt tuning dir)
    cached = _pt_get(model, pid, trial_idx, ctx)
    if cached:
        status = "PASS" if cached.get("success") else "FAIL"
        print(f"    ♻️ cached {status} | ans={cached.get('answer')} | "
              f"compactions={cached.get('n_compactions')}")
        return cached

    t0 = time.time()
    result = run_trial(
        model=model,
        problem=problem,
        token_limit=window,
        compaction=True,
        trial_idx=trial_idx,
        unbounded_run=unbounded_run,
        prompt_variant=variant,
    )
    elapsed = time.time() - t0

    status = "PASS" if result["success"] else "FAIL"
    print(f"    {status} | ans={result['answer']} (correct: {problem.get('correct_answer')}) | "
          f"compactions={result['n_compactions']} | {elapsed:.1f}s")

    # Show compaction summary preview
    for msg in result["conversation"]:
        content = msg.get("content", "")
        if "<compact>" in content:
            m = re.search(r"<compact>(.*?)</compact>", content, re.DOTALL)
            if m:
                summary = m.group(1).strip()
                preview = summary[:200].replace("\n", " ")
                print(f"      summary: {preview}{'...' if len(summary) > 200 else ''}")

    # Save to prompt tuning dir
    _pt_put(model, pid, trial_idx, result, ctx)
    return result


def load_unbounded_runs(model, problem, n_runs=3, verbose=True):
    """Load unbounded runs from file cache, running only missing ones."""
    pid = problem["problem_id"]
    runs = []
    need_run = False

    for i in range(n_runs):
        cached = _cache_get(model, pid, i, "Unbounded")
        if cached:
            runs.append(cached)
        else:
            need_run = True
            break

    if need_run:
        # Run all via run_unbounded (it checks per-trial cache internally)
        model_safe = model.replace("/", "_").replace(":", "_")
        out_path = RESULTS_DIR / f"{model_safe}_{pid}_Unbounded.json"
        ub_data = run_unbounded(model, problem, n_runs=n_runs,
                                context_max=131072, verbose=verbose, out_path=out_path)
        runs = ub_data.get("runs", [])

    return runs


def main():
    parser = argparse.ArgumentParser(description="Test compaction prompt variants")
    parser.add_argument("--model", default="mistral-small3.2")
    parser.add_argument("--problem", help="Single problem ID")
    parser.add_argument("--problem-type", help="Filter by prefix (e.g. scott)")
    parser.add_argument("--variant", help="Test single variant (default: all)")
    parser.add_argument("--window", type=int, help="Override window size")
    parser.add_argument("--trials", type=int, default=3, help="Trials per variant")
    parser.add_argument("--unbounded-runs", type=int, default=3)
    args = parser.parse_args()

    all_problems = load_all_problems()

    # Select problems
    if args.problem:
        if args.problem not in all_problems:
            print(f"Unknown problem: {args.problem}")
            print(f"Available: {sorted(all_problems.keys())}")
            sys.exit(1)
        problems = [all_problems[args.problem]]
    elif args.problem_type:
        problems = [p for pid, p in sorted(all_problems.items())
                     if pid.startswith(args.problem_type)]
    else:
        problems = [p for pid, p in sorted(all_problems.items())
                     if pid.startswith("scott_")]

    variants = [args.variant] if args.variant else list(PROMPT_VARIANTS.keys())

    print(f"{'='*60}")
    print(f"  Prompt Variant A/B Test")
    print(f"  Model: {args.model}")
    print(f"  Problems: {len(problems)}")
    print(f"  Variants: {variants}")
    print(f"  Trials per variant: {args.trials}")
    print(f"{'='*60}")

    grand_results = {}  # pid -> variant -> [trials]

    for problem in problems:
        pid = problem["problem_id"]
        print(f"\n{'─'*60}")
        print(f"  Problem: {pid} (answer: {problem.get('correct_answer')})")
        print(f"{'─'*60}")

        # Load/run unbounded
        ub_runs = load_unbounded_runs(args.model, problem,
                                       n_runs=args.unbounded_runs, verbose=True)
        if not ub_runs:
            print(f"  No unbounded runs — skipping")
            continue

        avg_tokens = sum(r.get("total_tokens", 0) for r in ub_runs) / len(ub_runs)
        solve_rate = sum(1 for r in ub_runs if r.get("success")) / len(ub_runs)
        print(f"  Unbounded: avg_tokens={avg_tokens:.0f}, solve_rate={solve_rate:.0%}")

        if solve_rate == 0:
            print(f"  Unbounded solve_rate=0 — skipping compaction test")
            continue

        # Pick window
        if args.window:
            window = args.window
        else:
            window = max(512, int(avg_tokens * 0.5))
            window = (window // 64) * 64
        print(f"  Window: {window} (half_budget={window//2})")

        problem_results = {}
        for variant in variants:
            print(f"\n  [{variant}]")
            trials = []
            for t in range(args.trials):
                ub_run = ub_runs[t % len(ub_runs)]
                print(f"    Trial {t+1}/{args.trials}: ", end="", flush=True)
                result = run_variant_trial(args.model, problem, variant, window,
                                           ub_run, trial_idx=t)
                trials.append(result)
            problem_results[variant] = trials

        grand_results[pid] = problem_results

    # Grand summary
    print(f"\n{'='*60}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Problem':<25} {'Variant':<15} {'Pass':>6} {'Compactions':>12} {'Time':>8}")
    print(f"  {'-'*66}")
    for pid, variants_data in grand_results.items():
        for variant, trials in variants_data.items():
            passes = sum(1 for t in trials if t["success"])
            total = len(trials)
            avg_comp = sum(t["n_compactions"] for t in trials) / total
            avg_time = sum(t["wall_time_s"] for t in trials) / total
            print(f"  {pid:<25} {variant:<15} {passes}/{total:>3} {avg_comp:>12.1f} {avg_time:>7.1f}s")
        print()


if __name__ == "__main__":
    main()
