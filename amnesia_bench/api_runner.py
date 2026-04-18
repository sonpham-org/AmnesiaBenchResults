#!/usr/bin/env python3
"""
AmnesiaBench API Runner — Run sweep experiments against cloud API models
(Gemini, Anthropic, OpenRouter) using the same pipeline as ollama_runner.

Uses the same file-based caching, prompt variants, and sweep logic.

Usage:
    # Gemini Flash Lite on all scott problems
    python3 amnesia_bench/api_runner.py \
        --model gemini://gemini-3.1-flash-lite-preview \
        --problem-type scott --full --sweep \
        --prompt-variant structured --trials 3

    # Anthropic (requires ANTHROPIC_OAUTHTOKEN)
    python3 amnesia_bench/api_runner.py \
        --model anthropic://claude-sonnet-4-6 \
        --problem scott_prime_sequence --unbounded-only

    # OpenRouter (requires OPENROUTER_API_KEY)
    python3 amnesia_bench/api_runner.py \
        --model openrouter://google/gemini-3.1-flash-lite-preview \
        --problem-type scott --full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Patch ollama_generate BEFORE importing the rest of ollama_runner
# so that run_trial, run_unbounded, etc. all use the API client.
import ollama_runner

# clients.py uses relative imports, so we need to handle it
import importlib, types
import backoff as _backoff_mod
# Patch the module so clients.py can find backoff via relative import
_pkg = types.ModuleType("amnesia_bench")
_pkg.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("amnesia_bench", _pkg)
sys.modules.setdefault("amnesia_bench.backoff", _backoff_mod)

from amnesia_bench.clients import create_client, GeminiClient, AnthropicClient, LLMClient
from ollama_runner import (
    load_all_problems,
    run_unbounded,
    binary_search_window,
    compaction_sweep,
    run_prediction,
    trial_context,
    _cache_get,
    RESULTS_DIR,
)
from compaction_prompts import PROMPT_VARIANTS, DEFAULT_PROMPT_VARIANT

RESULTS_DIR.mkdir(exist_ok=True)

_print_lock = threading.Lock()
def tprint(msg):
    with _print_lock:
        print(msg, flush=True)


def patch_ollama_generate(client, max_output_tokens: int = 0):
    """Replace ollama_runner.ollama_generate with an API client wrapper.

    max_output_tokens: if >0, clamp max_tokens to this ceiling (e.g. 8192 for Haiku).
    """
    def api_generate(model, system, user, max_tokens=4096, temperature=0.7, messages=None):
        if max_output_tokens > 0:
            max_tokens = min(max_tokens, max_output_tokens)
        # Multi-turn path: caller (e.g. compaction loop) supplied a full
        # conversation. Use it as-is so the model sees its own prior assistant
        # turns. Single-turn path: build [system?, user] from kwargs.
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
        start = time.time()
        result = client.generate(messages, max_tokens=max_tokens, stream=False)
        wall_time = time.time() - start

        response_text = result.get("final_content") or result.get("content", "")
        thinking_text = result.get("reasoning_content", "")

        return {
            "response": response_text,
            "thinking": thinking_text,
            "prompt_tokens": result.get("input_tokens", 0),
            "eval_tokens": result.get("output_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "wall_time_s": wall_time,
            "pp_tok_s": 0,
            "tg_tok_s": 0,
        }

    ollama_runner.ollama_generate = api_generate


def model_safe_name(model_url: str) -> str:
    """Convert model URL to a safe filename component."""
    # gemini://gemini-3.1-flash-lite-preview -> gemini-3.1-flash-lite-preview
    # anthropic://claude-sonnet-4-6 -> claude-sonnet-4-6
    for prefix in ["gemini://", "google://", "anthropic://", "openrouter://"]:
        if model_url.startswith(prefix):
            name = model_url[len(prefix):].strip("/")
            return name.replace("/", "_").replace(":", "_")
    return model_url.replace("/", "_").replace(":", "_")


def is_done(model_safe: str, problem_id: str, stage: str, variant: str = "vanilla") -> bool:
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
    model_url: str,
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
    pid = problem["problem_id"]
    # Use the safe name as "model" for ollama_runner functions
    model = model_safe
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
                                      context_max=context_max, verbose=True,
                                      out_path=ub_path,
                                      prompt_variant=prompt_variant)
            rate = ub_result.get("solve_rate", 0)
            avg = ub_result.get("avg_tokens", 0)
            tprint(f"  [{pid}] unbounded: done (rate={rate:.0%}, avg={avg} tok)")
        results["unbounded"] = True

    # Compact
    if "compact" in stages:
        co_path = RESULTS_DIR / f"{model_safe}_{pid}_NoTIR_Compact_{prompt_variant}.json"
        if co_path.exists():
            tprint(f"  [{pid}] compact: SKIP (exists)")
        else:
            ub_path = RESULTS_DIR / f"{model_safe}_{pid}_Unbounded.json"
            compact_upper = initial_window
            should_run = True
            if ub_path.exists():
                try:
                    with open(ub_path) as f:
                        ub_data = json.load(f)
                    if ub_data.get("solve_rate", 0) == 0:
                        tprint(f"  [{pid}] compact: SKIP (solve_rate=0)")
                        should_run = False
                    else:
                        max_tok = ub_data.get("max_tokens") or ub_data.get("avg_tokens")
                        if max_tok and max_tok > 0:
                            # Start binary search at context_window // 2
                            # but cap to unbounded tokens if smaller
                            compact_upper = min(context_max // 2, max_tok)
                            compact_upper = max(compact_upper, 512)
                except Exception:
                    pass

            if should_run:
                ub_runs = None
                if ub_path.exists():
                    try:
                        ub_runs = json.load(open(ub_path)).get("runs", [])
                    except Exception:
                        pass
                tprint(f"  [{pid}] compact: binary search [256, {compact_upper}]...")
                co_result = binary_search_window(
                    model=model, problem=problem, compaction=True,
                    trials_per_step=trials_per_step, threshold=threshold,
                    initial_window=compact_upper, verbose=False,
                    out_path=co_path, unbounded_runs=ub_runs,
                    prompt_variant=prompt_variant,
                )
                co_result["prediction"] = {
                    "success_prediction": prediction.get("success_prediction"),
                    "n_reliable_prediction": prediction.get("n_reliable_prediction"),
                }
                with open(co_path, "w") as f:
                    json.dump(co_result, f, indent=2)
                mw = co_result.get("minimum_window")
                tprint(f"  [{pid}] compact: done (min_window={mw})")
        results["compact"] = True

    # Sweep
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
                        tprint(f"  [{pid}] sweep: SKIP (solve_rate=0)")
                        should_sweep = False
                    else:
                        ub_runs = ub_data.get("runs", [])
                except Exception:
                    pass

            if should_sweep and ub_runs:
                tprint(f"  [{pid}] sweep: testing compaction curve...")
                compaction_sweep(
                    model=model, problem=problem,
                    unbounded_runs=ub_runs,
                    trials_per_point=trials_per_step,
                    verbose=False, out_path=sw_path,
                    prompt_variant=prompt_variant,
                    context_window=context_max,
                )
                tprint(f"  [{pid}] sweep: done")
        results["sweep"] = True

    return results


def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench API Runner")
    parser.add_argument("--model", required=True,
                        help="Model URL (gemini://model, anthropic://model, openrouter://model)")
    parser.add_argument("--api-key", help="API key (or use env: GEMINI_API_KEY, OPENROUTER_API_KEY)")
    parser.add_argument("--problem", help="Single problem ID")
    parser.add_argument("--problem-type", help="Problem type filter (scott, aimo3, etc.)")
    parser.add_argument("--max-problems", type=int, default=999)
    parser.add_argument("--unbounded-runs", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--initial-window", type=int, default=32768)
    parser.add_argument("--context-max", type=int, default=131072)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--unbounded-only", action="store_true")
    parser.add_argument("--compact-only", action="store_true")
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--prompt-variant", default="structured")

    args = parser.parse_args()

    # Create API client
    client = create_client(args.model, api_key=args.api_key)

    # Per-model output token limits (API enforced ceilings)
    MODEL_OUTPUT_LIMITS = {
        "claude-haiku-4-6": 64000,
        "claude-sonnet-4-6": 64000,
        "claude-opus-4-6": 64000,
    }
    # Extract model name from URL scheme
    for scheme in ("anthropic://", "gemini://", "google://", "openrouter://"):
        if args.model.startswith(scheme):
            _mname = args.model[len(scheme):].strip("/")
            break
    else:
        _mname = args.model
    max_out = MODEL_OUTPUT_LIMITS.get(_mname, 0)
    if max_out:
        # Also clamp context_max so unbounded runs don't request more than the model allows
        args.context_max = min(args.context_max, max_out)

    patch_ollama_generate(client, max_output_tokens=max_out)

    msafe = model_safe_name(args.model)

    # Determine stages
    if args.full:
        stages = ["prediction", "unbounded", "compact"]
        if args.sweep:
            stages.append("sweep")
    elif args.unbounded_only:
        stages = ["unbounded"]
    elif args.compact_only:
        stages = ["compact"]
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
            if pid.startswith(args.problem_type) or p.get("topic") == args.problem_type:
                problems.append(p)
        problems = problems[:args.max_problems]
    else:
        for pid, p in sorted(all_problems.items()):
            if pid.startswith("scott_"):
                problems.append(p)
        problems = problems[:args.max_problems]

    # Filter to problems that need work
    todo = [p for p in problems
            if any(not is_done(msafe, p["problem_id"], s, args.prompt_variant) for s in stages)]

    print(f"{'='*60}")
    print(f"  API Runner: {args.model}")
    print(f"  Model safe name: {msafe}")
    print(f"  Problems: {len(problems)} ({len(problems)-len(todo)} done, {len(todo)} to run)")
    print(f"  Stages: {stages}")
    print(f"  Prompt variant: {args.prompt_variant}")
    print(f"{'='*60}")

    if not todo:
        print("Nothing to do!")
        return

    t_start = time.time()
    completed = 0
    failed = 0

    for p in todo:
        pid = p["problem_id"]
        try:
            run_problem_pipeline(
                model_url=args.model,
                model_safe=msafe,
                problem=p,
                stages=stages,
                unbounded_runs=args.unbounded_runs,
                trials_per_step=args.trials,
                threshold=args.threshold,
                initial_window=args.initial_window,
                context_max=args.context_max,
                prompt_variant=args.prompt_variant,
            )
            completed += 1
            tprint(f"\n  DONE [{completed}/{len(todo)}]: {pid}")
        except Exception as e:
            failed += 1
            tprint(f"\n  FAILED [{pid}]: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Finished: {completed} done, {failed} failed")
    print(f"  Wall time: {elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
