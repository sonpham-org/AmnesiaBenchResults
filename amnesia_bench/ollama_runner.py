#!/usr/bin/env python3
"""
AmnesiaBench Ollama Runner — Run binary-search context-window experiments
against local models via Ollama API.

Usage:
    python3 ollama_runner.py --model qwen3:32b --problem crt_three_congruences
    python3 ollama_runner.py --model qwen3:32b --problem-type arc --max-problems 5
    python3 ollama_runner.py --model qwen3:32b --list-problems

Author: Sherlock (2026-04-04)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error
import threading

# ── Paths ─────────────────────────────────────────────────────────────────────

AMNESIA_DIR = Path(__file__).resolve().parent
REPO_ROOT = AMNESIA_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
ARC_PROBLEMS = AMNESIA_DIR / "arc_problems.json"

# Add this directory so we can import evaluator
sys.path.insert(0, str(AMNESIA_DIR))
from arc_evaluator import evaluate_arc_answer
from compaction_prompts import (
    PROMPT_VARIANTS, DEFAULT_PROMPT_VARIANT,
    build_unbounded_system, build_system_prompt, build_user_message,
    build_compact_prompt, build_resume_prompt,
    build_prediction_prompt as build_prediction_prompt_variant,
    get_variant,
)

# ── File-based trial cache ────────────────────────────────────────────────────
# Naming: {model}_{pid}_t{trial}_{context}.json
# Context = "Unbounded" | "w{window}_{variant}"
# File existence = cache hit.  No in-memory state needed.


def _model_safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def trial_path(model: str, pid: str, trial_idx: int,
               context: str = "Unbounded") -> Path:
    """Return the canonical file path for a single trial result."""
    return RESULTS_DIR / f"{_model_safe(model)}_{pid}_t{trial_idx}_{context}.json"


def trial_context(window: Optional[int] = None,
                  prompt_variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Build the context string for a trial file name."""
    if window is None:
        return "Unbounded"
    return f"w{window}_{prompt_variant}"


def _cache_get(model: str, pid: str, trial_idx: int,
               context: str = "Unbounded") -> Optional[dict]:
    """Load a cached trial result from disk, or return None."""
    p = trial_path(model, pid, trial_idx, context)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_put(model: str, pid: str, trial_idx: int, result: dict,
               context: str = "Unbounded"):
    """Save a trial result to disk."""
    p = trial_path(model, pid, trial_idx, context)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump(result, f, indent=2)


# ── Built-in problems (math/number theory) ────────────────────────────────────

MATH_PROBLEMS = {
    "crt_three_congruences": {
        "problem_id": "crt_three_congruences",
        "problem_text": (
            "Find the smallest positive integer $n$ satisfying all three "
            "congruences simultaneously:\n"
            "$$n \\equiv 3 \\pmod{7}$$\n"
            "$$n \\equiv 5 \\pmod{11}$$\n"
            "$$n \\equiv 8 \\pmod{13}$$"
        ),
        "correct_answer": 346,
        "topic": "number_theory",
    },
    "digit_sum_ten": {
        "problem_id": "digit_sum_ten",
        "problem_text": (
            "How many three-digit positive integers have digits that sum to 10?"
        ),
        "correct_answer": 54,
        "topic": "combinatorics",
    },
}

# Load AIMO3 problems from existing results (extract problem text from conversations)
def _load_aimo3_problems() -> dict:
    """Scan results/ for aimo3 problems and extract problem text from conversations."""
    probs = {}
    for fn in sorted(RESULTS_DIR.iterdir()):
        if "aimo3" in fn.name and fn.name.endswith("_Compact.json"):
            try:
                with open(fn) as f:
                    data = json.load(f)
                pid = data["problem_id"]
                if pid in probs:
                    continue
                # Extract problem text from first trial's conversation
                conv = data["binary_search"][0]["trials"][0]["conversation"]
                user_msg = next((m["content"] for m in conv if m["role"] == "user"), None)
                correct = data["binary_search"][0]["trials"][0].get("correct_answer")
                if user_msg and correct is not None:
                    probs[pid] = {
                        "problem_id": pid,
                        "problem_text": user_msg,
                        "correct_answer": correct,
                        "topic": "aimo3",
                    }
            except (json.JSONDecodeError, KeyError, StopIteration):
                continue
    return probs


def load_all_problems() -> dict:
    """Load all available problems."""
    all_probs = dict(MATH_PROBLEMS)

    # Load AIMO3 from results
    all_probs.update(_load_aimo3_problems())

    # Load problem JSON files from problems/ directory
    problems_dir = AMNESIA_DIR / "problems"
    if problems_dir.exists():
        for fn in sorted(problems_dir.iterdir()):
            if fn.suffix == ".json":
                try:
                    with open(fn) as f:
                        p = json.load(f)
                    pid = p.get("problem_id", fn.stem)
                    # Normalize: ensure ground_truth or correct_answer exists
                    if "correct_answer" not in p and "ground_truth" in p:
                        p["correct_answer"] = p["ground_truth"]
                    elif "ground_truth" not in p and "correct_answer" in p:
                        p["ground_truth"] = p["correct_answer"]
                    all_probs[pid] = p
                except (json.JSONDecodeError, OSError):
                    continue

    # Load ARC problems
    if ARC_PROBLEMS.exists():
        with open(ARC_PROBLEMS) as f:
            arc = json.load(f)
        for p in arc:
            all_probs[p["problem_id"]] = p

    return all_probs


# ── Ollama API ────────────────────────────────────────────────────────────────

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_generate(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    messages: list = None,
) -> dict:
    """Generation via Ollama /api/chat endpoint.

    If `messages` is provided, use it directly (multi-turn conversation).
    Otherwise build a single-turn [system, user] message list.
    """
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "flash_attn": True,
            "kv_cache_type": "q8_0",
        },
    }

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        result = json.loads(resp.read().decode())
    wall_time = time.time() - start

    msg = result.get("message", {})
    response_text = msg.get("content", "")
    thinking_text = msg.get("thinking", "") if "thinking" in msg else ""

    # Token counts from Ollama
    prompt_tokens = result.get("prompt_eval_count", 0)
    eval_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + eval_tokens

    # Timing details from Ollama
    prompt_dur = result.get("prompt_eval_duration", 0)
    eval_dur = result.get("eval_duration", 0)
    pp_tps = prompt_tokens / (prompt_dur / 1e9) if prompt_dur > 0 else 0
    tg_tps = eval_tokens / (eval_dur / 1e9) if eval_dur > 0 else 0

    return {
        "response": response_text,
        "thinking": thinking_text,
        "prompt_tokens": prompt_tokens,
        "eval_tokens": eval_tokens,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time,
        "pp_tok_s": round(pp_tps, 1),
        "tg_tok_s": round(tg_tps, 1),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def extract_boxed_answer(text: str) -> Optional[int]:
    """Extract integer from \\boxed{...}."""
    # Try \\boxed{N} first
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        try:
            return int(matches[-1].strip().replace(",", ""))
        except ValueError:
            pass
    # Try just a standalone number at the end
    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            pass
    return None


def evaluate_math_answer(response: str, correct_answer: int) -> dict:
    """Evaluate a math problem response."""
    predicted = extract_boxed_answer(response)
    return {
        "correct": predicted == correct_answer,
        "predicted": predicted,
        "expected": correct_answer,
    }


# ── System prompts ────────────────────────────────────────────────────────────

def make_system_prompt(token_limit: int, compaction: bool, topic: str,
                       variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Build the system prompt for a given config."""
    if topic == "arc":
        # ARC uses its own system prompt
        from arc_prompts import ARC_SYSTEM_PROMPT_SIMPLE
        base = ARC_SYSTEM_PROMPT_SIMPLE
        if compaction:
            base += (
                "\n\nYour context window is limited. "
                "To compact, write:\n<compact>your summary here</compact>\n\n"
                "When you call compact, the conversation resets to:\n"
                "  [this system prompt] + [the problem] + [your summary]\n"
                "You may compact at most 5 times."
            )
        return base

    # Use unified prompt variant system
    if compaction:
        return build_system_prompt(token_limit, variant)
    else:
        return build_unbounded_system(variant)



# ── Single trial ──────────────────────────────────────────────────────────────

def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens (4 chars per token estimate)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated at ~" + str(max_tokens) + " tokens]"


def run_trial(
    model: str,
    problem: dict,
    token_limit: int,
    compaction: bool,
    trial_idx: int,
    unbounded_run: Optional[dict] = None,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> dict:
    """Run a single trial of a problem at a given token limit.

    In compaction mode with unbounded_run provided:
      1. Reuse the unbounded run's output, truncated at 50% of N tokens
      2. Force compaction: model summarizes the truncated output
      3. Resume from summary with fresh 50% budget
      4. Repeat until solved or max compactions

    This avoids regenerating the first chunk and enables prompt caching
    (system + problem text is always the same base prompt).
    """
    topic = problem.get("topic", "math")
    is_arc = topic == "arc"
    half_budget = max(token_limit // 2, 256)
    max_compactions = 5
    max_turns = 40

    # Compaction trials use the bounded instructions (with N); unbounded uses its own.
    # Variants like refined3 keep system empty and put everything in user_msg.
    if compaction:
        system = build_system_prompt(token_limit, prompt_variant)
        user_msg = build_user_message(
            problem["problem_text"], prompt_variant,
            N=token_limit, tokens_left=half_budget,
        )
    else:
        system = build_unbounded_system(prompt_variant)
        user_msg = build_user_message(problem["problem_text"], prompt_variant, N=None)

    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    n_compactions = 0
    total_wall_time = 0.0
    n_turns = 0
    last_full_text = ""
    finish_reason = "stop"

    # ── COMPACTION MODE with unbounded reuse ───────��─────────────────────
    if compaction and unbounded_run:
        ub_conv = unbounded_run.get("conversation", [])
        ub_thinking = ""
        ub_response = ""
        for msg in ub_conv:
            if msg.get("role") == "assistant":
                ub_thinking = msg.get("thinking", "")
                ub_response = msg.get("content", "")
                break

        # Truncate the unbounded output at 50% of N tokens TOTAL.
        # Token budget is shared: thinking first, then response with remainder.
        # If truncation falls within thinking, response is EMPTY — the model
        # hadn't produced any visible output yet at that point.
        thinking_tok_est = len(ub_thinking) // 4 if ub_thinking else 0
        if ub_thinking and thinking_tok_est >= half_budget:
            # Truncation falls within thinking — no response visible
            trunc_thinking = _truncate_text_to_tokens(ub_thinking, half_budget)
            trunc_response = ""
        elif ub_thinking:
            # Thinking fits; response gets the remainder
            trunc_thinking = ub_thinking
            remaining = half_budget - thinking_tok_est
            trunc_response = _truncate_text_to_tokens(ub_response, remaining)
        else:
            # No thinking tokens — all budget goes to response
            trunc_thinking = ""
            trunc_response = _truncate_text_to_tokens(ub_response, half_budget)

        # Record the truncated first chunk in conversation
        conversation.append({
            "role": "assistant",
            "content": trunc_response,
            "thinking": trunc_thinking,
            "tokens": half_budget,
            "source": "unbounded_reuse_truncated",
        })

        # Always enter compaction loop — even if the answer is in the truncated text.
        # The point is to test: can the model solve AFTER compaction?
        # The truncated chunk is the "first generation." Now force compaction on it.
        prev_thinking = trunc_thinking
        prev_response = trunc_response
        # Track what user message preceded the current assistant output
        # (for multi-turn compaction — model sees: system + user_msg + assistant + compact_prompt)
        current_user_msg = user_msg

        while n_compactions < max_compactions and n_turns < max_turns:
            n_turns += 1

            # ── Force compaction: model summarizes its output ──────────
            # Build prev_output for prompt variants that embed it (structured, vanilla)
            prev_output = ""
            if prev_thinking:
                prev_output += f"<your_thinking>\n{prev_thinking}\n</your_thinking>\n\n"
            prev_output += f"<your_response>\n{prev_response}\n</your_response>"

            compact_prompt = build_compact_prompt(
                n=n_compactions + 1, prev_output=prev_output,
                variant=prompt_variant,
            )
            conversation.append({"role": "user", "content": compact_prompt})

            # Build multi-turn messages so model can see its own prior output
            # in context (system + current_user_msg + assistant response + compaction prompt).
            # Skip system role entirely when system is empty (e.g. refined3).
            compact_messages = []
            if system:
                compact_messages.append({"role": "system", "content": system})
            compact_messages.extend([
                {"role": "user", "content": current_user_msg},
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": compact_prompt},
            ])

            compact_result = ollama_generate(
                model=model,
                system=system,
                user=compact_prompt,
                messages=compact_messages,
                max_tokens=half_budget,
                temperature=0.3,
            )
            total_wall_time += compact_result["wall_time_s"]
            compact_response = compact_result["response"]
            compact_thinking = compact_result.get("thinking", "")

            conversation.append({
                "role": "assistant",
                "content": compact_response,
                "thinking": compact_thinking,
                "tokens": compact_result["eval_tokens"],
                "thinking_tokens": len(compact_thinking) // 4 if compact_thinking else 0,
                "prompt_tokens": compact_result["prompt_tokens"],
                "total_tokens": compact_result["total_tokens"],
                "pp_tok_s": compact_result.get("pp_tok_s"),
                "tg_tok_s": compact_result.get("tg_tok_s"),
            })

            # Extract summary. Some models (e.g. gemma4) emit everything to
            # `thinking` and leave `content` empty — fall back to thinking if
            # content has no <compact> match.
            compact_match = re.search(r"<compact>(.*?)</compact>", compact_response, re.DOTALL)
            if not compact_match and compact_thinking:
                compact_match = re.search(r"<compact>(.*?)</compact>", compact_thinking, re.DOTALL)
            if compact_match:
                summary = compact_match.group(1).strip()
            elif compact_response.strip():
                summary = compact_response[:500]
            else:
                summary = compact_thinking[:500]

            n_compactions += 1
            conversation.append({
                "role": "user",
                "content": f"[COMPACTION #{n_compactions} — context reset]",
                "forced_compact": True,
            })

            # ── Resume from summary: generate continuation ────────────
            restart_msg = build_resume_prompt(
                user_msg=user_msg, summary=summary, n_done=n_compactions,
                variant=prompt_variant, N=token_limit, tokens_left=half_budget,
            )
            conversation.append({
                "role": "user",
                "content": restart_msg,
                "label": f"[RESTART with summary — fresh budget]",
            })

            resume_result = ollama_generate(
                model=model,
                system=system,
                user=restart_msg,
                max_tokens=half_budget,
                temperature=0.7,
            )
            total_wall_time += resume_result["wall_time_s"]
            resume_response = resume_result["response"]
            resume_thinking = resume_result.get("thinking", "")
            resume_full = resume_thinking + "\n" + resume_response if resume_thinking else resume_response
            last_full_text = resume_full

            conversation.append({
                "role": "assistant",
                "content": resume_response,
                "thinking": resume_thinking,
                "tokens": resume_result["eval_tokens"],
                "thinking_tokens": len(resume_thinking) // 4 if resume_thinking else 0,
                "prompt_tokens": resume_result["prompt_tokens"],
                "total_tokens": resume_result["total_tokens"],
                "pp_tok_s": resume_result.get("pp_tok_s"),
                "tg_tok_s": resume_result.get("tg_tok_s"),
            })

            # Check if model solved after compaction
            if not is_arc:
                boxed = extract_boxed_answer(resume_full)
                if boxed is not None:
                    finish_reason = "solved"
                    break
            else:
                eval_result = evaluate_arc_answer(resume_full, problem["ground_truth"])
                if eval_result["correct"]:
                    finish_reason = "solved"
                    break

            # Not solved — next iteration will compact this output
            prev_thinking = resume_thinking
            prev_response = resume_response
            current_user_msg = restart_msg

        if finish_reason != "solved":
            finish_reason = "max_compactions"

        # Final evaluation
        if is_arc:
            eval_result = evaluate_arc_answer(last_full_text, problem["ground_truth"])
            success = eval_result["correct"]
            answer = None
        else:
            eval_result = evaluate_math_answer(last_full_text, problem["correct_answer"])
            success = eval_result["correct"]
            answer = eval_result["predicted"]

        return {
            "problem_id": problem["problem_id"],
            "correct_answer": problem.get("correct_answer"),
            "token_limit": token_limit,
            "compaction": True,
            "prompt_variant": prompt_variant,
            "trial_idx": trial_idx,
            "success": success,
            "answer": answer,
            "total_tokens_peak": half_budget * (n_compactions * 2 + 1),
            "n_turns": n_turns,
            "n_compactions": n_compactions,
            "wall_time_s": round(total_wall_time, 2),
            "finish_reason": finish_reason,
            "conversation": conversation,
        }

    # ── NON-COMPACTION MODE (or compaction without unbounded data) ────────
    result = ollama_generate(
        model=model,
        system=system,
        user=user_msg,
        max_tokens=max(token_limit, 256) if not compaction else half_budget,
        temperature=0.7,
    )

    total_wall_time = result["wall_time_s"]
    response_text = result["response"]
    thinking_text = result.get("thinking", "")
    full_text = thinking_text + "\n" + response_text if thinking_text else response_text
    thinking_tokens = len(thinking_text) // 4 if thinking_text else 0

    conversation.append({
        "role": "assistant",
        "content": response_text,
        "thinking": thinking_text,
        "tokens": result["eval_tokens"],
        "thinking_tokens": thinking_tokens,
        "prompt_tokens": result["prompt_tokens"],
        "total_tokens": result["total_tokens"],
        "pp_tok_s": result.get("pp_tok_s"),
        "tg_tok_s": result.get("tg_tok_s"),
    })

    if is_arc:
        eval_result = evaluate_arc_answer(full_text, problem["ground_truth"])
        success = eval_result["correct"]
        answer = None
    else:
        eval_result = evaluate_math_answer(full_text, problem["correct_answer"])
        success = eval_result["correct"]
        answer = eval_result["predicted"]

    return {
        "problem_id": problem["problem_id"],
        "correct_answer": problem.get("correct_answer"),
        "token_limit": token_limit,
        "compaction": compaction,
        "trial_idx": trial_idx,
        "success": success,
        "answer": answer,
        "total_tokens_peak": result["total_tokens"],
        "n_turns": 1,
        "n_compactions": 0,
        "wall_time_s": round(total_wall_time, 2),
        "finish_reason": "stop",
        "conversation": conversation,
    }


# ── Binary search ─────────────────────────────────────────────────────────────

def binary_search_window(
    model: str,
    problem: dict,
    compaction: bool = True,
    trials_per_step: int = 3,
    threshold: float = 0.6,  # fraction of trials that must succeed
    initial_window: int = 32768,
    min_window: int = 256,
    verbose: bool = True,
    out_path: Optional[Path] = None,
    unbounded_runs: Optional[list] = None,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> dict:
    """Binary search for minimum context window size."""

    lo = min_window
    hi = initial_window
    steps = []
    min_success_window = None

    # Compute max unbounded tokens — skip binary search steps at windows >= this
    max_ub_tokens = None
    if unbounded_runs:
        ub_toks = [r.get("total_tokens_peak") or r.get("eval_tokens", 0) for r in unbounded_runs]
        ub_toks = [t for t in ub_toks if t and t > 0]
        if ub_toks:
            max_ub_tokens = max(ub_toks)

    # First: verify the model can solve it at max window
    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem: {problem['problem_id']}")
        print(f"Model: {model}")
        print(f"Config: {'Compact' if compaction else 'HardCut'}")
        if max_ub_tokens:
            print(f"Max unbounded tokens: {max_ub_tokens} (skip windows >= this)")
        print(f"{'='*60}")

    while lo < hi:
        mid = (lo + hi) // 2
        # Round to nearest 16 for cleaner numbers
        mid = max(min_window, (mid // 64) * 64)

        # Skip windows >= max unbounded tokens (model solves without compaction at this size)
        if max_ub_tokens and mid >= max_ub_tokens:
            if verbose:
                print(f"\n  Window: {mid} tokens — SKIP (>= unbounded {max_ub_tokens})")
            hi = mid
            min_success_window = mid
            if hi - lo < 64:
                break
            continue

        if verbose:
            print(f"\n  Window: {mid} tokens (range [{lo}, {hi}])")

        trial_results = []
        successes = 0
        for t in range(trials_per_step):
            pid = problem["problem_id"]
            ctx = trial_context(mid, prompt_variant)
            cached = _cache_get(model, pid, t, ctx)
            if cached:
                trial = cached
                if verbose:
                    print(f"    Trial {t+1}/{trials_per_step}... ♻️ cached", end=" ")
            else:
                if verbose:
                    print(f"    Trial {t+1}/{trials_per_step}...", end=" ", flush=True)
                ub_run = None
                if compaction and unbounded_runs:
                    ub_run = unbounded_runs[t % len(unbounded_runs)]
                trial = run_trial(model, problem, mid, compaction, t,
                                  unbounded_run=ub_run, prompt_variant=prompt_variant)
                _cache_put(model, pid, t, trial, ctx)
            trial_results.append(trial)

            if trial["success"]:
                successes += 1
                if verbose:
                    print(f"✅ (answer={trial['answer']}, {trial['wall_time_s']:.1f}s)")
            else:
                if verbose:
                    print(f"❌ (answer={trial['answer']}, {trial['wall_time_s']:.1f}s)")

        success_rate = successes / trials_per_step
        passed = success_rate >= threshold
        steps.append({
            "window": mid,
            "trials": trial_results,
            "n_success": successes,
            "n_trials": trials_per_step,
            "pass_rate": success_rate,
            "passed": passed,
        })

        if verbose:
            print(f"    → {successes}/{trials_per_step} = {success_rate:.0%}")

        if success_rate >= threshold:
            hi = mid
            min_success_window = mid
        else:
            lo = mid + 64  # step up by at least 64

        # Write checkpoint after each step
        if out_path:
            partial = {
                "problem_id": problem["problem_id"],
                "model": model, "model_name": model,
                "config": {"tir": False, "compaction": compaction,
                           "name": "NoTIR_Compact" if compaction else "NoTIR_HardCut"},
                "prediction": {"success_prediction": None, "n_reliable_prediction": None},
                "binary_search": steps,
                "minimum_window": min_success_window,
                "search_range_final": [lo, hi],
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(out_path, "w") as f:
                json.dump(partial, f, indent=2)

        # Convergence check — granularity floor of 64 tokens
        if hi - lo < 64:
            break

    result = {
        "problem_id": problem["problem_id"],
        "model": model,
        "model_name": model,  # Railway API compat
        "config": {
            "tir": False,
            "compaction": compaction,
            "name": "NoTIR_Compact" if compaction else "NoTIR_HardCut",
        },
        "prediction": {
            "success_prediction": None,
            "n_reliable_prediction": None,
            "raw_response": None,
        },
        "binary_search": steps,
        "minimum_window": min_success_window,
        "search_range_final": [lo, hi],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if verbose:
        print(f"\n  Result: minimum window = {min_success_window}")
        print(f"  Final range: [{lo}, {hi}]")

    return result


# ── Prediction ───────────────────────────────────────────────────────────────

def run_prediction(
    model: str,
    problem: dict,
    verbose: bool = True,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> dict:
    """
    Prediction job: model sees the problem but NOT a concrete context limit.
    It predicts whether it can solve the problem and how many tokens it needs.
    300 token output limit.
    """
    topic = problem.get("topic", "math")
    is_arc = topic == "arc"

    if is_arc:
        task_desc = "an ARC (Abstraction and Reasoning Corpus) grid puzzle"
        system = (
            f"You are about to solve {task_desc} under context window constraints.\n"
            "Your context window will be restricted to N tokens.\n"
            "You do NOT know the value of N yet.\n\n"
            "Look at the problem below and answer TWO questions:\n"
            "1. Can you solve this problem? (True/False)\n"
            "2. What is the MINIMUM number of tokens N you need to solve it reliably?\n\n"
            "Make your determination in 300 tokens or less.\n"
            'Output exactly: {attempt: "True_or_False", N: your_N_value_or_0_if_False}'
        )
        user_content = problem["problem_text"]
    else:
        # Use unified prediction prompt (single user message, no system)
        system = ""
        user_content = build_prediction_prompt_variant(
            problem["problem_text"], variant=prompt_variant,
        )

    result = ollama_generate(
        model=model,
        system=system,
        user=user_content,
        max_tokens=300,
        temperature=0.3,
    )

    response = result["response"]

    # Parse attempt and N from response
    attempt = None
    predicted_n = None
    # Try to find {attempt: "...", N: ...}
    attempt_match = re.search(r'attempt["\s:]+(\w+)', response, re.IGNORECASE)
    if attempt_match:
        val = attempt_match.group(1).strip().lower()
        attempt = val in ("true", "yes", "1")

    n_match = re.search(r'["\s]N["\s:]+(\d+)', response)
    if n_match:
        predicted_n = int(n_match.group(1))

    if verbose:
        status = "ATTEMPT" if attempt else "SKIP"
        print(f"  [predict] {status} | predicted_N={predicted_n} | {result['wall_time_s']:.1f}s")

    return {
        "success_prediction": attempt,
        "n_reliable_prediction": predicted_n,
        "raw_response": response,
        "wall_time_s": result["wall_time_s"],
        "total_tokens": result["total_tokens"],
    }


# ── Unbounded multi-trial ────────────────────────────────────────────────────

def run_unbounded(
    model: str,
    problem: dict,
    n_runs: int = 3,
    context_max: int = 131072,
    verbose: bool = True,
    out_path: Optional[Path] = None,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> dict:
    """
    Run n_runs unbounded trials (no context restriction, no compaction).
    Returns Unbounded result JSON compatible with Railway API.
    """
    topic = problem.get("topic", "math")
    is_arc = topic == "arc"

    if is_arc:
        system = (
            "You are solving a pattern transformation puzzle.\n"
            "You will see grids of numbers (0-9). Training examples show input→output transformations.\n"
            "Discover the rule and apply it to the test input.\n"
            "Give each answer inside numbered tags: <answer_1> for test 1. "
            "Use space-separated values, one row per line."
        )
    else:
        system = build_unbounded_system(prompt_variant)

    if verbose:
        print(f"\n  Unbounded: {n_runs} runs for {problem['problem_id']}")

    pid = problem["problem_id"]
    runs = []
    for i in range(n_runs):
        # Check file cache first
        cached = _cache_get(model, pid, i, "Unbounded")
        if cached:
            if verbose:
                status = "PASS" if cached.get("success") else "FAIL"
                print(f"    Run {i+1}/{n_runs}: ♻️ cached {status} | ans={cached.get('answer')}")
            runs.append(cached)
            continue

        t0 = time.time()
        result = ollama_generate(
            model=model,
            system=system,
            user=build_user_message(problem["problem_text"], prompt_variant, N=None),
            max_tokens=context_max,
            temperature=0.7,
        )
        wall_time = time.time() - t0

        response_text = result["response"]
        thinking_text = result.get("thinking", "")
        full_text = thinking_text + "\n" + response_text if thinking_text else response_text
        thinking_tokens = len(thinking_text) // 4 if thinking_text else 0

        if is_arc:
            eval_result = evaluate_arc_answer(full_text, problem["ground_truth"])
            success = eval_result["correct"]
            answer = None
        else:
            eval_result = evaluate_math_answer(full_text, problem["correct_answer"])
            success = eval_result["correct"]
            answer = eval_result["predicted"]

        total_tokens = result["total_tokens"]
        status = "PASS" if success else "FAIL"
        if verbose:
            print(f"    Run {i+1}/{n_runs}: {status} | ans={answer} | {total_tokens} tok | think={thinking_tokens} | {wall_time:.1f}s")

        run_data = {
            "problem_id": pid,
            "model": model,
            "model_name": model,
            "trial_idx": i,
            "context": "Unbounded",
            "success": success,
            "answer": answer,
            "total_tokens": total_tokens,
            "prompt_tokens": result["prompt_tokens"],
            "eval_tokens": result["eval_tokens"],
            "thinking_tokens": thinking_tokens,
            "wall_time_s": round(wall_time, 2),
            "conversation": [
                {"role": "system", "content": system},
                {"role": "user", "content": problem["problem_text"]},
                {
                    "role": "assistant",
                    "content": response_text,
                    "thinking": thinking_text,
                    "tokens": result["eval_tokens"],
                    "thinking_tokens": thinking_tokens,
                    "total_tokens": total_tokens,
                },
            ],
        }
        runs.append(run_data)
        # Save individual trial file
        _cache_put(model, pid, i, run_data, "Unbounded")

        # Write grouped file incrementally so progress is visible
        if out_path:
            token_counts_so_far = [r["total_tokens"] for r in runs]
            successes_so_far = sum(1 for r in runs if r["success"])
            partial = {
                "model_name": model,
                "problem_id": problem["problem_id"],
                "config": "Unbounded",
                "avg_tokens": round(sum(token_counts_so_far) / len(token_counts_so_far)),
                "min_tokens": min(token_counts_so_far),
                "max_tokens": max(token_counts_so_far),
                "solve_rate": successes_so_far / (i + 1),
                "n_runs": i + 1,
                "n_runs_target": n_runs,
                "context_window": context_max,
                "runs": runs,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(out_path, "w") as f:
                json.dump(partial, f, indent=2)

    # Final aggregate
    token_counts = [r["total_tokens"] for r in runs]
    successes = sum(1 for r in runs if r["success"])

    unbounded_result = {
        "model_name": model,
        "problem_id": problem["problem_id"],
        "config": "Unbounded",
        "avg_tokens": round(sum(token_counts) / len(token_counts)) if token_counts else None,
        "min_tokens": min(token_counts) if token_counts else None,
        "max_tokens": max(token_counts) if token_counts else None,
        "solve_rate": successes / n_runs if n_runs > 0 else 0,
        "n_runs": n_runs,
        "context_window": context_max,
        "runs": runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(unbounded_result, f, indent=2)

    if verbose:
        print(f"    Avg: {unbounded_result['avg_tokens']} tok | "
              f"Rate: {successes}/{n_runs} = {unbounded_result['solve_rate']:.0%}")

    return unbounded_result


# ── Sweep: test compaction at multiple truncation points ──────────────────────

def compaction_sweep(
    model: str,
    problem: dict,
    unbounded_runs: list,
    sweep_points: Optional[list] = None,
    trials_per_point: int = 3,
    verbose: bool = True,
    out_path: Optional[Path] = None,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    context_window: int = 131072,
) -> dict:
    """
    Sweep compaction effectiveness at multiple truncation points.

    Instead of binary search, tests compaction at logarithmically-spaced
    token counts using the unbounded runs as the base. Gives the full
    curve of "can the model solve after compaction at N tokens?"

    For each truncation point N:
      1. Truncate unbounded run at N tokens (free — just string slice)
      2. Force compaction: model summarizes the truncated output
      3. Resume from summary: model continues solving
      4. Check if model gets the right answer

    Returns the full curve: [{point, pass_rate, trials}]
    """
    pid = problem["problem_id"]
    topic = problem.get("topic", "math")
    is_arc = topic == "arc"

    # Determine max tokens from unbounded runs
    max_tokens = 0
    for run in unbounded_runs:
        max_tokens = max(max_tokens, run.get("total_tokens", 0))
    if max_tokens == 0:
        max_tokens = 8192  # fallback

    # Estimate prompt size (system + problem text, ~4 chars/token)
    prompt_tokens = (len(problem.get("problem_text", "")) + 200) // 4

    # Default sweep points: powers of 2 from cw//2 down to 128
    # Based on the model's actual context window.
    if sweep_points is None:
        points = []
        p = context_window // 2
        while p >= 128:
            # Round to 64-token granularity
            p = max(128, (p // 64) * 64)
            points.append(p)
            p = p // 2
        sweep_points = points

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SWEEP: {model} / {pid}")
        print(f"  Points: {sweep_points}")
        print(f"  Trials per point: {trials_per_point}")
        print(f"  Unbounded runs: {len(unbounded_runs)}")
        print(f"{'='*60}")

    results = []
    total_wall_time = 0.0

    for pi, point in enumerate(sweep_points):
        if verbose:
            print(f"\n  Point {pi+1}/{len(sweep_points)}: truncate at {point} tokens")

        trial_results = []
        successes = 0

        window = point
        for t in range(trials_per_point):
            ctx = trial_context(window, prompt_variant)
            cached = _cache_get(model, pid, t, ctx)
            if cached:
                trial = cached
                if verbose:
                    status = "PASS" if trial["success"] else "FAIL"
                    print(f"    Trial {t+1}: ♻️ cached {status} | ans={trial['answer']}")
            else:
                ub_run = unbounded_runs[t % len(unbounded_runs)]
                trial = run_trial(
                    model=model,
                    problem=problem,
                    token_limit=window,
                    compaction=True,
                    trial_idx=t,
                    unbounded_run=ub_run,
                    prompt_variant=prompt_variant,
                )
                _cache_put(model, pid, t, trial, ctx)
            trial_results.append(trial)
            total_wall_time += trial["wall_time_s"]

            if trial["success"]:
                successes += 1
            if verbose and not cached:
                status = "PASS" if trial["success"] else "FAIL"
                print(f"    Trial {t+1}: {status} | ans={trial['answer']} | "
                      f"compactions={trial['n_compactions']} | {trial['wall_time_s']:.1f}s")

        pass_rate = successes / trials_per_point
        step = {
            "truncation_point": point // 2,
            "window": point,
            "trials": trial_results,
            "n_success": successes,
            "n_trials": trials_per_point,
            "pass_rate": pass_rate,
            "passed": pass_rate >= 0.6,
        }
        results.append(step)

        if verbose:
            print(f"    → {successes}/{trials_per_point} = {pass_rate:.0%}")

        # Write incrementally
        if out_path:
            sweep_result = _build_sweep_result(model, pid, results, sweep_points, total_wall_time)
            with open(out_path, "w") as f:
                json.dump(sweep_result, f, indent=2)

    sweep_result = _build_sweep_result(model, pid, results, sweep_points, total_wall_time)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(sweep_result, f, indent=2)

    if verbose:
        passing = [r for r in results if r["passed"]]
        min_point = min(r["truncation_point"] for r in passing) if passing else None
        print(f"\n  SWEEP DONE: {len(passing)}/{len(results)} points passed")
        print(f"  Min passing truncation: {min_point}")

    return sweep_result


def _build_sweep_result(model, pid, results, sweep_points, wall_time):
    passing = [r for r in results if r["passed"]]
    return {
        "problem_id": pid,
        "model": model,
        "model_name": model,
        "config": {"name": "Sweep", "type": "compaction_sweep"},
        "sweep_points": sweep_points,
        "sweep_results": results,
        "min_passing_truncation": min(r["truncation_point"] for r in passing) if passing else None,
        "pass_curve": [{"point": r["truncation_point"], "rate": r["pass_rate"]} for r in results],
        "wall_time_s": round(wall_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Where Is The Result: find earliest token position of \boxed{} ─────────────

def where_is_the_result(unbounded_runs: list, problem: dict) -> dict:
    """
    For each unbounded run, find the earliest token position where
    \\boxed{answer} appears in the output (thinking + response).

    This is free — no LLM calls, just string search on existing text.
    Tells us: how early in the reasoning does the model produce the answer?

    Returns:
    {
        "problem_id": str,
        "correct_answer": any,
        "runs": [
            {
                "trial_idx": int,
                "answer_token_position": int or null,  # token position of \\boxed{}
                "answer_char_position": int or null,    # char position
                "total_tokens": int,
                "answer_found": bool,
                "answer_correct": bool,
                "answer_value": any,
                "pct_of_total": float or null,  # what % through the output
            }
        ]
    }
    """
    pid = problem["problem_id"]
    correct = problem.get("correct_answer") or problem.get("ground_truth")
    results = []

    for i, run in enumerate(unbounded_runs):
        conv = run.get("conversation", [])
        thinking = ""
        response = ""
        for msg in conv:
            if msg.get("role") == "assistant":
                thinking = msg.get("thinking", "")
                response = msg.get("content", "")
                break

        full_text = thinking + "\n" + response if thinking else response
        total_tokens = run.get("total_tokens", len(full_text) // 4)

        # Find all \boxed{...} occurrences
        import re as _re
        matches = list(_re.finditer(r"\\boxed\{([^}]+)\}", full_text))

        if matches:
            # First \boxed{} — earliest point model produces an answer
            first_match = matches[0]
            first_char = first_match.start()
            first_tok = first_char // 4
            first_val = first_match.group(1).strip()
            try:
                first_int = int(first_val.replace(",", ""))
            except ValueError:
                first_int = first_val
            first_correct = str(first_int) == str(correct) if correct is not None else None
            first_pct = round(first_tok / max(total_tokens, 1) * 100, 1)

            # Last \boxed{} — final answer (what extract_boxed_answer returns)
            last_match = matches[-1]
            last_char = last_match.start()
            last_tok = last_char // 4
            last_val = last_match.group(1).strip()
            try:
                last_int = int(last_val.replace(",", ""))
            except ValueError:
                last_int = last_val
            last_correct = str(last_int) == str(correct) if correct is not None else None
            last_pct = round(last_tok / max(total_tokens, 1) * 100, 1)

            results.append({
                "trial_idx": i,
                "total_tokens": total_tokens,
                "answer_found": True,
                "n_boxed": len(matches),
                "first_answer": first_int,
                "first_token_pos": first_tok,
                "first_pct": first_pct,
                "first_correct": first_correct,
                "last_answer": last_int,
                "last_token_pos": last_tok,
                "last_pct": last_pct,
                "last_correct": last_correct,
            })
        else:
            results.append({
                "trial_idx": i,
                "total_tokens": total_tokens,
                "answer_found": False,
                "n_boxed": 0,
                "first_answer": None, "first_token_pos": None, "first_pct": None, "first_correct": None,
                "last_answer": None, "last_token_pos": None, "last_pct": None, "last_correct": None,
            })

    return {
        "problem_id": pid,
        "correct_answer": correct,
        "runs": results,
    }


# ── Full pipeline: predict + unbounded + compact ─────────────────────────────

def run_full_pipeline(
    model: str,
    problem: dict,
    results_dir: Path,
    unbounded_runs: int = 3,
    trials_per_step: int = 3,
    threshold: float = 0.6,
    initial_window: int = 32768,
    context_max: int = 131072,
    verbose: bool = True,
) -> None:
    """Run prediction + unbounded + compact binary search for one (model, problem) pair."""
    model_safe = model.replace("/", "_").replace(":", "_")
    pid = problem["problem_id"]
    results_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {model} / {pid}")
    print(f"{'='*60}")

    # 1. Prediction
    pred_path = results_dir / f"{model_safe}_{pid}_prediction.json"
    if pred_path.exists():
        print(f"  [predict] SKIP — already exists: {pred_path.name}")
        with open(pred_path) as f:
            prediction = json.load(f)
    else:
        prediction = run_prediction(model, problem, verbose=verbose)
        with open(pred_path, "w") as f:
            json.dump(prediction, f, indent=2)
        print(f"  [predict] Saved: {pred_path.name}")

    # 2. Unbounded (3 runs) — writes incrementally after each run
    ub_path = results_dir / f"{model_safe}_{pid}_Unbounded.json"
    if ub_path.exists():
        print(f"  [unbounded] SKIP — already exists: {ub_path.name}")
    else:
        run_unbounded(model, problem, n_runs=unbounded_runs,
                      context_max=context_max, verbose=verbose, out_path=ub_path)
        print(f"  [unbounded] Saved: {ub_path.name}")

    # 3. Compact binary search — writes checkpoint after each step
    config_name = "NoTIR_Compact"
    co_path = results_dir / f"{model_safe}_{pid}_{config_name}.json"
    if co_path.exists():
        print(f"  [compact] SKIP — already exists: {co_path.name}")
    else:
        co_result = binary_search_window(
            model=model,
            problem=problem,
            compaction=True,
            trials_per_step=trials_per_step,
            threshold=threshold,
            initial_window=initial_window,
            verbose=verbose,
            out_path=co_path,
        )
        # Attach prediction to compact result
        co_result["prediction"] = {
            "success_prediction": prediction.get("success_prediction"),
            "n_reliable_prediction": prediction.get("n_reliable_prediction"),
            "raw_response": prediction.get("raw_response"),
        }
        with open(co_path, "w") as f:
            json.dump(co_result, f, indent=2)
        print(f"  [compact] Saved: {co_path.name}")

    # 4. Summary — read compact result back if we skipped it
    co_data = None
    if co_path.exists():
        with open(co_path) as f:
            co_data = json.load(f)

    sfn = f"{model_safe}_{pid}_summary.json"
    sout = results_dir / sfn
    existing = []
    if sout.exists():
        with open(sout) as f:
            existing = json.load(f)
    existing_configs = {e.get("config") for e in existing}
    if config_name not in existing_configs and co_data:
        existing.append({
            "model_name": model,
            "problem_id": pid,
            "config": config_name,
            "minimum_window": co_data.get("minimum_window"),
            "search_range_final": co_data.get("search_range_final"),
            "n_reliable_prediction": prediction.get("n_reliable_prediction"),
            "success_prediction": prediction.get("success_prediction"),
        })
        with open(sout, "w") as f:
            json.dump(existing, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench Ollama Runner")
    parser.add_argument("--model", default="qwen3:32b", help="Ollama model name")
    parser.add_argument("--problem", help="Specific problem ID to run")
    parser.add_argument("--problem-type", help="Run all problems of a type (arc, aimo3, math)")
    parser.add_argument("--max-problems", type=int, default=999, help="Max problems to run")
    parser.add_argument("--trials", type=int, default=3, help="Trials per binary search step")
    parser.add_argument("--unbounded-runs", type=int, default=3, help="Number of unbounded runs")
    parser.add_argument("--threshold", type=float, default=0.6, help="Success threshold")
    parser.add_argument("--initial-window", type=int, default=32768, help="Starting window size")
    parser.add_argument("--context-max", type=int, default=131072, help="Max context for unbounded runs")
    parser.add_argument("--no-compact", action="store_true", help="Disable compaction (HardCut mode)")
    parser.add_argument("--list-problems", action="store_true", help="List available problems")
    parser.add_argument("--single-shot", action="store_true", help="Just run one trial, no binary search")
    parser.add_argument("--window", type=int, default=32768, help="Token window for single-shot")
    parser.add_argument("--full", action="store_true",
                        help="Full pipeline: predict + 3 unbounded + compact binary search")
    parser.add_argument("--predict-only", action="store_true", help="Only run prediction job")
    parser.add_argument("--unbounded-only", action="store_true", help="Only run unbounded trials")

    args = parser.parse_args()
    all_problems = load_all_problems()

    if args.list_problems:
        print(f"Available problems ({len(all_problems)}):\n")
        by_topic = {}
        for pid, p in sorted(all_problems.items()):
            topic = p.get("topic", "unknown")
            by_topic.setdefault(topic, []).append(pid)
        for topic in sorted(by_topic):
            print(f"  [{topic}]")
            for pid in by_topic[topic]:
                print(f"    {pid}")
        return

    # Select problems
    problems = []
    if args.problem:
        if args.problem not in all_problems:
            print(f"Unknown problem: {args.problem}")
            print(f"Use --list-problems to see available problems")
            sys.exit(1)
        problems = [all_problems[args.problem]]
    elif args.problem_type:
        for pid, p in sorted(all_problems.items()):
            if p.get("topic", "") == args.problem_type or pid.startswith(args.problem_type):
                problems.append(p)
        problems = problems[:args.max_problems]
    else:
        print("Specify --problem or --problem-type (or --list-problems)")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Problems: {len(problems)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    model_safe = args.model.replace("/", "_").replace(":", "_")

    # ── Full pipeline mode ────────────────────────────────────────────────
    if args.full:
        print(f"Mode: FULL PIPELINE (predict + {args.unbounded_runs} unbounded + compact)")
        for p in problems:
            run_full_pipeline(
                model=args.model,
                problem=p,
                results_dir=RESULTS_DIR,
                unbounded_runs=args.unbounded_runs,
                trials_per_step=args.trials,
                threshold=args.threshold,
                initial_window=args.initial_window,
                context_max=args.context_max,
            )
        print(f"\nAll done. Results in {RESULTS_DIR}/")
        return

    # ── Predict-only mode ─────────────────────────────────────────────────
    if args.predict_only:
        print("Mode: PREDICT ONLY")
        for p in problems:
            pred = run_prediction(args.model, p)
            fn = f"{model_safe}_{p['problem_id']}_prediction.json"
            out_path = RESULTS_DIR / fn
            with open(out_path, "w") as f:
                json.dump(pred, f, indent=2)
            print(f"  Saved: {out_path}")
        return

    # ── Unbounded-only mode ───────────────────────────────────────────────
    if args.unbounded_only:
        print(f"Mode: UNBOUNDED ONLY ({args.unbounded_runs} runs)")
        for p in problems:
            ub = run_unbounded(args.model, p, n_runs=args.unbounded_runs,
                               context_max=args.context_max)
            fn = f"{model_safe}_{p['problem_id']}_Unbounded.json"
            out_path = RESULTS_DIR / fn
            with open(out_path, "w") as f:
                json.dump(ub, f, indent=2)
            print(f"  Saved: {out_path}")
        return

    # ── Single-shot mode ──────────────────────────────────────────────────
    if args.single_shot:
        compaction = not args.no_compact
        for p in problems:
            print(f"\n--- {p['problem_id']} ---")
            trial = run_trial(args.model, p, args.window, compaction, 0)
            print(f"  Success: {trial['success']}")
            print(f"  Answer: {trial['answer']}")
            print(f"  Tokens: {trial['total_tokens_peak']}")
            print(f"  Time: {trial['wall_time_s']:.1f}s")
        return

    # ── Binary search only (legacy default) ───────────────────────────────
    compaction = not args.no_compact
    print(f"Config: {'HardCut' if args.no_compact else 'Compact'}")

    for p in problems:
        result = binary_search_window(
            model=args.model,
            problem=p,
            compaction=compaction,
            trials_per_step=args.trials,
            threshold=args.threshold,
            initial_window=args.initial_window,
        )

        config_name = "NoTIR_Compact" if compaction else "NoTIR_HardCut"
        fn = f"{model_safe}_{p['problem_id']}_{config_name}.json"
        out_path = RESULTS_DIR / fn
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out_path}")

        # Summary
        sfn = f"{model_safe}_{p['problem_id']}_summary.json"
        sout = RESULTS_DIR / sfn
        existing = []
        if sout.exists():
            with open(sout) as f:
                existing = json.load(f)
        existing.append({
            "model_name": args.model,
            "problem_id": p["problem_id"],
            "config": config_name,
            "minimum_window": result["minimum_window"],
            "search_range_final": result["search_range_final"],
            "steps": len(result["binary_search"]),
        })
        with open(sout, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Summary: {sout}")


if __name__ == "__main__":
    main()
