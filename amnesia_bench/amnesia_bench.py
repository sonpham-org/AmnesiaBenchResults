#!/usr/bin/env python3
"""
AmnesiaBench — How much context does a model actually need?

Binary-searches (log scale) for the minimum context window at which an LLM
can solve competition-math problems at a 20% success rate.

4 configurations: {TIR, No-TIR} x {Hard Cutoff, Compaction}
5 trials per window size. Full conversation traces saved.

Usage:
    # Start llama.cpp server first, then:
    python3 amnesia_bench.py --problem ab507a9f
    python3 amnesia_bench.py --all
    python3 amnesia_bench.py --analyze
"""

import argparse
import contextlib
import glob
import io
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ─── Defaults ────────────────────────────────────────────────────────────────

SERVER_URL = "http://localhost:8080"
MIN_WINDOW = 512
MAX_WINDOW = 32768
TRIALS_PER_WINDOW = 5
SUCCESS_THRESHOLD = 0.2          # 1/5 = 20%
CONVERGENCE_RATIO = 1.05         # stop when hi/lo < 5%
MAX_COMPACTIONS = 5
COMPACTION_TRIGGER = 0.70        # compact at 70% of budget
MAX_TURNS = 40                   # safety: max conversation turns
CODE_TIMEOUT = 30                # seconds per code execution
TEMPERATURE = 0.7
PROBLEMS_DIR = Path(__file__).parent / "problems"
RESULTS_DIR = Path(__file__).parent / "results"

# ─── Prompt Templates ────────────────────────────────────────────────────────

SYSTEM_HARD = """\
You are a mathematical problem solver.
Your context window is {token_limit} tokens total (this prompt + your output).
If you run out, generation stops and I take your last \\boxed{{}} answer.
You do not have access to any tools. Reason through the problem using only text.
Plan your reasoning to fit. Give your final answer as \\boxed{{integer}}."""

SYSTEM_COMPACT = """\
You are a mathematical problem solver.
Your context window is {token_limit} tokens total. If you exceed it without compacting, you FAIL with score 0.
You do not have access to any tools except compact.

To compact, write:
<compact>your summary here</compact>

When you call compact, the conversation resets to:
  [this system prompt] + [the problem] + [your summary]
You get a fresh {token_limit} budget, but the reset prompt eats into it.
The compact call itself costs tokens. You may compact at most {max_compactions} times.

Give your final answer as \\boxed{{integer}}."""

POST_COMPACT_USER = """\
{problem_text}

Your previous progress (from compact call):
---
{summary}
---
Continue solving. Give your final answer as \\boxed{{integer}}."""


# ─── Python Sandbox ──────────────────────────────────────────────────────────

class PythonSandbox:
    """In-process Python executor with persistent namespace."""

    def __init__(self, timeout: int = CODE_TIMEOUT):
        self.timeout = timeout
        self.namespace = {"__builtins__": __builtins__}

    def execute(self, code: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()

        def _alarm_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {self.timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(self.timeout)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, self.namespace)
            out = stdout.getvalue()
            err = stderr.getvalue()
            result = out if out else "(no output)"
            if err:
                result += f"\nSTDERR: {err}"
            return result
        except TimeoutError as e:
            return f"Error: {e}"
        except Exception:
            return f"Error: {traceback.format_exc()}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def reset(self):
        self.namespace = {"__builtins__": __builtins__}


# ─── LLM Client ─────────────────────────────────────────────────────────────

class LLMClient:
    """Wrapper for llama.cpp /v1/chat/completions."""

    def __init__(self, server_url: str = SERVER_URL, temperature: float = TEMPERATURE):
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature

    def generate(self, messages: list[dict], max_tokens: int) -> dict:
        """
        Send messages to the model. Returns:
        {
            "content": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "finish_reason": str,
        }
        """
        max_tokens = max(1, max_tokens)
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=3600,  # 1 hour — large windows with parallel slots are slow
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        msg = choice["message"]
        # Qwen3.5 splits thinking into reasoning_content, final answer into content
        reasoning = msg.get("reasoning_content", "") or ""
        content = msg.get("content", "") or ""
        # Combine both for our purposes — the model's full output
        full_content = ""
        if reasoning:
            full_content += f"<think>\n{reasoning}\n</think>\n"
        full_content += content

        return {
            "content": full_content,
            "reasoning_content": reasoning,
            "final_content": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "finish_reason": choice.get("finish_reason", "unknown"),
        }

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ─── Parsing Helpers ─────────────────────────────────────────────────────────

def extract_python_blocks(text: str) -> list[str]:
    """Extract all ```python code blocks from text."""
    pattern = r"```python\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def extract_compact_call(text: str) -> Optional[str]:
    """Extract <compact>...</compact> summary. Returns None if not found."""
    match = re.search(r"<compact>(.*?)</compact>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_boxed_answer(text: str) -> Optional[int]:
    """Extract the last \\boxed{...} answer from text, ignoring <think> blocks."""
    # Try outside <think> blocks first
    non_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    target = non_think if non_think.strip() else text

    matches = re.findall(r"\\boxed\{([^{}]+)\}", target)
    if not matches:
        # Fallback: try nested braces
        matches = re.findall(r"\\boxed\{(.+?)\}", target)
    if not matches:
        return None

    raw = matches[-1].strip()
    # Try direct int parse
    try:
        return int(raw)
    except ValueError:
        pass
    # Try float -> int
    try:
        f = float(raw)
        if f == int(f):
            return int(f)
    except ValueError:
        pass
    # Try simple eval (e.g. "2^10" or "3*5")
    try:
        cleaned = raw.replace("^", "**").replace(",", "")
        return int(eval(cleaned))
    except Exception:
        pass
    return None


# ─── Single Trial (one attempt at solving a problem) ─────────────────────────

@dataclass
class Turn:
    """One turn in the conversation."""
    role: str                   # "system", "user", "assistant"
    content: str
    tokens: Optional[int] = None          # completion_tokens (assistant only)
    prompt_tokens: Optional[int] = None   # prompt_tokens at this point
    total_tokens: Optional[int] = None    # total context at this point
    finish_reason: Optional[str] = None
    code_executed: Optional[str] = None   # code that was run (if any)
    code_output: Optional[str] = None     # output from code (if any)
    compact_summary: Optional[str] = None # summary extracted (if compact call)


@dataclass
class TrialResult:
    """Full result of one trial."""
    problem_id: str
    correct_answer: int
    token_limit: int
    tir: bool
    compaction: bool
    trial_idx: int
    success: bool
    answer: Optional[int]
    total_tokens_peak: int      # peak total_tokens seen
    n_turns: int
    n_compactions: int
    n_code_calls: int
    n_code_errors: int
    wall_time_s: float
    error: Optional[str]
    finish_reason: str          # "solved", "truncated", "budget_exceeded", "max_turns", "error"
    conversation: list = field(default_factory=list)  # list of Turn dicts


def run_trial(
    client: LLMClient,
    problem_id: str,
    problem_text: str,
    correct_answer: int,
    token_limit: int,
    tir: bool,
    compaction: bool,
    trial_idx: int,
) -> TrialResult:
    """Run one trial: try to solve the problem within the token budget."""

    t0 = time.time()
    sandbox = PythonSandbox() if tir else None
    conversation: list[Turn] = []  # full trace
    messages: list[dict] = []      # current API messages
    n_compactions = 0
    n_code_calls = 0
    n_code_errors = 0
    peak_tokens = 0
    last_content = ""
    error_msg = None
    finish = "max_turns"

    # Select system prompt
    if compaction:
        sys_prompt = SYSTEM_COMPACT.format(
            token_limit=token_limit, max_compactions=MAX_COMPACTIONS
        )
    else:
        sys_prompt = SYSTEM_HARD.format(token_limit=token_limit)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": problem_text},
    ]
    conversation.append(Turn(role="system", content=sys_prompt))
    conversation.append(Turn(role="user", content=problem_text))

    for turn_i in range(MAX_TURNS):
        # Calculate remaining budget
        # We estimate prompt_tokens from the last known total.
        # On the first call, we don't know yet — use a generous max_tokens.
        if peak_tokens > 0:
            estimated_prompt = peak_tokens  # last total ≈ next prompt
            remaining = token_limit - estimated_prompt
        else:
            remaining = token_limit  # first call, let API figure it out

        if remaining <= 0:
            if compaction:
                finish = "budget_exceeded"
            else:
                finish = "truncated"
            break

        # Generate
        try:
            resp = client.generate(messages, max_tokens=remaining)
        except Exception as e:
            error_msg = f"API error: {e}"
            finish = "error"
            break

        content = resp["content"]
        total_now = resp["total_tokens"]
        peak_tokens = max(peak_tokens, total_now)
        last_content = content

        turn = Turn(
            role="assistant",
            content=content,
            tokens=resp["completion_tokens"],
            prompt_tokens=resp["prompt_tokens"],
            total_tokens=total_now,
            finish_reason=resp["finish_reason"],
        )

        conversation.append(turn)

        # ── Check for boxed answer FIRST (highest priority) ──
        answer = extract_boxed_answer(content)
        if answer is not None:
            finish = "solved"
            break

        # ── Check for compact call ──
        compact_summary = extract_compact_call(content) if compaction else None
        if compact_summary is not None:
            turn.compact_summary = compact_summary
            n_compactions += 1

            if n_compactions > MAX_COMPACTIONS:
                finish = "max_compactions"
                break

            # Reset conversation with summary
            restart_user_msg = POST_COMPACT_USER.format(
                problem_text=problem_text,
                summary=compact_summary,
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": restart_user_msg},
            ]
            # Reset peak tracking for new window
            peak_tokens = 0
            conversation.append(Turn(
                role="user",
                content=f"[COMPACTION #{n_compactions} — context reset]",
            ))
            continue

        # ── Check budget exceeded (compaction mode = hard fail) ──
        if total_now >= token_limit:
            if compaction:
                finish = "budget_exceeded"
            else:
                finish = "truncated"
            break

        # ── Check for python code blocks (TIR mode) ──
        code_blocks = extract_python_blocks(content) if tir else []
        if code_blocks:
            # Execute ALL code blocks in order (variables persist)
            all_outputs = []
            for code in code_blocks:
                n_code_calls += 1
                output = sandbox.execute(code)
                if output.startswith("Error:"):
                    n_code_errors += 1
                all_outputs.append(output)
            combined_output = "\n---\n".join(all_outputs)

            # Truncate long output
            if len(combined_output) > 2000:
                combined_output = combined_output[:2000] + "\n... (truncated)"

            code_turn = Turn(
                role="user",
                content=f"Code output:\n{combined_output}",
                code_executed="\n---\n".join(code_blocks),
                code_output=combined_output,
            )
            conversation.append(code_turn)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Code output:\n{combined_output}"})
            continue

        # ── No code, no answer — prompt to continue ──
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Continue solving."})
        conversation.append(Turn(role="user", content="Continue solving."))

    # ── Extract answer ──
    # Try to find answer from the full conversation
    answer = None
    for t in reversed(conversation):
        if t.role == "assistant":
            answer = extract_boxed_answer(t.content)
            if answer is not None:
                break

    elapsed = time.time() - t0
    success = answer is not None and answer == correct_answer

    return TrialResult(
        problem_id=problem_id,
        correct_answer=correct_answer,
        token_limit=token_limit,
        tir=tir,
        compaction=compaction,
        trial_idx=trial_idx,
        success=success,
        answer=answer,
        total_tokens_peak=peak_tokens,
        n_turns=len([t for t in conversation if t.role == "assistant"]),
        n_compactions=n_compactions,
        n_code_calls=n_code_calls,
        n_code_errors=n_code_errors,
        wall_time_s=round(elapsed, 2),
        error=error_msg,
        finish_reason=finish,
        conversation=[asdict(t) for t in conversation],
    )


# ─── Binary Search ───────────────────────────────────────────────────────────

@dataclass
class WindowTest:
    """Result of testing one window size."""
    window: int
    trials: list  # list of TrialResult dicts
    n_success: int
    n_trials: int
    pass_rate: float
    passed: bool


def binary_search(
    client: LLMClient,
    problem_id: str,
    problem_text: str,
    correct_answer: int,
    tir: bool,
    compaction: bool,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    trials: int = TRIALS_PER_WINDOW,
) -> dict:
    """
    Binary search (log scale) for minimum context window.
    Returns full results dict with all trials.
    """
    config_name = f"{'Compact' if compaction else 'HardCut'}"
    print(f"\n{'='*60}")
    print(f"  {problem_id} | {config_name}")
    print(f"  Search range: [{min_window}, {max_window}]")
    print(f"{'='*60}")

    search_log: list[WindowTest] = []

    # First: verify solvable at max window
    print(f"\n  [Verify] Testing max window = {max_window} ...")
    test = _test_window(
        client, problem_id, problem_text, correct_answer,
        max_window, tir, compaction, trials
    )
    search_log.append(test)
    print(f"  [Verify] {test.n_success}/{test.n_trials} passed ({test.pass_rate:.0%})")

    if not test.passed:
        print(f"  UNSOLVABLE at max window. Skipping binary search.")
        return _build_result(
            problem_id, tir, compaction, search_log,
            minimum_window=None,
            search_range_final=(min_window, max_window),
        )

    # Binary search
    lo, hi = min_window, max_window
    step = 0
    while hi / lo > CONVERGENCE_RATIO:
        step += 1
        mid = int(math.exp((math.log(lo) + math.log(hi)) / 2))
        # Snap to multiples of 64 for cleanliness
        mid = max(min_window, (mid // 64) * 64)

        # Avoid re-testing same values
        if mid == lo or mid == hi:
            break

        print(f"\n  [Step {step}] Testing window = {mid}  (range [{lo}, {hi}], ratio {hi/lo:.3f})")
        test = _test_window(
            client, problem_id, problem_text, correct_answer,
            mid, tir, compaction, trials
        )
        search_log.append(test)
        print(f"  [Step {step}] {test.n_success}/{test.n_trials} passed ({test.pass_rate:.0%}) → {'hi=mid' if test.passed else 'lo=mid'}")

        if test.passed:
            hi = mid
        else:
            lo = mid

    print(f"\n  RESULT: minimum window ≈ {hi} tokens (range [{lo}, {hi}])")

    return _build_result(
        problem_id, tir, compaction, search_log,
        minimum_window=hi,
        search_range_final=(lo, hi),
    )


def _test_window(
    client, problem_id, problem_text, correct_answer,
    window, tir, compaction, n_trials,
) -> WindowTest:
    """Run N trials at a given window size, in parallel."""
    t0 = time.time()

    def _run_one(i):
        return run_trial(
            client, problem_id, problem_text, correct_answer,
            token_limit=window,
            tir=tir,
            compaction=compaction,
            trial_idx=i,
        )

    # Run all trials in parallel (server has enough slots)
    trials_results = [None] * n_trials
    n_success = 0
    with ThreadPoolExecutor(max_workers=n_trials) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n_trials)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            trials_results[i] = asdict(result)
            if result.success:
                n_success += 1
            status = "OK" if result.success else "FAIL"
            ans_str = f"ans={result.answer}" if result.answer is not None else "no answer"
            print(f"    trial {i}: {status} | {ans_str} | {result.finish_reason} | {result.total_tokens_peak} tok | {result.wall_time_s}s")

    elapsed = time.time() - t0
    pass_rate = n_success / n_trials
    print(f"    [{n_trials} trials in {elapsed:.1f}s wall, {n_success}/{n_trials} passed]")
    return WindowTest(
        window=window,
        trials=trials_results,
        n_success=n_success,
        n_trials=n_trials,
        pass_rate=pass_rate,
        passed=pass_rate >= SUCCESS_THRESHOLD,
    )


def _build_result(problem_id, tir, compaction, search_log, minimum_window, search_range_final):
    return {
        "problem_id": problem_id,
        "config": {
            "tir": tir,
            "compaction": compaction,
            "name": f"{'TIR' if tir else 'NoTIR'}_{'Compact' if compaction else 'HardCut'}",
        },
        "binary_search": [asdict(w) for w in search_log],
        "minimum_window": minimum_window,
        "search_range_final": list(search_range_final),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def load_problem(problem_id: str) -> dict:
    """Load a problem JSON from the problems/ directory."""
    # Try exact match
    path = PROBLEMS_DIR / f"{problem_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    # Try fuzzy match (e.g., "ab507a9f" matches "aimo3_hard_ab507a9f.json")
    for p in PROBLEMS_DIR.glob("*.json"):
        if problem_id in p.stem:
            return json.loads(p.read_text())
    raise FileNotFoundError(f"No problem matching '{problem_id}' in {PROBLEMS_DIR}")


def load_all_problems() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(PROBLEMS_DIR.glob("*.json"))]


def run_problem(
    client: LLMClient,
    problem: dict,
    configs: list[tuple[bool, bool]] = None,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    trials: int = TRIALS_PER_WINDOW,
):
    """Run binary search for all configs on one problem. Save results."""
    if configs is None:
        configs = [
            (False, False),  # Hard Cutoff (no tools)
            (False, True),   # Compaction (compact tool only)
        ]

    pid = problem["problem_id"]
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    for tir, compaction in configs:
        config_name = f"{'Compact' if compaction else 'HardCut'}"
        result = binary_search(
            client,
            problem_id=pid,
            problem_text=problem["problem_text"],
            correct_answer=problem["ground_truth"],
            tir=tir,
            compaction=compaction,
            min_window=min_window,
            max_window=max_window,
            trials=trials,
        )
        result["model"] = "Qwen3.5-35B-A3B-Q4_K_M"
        all_results.append(result)

        # Save per-config result (with full traces)
        outpath = RESULTS_DIR / f"{pid}_{config_name}.json"
        outpath.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Saved: {outpath}")

    # Save combined summary (without conversation traces for readability)
    summary = []
    for r in all_results:
        summary.append({
            "problem_id": r["problem_id"],
            "config": r["config"]["name"],
            "minimum_window": r["minimum_window"],
            "search_range_final": r["search_range_final"],
            "steps": len(r["binary_search"]),
        })

    summary_path = RESULTS_DIR / f"{pid}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary: {summary_path}")

    return all_results


def analyze_results():
    """Print a summary table of all completed results."""
    results_files = sorted(RESULTS_DIR.glob("*_summary.json"))
    if not results_files:
        print("No results found. Run experiments first.")
        return

    print(f"\n{'Problem':<30} {'Config':<20} {'Min Window':>10} {'Range':>16}")
    print("-" * 80)
    for f in results_files:
        data = json.loads(f.read_text())
        for entry in data:
            lo, hi = entry["search_range_final"]
            mw = entry["minimum_window"]
            mw_str = str(mw) if mw is not None else "UNSOLVABLE"
            print(f"{entry['problem_id']:<30} {entry['config']:<20} {mw_str:>10} [{lo:>6}, {hi:>6}]")


def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench — context window binary search")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", type=str, help="Problem ID (or substring) to test")
    group.add_argument("--all", action="store_true", help="Run all problems")
    group.add_argument("--analyze", action="store_true", help="Analyze existing results")

    parser.add_argument("--server", type=str, default=SERVER_URL, help="llama.cpp server URL")
    parser.add_argument("--min-window", type=int, default=MIN_WINDOW)
    parser.add_argument("--max-window", type=int, default=MAX_WINDOW)
    parser.add_argument("--trials", type=int, default=TRIALS_PER_WINDOW)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--config", type=str, default=None,
                        help="Run specific config only: NoTIR_HardCut, TIR_HardCut, NoTIR_Compact, TIR_Compact")

    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        return

    min_window = args.min_window
    max_window = args.max_window
    trials_per_window = args.trials

    client = LLMClient(server_url=args.server, temperature=args.temperature)
    if not client.ping():
        print(f"ERROR: Cannot reach llama.cpp server at {args.server}")
        print(f"Start it first:\n  llama-server --model <path> --host 0.0.0.0 --port 8080 --ctx-size 65536")
        sys.exit(1)
    print(f"Server OK: {args.server}")

    # Parse config filter
    configs = None
    if args.config:
        config_map = {
            "HardCut": (False, False),
            "Compact": (False, True),
        }
        if args.config not in config_map:
            print(f"ERROR: Unknown config '{args.config}'. Choose from: {list(config_map.keys())}")
            sys.exit(1)
        configs = [config_map[args.config]]

    if args.all:
        problems = load_all_problems()
    else:
        problems = [load_problem(args.problem)]

    print(f"Problems: {[p['problem_id'] for p in problems]}")
    print(f"Search range: [{min_window}, {max_window}]")
    print(f"Trials per window: {trials_per_window}")
    print(f"Configs: {[c for c in (configs or [(False,False),(False,True)])]}")
    print()

    for problem in problems:
        print(f"\n{'#'*60}")
        print(f"  PROBLEM: {problem['problem_id']}")
        print(f"  Answer: {problem['ground_truth']}")
        print(f"  120B pass rate: {problem.get('gptoss_120b_pass_rate', '?')}")
        print(f"{'#'*60}")
        run_problem(client, problem, configs=configs,
                    min_window=min_window, max_window=max_window,
                    trials=trials_per_window)

    print("\n\nAll done. Run --analyze to see summary.")


if __name__ == "__main__":
    main()
