#!/usr/bin/env python3
# Author: Claude Sonnet 4.6 (Bubba)
# Date: 28-March-2026
# PURPOSE: AmnesiaBench v2 — multi-model, multi-problem benchmark runner that binary-searches
#   for the minimum context window (n_reliable) at which each LLM can solve competition-math
#   problems at 60% success rate. Supports 10 problems × N models for overnight runs.
#   Features: prediction phase, composite Scott scoring, --model / --model-name flags,
#   --run-all-models mode reading models.json, per-model result namespacing, full scoring table.
#   Integration points: run_prediction_phase() → run_problem() → binary_search() → run_trial().
# SRP/DRY check: Pass — prediction phase, scoring, model iteration all isolated. No duplication
#   of result I/O. calculate_scores() is the single scoring engine. run_all_models() delegates
#   to run_problem() so the multi-model path is just a loop around the single-model path.
"""
AmnesiaBench v2 — How much context does a model actually need?

Binary-searches (log scale) for the minimum context window at which an LLM
can solve competition-math problems at a 60% success rate. Runs prediction
phase, computes composite Scott scores, supports multi-model overnight runs.

Usage:
    # Single problem, single model:
    python3 amnesia_bench.py --problem ab507a9f --model http://localhost:8080 --model-name Qwen35B

    # All problems, single model:
    python3 amnesia_bench.py --all --model http://localhost:8080 --model-name Qwen35B

    # All problems, all models from models.json:
    python3 amnesia_bench.py --all --run-all-models

    # Analysis:
    python3 amnesia_bench.py --analyze
    python3 amnesia_bench.py --scores
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
TRIALS_PER_WINDOW = 3
SUCCESS_THRESHOLD = 0.6          # 60%
CONVERGENCE_RATIO = 1.05         # stop when hi/lo < 5% (fallback)
CONVERGENCE_ABS = 50             # stop when hi - lo < 50 tokens (primary)
MAX_COMPLETION_TOKENS = 16384    # hard cap per generation turn
MAX_COMPACTIONS = 5
COMPACTION_TRIGGER = 0.70
MAX_TURNS = 40
CODE_TIMEOUT = 30
TEMPERATURE = 0.7
PROBLEMS_DIR = Path(__file__).parent / "problems"
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_JSON = Path(__file__).parent / "models.json"

DEFAULT_COMPACTION_PROMPT = "Compact your context window to continue."

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

PREDICTION_PROMPT = """\
You are about to solve a math problem under context window constraints.

--- PROBLEM ---
{problem_text}

--- SCORING ---
Your score depends on:
- n_reliable: minimum context window where you solve this 60%+ of the time
- n_reliable_prediction: your prediction of n_reliable (before we test)
- Lower n_reliable_prediction = better score IF correct; if too low and you fail = infinity penalty
- success_prediction: whether you can solve this at all

You may opt out (success_prediction=False) if you think you cannot solve this problem.

--- INSTRUCTIONS ---
Respond in 300 tokens or less. Include these tags:
<success_prediction>True or False</success_prediction>
<n_reliable_prediction>integer (tokens)</n_reliable_prediction>
<compaction_prompt>one sentence describing what to preserve when compacting</compaction_prompt>"""


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
        Send messages to the model. Returns usage + content dict.
        """
        max_tokens = max(1, max_tokens)
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=3600,
            stream=True,
        )
        resp.raise_for_status()

        full_content = ""
        reasoning = ""
        content = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        finish_reason = "unknown"

        print("    [stream] ", end="", flush=True)
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            r_piece = delta.get("reasoning_content", "") or ""
            c_piece = delta.get("content", "") or ""
            if r_piece:
                reasoning += r_piece
                sys.stdout.write(r_piece)
                sys.stdout.flush()
            if c_piece:
                content += c_piece
                sys.stdout.write(c_piece)
                sys.stdout.flush()
            finish_reason = choice.get("finish_reason") or finish_reason
            usage = chunk.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
        print()

        if reasoning:
            full_content = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            full_content = content

        return {
            "content": full_content,
            "reasoning_content": reasoning,
            "final_content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
        }

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ─── Parsing Helpers ─────────────────────────────────────────────────────────

def extract_python_blocks(text: str) -> list[str]:
    pattern = r"```python\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def extract_compact_call(text: str) -> Optional[str]:
    match = re.search(r"<compact>(.*?)</compact>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_boxed_answer(text: str) -> Optional[int]:
    """Extract the last \\boxed{...} answer from text, ignoring <think> blocks."""
    non_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    target = non_think if non_think.strip() else text

    matches = re.findall(r"\\boxed\{([^{}]+)\}", target)
    if not matches:
        matches = re.findall(r"\\boxed\{(.+?)\}", target)
    if not matches:
        return None

    raw = matches[-1].strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        f = float(raw)
        if f == int(f):
            return int(f)
    except ValueError:
        pass
    try:
        cleaned = raw.replace("^", "**").replace(",", "")
        return int(eval(cleaned))
    except Exception:
        pass
    return None


# ─── Prediction Phase ────────────────────────────────────────────────────────

def run_prediction_phase(
    client: LLMClient,
    problem: dict,
    max_tokens: int = 300,
) -> dict:
    """
    Ask the model to predict its own performance before testing begins.
    Returns parsed prediction dict with keys: success_prediction, n_reliable_prediction,
    compaction_prompt, raw_response. Falls back to safe defaults on parse failure.
    """
    problem_text = problem.get("problem_text", "")
    prompt = PREDICTION_PROMPT.format(problem_text=problem_text)
    messages = [{"role": "user", "content": prompt}]

    print(f"\n  [Prediction Phase] Asking model to predict performance...")
    try:
        resp = client.generate(messages, max_tokens=max_tokens)
    except Exception as e:
        print(f"  [Prediction Phase] API error: {e} — using defaults")
        return _prediction_defaults(raw_response=f"ERROR: {e}")

    raw = resp.get("content", "")
    completion_tokens = resp.get("completion_tokens", 0)

    if completion_tokens > max_tokens:
        print(f"  [Prediction Phase] Response too long ({completion_tokens} > {max_tokens}) — using defaults")
        return _prediction_defaults(raw_response=raw)

    success_match = re.search(
        r"<success_prediction>\s*(True|False)\s*</success_prediction>",
        raw, re.IGNORECASE
    )
    if not success_match:
        print("  [Prediction Phase] Missing <success_prediction> tag — using defaults")
        return _prediction_defaults(raw_response=raw)
    success_prediction = success_match.group(1).strip().lower() == "true"

    n_reliable_match = re.search(
        r"<n_reliable_prediction>\s*(\d+)\s*</n_reliable_prediction>",
        raw
    )
    if not n_reliable_match:
        print("  [Prediction Phase] Missing <n_reliable_prediction> tag — using defaults")
        return _prediction_defaults(raw_response=raw)
    n_reliable_prediction = int(n_reliable_match.group(1))

    compaction_match = re.search(
        r"<compaction_prompt>(.*?)</compaction_prompt>",
        raw, re.DOTALL
    )
    if not compaction_match:
        print("  [Prediction Phase] Missing <compaction_prompt> tag — using defaults")
        return _prediction_defaults(raw_response=raw)
    compaction_prompt = compaction_match.group(1).strip() or DEFAULT_COMPACTION_PROMPT

    print(f"  [Prediction Phase] success={success_prediction}, n_reliable={n_reliable_prediction}, compaction='{compaction_prompt[:60]}'")

    return {
        "success_prediction": success_prediction,
        "n_reliable_prediction": n_reliable_prediction,
        "compaction_prompt": compaction_prompt,
        "raw_response": raw,
    }


def _prediction_defaults(raw_response: str = "") -> dict:
    """Return safe prediction defaults (n_reliable=None means infinity)."""
    return {
        "success_prediction": True,
        "n_reliable_prediction": None,
        "compaction_prompt": DEFAULT_COMPACTION_PROMPT,
        "raw_response": raw_response,
    }


# ─── Scoring Engine ──────────────────────────────────────────────────────────

def calculate_scores(results_dir: Optional[Path] = None) -> None:
    """
    Load all per-model result files and compute Scott's composite benchmark scores.

    Scott's formula:
        Per-problem score:
            problem_score = baseline_n_reliable / n_reliable
            prediction_score = baseline_n_reliable_prediction / n_reliable_prediction

        Where baseline = lowest n_reliable (or n_reliable_prediction) across all models
        that solved that problem (i.e. the best-performing model sets the baseline).

        Composite scores:
            composite = mean(problem_scores over all solved problems)
            prediction_composite = mean(prediction_scores over all problems)

        Coverage = problems_attempted / problems_eligible
            eligible = problems where model context_max >= baseline_n_reliable

        Accuracy = problems_solved / problems_attempted

        Prediction accuracy = correct_success_predictions / total_problems
            (correct = predicted True and solved, OR predicted False and unsolvable)

        Final score = composite * prediction_composite * coverage * accuracy
                    * prediction_accuracy * 10000

        NOTE: FLOPs not tracked yet — omitted from scoring, noted in output.

    Prints both a per-problem table and a per-model composite score table.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir = Path(results_dir)

    # ── Load all per-config result files (not summary files) ──
    # File naming: results/{model_name}_{problem_id}_{config}.json
    # or legacy: results/{problem_id}_{config}.json (no model prefix)
    result_files = sorted(results_dir.glob("*.json"))
    result_files = [f for f in result_files if not f.name.endswith("_summary.json")]

    if not result_files:
        print("No result files found. Run experiments first.")
        return

    # Structure: {model_name: {problem_id: {config_name: result_dict}}}
    by_model: dict[str, dict[str, dict[str, dict]]] = {}

    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
        except Exception as e:
            print(f"  [scores] Could not read {rf.name}: {e}")
            continue

        model_name = data.get("model_name") or data.get("model") or "unknown"
        pid = data.get("problem_id", rf.stem)
        config = data.get("config", {})
        config_name = config.get("name", "unknown") if isinstance(config, dict) else str(config)

        by_model.setdefault(model_name, {}).setdefault(pid, {})[config_name] = data

    if not by_model:
        print("No parseable result files found.")
        return

    all_problem_ids = sorted({pid for m in by_model.values() for pid in m})

    # ── Compute baselines: best (lowest) n_reliable per problem across all models ──
    # baseline_n_reliable[pid] = min n_reliable across models that have a finite n_reliable
    # baseline_n_reliable_prediction[pid] = min n_reliable_prediction across models
    baseline_n_reliable: dict[str, Optional[int]] = {}
    baseline_n_pred: dict[str, Optional[int]] = {}

    for pid in all_problem_ids:
        n_reliables = []
        n_preds = []
        for model_data in by_model.values():
            for config_name, result in model_data.get(pid, {}).items():
                mw = result.get("minimum_window")
                if mw is not None:
                    n_reliables.append(mw)
                pred = result.get("prediction", {}) or {}
                np_val = pred.get("n_reliable_prediction")
                if np_val is not None:
                    n_preds.append(np_val)
        baseline_n_reliable[pid] = min(n_reliables) if n_reliables else None
        baseline_n_pred[pid] = min(n_preds) if n_preds else None

    # ── Per-problem detail table ──
    print(f"\n{'='*110}")
    print(f"  AmnesiaBench v2 — Per-Problem Detail")
    print(f"{'='*110}")
    print(f"{'Model':<25} {'Problem':<28} {'Config':<22} {'MinWin':>7} {'Baseline':>8} {'ProbScore':>10} {'N_Pred':>8} {'PredScore':>10}")
    print(f"{'-'*110}")

    # ── Per-model composite score computation ──
    model_scores = {}

    for model_name in sorted(by_model.keys()):
        model_data = by_model[model_name]
        problem_scores = []
        prediction_scores = []
        total_problems = len(all_problem_ids)
        problems_attempted = 0
        problems_solved = 0
        problems_eligible = 0
        correct_success_preds = 0

        for pid in all_problem_ids:
            if pid not in model_data:
                continue

            baseline = baseline_n_reliable.get(pid)
            base_pred = baseline_n_pred.get(pid)

            # Count as eligible if baseline exists (some model solved it)
            if baseline is not None:
                problems_eligible += 1

            # Use the best config for this problem (lowest minimum_window)
            configs_for_pid = model_data[pid]
            best_result = None
            best_mw = None
            for config_name, result in configs_for_pid.items():
                mw = result.get("minimum_window")
                if mw is not None:
                    if best_mw is None or mw < best_mw:
                        best_mw = mw
                        best_result = result

            if best_result is None:
                # Model didn't solve this problem in any config
                pred = list(configs_for_pid.values())[0].get("prediction", {}) or {}
                success_pred = pred.get("success_prediction", True)
                if not success_pred and baseline is None:
                    correct_success_preds += 1  # correctly predicted failure
                # Still attempted
                problems_attempted += 1
                continue

            problems_attempted += 1
            problems_solved += 1

            # Problem score
            if baseline is not None and best_mw is not None:
                prob_score = baseline / best_mw
            else:
                prob_score = 0.0
            problem_scores.append(prob_score)

            # Prediction score
            pred = best_result.get("prediction", {}) or {}
            n_pred_val = pred.get("n_reliable_prediction")
            success_pred = pred.get("success_prediction", True)

            if success_pred:
                correct_success_preds += 1  # correctly predicted success (and solved)

            if n_pred_val is not None and base_pred is not None and n_pred_val > 0:
                pred_score = base_pred / n_pred_val
            else:
                pred_score = 0.0
            prediction_scores.append(pred_score)

            prob_score_str = f"{prob_score:.3f}"
            pred_score_str = f"{pred_score:.3f}" if n_pred_val is not None else "N/A"
            baseline_str = str(baseline) if baseline is not None else "N/A"
            n_pred_str = str(n_pred_val) if n_pred_val is not None else "inf"

            # Use config name from best result
            cfg = best_result.get("config", {})
            cfg_name = cfg.get("name", "unknown") if isinstance(cfg, dict) else str(cfg)

            print(
                f"{model_name:<25} {pid:<28} {cfg_name:<22} {str(best_mw):>7} "
                f"{baseline_str:>8} {prob_score_str:>10} {n_pred_str:>8} {pred_score_str:>10}"
            )

        # ── Composite scores ──
        composite = sum(problem_scores) / len(problem_scores) if problem_scores else 0.0
        pred_composite = sum(prediction_scores) / len(prediction_scores) if prediction_scores else 0.0
        coverage = problems_attempted / problems_eligible if problems_eligible > 0 else 0.0
        accuracy = problems_solved / problems_attempted if problems_attempted > 0 else 0.0
        pred_accuracy = correct_success_preds / total_problems if total_problems > 0 else 0.0

        final_score = composite * pred_composite * coverage * accuracy * pred_accuracy * 10000

        model_scores[model_name] = {
            "composite": composite,
            "pred_composite": pred_composite,
            "coverage": coverage,
            "accuracy": accuracy,
            "pred_accuracy": pred_accuracy,
            "final_score": final_score,
            "problems_attempted": problems_attempted,
            "problems_solved": problems_solved,
            "problems_eligible": problems_eligible,
            "total_problems": total_problems,
        }

    # ── Per-model composite table ──
    print(f"\n{'='*100}")
    print(f"  AmnesiaBench v2 — Composite Scores (Scott's Formula)")
    print(f"  NOTE: FLOPs not tracked — omitted from scoring.")
    print(f"{'='*100}")
    print(f"{'Model':<25} {'Composite':>10} {'PredComp':>10} {'Coverage':>9} {'Accuracy':>9} {'PredAcc':>8} {'FinalScore':>12}")
    print(f"{'-'*100}")

    for model_name in sorted(model_scores.keys()):
        s = model_scores[model_name]
        print(
            f"{model_name:<25} "
            f"{s['composite']:>10.4f} "
            f"{s['pred_composite']:>10.4f} "
            f"{s['coverage']:>9.3f} "
            f"{s['accuracy']:>9.3f} "
            f"{s['pred_accuracy']:>8.3f} "
            f"{s['final_score']:>12.2f}"
        )
    print(f"{'='*100}")
    print(f"\nFormula: final_score = composite × pred_composite × coverage × accuracy × pred_accuracy × 10000")
    print(f"  composite = mean(baseline_n_reliable / model_n_reliable) over solved problems")
    print(f"  pred_composite = mean(baseline_n_pred / model_n_pred) over all problems")
    print(f"  coverage = attempted / eligible (eligible: baseline exists for problem)")
    print(f"  accuracy = solved / attempted")
    print(f"  pred_accuracy = correct_success_predictions / total_problems\n")


# ─── Single Trial ─────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: str
    tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    code_executed: Optional[str] = None
    code_output: Optional[str] = None
    compact_summary: Optional[str] = None


@dataclass
class TrialResult:
    problem_id: str
    correct_answer: int
    token_limit: int
    tir: bool
    compaction: bool
    trial_idx: int
    success: bool
    answer: Optional[int]
    total_tokens_peak: int
    n_turns: int
    n_compactions: int
    n_code_calls: int
    n_code_errors: int
    wall_time_s: float
    error: Optional[str]
    finish_reason: str
    conversation: list = field(default_factory=list)


def run_trial(
    client: LLMClient,
    problem_id: str,
    problem_text: str,
    correct_answer: int,
    token_limit: int,
    tir: bool,
    compaction: bool,
    trial_idx: int,
    compaction_hint: str = "",
) -> TrialResult:
    t0 = time.time()
    sandbox = PythonSandbox() if tir else None
    conversation: list[Turn] = []
    messages: list[dict] = []
    n_compactions = 0
    n_code_calls = 0
    n_code_errors = 0
    peak_tokens = 0
    error_msg = None
    finish = "max_turns"

    active_compaction_hint = compaction_hint.strip() if compaction_hint else DEFAULT_COMPACTION_PROMPT

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
        if peak_tokens > 0:
            remaining = token_limit - peak_tokens
        else:
            remaining = token_limit

        if remaining <= 0:
            finish = "budget_exceeded" if compaction else "truncated"
            break

        capped_tokens = min(remaining, MAX_COMPLETION_TOKENS)
        try:
            resp = client.generate(messages, max_tokens=capped_tokens)
        except Exception as e:
            error_msg = f"API error: {e}"
            finish = "error"
            break

        if resp["finish_reason"] in ("length", "truncated") and extract_boxed_answer(resp["content"]) is None:
            finish = "truncated"

        content = resp["content"]
        total_now = resp["total_tokens"]
        peak_tokens = max(peak_tokens, total_now)

        turn = Turn(
            role="assistant",
            content=content,
            tokens=resp["completion_tokens"],
            prompt_tokens=resp["prompt_tokens"],
            total_tokens=total_now,
            finish_reason=resp["finish_reason"],
        )
        conversation.append(turn)

        answer = extract_boxed_answer(content)
        if answer is not None:
            finish = "solved"
            break

        compact_summary = extract_compact_call(content) if compaction else None
        if compact_summary is not None:
            turn.compact_summary = compact_summary
            n_compactions += 1
            if n_compactions > MAX_COMPACTIONS:
                finish = "max_compactions"
                break
            hint_line = f"\nHint: {active_compaction_hint}" if active_compaction_hint else ""
            restart_user_msg = POST_COMPACT_USER.format(
                problem_text=problem_text + hint_line,
                summary=compact_summary,
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": restart_user_msg},
            ]
            peak_tokens = 0
            conversation.append(Turn(role="user", content=f"[COMPACTION #{n_compactions} — context reset]"))
            continue

        if total_now >= token_limit:
            finish = "budget_exceeded" if compaction else "truncated"
            break

        code_blocks = extract_python_blocks(content) if tir else []
        if code_blocks:
            all_outputs = []
            for code in code_blocks:
                n_code_calls += 1
                output = sandbox.execute(code)
                if output.startswith("Error:"):
                    n_code_errors += 1
                all_outputs.append(output)
            combined_output = "\n---\n".join(all_outputs)
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

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Continue solving."})
        conversation.append(Turn(role="user", content="Continue solving."))

    # Extract final answer from conversation
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
    window: int
    trials: list
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
    compaction_hint: str = "",
) -> dict:
    config_name = f"{'Compact' if compaction else 'HardCut'}"
    print(f"\n{'='*60}")
    print(f"  {problem_id} | {config_name}")
    print(f"  Search range: [{min_window}, {max_window}]")
    print(f"{'='*60}")

    search_log: list[WindowTest] = []

    # Verify solvable at max window
    print(f"\n  [Verify] Testing max window = {max_window} ...")
    test = _test_window(
        client, problem_id, problem_text, correct_answer,
        max_window, tir, compaction, trials, compaction_hint
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

    lo, hi = min_window, max_window
    step = 0
    while hi / lo > CONVERGENCE_RATIO and (hi - lo) > CONVERGENCE_ABS:
        step += 1
        mid = int(math.exp((math.log(lo) + math.log(hi)) / 2))
        mid = max(min_window, (mid // 64) * 64)
        if mid == lo or mid == hi:
            break

        print(f"\n  [Step {step}] Testing window = {mid}  (range [{lo}, {hi}], gap {hi-lo}, ratio {hi/lo:.3f})")
        test = _test_window(
            client, problem_id, problem_text, correct_answer,
            mid, tir, compaction, trials, compaction_hint
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
    compaction_hint: str = "",
) -> WindowTest:
    t0 = time.time()

    def _run_one(i):
        return run_trial(
            client, problem_id, problem_text, correct_answer,
            token_limit=window, tir=tir, compaction=compaction,
            trial_idx=i, compaction_hint=compaction_hint,
        )

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
        window=window, trials=trials_results,
        n_success=n_success, n_trials=n_trials,
        pass_rate=pass_rate, passed=pass_rate >= SUCCESS_THRESHOLD,
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


# ─── Problem Loading ─────────────────────────────────────────────────────────

def load_problem(problem_id: str) -> dict:
    """Load a problem JSON from problems/. Matches exact stem or substring."""
    path = PROBLEMS_DIR / f"{problem_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    for p in PROBLEMS_DIR.glob("*.json"):
        if problem_id in p.stem:
            return json.loads(p.read_text())
    raise FileNotFoundError(f"No problem matching '{problem_id}' in {PROBLEMS_DIR}")


def load_all_problems() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(PROBLEMS_DIR.glob("*.json"))]


# ─── Result Filename Helpers ─────────────────────────────────────────────────

def result_filename(model_name: str, problem_id: str, config_name: str) -> Path:
    """
    Build result file path for a given model/problem/config combination.
    Format: results/{model_name}_{problem_id}_{config_name}.json
    Model name is sanitized (spaces → underscores, slashes → dashes).
    """
    safe_model = re.sub(r"[^\w\-]", "_", model_name)
    return RESULTS_DIR / f"{safe_model}_{problem_id}_{config_name}.json"


def summary_filename(model_name: str, problem_id: str) -> Path:
    safe_model = re.sub(r"[^\w\-]", "_", model_name)
    return RESULTS_DIR / f"{safe_model}_{problem_id}_summary.json"


# ─── Single-Problem Runner ───────────────────────────────────────────────────

def run_problem(
    client: LLMClient,
    problem: dict,
    model_name: str = "unknown",
    configs: list[tuple[bool, bool]] = None,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    trials: int = TRIALS_PER_WINDOW,
):
    """
    Run binary search for all configs on one problem. Save per-config and summary results.
    Results namespaced by model_name to prevent multi-model collisions.
    """
    if configs is None:
        configs = [
            (False, False),  # NoTIR + HardCut
            (False, True),   # NoTIR + Compact
        ]

    pid = problem["problem_id"]
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    for tir, compaction in configs:
        config_name = f"{'TIR' if tir else 'NoTIR'}_{'Compact' if compaction else 'HardCut'}"
        outpath = result_filename(model_name, pid, config_name)

        # Resume: skip if valid completed result exists
        if outpath.exists():
            try:
                existing = json.loads(outpath.read_text())
                if existing.get("minimum_window") is not None or existing.get("binary_search"):
                    print(f"\n  [SKIP] {model_name} | {pid} | {config_name} — result exists at {outpath.name}")
                    all_results.append(existing)
                    continue
            except Exception:
                pass

        # Prediction phase
        prediction = run_prediction_phase(client, problem, max_tokens=300)
        compaction_hint = prediction.get("compaction_prompt", DEFAULT_COMPACTION_PROMPT)

        if not prediction.get("success_prediction", True):
            print(f"\n  [Prediction Phase] Model opted out. Skipping binary search for {pid} | {config_name}.")
            result = _build_result(
                pid, tir, compaction, [],
                minimum_window=None,
                search_range_final=(min_window, max_window),
            )
            result["prediction"] = prediction
            result["model_name"] = model_name
            all_results.append(result)
            outpath.write_text(json.dumps(result, indent=2, default=str))
            print(f"\n  Saved (opt-out): {outpath.name}")
            continue

        # Binary search
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
            compaction_hint=compaction_hint,
        )
        result["model_name"] = model_name
        result["prediction"] = prediction
        all_results.append(result)

        outpath.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Saved: {outpath.name}")

    # Save combined summary (compact, no conversation traces)
    summary = []
    for r in all_results:
        entry = {
            "model_name": model_name,
            "problem_id": r["problem_id"],
            "config": r["config"]["name"] if isinstance(r.get("config"), dict) else r.get("config"),
            "minimum_window": r["minimum_window"],
            "search_range_final": r.get("search_range_final"),
            "steps": len(r.get("binary_search", [])),
        }
        pred = r.get("prediction")
        if pred:
            entry["n_reliable_prediction"] = pred.get("n_reliable_prediction")
            entry["success_prediction"] = pred.get("success_prediction")
        summary.append(entry)

    sp = summary_filename(model_name, pid)
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary: {sp.name}")

    return all_results


# ─── Multi-Model Runner ──────────────────────────────────────────────────────

def load_models_json() -> list[dict]:
    """Load models.json from the AmnesiaBench directory. Returns list of {name, url} dicts."""
    if not MODELS_JSON.exists():
        raise FileNotFoundError(
            f"models.json not found at {MODELS_JSON}. "
            "Create it with a list of {{name, url}} entries."
        )
    models = json.loads(MODELS_JSON.read_text())
    if not isinstance(models, list) or not models:
        raise ValueError("models.json must be a non-empty list of {name, url} objects.")
    for m in models:
        if "name" not in m or "url" not in m:
            raise ValueError(f"Each model entry must have 'name' and 'url' keys. Got: {m}")
    return models


def run_all_models(
    problems: list[dict],
    configs: list[tuple[bool, bool]] = None,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    trials: int = TRIALS_PER_WINDOW,
    temperature: float = TEMPERATURE,
):
    """
    Iterate over all models in models.json, run all problems for each model.
    Models are run sequentially (one model at a time, all problems per model).
    If a model's server is unreachable, it is skipped with a warning.
    """
    models = load_models_json()
    print(f"\n{'#'*70}")
    print(f"  --run-all-models: {len(models)} model(s) × {len(problems)} problem(s)")
    for m in models:
        print(f"    {m['name']} → {m['url']}")
    print(f"{'#'*70}\n")

    for model_entry in models:
        mname = model_entry["name"]
        murl = model_entry["url"]
        print(f"\n{'#'*70}")
        print(f"  MODEL: {mname}")
        print(f"  URL:   {murl}")
        print(f"{'#'*70}")

        client = LLMClient(server_url=murl, temperature=temperature)
        if not client.ping():
            print(f"  WARNING: Cannot reach server at {murl} — skipping {mname}")
            continue

        print(f"  Server OK: {murl}")
        for problem in problems:
            print(f"\n{'='*60}")
            print(f"  PROBLEM: {problem['problem_id']}")
            print(f"  Answer: {problem['ground_truth']}")
            print(f"{'='*60}")
            run_problem(
                client, problem,
                model_name=mname,
                configs=configs,
                min_window=min_window,
                max_window=max_window,
                trials=trials,
            )

    print("\n\nAll models done. Run --scores for composite scoring table.")


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze_results():
    """Print a per-model summary table of all completed results."""
    summary_files = sorted(RESULTS_DIR.glob("*_summary.json"))
    if not summary_files:
        print("No results found. Run experiments first.")
        return

    print(f"\n{'Model':<25} {'Problem':<30} {'Config':<24} {'Min Window':>10} {'Range':>18} {'Steps':>6}")
    print("-" * 118)

    current_model = None
    for f in summary_files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  [analyze] Could not read {f.name}: {e}")
            continue

        for entry in data:
            model = entry.get("model_name", "unknown")
            pid = entry.get("problem_id", "?")
            config = entry.get("config", "?")
            mw = entry.get("minimum_window")
            mw_str = str(mw) if mw is not None else "UNSOLVABLE"
            sr = entry.get("search_range_final", ["-", "-"])
            lo, hi = sr if sr else ("-", "-")
            steps = entry.get("steps", "?")

            if model != current_model:
                if current_model is not None:
                    print()
                current_model = model

            print(f"{model:<25} {pid:<30} {config:<24} {mw_str:>10} [{str(lo):>6}, {str(hi):>6}] {str(steps):>6}")


# ─── Main ────────────────────────────────────────────────────────────────────

def derive_model_name(url: str) -> str:
    """Derive a short model name from the server URL."""
    url = url.rstrip("/")
    # Extract host:port, replace dots/colons with underscores
    host_port = url.split("//")[-1]
    return re.sub(r"[^\w]", "_", host_port)


def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench v2 — multi-model context window benchmark")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", type=str, help="Problem ID (or substring) to test")
    group.add_argument("--all", action="store_true", help="Run all problems")
    group.add_argument("--analyze", action="store_true", help="Analyze existing results")
    group.add_argument("--scores", action="store_true", help="Print composite Scott scoring table")

    parser.add_argument("--model", type=str, default=SERVER_URL,
                        help=f"llama.cpp server URL (default: {SERVER_URL})")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Label for this model in results (default: derived from --model URL)")
    parser.add_argument("--run-all-models", action="store_true",
                        help="Iterate over all models in models.json (overrides --model/--model-name)")

    parser.add_argument("--min-window", type=int, default=MIN_WINDOW)
    parser.add_argument("--max-window", type=int, default=MAX_WINDOW)
    parser.add_argument("--trials", type=int, default=TRIALS_PER_WINDOW)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--config", type=str, default=None,
                        help="Run specific config only: NoTIR_HardCut, TIR_HardCut, NoTIR_Compact, TIR_Compact")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory for --scores / --analyze (default: ./results)")

    args = parser.parse_args()

    # Redirect results dir if specified
    if args.results_dir:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.results_dir)

    if args.analyze:
        analyze_results()
        return

    if args.scores:
        rd = Path(args.results_dir) if args.results_dir else None
        calculate_scores(rd)
        return

    min_window = args.min_window
    max_window = args.max_window
    trials_per_window = args.trials

    # Config filter
    configs = None
    if args.config:
        config_map = {
            "NoTIR_HardCut": (False, False),
            "TIR_HardCut": (True, False),
            "NoTIR_Compact": (False, True),
            "TIR_Compact": (True, True),
            # Legacy short names
            "HardCut": (False, False),
            "Compact": (False, True),
        }
        if args.config not in config_map:
            print(f"ERROR: Unknown config '{args.config}'. Choose from: {list(config_map.keys())}")
            sys.exit(1)
        configs = [config_map[args.config]]

    # Load problems
    if args.all:
        problems = load_all_problems()
    else:
        problems = [load_problem(args.problem)]

    # Multi-model mode
    if args.run_all_models:
        run_all_models(
            problems=problems,
            configs=configs,
            min_window=min_window,
            max_window=max_window,
            trials=trials_per_window,
            temperature=args.temperature,
        )
        return

    # Single-model mode
    model_url = args.model
    model_name = args.model_name or derive_model_name(model_url)

    client = LLMClient(server_url=model_url, temperature=args.temperature)
    if not client.ping():
        print(f"ERROR: Cannot reach llama.cpp server at {model_url}")
        print(f"Start it first:\n  llama-server --model <path> --host 0.0.0.0 --port 8080 --ctx-size 65536")
        sys.exit(1)
    print(f"Server OK: {model_url}  (model_name: {model_name})")

    print(f"Problems: {[p['problem_id'] for p in problems]}")
    print(f"Search range: [{min_window}, {max_window}]")
    print(f"Trials per window: {trials_per_window}")
    print(f"Configs: {configs or [(False,False),(False,True)]}")
    print()

    for problem in problems:
        print(f"\n{'#'*60}")
        print(f"  PROBLEM: {problem['problem_id']}")
        print(f"  Answer: {problem['ground_truth']}")
        print(f"  120B pass rate: {problem.get('gptoss_120b_pass_rate', '?')}")
        print(f"{'#'*60}")
        run_problem(
            client, problem,
            model_name=model_name,
            configs=configs,
            min_window=min_window,
            max_window=max_window,
            trials=trials_per_window,
        )

    print("\n\nAll done. Run --analyze to see summary or --scores for composite score table.")


if __name__ == "__main__":
    main()
