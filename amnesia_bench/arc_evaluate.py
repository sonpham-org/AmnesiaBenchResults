# Author: Claude Sonnet 4.6 (Bubba)
# Date: 30-March-2026
# PURPOSE: ARC-specific evaluation module for AmnesiaBench v3.1. Mirrors the nested
#   binary search architecture of evaluate.py but adapted for ARC grid puzzles:
#   - Problem text is a grid-based training+test prompt (no math/text answer)
#   - Model must output 2 grid attempts; success if EITHER matches exactly
#   - Answer extraction handles {attempt_1: [[...]]} and {attempt_2: [[...]]} formats
#   - Compaction loop is identical to evaluate.py; only system prompt / answer
#     extraction differ
#   Integration points: cli.py (arc-predict, arc-evaluate subcommands); imports
#   arc_prompts, utils (grid utilities), backoff, clients.
#   Checkpointing, resume logic, and trace saving are identical to evaluate.py.
# SRP/DRY check: Pass — ARC-specific logic is isolated here; grid utilities in utils.py;
#   prompt text in arc_prompts.py; no duplication with evaluate.py business logic.

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .arc_prompts import build_arc_evaluation_prompt, build_arc_prediction_prompt
from .backoff import ResumptionQueue
from .utils import (
    extract_arc_answers,
    grids_match,
    arc_evaluation_filename,
    arc_prediction_filename,
    sanitize_model_name,
)

_PACKAGE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = _PACKAGE_DIR.parent / "results"

# Search parameters — identical to evaluate.py
OUTER_CHECKS_PER_N = 1
INNER_CHECKS_PER_N = 3
INNER_PASS_THRESHOLD = 2
OUTER_STOP_RATIO = 0.05
INNER_STOP_ABS = 1
N_MIN = 1
TEMPERATURE = 0.7

# Compaction parameters — identical to evaluate.py
MAX_TURNS = 40
COMPACTION_TRIGGER = 0.50


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run_arc_evaluation(
    client,
    model_name: str,
    problem: dict,
    context_max: int,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
) -> dict:
    """
    Run the nested binary search evaluation for one ARC (model, problem) pair.

    Flow:
      1. Check ARC prediction file — if attempt=False, skip.
      2. Test at full context_max (unbounded, no compaction). Record actual tokens.
      3. Run binary search WITHOUT compaction → n_reliable_no_compact.
      4. Run binary search WITH compaction → n_reliable_compact.
      5. Save and return result.

    Result schema mirrors evaluate.py's _build_result() output, plus:
      - problem_type: "arc"

    The problem dict must have:
      - problem_id (str)
      - problem_text (str) — LLM-ready formatted training+test prompt block
      - test[0]["output"] — the expected output grid (list[list[int]])
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    problem_id = problem["problem_id"]
    out_path = arc_evaluation_filename(results_dir, model_name, problem_id)

    # Resume-friendly: check for completed or in-progress evaluation
    checkpoint = None
    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        status = existing.get("status", "completed")
        if status == "completed":
            print(f"  [arc-eval] SKIP {model_name} / {problem_id} — completed: {out_path.name}")
            return existing
        elif status == "running":
            print(f"  [arc-eval] RESUMING {model_name} / {problem_id} from checkpoint")
            checkpoint = existing

    # Check ARC prediction — if attempt=False, skip evaluation
    pred_path = arc_prediction_filename(results_dir, model_name, problem_id)
    if pred_path.exists():
        pred = json.loads(pred_path.read_text())
        if not pred.get("attempt", True):
            print(
                f"  [arc-eval] SKIP {model_name} / {problem_id} "
                f"— prediction says attempt=False"
            )
            result = _build_result(
                model_name, problem_id,
                n_while_unbounded=None,
                n_reliable_no_compact=None,
                n_reliable_compact=None,
                no_compact_outer_log=[],
                no_compact_inner_log=[],
                compact_outer_log=[],
                compact_inner_log=[],
                search_range=[N_MIN, context_max],
                api_calls=0, input_tokens=0, output_tokens=0,
                cost=0.0, wall_time=0.0, skipped=True,
            )
            out_path.write_text(json.dumps(result, indent=2))
            return result

    print(f"\n  [arc-eval] {model_name} / {problem_id} | context_max={context_max}")

    t_start = time.time()
    state = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0}

    problem_text = problem["problem_text"]
    # Ground truth: the expected output grid for the first test case
    expected_grid = problem["test"][0]["output"]

    safe_model = sanitize_model_name(model_name)
    traces_dir = results_dir / "traces" / f"{safe_model}_{problem_id}_arc"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if checkpoint:
        n_while_unbounded = checkpoint.get("n_while_unbounded")
        no_compact_outer_log = checkpoint.get("no_compact_outer_log", [])
        no_compact_inner_log = checkpoint.get("no_compact_inner_log", [])
        compact_outer_log = checkpoint.get("compact_outer_log", [])
        compact_inner_log = checkpoint.get("compact_inner_log", [])
        phase = checkpoint.get("phase", "unbounded")
        state["api_calls"] = checkpoint.get("total_api_calls", 0)
        state["input_tokens"] = checkpoint.get("total_input_tokens", 0)
        state["output_tokens"] = checkpoint.get("total_output_tokens", 0)
    else:
        n_while_unbounded = None
        no_compact_outer_log = []
        no_compact_inner_log = []
        compact_outer_log = []
        compact_inner_log = []
        phase = "unbounded"

    # ── Step 1: Unbounded test ────────────────────────────────────────────────
    if phase == "unbounded" or n_while_unbounded is None:
        print(f"  [arc-eval] Unbounded test — no context restriction, measuring natural usage...")
        unbounded_pass, unbounded_log = _test_arc_unbounded(
            client, problem_text, expected_grid, state=state,
            context_max=context_max,
        )
        if not unbounded_pass:
            print(f"  [arc-eval] UNSOLVABLE at context_max={context_max} — skipping search")
            result = _build_result(
                model_name, problem_id,
                n_while_unbounded=None,
                n_reliable_no_compact=None,
                n_reliable_compact=None,
                no_compact_outer_log=unbounded_log,
                no_compact_inner_log=[],
                compact_outer_log=[],
                compact_inner_log=[],
                search_range=[N_MIN, context_max],
                api_calls=state["api_calls"],
                input_tokens=state["input_tokens"],
                output_tokens=state["output_tokens"],
                cost=0.0,
                wall_time=round(time.time() - t_start, 2),
                status="completed",
            )
            out_path.write_text(json.dumps(result, indent=2))
            return result

        actual_tokens = max(
            (t.get("total_tokens", 0) for t in unbounded_log if t.get("success")),
            default=0,
        )
        if actual_tokens == 0:
            for t in unbounded_log:
                if t.get("success"):
                    content_len = len(str(t.get("content_snapshot", "")))
                    actual_tokens = max(actual_tokens, content_len // 4)
        if actual_tokens == 0:
            actual_tokens = context_max
        n_while_unbounded = actual_tokens
        print(f"  [arc-eval] n_while_unbounded={n_while_unbounded} (actual tokens used)")
        phase = "no_compact_outer"

        _write_checkpoint(
            out_path, {"model_name": model_name, "problem_id": problem_id},
            phase=phase,
            no_compact_outer_log=no_compact_outer_log,
            no_compact_inner_log=no_compact_inner_log,
            compact_outer_log=compact_outer_log,
            compact_inner_log=compact_inner_log,
            state=state,
            extra={"n_while_unbounded": n_while_unbounded},
        )

    # ── Step 2: Binary search WITHOUT compaction ──────────────────────────────
    if phase in ("no_compact_outer", "no_compact_inner") or not no_compact_outer_log:
        print(f"  [arc-eval] Outer binary search (NO compaction) [{N_MIN}, {n_while_unbounded}] ...")
        if phase == "no_compact_outer":
            nc_outer_lo, nc_outer_hi = N_MIN, n_while_unbounded
        else:
            nc_outer_lo, nc_outer_hi = _replay_search_log(
                no_compact_outer_log, N_MIN, n_while_unbounded
            )

        no_compact_outer_log, nc_transition_lo, nc_transition_hi = _outer_binary_search(
            client, problem_text, expected_grid,
            lo=nc_outer_lo, hi=nc_outer_hi,
            state=state,
            compaction_enabled=False,
            existing_log=no_compact_outer_log if phase == "no_compact_outer" else [],
            checkpoint_path=out_path,
            checkpoint_data={
                "model_name": model_name, "problem_id": problem_id,
                "n_while_unbounded": n_while_unbounded,
                "no_compact_outer_log": no_compact_outer_log,
                "no_compact_inner_log": no_compact_inner_log,
                "compact_outer_log": compact_outer_log,
                "compact_inner_log": compact_inner_log,
            },
            traces_dir=traces_dir,
        )
        phase = "no_compact_inner"

        mid = (nc_transition_lo + nc_transition_hi) // 2
        nc_inner_lo = max(N_MIN, mid - (nc_transition_hi - nc_transition_lo) * 3 // 2)
        nc_inner_hi = min(n_while_unbounded, mid + (nc_transition_hi - nc_transition_lo) * 3 // 2)
        nc_inner_hi = max(nc_inner_hi, nc_transition_hi)

        print(
            f"  [arc-eval] Inner binary search (NO compaction) [{nc_inner_lo}, {nc_inner_hi}] ..."
        )
        no_compact_inner_log, n_reliable_no_compact = _inner_binary_search(
            client, problem_text, expected_grid,
            lo=nc_inner_lo, hi=nc_inner_hi,
            state=state,
            compaction_enabled=False,
            existing_log=[],
            checkpoint_path=out_path,
            checkpoint_data={
                "model_name": model_name, "problem_id": problem_id,
                "n_while_unbounded": n_while_unbounded,
                "no_compact_outer_log": no_compact_outer_log,
                "no_compact_inner_log": no_compact_inner_log,
                "compact_outer_log": compact_outer_log,
                "compact_inner_log": compact_inner_log,
            },
            traces_dir=traces_dir,
        )
        print(f"  [arc-eval] n_reliable_no_compact={n_reliable_no_compact}")
        phase = "compact_outer"

        _write_checkpoint(
            out_path, {"model_name": model_name, "problem_id": problem_id},
            phase=phase,
            no_compact_outer_log=no_compact_outer_log,
            no_compact_inner_log=no_compact_inner_log,
            compact_outer_log=compact_outer_log,
            compact_inner_log=compact_inner_log,
            state=state,
            extra={
                "n_while_unbounded": n_while_unbounded,
                "n_reliable_no_compact": n_reliable_no_compact,
            },
        )
    else:
        n_reliable_no_compact = checkpoint.get("n_reliable_no_compact")

    # ── Step 3: Binary search WITH compaction ─────────────────────────────────
    if phase in ("compact_outer", "compact_inner") or not compact_outer_log:
        print(f"  [arc-eval] Outer binary search (WITH compaction) [{N_MIN}, {n_while_unbounded}] ...")
        if phase == "compact_outer":
            c_outer_lo, c_outer_hi = N_MIN, n_while_unbounded
        else:
            c_outer_lo, c_outer_hi = _replay_search_log(
                compact_outer_log, N_MIN, n_while_unbounded
            )

        compact_outer_log, c_transition_lo, c_transition_hi = _outer_binary_search(
            client, problem_text, expected_grid,
            lo=c_outer_lo, hi=c_outer_hi,
            state=state,
            compaction_enabled=True,
            existing_log=compact_outer_log if phase == "compact_outer" else [],
            checkpoint_path=out_path,
            checkpoint_data={
                "model_name": model_name, "problem_id": problem_id,
                "n_while_unbounded": n_while_unbounded,
                "n_reliable_no_compact": n_reliable_no_compact,
                "no_compact_outer_log": no_compact_outer_log,
                "no_compact_inner_log": no_compact_inner_log,
                "compact_outer_log": compact_outer_log,
                "compact_inner_log": compact_inner_log,
            },
            traces_dir=traces_dir,
        )
        phase = "compact_inner"

        mid = (c_transition_lo + c_transition_hi) // 2
        c_inner_lo = max(N_MIN, mid - (c_transition_hi - c_transition_lo) * 3 // 2)
        c_inner_hi = min(n_while_unbounded, mid + (c_transition_hi - c_transition_lo) * 3 // 2)
        c_inner_hi = max(c_inner_hi, c_transition_hi)

        print(
            f"  [arc-eval] Inner binary search (WITH compaction) [{c_inner_lo}, {c_inner_hi}] ..."
        )
        compact_inner_log, n_reliable_compact = _inner_binary_search(
            client, problem_text, expected_grid,
            lo=c_inner_lo, hi=c_inner_hi,
            state=state,
            compaction_enabled=True,
            existing_log=[],
            checkpoint_path=out_path,
            checkpoint_data={
                "model_name": model_name, "problem_id": problem_id,
                "n_while_unbounded": n_while_unbounded,
                "n_reliable_no_compact": n_reliable_no_compact,
                "no_compact_outer_log": no_compact_outer_log,
                "no_compact_inner_log": no_compact_inner_log,
                "compact_outer_log": compact_outer_log,
                "compact_inner_log": compact_inner_log,
            },
            traces_dir=traces_dir,
        )
        print(f"  [arc-eval] n_reliable_compact={n_reliable_compact}")
    else:
        n_reliable_compact = checkpoint.get("n_reliable_compact")

    wall_time = round(time.time() - t_start, 2)
    result = _build_result(
        model_name, problem_id,
        n_while_unbounded=n_while_unbounded,
        n_reliable_no_compact=n_reliable_no_compact,
        n_reliable_compact=n_reliable_compact,
        no_compact_outer_log=no_compact_outer_log,
        no_compact_inner_log=no_compact_inner_log,
        compact_outer_log=compact_outer_log,
        compact_inner_log=compact_inner_log,
        search_range=[N_MIN, n_while_unbounded],
        api_calls=state["api_calls"],
        input_tokens=state["input_tokens"],
        output_tokens=state["output_tokens"],
        cost=0.0,
        wall_time=wall_time,
        status="completed",
    )

    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"  [arc-eval] DONE — n_reliable_no_compact={n_reliable_no_compact} "
        f"n_reliable_compact={n_reliable_compact} | "
        f"api_calls={state['api_calls']} | {wall_time}s → {out_path.name}"
    )
    return result


def run_arc_evaluations_for_problems(
    client,
    model_name: str,
    problems: list,
    context_max: int,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
) -> list:
    """Run ARC evaluation for a list of problems. Returns list of result dicts."""
    results = []
    for problem in problems:
        try:
            result = run_arc_evaluation(
                client, model_name, problem, context_max,
                results_dir=results_dir,
                queue=queue,
                force=force,
            )
        except Exception as e:
            err = str(e)
            print(f"  [arc-eval] FAILED {model_name} / {problem['problem_id']}: {err}")
            if queue:
                queue.push(model_name, problem["problem_id"], "arc_evaluation", err)
            continue
        results.append(result)
    return results


# ─── ARC Prediction Job ───────────────────────────────────────────────────────

def run_arc_prediction(
    client,
    model_name: str,
    problem: dict,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
) -> dict:
    """
    Run the ARC prediction job for one (model, problem) pair.

    Asks the model: can you solve this puzzle? What N do you need?
    Saves and returns the prediction result dict.

    Result schema matches predict.py's run_prediction() output.
    """
    from .predict import _parse_prediction_response, _fallback_result

    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    problem_id = problem["problem_id"]
    out_path = arc_prediction_filename(results_dir, model_name, problem_id)

    if out_path.exists() and not force:
        print(f"  [arc-predict] SKIP {model_name} / {problem_id} — file exists: {out_path.name}")
        return json.loads(out_path.read_text())

    print(f"\n  [arc-predict] {model_name} / {problem_id}")

    prompt = build_arc_prediction_prompt(problem["problem_text"])
    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.generate(messages, max_tokens=300)
    except Exception as e:
        err_str = str(e)
        print(f"  [arc-predict] API ERROR: {err_str}")
        if queue:
            queue.push(model_name, problem_id, "arc_prediction", err_str)
        return _fallback_result(model_name, problem_id, raw_response=f"ERROR: {err_str}")

    raw = resp.get("content", "") or ""
    input_tokens = resp.get("input_tokens", 0)
    output_tokens = resp.get("output_tokens", 0)
    total_tokens = resp.get("total_tokens", input_tokens + output_tokens)

    attempt, n_predicted, parse_success = _parse_prediction_response(raw)
    fallback_used = not parse_success

    result = {
        "model_name": model_name,
        "problem_id": problem_id,
        "problem_type": "arc",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "n_predicted": n_predicted,
        "raw_response": raw,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "parse_success": parse_success,
        "fallback_used": fallback_used,
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"  [arc-predict] attempt={attempt}, n_predicted={n_predicted}, "
        f"parse_success={parse_success} → {out_path.name}"
    )
    return result


# ─── Checkpoint Writer ────────────────────────────────────────────────────────

def _write_checkpoint(
    checkpoint_path: Optional[Path],
    checkpoint_data: dict,
    phase: str,
    no_compact_outer_log: list,
    no_compact_inner_log: list,
    compact_outer_log: list,
    compact_inner_log: list,
    state: dict,
    extra: Optional[dict] = None,
):
    """Write a running checkpoint to disk after each search step."""
    if checkpoint_path is None:
        return
    data = {
        **checkpoint_data,
        "status": "running",
        "problem_type": "arc",
        "phase": phase,
        "no_compact_outer_log": no_compact_outer_log,
        "no_compact_inner_log": no_compact_inner_log,
        "compact_outer_log": compact_outer_log,
        "compact_inner_log": compact_inner_log,
        "total_api_calls": state["api_calls"],
        "total_input_tokens": state["input_tokens"],
        "total_output_tokens": state["output_tokens"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        data.update(extra)
    checkpoint_path.write_text(json.dumps(data, indent=2))


# ─── Search Log Replay ────────────────────────────────────────────────────────

def _replay_search_log(log: list, initial_lo: int, initial_hi: int) -> tuple:
    """Replay a search log to reconstruct current lo/hi state."""
    lo, hi = initial_lo, initial_hi
    for entry in log:
        n = entry["N"]
        if entry["passed"]:
            hi = n
        else:
            lo = n
    return lo, hi


# ─── Outer Binary Search ──────────────────────────────────────────────────────

def _outer_binary_search(
    client, problem_text: str, expected_grid: list,
    lo: int, hi: int, state: dict,
    compaction_enabled: bool,
    existing_log: Optional[list] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_data: Optional[dict] = None,
    traces_dir: Optional[Path] = None,
) -> tuple:
    """
    Find the rough fail→pass transition zone using a single trial per N.
    Stop when step < 5% of current N.
    Returns (log, transition_lo, transition_hi).
    """
    log = existing_log if existing_log else []
    step = len(log)
    last_fail_lo = lo
    first_pass_hi = hi
    label = "compact" if compaction_enabled else "nocompact"

    while True:
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break
        step_size = hi - lo
        if step_size < OUTER_STOP_RATIO * mid:
            break

        step += 1
        print(f"  [arc-outer-{label} step {step}] N={mid}  range=[{lo},{hi}]")
        passed, trial_log = _test_arc_n(
            client, problem_text, expected_grid, mid,
            n_trials=OUTER_CHECKS_PER_N, state=state,
            compaction_enabled=compaction_enabled,
            traces_dir=traces_dir,
        )
        log.append({"N": mid, "passed": passed, "trials": trial_log})

        _write_checkpoint(
            checkpoint_path, checkpoint_data or {},
            phase=f"{'compact' if compaction_enabled else 'no_compact'}_outer",
            no_compact_outer_log=(checkpoint_data or {}).get("no_compact_outer_log", []) if compaction_enabled else log,
            no_compact_inner_log=(checkpoint_data or {}).get("no_compact_inner_log", []),
            compact_outer_log=log if compaction_enabled else (checkpoint_data or {}).get("compact_outer_log", []),
            compact_inner_log=(checkpoint_data or {}).get("compact_inner_log", []),
            state=state,
            extra={"n_while_unbounded": (checkpoint_data or {}).get("n_while_unbounded")},
        )

        if passed:
            first_pass_hi = mid
            hi = mid
        else:
            last_fail_lo = mid
            lo = mid

    print(f"  [arc-outer-{label}] transition zone: [{last_fail_lo}, {first_pass_hi}]")
    return log, last_fail_lo, first_pass_hi


# ─── Inner Binary Search ──────────────────────────────────────────────────────

def _inner_binary_search(
    client, problem_text: str, expected_grid: list,
    lo: int, hi: int, state: dict,
    compaction_enabled: bool,
    existing_log: Optional[list] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_data: Optional[dict] = None,
    traces_dir: Optional[Path] = None,
) -> tuple:
    """
    Refine the transition zone using 3 trials per N; require 2/3 to pass.
    Stop when hi - lo <= 1 token.
    Returns (log, n_reliable).
    """
    log = existing_log if existing_log else []
    step = len(log)
    n_reliable = hi
    label = "compact" if compaction_enabled else "nocompact"

    while hi - lo > INNER_STOP_ABS:
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break

        step += 1
        print(f"  [arc-inner-{label} step {step}] N={mid}  range=[{lo},{hi}]")
        passed, trial_log = _test_arc_n(
            client, problem_text, expected_grid, mid,
            n_trials=INNER_CHECKS_PER_N, state=state,
            pass_threshold=INNER_PASS_THRESHOLD,
            compaction_enabled=compaction_enabled,
            traces_dir=traces_dir,
        )
        log.append({"N": mid, "passed": passed, "n_trials": INNER_CHECKS_PER_N, "trials": trial_log})

        _write_checkpoint(
            checkpoint_path, checkpoint_data or {},
            phase=f"{'compact' if compaction_enabled else 'no_compact'}_inner",
            no_compact_outer_log=(checkpoint_data or {}).get("no_compact_outer_log", []),
            no_compact_inner_log=log if not compaction_enabled else (checkpoint_data or {}).get("no_compact_inner_log", []),
            compact_outer_log=(checkpoint_data or {}).get("compact_outer_log", []),
            compact_inner_log=log if compaction_enabled else (checkpoint_data or {}).get("compact_inner_log", []),
            state=state,
            extra={
                "n_while_unbounded": (checkpoint_data or {}).get("n_while_unbounded"),
                "inner_lo": lo, "inner_hi": hi,
            },
        )

        if passed:
            n_reliable = mid
            hi = mid
        else:
            lo = mid

    print(f"  [arc-inner-{label}] n_reliable={n_reliable}")
    return log, n_reliable


# ─── Unbounded Test ───────────────────────────────────────────────────────────

def _test_arc_unbounded(
    client, problem_text: str, expected_grid: list,
    state: dict,
    context_max: int = 262144,
) -> tuple:
    """
    Run ONE ARC trial with NO context window restriction.
    System prompt does not mention N or token limits.
    Measures natural token usage.
    Returns (passed: bool, trial_log: list with one entry).
    """
    t0 = time.time()
    system_prompt = (
        "You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.\n"
        "You will see training examples showing input→output grid transformations.\n"
        "Your task is to figure out the pattern and apply it to the test input.\n\n"
        "Grids use integers 0-9 where 0 is typically the background color.\n\n"
        "Provide exactly 2 answer attempts. Output each as a JSON grid:\n"
        "{attempt_1: [[row1], [row2], ...]}\n"
        "{attempt_2: [[row1], [row2], ...]}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text},
    ]

    try:
        resp = client.generate(messages, max_tokens=context_max, stream=False)
    except Exception as e:
        result = {
            "trial_idx": 0, "N": "unbounded", "success": False,
            "answer": None, "expected": expected_grid,
            "finish_reason": "error", "input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0,
            "wall_time_s": round(time.time() - t0, 2), "error": str(e),
        }
        state["api_calls"] += 1
        return False, [result]

    content = resp.get("content", "") or ""
    answers = extract_arc_answers(content)
    success = _check_arc_success(answers, expected_grid)

    total_tokens = resp.get("total_tokens", 0)
    input_tokens = resp.get("input_tokens", 0)
    output_tokens = resp.get("output_tokens", 0)
    timings = resp.get("timings", {})

    if total_tokens == 0:
        input_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        output_tokens = len(content) // 4
        total_tokens = input_tokens + output_tokens

    wall_time = round(time.time() - t0, 2)
    state["api_calls"] += 1
    state["input_tokens"] += input_tokens
    state["output_tokens"] += output_tokens

    result = {
        "trial_idx": 0, "N": "unbounded", "success": success,
        "answer": answers[0] if answers else None,
        "expected": expected_grid,
        "finish_reason": resp.get("finish_reason", "unknown"),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "timings": timings,
        "wall_time_s": wall_time, "error": None,
        "n_attempts": len(answers),
        "content_snapshot": content[:2000],
    }
    status = "PASS" if success else "FAIL"
    print(f"    unbounded: {status} | {len(answers)} attempt(s) | {total_tokens} tokens | {wall_time}s")
    return success, [result]


# ─── Single N Test ────────────────────────────────────────────────────────────

def _test_arc_n(
    client, problem_text: str, expected_grid: list,
    N: int, n_trials: int, state: dict,
    compaction_enabled: bool,
    pass_threshold: int = 1,
    traces_dir: Optional[Path] = None,
) -> tuple:
    """
    Run n_trials trials at context window N in parallel.
    Returns (passed: bool, trial_log: list).
    passed = True if successes >= pass_threshold.
    """
    results = [None] * n_trials
    label = "compact" if compaction_enabled else "nocompact"

    def _run_one(idx):
        return _run_arc_trial(
            client, problem_text, expected_grid, N, idx,
            compaction_enabled=compaction_enabled,
        )

    with ThreadPoolExecutor(max_workers=n_trials) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n_trials)}
        for future in as_completed(futures):
            idx = futures[future]
            r = future.result()
            results[idx] = r
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"    trial {idx}: {status} | {r['n_attempts']} attempt(s) | "
                f"{r['finish_reason']} | {r['total_tokens']} tok | {r['wall_time_s']:.1f}s"
            )
            state["api_calls"] += 1
            state["input_tokens"] += r.get("input_tokens", 0)
            state["output_tokens"] += r.get("output_tokens", 0)

            if traces_dir is not None:
                trace_file = traces_dir / f"{label}_N{N}_trial{idx}.json"
                try:
                    # Trim large grids from conversation log for trace storage
                    trace_file.write_text(json.dumps(r, indent=2))
                except Exception as e:
                    print(f"    [trace] WARNING — failed to save trace: {e}")

    n_pass = sum(1 for r in results if r["success"])
    passed = n_pass >= pass_threshold
    print(f"    [{n_pass}/{n_trials} passed — {'PASS' if passed else 'FAIL'}]")
    return passed, results


# ─── Single ARC Trial ─────────────────────────────────────────────────────────

def _check_arc_success(answers: list, expected_grid: list) -> bool:
    """Return True if any answer in answers is an exact grid match for expected_grid."""
    for attempt in answers:
        if grids_match(attempt, expected_grid):
            return True
    return False


def _run_arc_trial(
    client, problem_text: str, expected_grid: list,
    N: int, trial_idx: int,
    compaction_enabled: bool = True,
) -> dict:
    """
    Run one ARC evaluation trial at context window N.

    When compaction_enabled=False:
        Single-shot: send prompt, get response, extract up to 2 grid answers.

    When compaction_enabled=True:
        Full compaction loop (same as evaluate.py _run_trial):
          1. Generate response.
          2. Check for grid answers → success if either matches.
          3. If tokens >= 50% of N → inject compaction prompt.
          4. Reset conversation to: system + problem + summary.
          5. Probe reset cost. If >= 50% of N → FAIL.

    Returns a trace dict with full conversation log included.
    """
    t0 = time.time()

    # ── ARC-specific system prompt ────────────────────────────────────────────
    if compaction_enabled:
        system_prompt = (
            f"You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.\n"
            f"Your context window is restricted to N = {N} tokens "
            f"(including these instructions). "
            f"When you reach 50% of this limit, you will be asked to compact "
            f"your context so that you have room to continue working.\n\n"
            f"To compact, write your working notes inside <compact>...</compact> tags.\n"
            f"When you compact, the conversation resets to: "
            f"[this system prompt] + [the puzzle] + [your summary].\n\n"
            f"Grids use integers 0-9 where 0 is typically the background color.\n\n"
            f"When you finish, provide exactly 2 answer attempts:\n"
            f"{{attempt_1: [[row1], [row2], ...]}}\n"
            f"{{attempt_2: [[row1], [row2], ...]}}"
        )
    else:
        system_prompt = (
            f"You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.\n"
            f"Your context window is restricted to N = {N} tokens.\n\n"
            f"Grids use integers 0-9 where 0 is typically the background color.\n\n"
            f"Provide exactly 2 answer attempts:\n"
            f"{{attempt_1: [[row1], [row2], ...]}}\n"
            f"{{attempt_2: [[row1], [row2], ...]}}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text},
    ]

    conversation_log = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text},
    ]

    n_compactions = 0
    compaction_summaries = []
    per_turn_tokens = []
    all_timings = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    total_tokens_peak = 0
    answers = []
    finish_reason = "max_turns"
    error_msg = None
    content_snapshot = ""

    # ── No-compaction path: single shot ──────────────────────────────────────
    if not compaction_enabled:
        turn_t0 = time.time()
        try:
            resp = client.generate(messages, max_tokens=N, stream=False)
        except Exception as e:
            error_msg = str(e)
            return _arc_trial_result(
                trial_idx, N, False, [], expected_grid,
                "error", 0, 0, 0, 0, 0, [], [], [], conversation_log,
                round(time.time() - t0, 2), str(e),
                compaction_enabled=False,
            )

        turn_time = round(time.time() - turn_t0, 2)
        content = resp.get("content", "") or ""
        content_snapshot = content
        raw_input = resp.get("input_tokens", 0)
        raw_output = resp.get("output_tokens", 0)
        raw_total = resp.get("total_tokens", 0)
        thinking_tok = resp.get("thinking_tokens", 0)
        timings = resp.get("timings", {})
        fr = resp.get("finish_reason", "stop")

        if raw_total == 0:
            input_len = sum(len(m.get("content", "")) for m in messages)
            raw_input = input_len // 4
            raw_output = len(content) // 4
            raw_total = raw_input + raw_output

        total_input_tokens = raw_input
        total_output_tokens = raw_output
        total_thinking_tokens = thinking_tok
        total_tokens_peak = raw_total
        per_turn_tokens.append(raw_output)
        all_timings.append(timings)

        conversation_log.append({
            "role": "assistant",
            "content": content,
            "input_tokens": raw_input,
            "output_tokens": raw_output,
            "thinking_tokens": thinking_tok,
            "finish_reason": fr,
            "wall_time_s": turn_time,
            "timings": timings,
        })

        answers = extract_arc_answers(content)
        success = _check_arc_success(answers, expected_grid)
        finish_reason = "solved" if success else (fr if fr else "no_answer")

        return _arc_trial_result(
            trial_idx, N, success, answers, expected_grid,
            finish_reason, total_input_tokens, total_output_tokens,
            total_thinking_tokens, total_tokens_peak, 0,
            per_turn_tokens, [], all_timings, conversation_log,
            round(time.time() - t0, 2), None,
            content_snapshot=content_snapshot,
            compaction_enabled=False,
        )

    # ── Compaction-enabled path ───────────────────────────────────────────────
    awaiting_compact = False

    for turn_i in range(MAX_TURNS):
        remaining = N - total_tokens_peak if total_tokens_peak > 0 else N
        if remaining <= 0:
            finish_reason = "budget_exceeded"
            break

        max_gen = min(remaining, 16384)
        turn_t0 = time.time()

        try:
            resp = client.generate(messages, max_tokens=max_gen, stream=False)
        except Exception as e:
            error_msg = str(e)
            finish_reason = "error"
            break

        turn_time = round(time.time() - turn_t0, 2)
        content = resp.get("content", "") or ""
        content_snapshot = content
        raw_input = resp.get("input_tokens", 0)
        raw_output = resp.get("output_tokens", 0)
        raw_total = resp.get("total_tokens", 0)
        thinking_tok = resp.get("thinking_tokens", 0)
        timings = resp.get("timings", {})
        fr = resp.get("finish_reason", "stop")

        if raw_total == 0:
            input_len = sum(len(m.get("content", "")) for m in messages)
            raw_input = input_len // 4
            raw_output = len(content) // 4
            raw_total = raw_input + raw_output

        total_tokens_peak = max(total_tokens_peak, raw_total)
        total_input_tokens += raw_input
        total_output_tokens += raw_output
        total_thinking_tokens += thinking_tok
        per_turn_tokens.append(raw_output)
        all_timings.append(timings)

        conversation_log.append({
            "role": "assistant",
            "content": content,
            "input_tokens": raw_input,
            "output_tokens": raw_output,
            "thinking_tokens": thinking_tok,
            "finish_reason": fr,
            "wall_time_s": turn_time,
            "timings": timings,
        })

        # ── Check for grid answers ────────────────────────────────────────────
        candidate_answers = extract_arc_answers(content)
        if candidate_answers:
            answers = candidate_answers
            if _check_arc_success(answers, expected_grid):
                finish_reason = "solved"
                break
            # Answers found but wrong — don't break; model may refine on next turn
            # However if this is NOT the compaction turn, treat it as final answer
            if not awaiting_compact:
                finish_reason = "answered_wrong"
                break

        # ── Handle compaction response ────────────────────────────────────────
        if awaiting_compact:
            compact_match = re.search(r"<compact>(.*?)</compact>", content, re.DOTALL)
            if compact_match:
                summary = compact_match.group(1).strip()
            else:
                summary = content.strip()

            n_compactions += 1
            compaction_summaries.append(summary)

            conversation_log.append({
                "role": "system",
                "content": f"[COMPACTION #{n_compactions} — context reset]",
                "summary": summary,
            })

            reset_user = (
                f"{problem_text}\n\n"
                f"Your previous working notes (from compact call):\n"
                f"---\n{summary}\n---\n"
                f"Continue solving. Provide exactly 2 answer attempts:\n"
                f"{{attempt_1: [[row1], [row2], ...]}}\n"
                f"{{attempt_2: [[row1], [row2], ...]}}"
            )
            reset_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": reset_user},
            ]

            probe_t0 = time.time()
            try:
                probe = client.generate(reset_messages, max_tokens=1, stream=False)
                reset_tokens = probe.get("total_tokens", 0)
                if reset_tokens == 0:
                    input_len = sum(len(m.get("content", "")) for m in reset_messages)
                    reset_tokens = input_len // 4
                total_input_tokens += probe.get("input_tokens", 0)
                total_output_tokens += probe.get("output_tokens", 0)
            except Exception as e:
                reset_tokens = N  # conservative fail-safe
                error_msg = f"probe error: {e}"

            probe_time = round(time.time() - probe_t0, 2)
            conversation_log.append({
                "role": "system",
                "content": "[PROBE — reset context token count]",
                "reset_tokens": reset_tokens,
                "wall_time_s": probe_time,
            })

            if reset_tokens >= N * COMPACTION_TRIGGER:
                finish_reason = "compaction_insufficient"
                total_tokens_peak = max(total_tokens_peak, reset_tokens)
                break

            messages = reset_messages
            conversation_log.append({"role": "user", "content": reset_user})
            total_tokens_peak = reset_tokens
            awaiting_compact = False
            continue

        # ── Check if we've hit 50% — trigger compaction ───────────────────────
        if total_tokens_peak >= N * COMPACTION_TRIGGER:
            compaction_prompt = (
                "You have reached 50% of your context window. "
                "Please compact your context now by writing a condensed summary "
                "of your analysis so far inside <compact>...</compact> tags. "
                "Be as concise as possible — preserve only the key pattern insights "
                "and any partial solution you've identified."
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": compaction_prompt})
            conversation_log.append({"role": "user", "content": compaction_prompt})
            awaiting_compact = True
            continue

        # ── Truncated without answering ───────────────────────────────────────
        if fr in ("length", "truncated"):
            finish_reason = "truncated"
            break

        # ── Continue working ──────────────────────────────────────────────────
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Continue solving."})
        conversation_log.append({"role": "user", "content": "Continue solving."})

    wall_time = round(time.time() - t0, 2)
    success = _check_arc_success(answers, expected_grid)

    return _arc_trial_result(
        trial_idx, N, success, answers, expected_grid,
        finish_reason=finish_reason,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_thinking_tokens=total_thinking_tokens,
        total_tokens_peak=total_tokens_peak,
        n_compactions=n_compactions,
        per_turn_tokens=per_turn_tokens,
        compaction_summaries=compaction_summaries,
        all_timings=all_timings,
        conversation_log=conversation_log,
        wall_time=wall_time,
        error_msg=error_msg,
        content_snapshot=content_snapshot[:2000] if content_snapshot else "",
        compaction_enabled=True,
    )


def _arc_trial_result(
    trial_idx: int,
    N: int,
    success: bool,
    answers: list,
    expected_grid: list,
    finish_reason: str,
    total_input_tokens: int,
    total_output_tokens: int,
    total_thinking_tokens: int,
    total_tokens_peak: int,
    n_compactions: int,
    per_turn_tokens: list,
    compaction_summaries: list,
    all_timings: list,
    conversation_log: list,
    wall_time: float,
    error_msg: Optional[str],
    content_snapshot: str = "",
    compaction_enabled: bool = True,
) -> dict:
    """Build the standardized ARC trial result / trace dict."""
    return {
        "trial_idx": trial_idx,
        "N": N,
        "problem_type": "arc",
        "compaction_enabled": compaction_enabled,
        "success": success,
        "answer": answers[0] if answers else None,    # first attempt grid
        "answer_2": answers[1] if len(answers) > 1 else None,  # second attempt grid
        "n_attempts": len(answers),
        "expected": expected_grid,
        "finish_reason": finish_reason,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "thinking_tokens": total_thinking_tokens,
        "total_tokens": total_tokens_peak,
        "n_compactions": n_compactions,
        "compaction_summaries": compaction_summaries,
        "per_turn_tokens": per_turn_tokens,
        "all_timings": all_timings,
        "conversation_log": conversation_log,
        "content_snapshot": content_snapshot,
        "wall_time_s": wall_time,
        "error": error_msg,
        "cost_usd": 0.0,
    }


# ─── Result Builder ───────────────────────────────────────────────────────────

def _build_result(
    model_name: str,
    problem_id: str,
    n_while_unbounded,
    n_reliable_no_compact,
    n_reliable_compact,
    no_compact_outer_log: list,
    no_compact_inner_log: list,
    compact_outer_log: list,
    compact_inner_log: list,
    search_range: list,
    api_calls: int,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    wall_time: float,
    skipped: bool = False,
    status: str = "completed",
) -> dict:
    return {
        "status": status,
        "problem_type": "arc",
        "model_name": model_name,
        "problem_id": problem_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_while_unbounded": n_while_unbounded,
        "n_reliable_no_compact": n_reliable_no_compact,
        "n_reliable_compact": n_reliable_compact,
        "no_compact_outer_log": no_compact_outer_log,
        "no_compact_inner_log": no_compact_inner_log,
        "compact_outer_log": compact_outer_log,
        "compact_inner_log": compact_inner_log,
        "search_range_final": search_range,
        "total_api_calls": api_calls,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cost_usd": cost,
        "wall_time_s": wall_time,
        "skipped": skipped,
    }
