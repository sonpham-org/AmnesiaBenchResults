# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Evaluation job for AmnesiaBench v3.1. Implements nested binary search to find
#   n_reliable — the smallest context window N where the model can solve a problem at
#   >=66.7% success rate (2/3 passes). Runs the binary search TWICE: once without
#   compaction (n_reliable_no_compact) and once with compaction (n_reliable_compact).
#   Saves full conversation traces per trial to traces/{model}_{problem_id}/.
#   Uses stream=False for all evaluation calls to get exact token counts + llama-server
#   timings (prefill/decode speed, KV cache hits).
#   Integration points: called by cli.py; imports clients, prompts, utils, backoff.
#   Checks prediction file first — if attempt=False, skips evaluation entirely.
#   Resume-friendly: skips if evaluation file already exists with status=completed.
# SRP/DRY check: Pass — binary search logic is isolated here; no prediction or scoring
#   code. Prompt construction delegates to prompts.py.

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .backoff import ResumptionQueue
from .prompts import build_unbounded_system, build_system_prompt, build_user_message, build_compact_prompt, build_resume_prompt, DEFAULT_PROMPT_VARIANT
from .utils import extract_final_answer, prediction_filename, evaluation_filename, sanitize_model_name

_PACKAGE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = _PACKAGE_DIR.parent / "results"

# Search parameters
OUTER_CHECKS_PER_N = 1          # single trial in outer search
INNER_CHECKS_PER_N = 3          # 3 trials in inner search
INNER_PASS_THRESHOLD = 2        # need 2/3 = 66.7% in inner search
OUTER_STOP_RATIO = 0.05         # outer stops when step < 5% of current N
INNER_STOP_ABS = 1              # inner stops when hi - lo <= 1 token
N_MIN = 1
TEMPERATURE = 0.7

# Compaction scheme parameters
MAX_TURNS = 40                  # safety cap on total turns per trial
COMPACTION_TRIGGER = 0.50       # compact at 50% of N


def run_evaluation(
    client,
    model_name: str,
    problem: dict,
    context_max: int,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> dict:
    """
    Run the nested binary search evaluation for one (model, problem) pair.

    Flow:
      1. Check prediction file — if attempt=False, skip.
      2. Test at full context_max (unbounded, no compaction). Record actual tokens used.
      3. Run binary search WITHOUT compaction → n_reliable_no_compact.
      4. Run binary search WITH compaction → n_reliable_compact.
      5. Save and return result.

    Result schema:
    {
        "model_name": str,
        "problem_id": str,
        "timestamp": ISO-8601 str,
        "n_while_unbounded": int or null,
        "n_reliable_no_compact": int or null,
        "n_reliable_compact": int or null,
        "no_compact_outer_log": [...],
        "no_compact_inner_log": [...],
        "compact_outer_log": [...],
        "compact_inner_log": [...],
        "search_range_final": [lo, hi],
        "total_api_calls": int,
        "total_input_tokens": int,
        "total_output_tokens": int,
        "total_cost_usd": float,
        "wall_time_s": float,
    }
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    problem_id = problem["problem_id"]
    out_path = evaluation_filename(results_dir, model_name, problem_id)

    # Resume-friendly: check for completed or in-progress evaluation
    checkpoint = None
    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        status = existing.get("status", "completed")
        if status == "completed":
            print(f"  [evaluate] SKIP {model_name} / {problem_id} — completed: {out_path.name}")
            return existing
        elif status == "running":
            print(f"  [evaluate] RESUMING {model_name} / {problem_id} from checkpoint")
            checkpoint = existing

    # Check prediction — if attempt=False, skip evaluation
    pred_path = prediction_filename(results_dir, model_name, problem_id)
    if pred_path.exists():
        pred = json.loads(pred_path.read_text())
        if not pred.get("attempt", True):
            print(
                f"  [evaluate] SKIP {model_name} / {problem_id} "
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

    print(f"\n  [evaluate] {model_name} / {problem_id} | context_max={context_max}")

    t_start = time.time()
    state = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    problem_text = problem["problem_text"]
    ground_truth = str(problem.get("ground_truth", ""))
    safe_model = sanitize_model_name(model_name)
    traces_dir = results_dir / "traces" / f"{safe_model}_{problem_id}"
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
        # Resume logic falls through to the appropriate phase below
    else:
        n_while_unbounded = None
        no_compact_outer_log = []
        no_compact_inner_log = []
        compact_outer_log = []
        compact_inner_log = []
        phase = "unbounded"

    # ── Step 1: Unbounded test — no restriction, measure natural token usage ──
    if phase == "unbounded" or n_while_unbounded is None:
        print(f"  [evaluate] Unbounded test — no context restriction, measuring natural usage...")
        unbounded_pass, unbounded_log = _test_unbounded(
            client, problem_text, ground_truth, state=state,
            context_max=context_max, variant=variant,
        )
        if not unbounded_pass:
            print(f"  [evaluate] UNSOLVABLE at context_max={context_max} — skipping search")
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

        # Use actual tokens from the successful trial, not context_max
        actual_tokens = max(
            (t.get("total_tokens", 0) for t in unbounded_log if t.get("success")),
            default=0,
        )
        if actual_tokens == 0:
            # Token counting failed or streaming fallback — estimate from content
            for t in unbounded_log:
                if t.get("success"):
                    content_len = len(t.get("content_snapshot", ""))
                    actual_tokens = max(actual_tokens, content_len // 4)
        if actual_tokens == 0:
            actual_tokens = context_max
        n_while_unbounded = actual_tokens
        print(f"  [evaluate] n_while_unbounded={n_while_unbounded} (actual tokens used)")
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
        print(f"  [evaluate] Outer binary search (NO compaction) [{N_MIN}, {n_while_unbounded}] ...")
        if phase == "no_compact_outer":
            nc_outer_lo, nc_outer_hi = N_MIN, n_while_unbounded
        else:
            nc_outer_lo, nc_outer_hi = _replay_search_log(
                no_compact_outer_log, N_MIN, n_while_unbounded
            )

        no_compact_outer_log, nc_transition_lo, nc_transition_hi = _outer_binary_search(
            client, problem_text, ground_truth,
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
            variant=variant,
        )
        phase = "no_compact_inner"

        mid = (nc_transition_lo + nc_transition_hi) // 2
        nc_inner_lo = max(N_MIN, mid - (nc_transition_hi - nc_transition_lo) * 3 // 2)
        nc_inner_hi = min(n_while_unbounded, mid + (nc_transition_hi - nc_transition_lo) * 3 // 2)
        nc_inner_hi = max(nc_inner_hi, nc_transition_hi)

        print(
            f"  [evaluate] Inner binary search (NO compaction) [{nc_inner_lo}, {nc_inner_hi}] ..."
        )
        no_compact_inner_log, n_reliable_no_compact = _inner_binary_search(
            client, problem_text, ground_truth,
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
            variant=variant,
        )
        print(f"  [evaluate] n_reliable_no_compact={n_reliable_no_compact}")
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
        print(f"  [evaluate] Outer binary search (WITH compaction) [{N_MIN}, {n_while_unbounded}] ...")
        if phase == "compact_outer":
            c_outer_lo, c_outer_hi = N_MIN, n_while_unbounded
        else:
            c_outer_lo, c_outer_hi = _replay_search_log(
                compact_outer_log, N_MIN, n_while_unbounded
            )

        compact_outer_log, c_transition_lo, c_transition_hi = _outer_binary_search(
            client, problem_text, ground_truth,
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
            variant=variant,
        )
        phase = "compact_inner"

        mid = (c_transition_lo + c_transition_hi) // 2
        c_inner_lo = max(N_MIN, mid - (c_transition_hi - c_transition_lo) * 3 // 2)
        c_inner_hi = min(n_while_unbounded, mid + (c_transition_hi - c_transition_lo) * 3 // 2)
        c_inner_hi = max(c_inner_hi, c_transition_hi)

        print(
            f"  [evaluate] Inner binary search (WITH compaction) [{c_inner_lo}, {c_inner_hi}] ..."
        )
        compact_inner_log, n_reliable_compact = _inner_binary_search(
            client, problem_text, ground_truth,
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
            variant=variant,
        )
        print(f"  [evaluate] n_reliable_compact={n_reliable_compact}")
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
        f"  [evaluate] DONE — n_reliable_no_compact={n_reliable_no_compact} "
        f"n_reliable_compact={n_reliable_compact} | "
        f"api_calls={state['api_calls']} | {wall_time}s → {out_path.name}"
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
    client, problem_text: str, ground_truth: str,
    lo: int, hi: int, state: dict,
    compaction_enabled: bool,
    existing_log: Optional[list] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_data: Optional[dict] = None,
    traces_dir: Optional[Path] = None,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> tuple:
    """
    Find the rough fail→pass transition zone using a single trial per N.
    Stop when step < 5% of current N.
    Returns (log, transition_lo, transition_hi).
    Writes checkpoint after each step for resumability.
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
        print(f"  [outer-{label} step {step}] N={mid}  range=[{lo},{hi}]")
        passed, trial_log = _test_n(
            client, problem_text, ground_truth, mid,
            n_trials=OUTER_CHECKS_PER_N, state=state,
            compaction_enabled=compaction_enabled,
            traces_dir=traces_dir,
            variant=variant,
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

    print(f"  [outer-{label}] transition zone: [{last_fail_lo}, {first_pass_hi}]")
    return log, last_fail_lo, first_pass_hi


# ─── Inner Binary Search ──────────────────────────────────────────────────────

def _inner_binary_search(
    client, problem_text: str, ground_truth: str,
    lo: int, hi: int, state: dict,
    compaction_enabled: bool,
    existing_log: Optional[list] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_data: Optional[dict] = None,
    traces_dir: Optional[Path] = None,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> tuple:
    """
    Refine the transition zone using 3 trials per N; require 2/3 to pass.
    Stop when hi - lo <= 1 token.
    Returns (log, n_reliable) where n_reliable is the smallest passing N.
    Writes checkpoint after each step for resumability.
    """
    log = existing_log if existing_log else []
    step = len(log)
    n_reliable = hi  # conservative default
    label = "compact" if compaction_enabled else "nocompact"

    while hi - lo > INNER_STOP_ABS:
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break

        step += 1
        print(f"  [inner-{label} step {step}] N={mid}  range=[{lo},{hi}]")
        passed, trial_log = _test_n(
            client, problem_text, ground_truth, mid,
            n_trials=INNER_CHECKS_PER_N, state=state,
            pass_threshold=INNER_PASS_THRESHOLD,
            compaction_enabled=compaction_enabled,
            traces_dir=traces_dir,
            variant=variant,
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

    print(f"  [inner-{label}] n_reliable={n_reliable}")
    return log, n_reliable


# ─── Unbounded Test ───────────────────────────────────────────────────────────

def _test_unbounded(
    client, problem_text: str, ground_truth: str, state: dict,
    context_max: int = 262144,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> tuple:
    """
    Run ONE trial with NO context window restriction.
    Just: 'solve this problem'. No mention of N, no compaction, no constraints.
    Measures how many tokens the model naturally uses.
    Returns (passed: bool, trial_log: list with one entry).
    """
    t0 = time.time()
    system_prompt = build_unbounded_system(variant)
    user_msg = build_user_message(problem_text, variant, N=None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_msg})

    try:
        # Use a generous max_tokens but don't tell the model about any limit
        resp = client.generate(messages, max_tokens=context_max, stream=False)
    except Exception as e:
        result = {
            "trial_idx": 0, "N": "unbounded", "success": False,
            "answer": None, "expected": ground_truth,
            "finish_reason": "error", "input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0,
            "wall_time_s": round(time.time() - t0, 2), "error": str(e),
        }
        state["api_calls"] += 1
        return False, [result]

    content = resp.get("content", "") or ""
    answer = _extract_final_answer_from_content(content)
    success = answer is not None and str(answer).strip() == str(ground_truth).strip()

    total_tokens = resp.get("total_tokens", 0)
    input_tokens = resp.get("input_tokens", 0)
    output_tokens = resp.get("output_tokens", 0)
    timings = resp.get("timings", {})

    # If token counting failed (streaming fallback), estimate
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
        "answer": answer, "expected": ground_truth,
        "finish_reason": resp.get("finish_reason", "unknown"),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "timings": timings,
        "wall_time_s": wall_time, "error": None,
    }
    status = "PASS" if success else "FAIL"
    print(f"    unbounded: {status} | ans={answer!r} | {total_tokens} total tokens | {wall_time}s")

    return success, [result]


# ─── Single N Test ────────────────────────────────────────────────────────────

def _test_n(
    client, problem_text: str, ground_truth: str,
    N: int, n_trials: int, state: dict,
    compaction_enabled: bool,
    pass_threshold: int = 1,
    traces_dir: Optional[Path] = None,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> tuple:
    """
    Run n_trials trials at context window N in parallel.
    Returns (passed: bool, trial_log: list).
    passed = True if successes >= pass_threshold.
    Saves each trial's full conversation trace to traces_dir if provided.
    """
    results = [None] * n_trials
    label = "compact" if compaction_enabled else "nocompact"

    def _run_one(idx):
        return _run_trial(
            client, problem_text, ground_truth, N, idx,
            compaction_enabled=compaction_enabled,
            variant=variant,
        )

    with ThreadPoolExecutor(max_workers=n_trials) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n_trials)}
        for future in as_completed(futures):
            idx = futures[future]
            r = future.result()
            results[idx] = r
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"    trial {idx}: {status} | ans={r['answer']!r} | "
                f"{r['finish_reason']} | {r['total_tokens']} tok | {r['wall_time_s']:.1f}s"
            )
            state["api_calls"] += 1
            state["input_tokens"] += r.get("input_tokens", 0)
            state["output_tokens"] += r.get("output_tokens", 0)

            # Save trace to disk
            if traces_dir is not None:
                trace_file = traces_dir / f"{label}_N{N}_trial{idx}.json"
                try:
                    trace_file.write_text(json.dumps(r, indent=2))
                except Exception as e:
                    print(f"    [trace] WARNING — failed to save trace: {e}")

    n_pass = sum(1 for r in results if r["success"])
    passed = n_pass >= pass_threshold
    print(f"    [{n_pass}/{n_trials} passed — {'PASS' if passed else 'FAIL'}]")
    return passed, results


# ─── Single Trial ─────────────────────────────────────────────────────────────

def _extract_final_answer_from_content(text: str):
    """Extract answer from {final_answer: "ANSWER"} format, with fallback."""
    if not text:
        return None
    match = re.search(r'\{final_answer:\s*"([^"]+)"\}', text)
    if match:
        return match.group(1).strip()
    return extract_final_answer(text)


def _run_trial(
    client, problem_text: str, ground_truth: str,
    N: int, trial_idx: int,
    compaction_enabled: bool = True,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> dict:
    """
    Run one evaluation trial at context window N.

    When compaction_enabled=False:
        Single-shot: send prompt, get response, done.
        No compaction loop. No context resets.
        Used for n_reliable_no_compact measurement.

    When compaction_enabled=True:
        Full compaction loop:
          1. Generate response.
          2. If final_answer found → success.
          3. If tokens >= 50% of N → inject compaction prompt, get summary.
          4. Reset conversation to: system + problem + summary.
          5. Probe reset cost. If >= 50% of N → FAIL (compaction_insufficient).
          6. Else continue.

    All calls use stream=False for exact token counts and llama-server timings.

    Returns a dict that IS the trace file — full conversation log included.
    """
    t0 = time.time()

    system_prompt = build_system_prompt(N, variant)
    half_budget = max(N // 2, 256)
    user_msg = build_user_message(problem_text, variant, N=N, tokens_left=half_budget)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_msg})

    # Full conversation log — becomes the trace file
    conversation_log = list(messages)

    n_compactions = 0
    compaction_summaries = []
    per_turn_tokens = []
    all_timings = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    total_tokens_peak = 0
    answer = None
    finish_reason = "max_turns"
    error_msg = None
    content_snapshot = ""  # last assistant content, for token estimation fallback

    # ── No-compaction path: single shot ──────────────────────────────────────
    if not compaction_enabled:
        turn_t0 = time.time()
        try:
            resp = client.generate(messages, max_tokens=N, stream=False)
        except Exception as e:
            error_msg = str(e)
            finish_reason = "error"
            wall_time = round(time.time() - t0, 2)
            return _trial_result(
                trial_idx, N, False, None, ground_truth,
                finish_reason, 0, 0, 0, 0, 0,
                [], [], [], conversation_log, wall_time, error_msg,
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

        # Token estimation fallback if server returned 0 (should not happen with stream=False)
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

        answer = _extract_final_answer_from_content(content)
        finish_reason = "solved" if answer is not None else (fr if fr else "no_answer")

        wall_time = round(time.time() - t0, 2)
        return _trial_result(
            trial_idx, N,
            success=(answer is not None and str(answer).strip() == str(ground_truth).strip()),
            answer=answer, ground_truth=ground_truth,
            finish_reason=finish_reason,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_thinking_tokens=total_thinking_tokens,
            total_tokens_peak=total_tokens_peak,
            n_compactions=0,
            per_turn_tokens=per_turn_tokens,
            compaction_summaries=[],
            all_timings=all_timings,
            conversation_log=conversation_log,
            wall_time=wall_time,
            error_msg=error_msg,
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

        # Token estimation fallback
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

        # ── Check for final answer ────────────────────────────────────────────
        answer = _extract_final_answer_from_content(content)
        if answer is not None:
            finish_reason = "solved"
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

            reset_user = build_resume_prompt(
                user_msg=user_msg, summary=summary,
                n_done=n_compactions, variant=variant, N=N,
                tokens_left=half_budget,
            )
            reset_messages = []
            if system_prompt:
                reset_messages.append({"role": "system", "content": system_prompt})
            reset_messages.append({"role": "user", "content": reset_user})

            # Probe how many tokens the reset prompt costs (stream=False for exact count)
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
                "content": f"[PROBE — reset context token count]",
                "reset_tokens": reset_tokens,
                "wall_time_s": probe_time,
            })

            if reset_tokens >= N * COMPACTION_TRIGGER:
                finish_reason = "compaction_insufficient"
                total_tokens_peak = max(total_tokens_peak, reset_tokens)
                break

            # Compaction succeeded — reset context and continue
            messages = reset_messages
            conversation_log.append({"role": "user", "content": reset_user})
            total_tokens_peak = reset_tokens
            awaiting_compact = False
            continue

        # ── Check if we've hit 50% — trigger compaction ───────────────────────
        if total_tokens_peak >= N * COMPACTION_TRIGGER:
            compaction_prompt = build_compact_prompt(
                n=n_compactions + 1, prev_output=content, variant=variant,
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
    success = answer is not None and str(answer).strip() == str(ground_truth).strip()

    return _trial_result(
        trial_idx, N, success, answer, ground_truth,
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
        content_snapshot=content_snapshot,
        compaction_enabled=True,
    )


def _trial_result(
    trial_idx: int,
    N: int,
    success: bool,
    answer,
    ground_truth: str,
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
    """Build the standardized trial result / trace dict."""
    return {
        "trial_idx": trial_idx,
        "N": N,
        "compaction_enabled": compaction_enabled,
        "success": success,
        "answer": answer,
        "expected": ground_truth,
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
        "cost_usd": 0.0,  # placeholder; priced per-model by caller if needed
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


def run_evaluations_for_problems(
    client,
    model_name: str,
    problems: list,
    context_max: int,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> list:
    """Run evaluation job for a list of problems. Returns list of result dicts."""
    results = []
    for problem in problems:
        try:
            result = run_evaluation(
                client, model_name, problem, context_max,
                results_dir=results_dir,
                queue=queue,
                force=force,
                variant=variant,
            )
        except Exception as e:
            err = str(e)
            print(f"  [evaluate] FAILED {model_name} / {problem['problem_id']}: {err}")
            if queue:
                queue.push(model_name, problem["problem_id"], "evaluation", err)
            continue
        results.append(result)
    return results
