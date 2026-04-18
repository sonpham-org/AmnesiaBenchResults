# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Prediction job for AmnesiaBench v3. Asks the model to predict whether it can
#   solve a problem and what minimum N it needs, before any evaluation trials run.
#   Saves results to {model}_{problem}_prediction.json.
#   Integration points: called by cli.py; imports clients, prompts, utils, backoff.
#   Resume-friendly: skips if prediction file already exists.
# SRP/DRY check: Pass — prediction I/O and parse logic are isolated here; no binary
#   search or scoring code. Prompt text lives in prompts.py only.

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .backoff import ResumptionQueue
from .prompts import build_prediction_prompt, DEFAULT_PROMPT_VARIANT
from .utils import prediction_filename

# Default results directory (relative to package parent)
_PACKAGE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = _PACKAGE_DIR.parent / "results"

PREDICTION_MAX_TOKENS = 300


def run_prediction(
    client,
    model_name: str,
    problem: dict,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> dict:
    """
    Run the prediction job for one (model, problem) pair.

    Returns the prediction result dict (also saves to disk).
    If the prediction file already exists and force=False, loads and returns it.

    Result schema:
    {
        "model_name": str,
        "problem_id": str,
        "timestamp": ISO-8601 str,
        "attempt": bool,
        "n_predicted": int or null,   # null = inf (model couldn't/wouldn't predict)
        "raw_response": str,
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int,
        "parse_success": bool,
        "fallback_used": bool,
    }
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    problem_id = problem["problem_id"]
    out_path = prediction_filename(results_dir, model_name, problem_id)

    # Resume-friendly skip
    if out_path.exists() and not force:
        print(f"  [predict] SKIP {model_name} / {problem_id} — file exists: {out_path.name}")
        return json.loads(out_path.read_text())

    print(f"\n  [predict] {model_name} / {problem_id}")

    prompt = build_prediction_prompt(problem["problem_text"], variant=variant)
    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.generate(messages, max_tokens=PREDICTION_MAX_TOKENS)
    except Exception as e:
        err_str = str(e)
        print(f"  [predict] API ERROR: {err_str}")
        if queue:
            queue.push(model_name, problem_id, "prediction", err_str)
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
        f"  [predict] attempt={attempt}, n_predicted={n_predicted}, "
        f"parse_success={parse_success} → {out_path.name}"
    )
    return result


def _parse_prediction_response(raw: str) -> tuple:
    """
    Parse the model's prediction response.

    Expected format:
        {attempt: "True", N: "2048"}
        {attempt: "False", N: "0"}

    Returns (attempt: bool, n_predicted: int|None, parse_success: bool).
    On parse failure: attempt=True, n_predicted=None (=inf), parse_success=False.
    """
    # Try to extract attempt field
    attempt_match = re.search(
        r'\{attempt:\s*"(True|False)"', raw, re.IGNORECASE
    )
    n_match = re.search(
        r'N:\s*"(\d+)"', raw
    )

    if not attempt_match or not n_match:
        print(f"  [predict] Parse failed — fallback: attempt=True, N=inf")
        return True, None, False

    attempt_str = attempt_match.group(1).strip().lower()
    attempt = attempt_str == "true"

    n_raw = int(n_match.group(1))
    if not attempt or n_raw == 0:
        n_predicted = None  # inf / opted out
    else:
        n_predicted = n_raw

    return attempt, n_predicted, True


def _fallback_result(model_name: str, problem_id: str, raw_response: str = "") -> dict:
    """Return a safe fallback prediction (attempt=True, N=inf) on API error."""
    return {
        "model_name": model_name,
        "problem_id": problem_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt": True,
        "n_predicted": None,
        "raw_response": raw_response,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "parse_success": False,
        "fallback_used": True,
    }


def run_predictions_for_problems(
    client,
    model_name: str,
    problems: list,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> list:
    """Run prediction job for a list of problems. Returns list of result dicts."""
    results = []
    for problem in problems:
        result = run_prediction(
            client, model_name, problem,
            results_dir=results_dir,
            queue=queue,
            force=force,
            variant=variant,
        )
        results.append(result)
    return results
