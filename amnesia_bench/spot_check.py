#!/usr/bin/env python3
"""
AmnesiaBench Spot Check — Validate result files before pushing to Railway.

Usage:
    python3 spot_check.py --model qwen3_32b
    python3 spot_check.py --model qwen3_32b --results-dir results/
    python3 spot_check.py --all

Checks:
  1. File count (Unbounded + Compact per problem)
  2. JSON validity
  3. Required fields present and non-null
  4. Thinking tokens counted (for models that use them)
  5. Conversation traces non-empty
  6. Solve rate / minimum_window sanity
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"  # repo_root/results/

# 10 AIMO problems
AIMO_PROBLEMS = [
    "aimo3_hard_271f3da5", "aimo3_hard_6111f603", "aimo3_hard_e2001d21",
    "crt_three_congruences", "digit_sum_ten", "handshakes_10",
    "milly_grid_walk", "modular_power_2_1000", "sum_divisors_720",
    "essay_questions",
]


def check_json(path: Path) -> tuple[bool, str]:
    """Try to load a JSON file. Returns (ok, error_msg)."""
    try:
        with open(path) as f:
            json.load(f)
        return True, ""
    except (json.JSONDecodeError, OSError) as e:
        return False, str(e)


def check_unbounded(path: Path) -> list[str]:
    """Check an Unbounded result file. Returns list of issues."""
    issues = []
    with open(path) as f:
        data = json.load(f)

    for field in ("model_name", "problem_id", "config", "avg_tokens", "min_tokens",
                  "max_tokens", "solve_rate", "n_runs"):
        if data.get(field) is None:
            issues.append(f"  Missing/null: {field}")

    if data.get("config") != "Unbounded":
        issues.append(f"  config should be 'Unbounded', got: {data.get('config')}")

    avg = data.get("avg_tokens")
    if avg is not None and (avg < 10 or avg > 500000):
        issues.append(f"  Suspicious avg_tokens: {avg}")

    rate = data.get("solve_rate")
    if rate is not None and rate == 0:
        issues.append(f"  WARNING: solve_rate=0 (model never solved this problem)")

    # Check traces exist
    runs = data.get("runs", [])
    if not runs:
        issues.append("  No run traces (runs field empty)")
    else:
        for i, run in enumerate(runs):
            conv = run.get("conversation", [])
            if len(conv) < 3:
                issues.append(f"  Run {i}: conversation too short ({len(conv)} messages)")
            # Check thinking tokens tracked
            for msg in conv:
                if msg.get("role") == "assistant" and msg.get("thinking"):
                    if msg.get("thinking_tokens", 0) == 0:
                        issues.append(f"  Run {i}: has thinking text but thinking_tokens=0")

    return issues


def check_compact(path: Path) -> list[str]:
    """Check a Compact result file. Returns list of issues."""
    issues = []
    with open(path) as f:
        data = json.load(f)

    for field in ("model_name", "problem_id", "config", "binary_search",
                  "minimum_window", "search_range_final", "prediction"):
        if field not in data:
            issues.append(f"  Missing field: {field}")

    # Check prediction
    pred = data.get("prediction", {})
    if pred.get("n_reliable_prediction") is None:
        issues.append("  prediction.n_reliable_prediction is null")

    # Check binary search steps
    bs = data.get("binary_search", [])
    if not bs:
        issues.append("  binary_search is empty (no steps)")
    else:
        for i, step in enumerate(bs):
            trials = step.get("trials", [])
            if not trials:
                issues.append(f"  Step {i}: no trials")
                continue
            for j, trial in enumerate(trials):
                conv = trial.get("conversation", [])
                if not conv:
                    issues.append(f"  Step {i} trial {j}: empty conversation")
                peak = trial.get("total_tokens_peak", 0)
                if peak == 0:
                    issues.append(f"  Step {i} trial {j}: total_tokens_peak=0")
                # Check thinking tokens
                for msg in conv:
                    if isinstance(msg, dict) and msg.get("thinking") and msg.get("thinking_tokens", 0) == 0:
                        issues.append(f"  Step {i} trial {j}: thinking text but thinking_tokens=0")

    # Sanity check minimum_window
    mw = data.get("minimum_window")
    if mw is not None and (mw < 64 or mw > 500000):
        issues.append(f"  Suspicious minimum_window: {mw}")

    return issues


def spot_check_model(model_safe: str, results_dir: Path, problems: list[str]) -> bool:
    """Run all checks for a model. Returns True if all pass."""
    print(f"\n{'='*60}")
    print(f"  Spot check: {model_safe}")
    print(f"  Results dir: {results_dir}")
    print(f"  Expected problems: {len(problems)}")
    print(f"{'='*60}")

    total_issues = 0

    # Check file counts
    expected_ub = len(problems)
    expected_co = len(problems)
    found_ub = 0
    found_co = 0

    for pid in problems:
        ub_path = results_dir / f"{model_safe}_{pid}_Unbounded.json"
        co_path = results_dir / f"{model_safe}_{pid}_NoTIR_Compact.json"

        # Unbounded
        if ub_path.exists():
            found_ub += 1
            ok, err = check_json(ub_path)
            if not ok:
                print(f"  INVALID JSON: {ub_path.name}: {err}")
                total_issues += 1
                continue
            issues = check_unbounded(ub_path)
            if issues:
                print(f"\n  {ub_path.name}:")
                for iss in issues:
                    print(iss)
                total_issues += len(issues)
        else:
            print(f"  MISSING: {ub_path.name}")
            total_issues += 1

        # Compact
        if co_path.exists():
            found_co += 1
            ok, err = check_json(co_path)
            if not ok:
                print(f"  INVALID JSON: {co_path.name}: {err}")
                total_issues += 1
                continue
            issues = check_compact(co_path)
            if issues:
                print(f"\n  {co_path.name}:")
                for iss in issues:
                    print(iss)
                total_issues += len(issues)
        else:
            print(f"  MISSING: {co_path.name}")
            total_issues += 1

    print(f"\n  Summary:")
    print(f"    Unbounded: {found_ub}/{expected_ub}")
    print(f"    Compact:   {found_co}/{expected_co}")
    print(f"    Issues:    {total_issues}")

    if total_issues == 0:
        print(f"    PASS - ready to push")
    else:
        print(f"    FAIL - fix {total_issues} issue(s) before pushing")

    return total_issues == 0


def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench Spot Check")
    parser.add_argument("--model", help="Model name (as used in filenames, e.g. qwen3_32b)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Results directory")
    parser.add_argument("--all", action="store_true", help="Check all models found in results/")
    parser.add_argument("--problems", default="aimo", choices=["aimo", "all"],
                        help="Problem set to check (default: aimo)")
    args = parser.parse_args()

    problems = AIMO_PROBLEMS  # Phase 1: AIMO only

    if args.all:
        # Find all unique model prefixes
        models = set()
        for f in args.results_dir.iterdir():
            if f.name.endswith("_Unbounded.json") or f.name.endswith("_NoTIR_Compact.json"):
                # Strip problem_id and config suffix to get model name
                for pid in problems:
                    if f"_{pid}_" in f.name:
                        model = f.name.split(f"_{pid}_")[0]
                        models.add(model)
                        break
        if not models:
            print("No model results found.")
            sys.exit(1)
        all_pass = True
        for model in sorted(models):
            if not spot_check_model(model, args.results_dir, problems):
                all_pass = False
        sys.exit(0 if all_pass else 1)

    if not args.model:
        print("Specify --model or --all")
        sys.exit(1)

    ok = spot_check_model(args.model, args.results_dir, problems)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
