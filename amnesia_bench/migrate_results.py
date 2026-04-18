#!/usr/bin/env python3
"""
Migrate existing grouped result files into individual trial files.

Old format:
  {model}_{pid}_Unbounded.json          → contains runs[]
  {model}_{pid}_NoTIR_Compact.json      → contains binary_search[].trials[]

New format:
  {model}_{pid}_t{i}_Unbounded.json     → one per unbounded trial
  {model}_{pid}_t{i}_w{window}_baseline.json → one per compact trial

Usage:
    python3 amnesia_bench/migrate_results.py              # dry run
    python3 amnesia_bench/migrate_results.py --write      # actually write files
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def migrate_unbounded(path: Path, dry_run: bool) -> int:
    """Extract individual trial files from a grouped Unbounded result."""
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return 0

    if data.get("config") != "Unbounded":
        return 0

    model = data.get("model_name") or data.get("model", "")
    pid = data.get("problem_id", "")
    runs = data.get("runs", [])
    if not model or not pid or not runs:
        return 0

    model_safe = model.replace("/", "_").replace(":", "_")
    count = 0
    for run in runs:
        i = run.get("trial_idx", count)
        out = RESULTS_DIR / f"{model_safe}_{pid}_t{i}_Unbounded.json"
        if out.exists():
            continue
        trial_data = dict(run)
        trial_data["problem_id"] = pid
        trial_data["model"] = model
        trial_data["model_name"] = model
        trial_data["context"] = "Unbounded"
        if dry_run:
            print(f"  [DRY] {out.name}")
        else:
            with open(out, "w") as f:
                json.dump(trial_data, f, indent=2)
            print(f"  [OK]  {out.name}")
        count += 1
    return count


def migrate_compact(path: Path, dry_run: bool) -> int:
    """Extract individual trial files from a grouped Compact binary search result."""
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return 0

    config = data.get("config", {})
    if not isinstance(config, dict) or "Compact" not in config.get("name", ""):
        return 0

    model = data.get("model") or data.get("model_name", "")
    pid = data.get("problem_id", "")
    steps = data.get("binary_search", [])
    if not model or not pid or not steps:
        return 0

    model_safe = model.replace("/", "_").replace(":", "_")
    count = 0
    for step in steps:
        window = step.get("window", 0)
        trials = step.get("trials", [])
        for trial in trials:
            i = trial.get("trial_idx", 0)
            variant = trial.get("prompt_variant", "baseline")
            ctx = f"w{window}_{variant}"
            out = RESULTS_DIR / f"{model_safe}_{pid}_t{i}_{ctx}.json"
            if out.exists():
                continue
            trial_data = dict(trial)
            trial_data["problem_id"] = pid
            trial_data["model"] = model
            trial_data["model_name"] = model
            trial_data["context"] = ctx
            trial_data["window"] = window
            if dry_run:
                print(f"  [DRY] {out.name}")
            else:
                with open(out, "w") as f:
                    json.dump(trial_data, f, indent=2)
                print(f"  [OK]  {out.name}")
            count += 1
    return count


def migrate_sweep(path: Path, dry_run: bool) -> int:
    """Extract individual trial files from a Sweep result."""
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return 0

    config = data.get("config", {})
    if not isinstance(config, dict) or config.get("name") != "Sweep":
        return 0

    model = data.get("model") or data.get("model_name", "")
    pid = data.get("problem_id", "")
    results = data.get("sweep_results", [])
    if not model or not pid or not results:
        return 0

    model_safe = model.replace("/", "_").replace(":", "_")
    count = 0
    for step in results:
        window = step.get("window", 0)
        trials = step.get("trials", [])
        for trial in trials:
            i = trial.get("trial_idx", 0)
            variant = trial.get("prompt_variant", "baseline")
            ctx = f"w{window}_{variant}"
            out = RESULTS_DIR / f"{model_safe}_{pid}_t{i}_{ctx}.json"
            if out.exists():
                continue
            trial_data = dict(trial)
            trial_data["problem_id"] = pid
            trial_data["model"] = model
            trial_data["model_name"] = model
            trial_data["context"] = ctx
            trial_data["window"] = window
            if dry_run:
                print(f"  [DRY] {out.name}")
            else:
                with open(out, "w") as f:
                    json.dump(trial_data, f, indent=2)
                print(f"  [OK]  {out.name}")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Migrate grouped results to individual trial files")
    parser.add_argument("--write", action="store_true", help="Actually write files (default: dry run)")
    args = parser.parse_args()

    dry_run = not args.write
    if dry_run:
        print("DRY RUN — pass --write to actually create files\n")

    total = 0
    for f in sorted(RESULTS_DIR.iterdir()):
        if not f.suffix == ".json":
            continue
        # Skip files that are already individual trials
        if "_t" in f.stem and ("_Unbounded" in f.stem or "_w" in f.stem):
            continue
        # Skip prediction/summary files
        if f.name.endswith("_prediction.json") or f.name.endswith("_summary.json"):
            continue

        if "_Unbounded" in f.name:
            n = migrate_unbounded(f, dry_run)
            if n:
                total += n
        elif "_Compact" in f.name or "_NoTIR_Compact" in f.name:
            n = migrate_compact(f, dry_run)
            if n:
                total += n
        elif "_Sweep" in f.name:
            n = migrate_sweep(f, dry_run)
            if n:
                total += n

    print(f"\n{'Would create' if dry_run else 'Created'} {total} individual trial files")


if __name__ == "__main__":
    main()
