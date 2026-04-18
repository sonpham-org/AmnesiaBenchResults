#!/usr/bin/env python3
"""
Upload local AmnesiaBench JSON results to Railway Postgres.

Reads results/*.json, parses Unbounded and Compact files,
and upserts into amnesia_results + amnesia_traces tables.

Usage:
    python3 upload_results.py                      # upload all
    python3 upload_results.py --model qwen3.5:9b   # upload one model
    python3 upload_results.py --dry-run             # show what would be uploaded
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DB_URL_FILE = Path("/tmp/railway_db_url.txt")


def get_db_url() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if DB_URL_FILE.exists():
        return DB_URL_FILE.read_text().strip()
    raise RuntimeError("No DATABASE_URL env var or /tmp/railway_db_url.txt found")


def upload_unbounded(cur, path: Path, dry_run: bool = False) -> bool:
    """Upload an Unbounded result file. Returns True if uploaded."""
    with open(path) as f:
        d = json.load(f)

    model = d.get("model_name") or d.get("model", "")
    pid = d.get("problem_id", "")
    if not model or not pid:
        return False

    config = "Unbounded"
    avg_tokens = d.get("avg_tokens")
    min_tokens = d.get("min_tokens")
    max_tokens = d.get("max_tokens")
    solve_rate = d.get("solve_rate")
    n_runs = d.get("n_runs")
    context_window = d.get("context_window")
    timestamp = d.get("timestamp")

    if dry_run:
        print(f"  [DRY] {model} / {pid} / {config}: avg={avg_tokens} rate={solve_rate}")
        return True

    # Upsert result row
    cur.execute("""
        INSERT INTO amnesia_results (model_name, problem_id, config,
            avg_tokens, min_tokens, max_tokens, solve_rate, n_runs, context_window, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, problem_id, config) DO UPDATE SET
            avg_tokens = EXCLUDED.avg_tokens,
            min_tokens = EXCLUDED.min_tokens,
            max_tokens = EXCLUDED.max_tokens,
            solve_rate = EXCLUDED.solve_rate,
            n_runs = EXCLUDED.n_runs,
            context_window = EXCLUDED.context_window,
            timestamp = EXCLUDED.timestamp
        RETURNING id
    """, (model, pid, config, avg_tokens, min_tokens, max_tokens,
          solve_rate, n_runs, context_window, timestamp))
    result_id = cur.fetchone()[0]

    # Upload individual run traces
    runs = d.get("runs", [])
    for i, run in enumerate(runs):
        cur.execute("""
            INSERT INTO amnesia_traces (result_id, trace_type, step_index,
                success, answer, total_tokens, wall_time_s, conversation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (result_id, trace_type, step_index) DO UPDATE SET
                success = EXCLUDED.success,
                answer = EXCLUDED.answer,
                total_tokens = EXCLUDED.total_tokens,
                wall_time_s = EXCLUDED.wall_time_s,
                conversation = EXCLUDED.conversation
        """, (result_id, "unbounded_run", i,
              run.get("success"), str(run.get("answer")),
              run.get("total_tokens"), run.get("wall_time_s"),
              Json(run.get("conversation"))))

    return True


def upload_compact(cur, path: Path, dry_run: bool = False) -> bool:
    """Upload a Compact result file. Returns True if uploaded."""
    with open(path) as f:
        d = json.load(f)

    model = d.get("model_name") or d.get("model", "")
    pid = d.get("problem_id", "")
    if not model or not pid:
        return False

    config_raw = d.get("config", {})
    config_name = config_raw.get("name", "") if isinstance(config_raw, dict) else str(config_raw)
    if "Compact" not in config_name:
        return False

    config = "NoTIR_Compact"
    bs = d.get("binary_search", [])
    minimum_window = d.get("minimum_window")
    search_range = d.get("search_range_final", [None, None])
    prediction = d.get("prediction", {})
    timestamp = d.get("timestamp")

    # Determine result state
    if len(bs) == 0:
        result_state = "OPTED_OUT"
    elif minimum_window is not None:
        result_state = "CONVERGED"
    else:
        # Model attempted but never passed at any window → FAILED (not "unsolvable")
        # It may have produced answers, just wrong ones
        total_successes = sum(s.get("n_success", 0) for s in bs)
        total_trials = sum(s.get("n_trials", 0) for s in bs)
        if total_successes > 0:
            result_state = f"PARTIAL ({total_successes}/{total_trials})"
        else:
            result_state = "FAILED"

    # Build summary (step windows + pass/fail, no traces)
    summary = []
    for step in bs:
        summary.append({
            "window": step.get("window"),
            "passed": step.get("passed"),
            "n_success": step.get("n_success"),
            "n_trials": step.get("n_trials"),
        })

    if dry_run:
        print(f"  [DRY] {model} / {pid} / {config}: min_window={minimum_window} state={result_state} steps={len(bs)}")
        return True

    # Upsert result row
    cur.execute("""
        INSERT INTO amnesia_results (model_name, problem_id, config,
            minimum_window, result_state, search_range_lo, search_range_hi,
            predicted_n, success_prediction, timestamp, summary_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, problem_id, config) DO UPDATE SET
            minimum_window = EXCLUDED.minimum_window,
            result_state = EXCLUDED.result_state,
            search_range_lo = EXCLUDED.search_range_lo,
            search_range_hi = EXCLUDED.search_range_hi,
            predicted_n = EXCLUDED.predicted_n,
            success_prediction = EXCLUDED.success_prediction,
            timestamp = EXCLUDED.timestamp,
            summary_json = EXCLUDED.summary_json
        RETURNING id
    """, (model, pid, config, minimum_window, result_state,
          search_range[0] if len(search_range) > 0 else None,
          search_range[1] if len(search_range) > 1 else None,
          prediction.get("n_reliable_prediction"),
          prediction.get("success_prediction"),
          timestamp, Json(summary)))
    result_id = cur.fetchone()[0]

    # Upload prediction trace
    if prediction.get("raw_response"):
        cur.execute("""
            INSERT INTO amnesia_traces (result_id, trace_type, step_index,
                answer, conversation)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (result_id, trace_type, step_index) DO UPDATE SET
                answer = EXCLUDED.answer,
                conversation = EXCLUDED.conversation
        """, (result_id, "prediction", 0,
              str(prediction.get("n_reliable_prediction")),
              Json({"raw_response": prediction.get("raw_response")})))

    # Upload binary search step traces (1 trial per step — the first one)
    for i, step in enumerate(bs):
        trials = step.get("trials", [])
        if not trials:
            continue
        trial = trials[0]  # Only store the first/representative trial
        cur.execute("""
            INSERT INTO amnesia_traces (result_id, trace_type, step_index,
                window_size, success, answer, total_tokens, n_compactions,
                wall_time_s, conversation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (result_id, trace_type, step_index) DO UPDATE SET
                window_size = EXCLUDED.window_size,
                success = EXCLUDED.success,
                answer = EXCLUDED.answer,
                total_tokens = EXCLUDED.total_tokens,
                n_compactions = EXCLUDED.n_compactions,
                wall_time_s = EXCLUDED.wall_time_s,
                conversation = EXCLUDED.conversation
        """, (result_id, "compact_step", i,
              step.get("window"),
              trial.get("success"),
              str(trial.get("answer")),
              trial.get("total_tokens_peak") or trial.get("total_tokens"),
              trial.get("n_compactions", 0),
              trial.get("wall_time_s"),
              Json(trial.get("conversation"))))

    return True


def upload_sweep(cur, path: Path, dry_run: bool = False) -> bool:
    """Upload a Sweep result file. Returns True if uploaded."""
    with open(path) as f:
        d = json.load(f)

    model = d.get("model_name") or d.get("model", "")
    pid = d.get("problem_id", "")
    if not model or not pid:
        return False

    config = "Sweep"
    sweep_results = d.get("sweep_results", [])
    min_trunc = d.get("min_passing_truncation")
    pass_curve = d.get("pass_curve", [])
    wall_time = d.get("wall_time_s")
    timestamp = d.get("timestamp")

    if dry_run:
        print(f"  [DRY] {model} / {pid} / {config}: {len(sweep_results)} points, min_trunc={min_trunc}")
        return True

    # Build summary
    summary = [{
        "truncation_point": s.get("truncation_point"),
        "window": s.get("window"),
        "pass_rate": s.get("pass_rate"),
        "n_success": s.get("n_success"),
        "n_trials": s.get("n_trials"),
        "passed": s.get("passed"),
    } for s in sweep_results]

    cur.execute("""
        INSERT INTO amnesia_results (model_name, problem_id, config,
            minimum_window, result_state, timestamp, summary_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, problem_id, config) DO UPDATE SET
            minimum_window = EXCLUDED.minimum_window,
            result_state = EXCLUDED.result_state,
            timestamp = EXCLUDED.timestamp,
            summary_json = EXCLUDED.summary_json
        RETURNING id
    """, (model, pid, config,
          min_trunc,
          "CONVERGED" if min_trunc else "NO_PASS",
          timestamp,
          Json(summary)))
    result_id = cur.fetchone()[0]

    # Upload individual sweep point traces
    for i, step in enumerate(sweep_results):
        trials = step.get("trials", [])
        if not trials:
            continue
        trial = trials[0]
        cur.execute("""
            INSERT INTO amnesia_traces (result_id, trace_type, step_index,
                window_size, success, answer, total_tokens, n_compactions,
                wall_time_s, conversation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (result_id, trace_type, step_index) DO UPDATE SET
                window_size = EXCLUDED.window_size,
                success = EXCLUDED.success,
                answer = EXCLUDED.answer,
                total_tokens = EXCLUDED.total_tokens,
                n_compactions = EXCLUDED.n_compactions,
                wall_time_s = EXCLUDED.wall_time_s,
                conversation = EXCLUDED.conversation
        """, (result_id, "sweep_point", i,
              step.get("window"),
              trial.get("success"),
              str(trial.get("answer")),
              trial.get("total_tokens_peak") or trial.get("total_tokens"),
              trial.get("n_compactions", 0),
              trial.get("wall_time_s"),
              Json(trial.get("conversation"))))

    return True


def main():
    parser = argparse.ArgumentParser(description="Upload AmnesiaBench results to Railway DB")
    parser.add_argument("--model", help="Only upload results for this model")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if not args.dry_run:
        db_url = get_db_url()
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()
    else:
        conn = cur = None

    model_filter = args.model.replace("/", "_").replace(":", "_") if args.model else None

    ub_count = 0
    co_count = 0
    sw_count = 0
    skip_count = 0

    for f in sorted(args.results_dir.iterdir()):
        if not f.suffix == ".json":
            continue
        if f.name.endswith("_summary.json"):
            continue
        if f.name.endswith("_prediction.json"):
            continue
        # Skip individual trial files (new format: _t{N}_ in name)
        if "_t" in f.stem and ("_Unbounded" in f.stem or "_w" in f.stem):
            continue

        if model_filter and not f.name.startswith(model_filter):
            skip_count += 1
            continue

        try:
            if f.name.endswith("_Unbounded.json"):
                if upload_unbounded(cur, f, dry_run=args.dry_run):
                    ub_count += 1
            elif "_Compact" in f.name:
                if upload_compact(cur, f, dry_run=args.dry_run):
                    co_count += 1
            elif f.name.endswith("_Sweep.json"):
                if upload_sweep(cur, f, dry_run=args.dry_run):
                    sw_count += 1
        except Exception as e:
            print(f"  ERROR: {f.name}: {e}")

    print(f"\nUploaded: {ub_count} Unbounded, {co_count} Compact, {sw_count} Sweep (skipped {skip_count})")

    if conn:
        # Print DB counts
        cur.execute("SELECT config, COUNT(*) FROM amnesia_results GROUP BY config")
        for config, count in cur.fetchall():
            print(f"  DB total {config}: {count}")
        cur.execute("SELECT trace_type, COUNT(*) FROM amnesia_traces GROUP BY trace_type")
        for ttype, count in cur.fetchall():
            print(f"  DB traces {ttype}: {count}")
        conn.close()


if __name__ == "__main__":
    main()
