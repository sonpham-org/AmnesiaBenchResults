# Author: Claude Sonnet 4.6
# Date: 30-March-2026
# PURPOSE: ARC Problem Generator for AmnesiaBench. Fetches ARC puzzle tasks from
#          the public API (arc.markbarney.net), formats them as AmnesiaBench-compatible
#          problem JSON with space-separated numeric grids, and saves to arc_problems.json.
#          Handles 15 unsolved tasks (1 ARC1-Eval + 14 ARC2-Eval) and 8 hardest solved
#          ARC2-Eval tasks (23 tasks = 23 problems). All test pairs are included in each
#          problem — model must solve ALL test pairs correctly for the task to count as solved.
#          No external dependencies beyond stdlib.
# SRP/DRY check: Pass — single responsibility (fetch + format ARC tasks), grid formatting
#                 extracted to helper function, no duplication.

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error


# ── Task Definitions ──────────────────────────────────────────────────────────

UNSOLVED_ARC1_EVAL = ["50f325b5"]

UNSOLVED_ARC2_EVAL = [
    "62593bfd", "2b83f449", "88bcf3b4", "8b7bacbf", "faa9f03d",
    "269e22fb", "4e34c42c", "21897d95", "abc82100", "9bbf930d",
    "a32d8b75", "e12f9a14", "13e47133", "88e364bc",
]

HARDEST_SOLVED_ARC2_EVAL = [
    "e3721c99", "5dbc8537", "d35bdbdc", "8e5c0c38",
    "d8e07eb2", "a25697e4", "71e489b6", "446ef5d2",
]

API_BASE = "https://arc.markbarney.net/api/puzzle/task"


# ── Helpers ───────────────────────────────────────────────────────────────────

def grid_to_text(grid: list[list[int]]) -> str:
    """Convert a 2D int grid to space-separated rows."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def fetch_task(task_id: str, retries: int = 3) -> dict:
    """Fetch a single ARC task from the API with retry logic."""
    url = f"{API_BASE}/{task_id}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AmnesiaBench/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not data.get("success"):
                raise ValueError(f"API returned success=false for {task_id}")
            return data["data"]
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} for {task_id} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to fetch {task_id} after {retries} attempts: {e}")


def build_problem_text(train_pairs: list[dict], test_pairs: list[dict]) -> str:
    """Build the full problem text with training examples and ALL test inputs.

    All test pairs are presented together. The model must solve each one.
    For single-test tasks, phrasing is singular. For multi-test, each is numbered.
    """
    parts = []

    if len(test_pairs) == 1:
        parts.append("ARC Puzzle: Predict the test output grid.")
    else:
        parts.append(f"ARC Puzzle: Predict the output grid for each of the {len(test_pairs)} test inputs.")

    for i, pair in enumerate(train_pairs, 1):
        parts.append(f"\nTraining Example {i}:")
        parts.append(f"Input:\n{grid_to_text(pair['input'])}")
        parts.append(f"\nOutput:\n{grid_to_text(pair['output'])}")

    if len(test_pairs) == 1:
        parts.append(f"\nTest Input:\n{grid_to_text(test_pairs[0]['input'])}")
        parts.append("\nGive your answer in <answer_1>...</answer_1> tags, space-separated values, one row per line.")
    else:
        for t_idx, test_pair in enumerate(test_pairs, 1):
            parts.append(f"\nTest Input {t_idx}:\n{grid_to_text(test_pair['input'])}")
        parts.append(f"\nFor each test input, give your answer in a numbered tag: <answer_1> for test 1, <answer_2> for test 2, etc.")
        parts.append("Use space-separated values, one row per line.")

    return "\n".join(parts)


def determine_source(task_id: str) -> str:
    """Determine the source label for a task ID."""
    if task_id in UNSOLVED_ARC1_EVAL:
        return "ARC1-Eval"
    return "ARC2-Eval"


def determine_problem_id(task_id: str) -> str:
    """Generate problem_id with appropriate prefix."""
    if task_id in UNSOLVED_ARC1_EVAL:
        return f"arc1_{task_id}"
    return f"arc2_{task_id}"


def build_problem(task_id: str, task_data: dict) -> dict:
    """Build a single AmnesiaBench problem dict from fetched task data.

    All test pairs are included in one problem. ground_truth is a list of grids
    (one per test pair). Model must solve ALL test pairs correctly = task solved.
    """
    train_pairs = task_data["train"]
    test_pairs = task_data["test"]

    problem_text = build_problem_text(train_pairs, test_pairs)

    # ground_truth: list of 2D grids (one per test pair)
    # Single test pair: still a list with one element for consistency
    ground_truth = [tp["output"] for tp in test_pairs]

    return {
        "problem_id": determine_problem_id(task_id),
        "problem_text": problem_text,
        "ground_truth": ground_truth,
        "num_test_pairs": len(test_pairs),
        "topic": "arc",
        "source": determine_source(task_id),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_task_ids = UNSOLVED_ARC1_EVAL + UNSOLVED_ARC2_EVAL + HARDEST_SOLVED_ARC2_EVAL
    problems = []
    total_tokens = 0
    grid_sizes = []

    print(f"Fetching {len(all_task_ids)} ARC tasks from {API_BASE}...")
    print()

    for i, task_id in enumerate(all_task_ids, 1):
        source = determine_source(task_id)
        print(f"  [{i:2d}/{len(all_task_ids)}] {task_id} ({source})...", end=" ", flush=True)

        task_data = fetch_task(task_id)
        problem = build_problem(task_id, task_data)
        problems.append(problem)

        # Stats
        n_pairs = problem["num_test_pairs"]
        pair_info = []
        for gt in problem["ground_truth"]:
            rows, cols = len(gt), len(gt[0]) if gt else 0
            grid_sizes.append((rows, cols))
            pair_info.append(f"{rows}x{cols}")
        tokens = estimate_tokens(problem["problem_text"])
        total_tokens += tokens

        grids_str = ", ".join(pair_info)
        print(f"OK — {n_pairs} test pair{'s' if n_pairs > 1 else ''} ({grids_str}), ~{tokens} tokens")

        # Be polite to the API
        if i < len(all_task_ids):
            time.sleep(0.3)

    # Save
    output_path = "arc_problems.json"
    with open(output_path, "w") as f:
        json.dump(problems, f, indent=2)

    # Summary
    print()
    print("=" * 60)
    print(f"  Problems generated: {len(problems)}")
    print(f"  ARC1-Eval:          {sum(1 for p in problems if p['source'] == 'ARC1-Eval')}")
    print(f"  ARC2-Eval:          {sum(1 for p in problems if p['source'] == 'ARC2-Eval')}")
    print(f"  Grid sizes:         {min(r for r, c in grid_sizes)}x{min(c for r, c in grid_sizes)} "
          f"to {max(r for r, c in grid_sizes)}x{max(c for r, c in grid_sizes)}")
    print(f"  Total token est:    ~{total_tokens:,}")
    print(f"  Avg tokens/problem: ~{total_tokens // len(problems):,}")
    print(f"  Saved to:           {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
