# Author: Claude Sonnet 4.6
# Date: 30-March-2026
# PURPOSE: ARC Answer Evaluator for AmnesiaBench. Parses model responses containing
#          grid answers inside numbered <answer_N> tags — one attempt per test pair.
#          Single test: <answer_1>. Two tests: <answer_1> + <answer_2>. Etc.
#          Task is solved if ALL test pairs match exactly. No external dependencies.
# SRP/DRY check: Pass — extraction and evaluation are cleanly separated functions,
#                 grid parsing logic is not duplicated.

from __future__ import annotations

import json
import re
import sys
from typing import Optional


def _parse_grid(raw: str) -> Optional[list[list[int]]]:
    """Parse space-separated rows into a 2D list of ints."""
    raw = raw.strip()
    if not raw:
        return None
    grid = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            row = [int(x) for x in line.split()]
            if row:
                grid.append(row)
        except ValueError:
            return None
    return grid if grid else None


def extract_grid_answer(response_text: str, index: int = 1) -> Optional[list[list[int]]]:
    """Extract grid from <answer_N> tag where N = index."""
    match = re.search(rf"<answer_{index}>\s*(.*?)\s*</answer_{index}>", response_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return _parse_grid(match.group(1))


def extract_all_numbered_answers(response_text: str) -> dict[int, Optional[list[list[int]]]]:
    """
    Extract all <answer_N>...</answer_N> blocks from response.
    Returns dict mapping N -> parsed grid (or None if unparseable).
    """
    matches = re.finditer(r"<answer_(\d+)>\s*(.*?)\s*</answer_\1>", response_text, re.DOTALL | re.IGNORECASE)
    results = {}
    for m in matches:
        idx = int(m.group(1))
        results[idx] = _parse_grid(m.group(2))
    return results


def evaluate_single_grid(predicted: Optional[list[list[int]]], ground_truth: list[list[int]]) -> dict:
    """Evaluate one predicted grid against one ground truth grid."""
    if predicted is None:
        return {"correct": False, "predicted_grid": None, "expected_grid": ground_truth, "dimensions_match": False}

    pred_rows, gt_rows = len(predicted), len(ground_truth)
    pred_cols = len(predicted[0]) if predicted else 0
    gt_cols = len(ground_truth[0]) if ground_truth else 0

    return {
        "correct": predicted == ground_truth,
        "predicted_grid": predicted,
        "expected_grid": ground_truth,
        "dimensions_match": (pred_rows == gt_rows) and (pred_cols == gt_cols),
    }


def evaluate_arc_answer(response_text: str, ground_truth: list) -> dict:
    """
    Evaluate model's ARC answer against ground truth.

    ground_truth: list of 2D grids (one per test pair).
    Model provides <answer_1> for test 1, <answer_2> for test 2, etc.
    One attempt per test pair. Task solved if all pairs match exactly.
    """
    if not ground_truth:
        return {"correct": False, "num_test_pairs": 0, "per_pair": []}

    # Detect legacy single-grid format
    is_multi = isinstance(ground_truth[0], list) and ground_truth[0] and isinstance(ground_truth[0][0], list)
    if not is_multi:
        ground_truth = [ground_truth]

    answers = extract_all_numbered_answers(response_text)

    per_pair = []
    for i, gt in enumerate(ground_truth):
        # answer_1 = test pair 0, answer_2 = test pair 1, etc.
        pred = answers.get(i + 1)
        result = evaluate_single_grid(pred, gt)
        result["test_pair_index"] = i
        result["answer_tag"] = f"answer_{i + 1}"
        per_pair.append(result)

    return {
        "correct": all(p["correct"] for p in per_pair),
        "num_test_pairs": len(ground_truth),
        "num_answers_found": len(answers),
        "per_pair": per_pair,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ARC Evaluator — Self-Test")
    print("=" * 50)

    # 1: Single test correct
    gt1 = [[[0, 0, 2], [0, 1, 0], [2, 2, 2]]]
    r1 = evaluate_arc_answer("<answer_1>\n0 0 2\n0 1 0\n2 2 2\n</answer_1>", gt1)
    assert r1["correct"]; print("  [PASS] 1: Single test correct")

    # 2: Single test wrong
    r2 = evaluate_arc_answer("<answer_1>\n0 0 0\n0 0 0\n0 0 0\n</answer_1>", gt1)
    assert not r2["correct"]; print("  [PASS] 2: Single test wrong")

    # 3: No tags
    r3 = evaluate_arc_answer("no answer", gt1)
    assert not r3["correct"]; print("  [PASS] 3: No tags")

    # 4: Two tests both correct
    gt4 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    r4 = evaluate_arc_answer("<answer_1>\n1 2\n3 4\n</answer_1>\n<answer_2>\n5 6\n7 8\n</answer_2>", gt4)
    assert r4["correct"]; print("  [PASS] 4: Two tests both correct")

    # 5: Two tests, second wrong
    r5 = evaluate_arc_answer("<answer_1>\n1 2\n3 4\n</answer_1>\n<answer_2>\n0 0\n0 0\n</answer_2>", gt4)
    assert not r5["correct"]
    assert r5["per_pair"][0]["correct"] and not r5["per_pair"][1]["correct"]
    print("  [PASS] 5: Two tests, second wrong")

    # 6: Missing answer_2
    r6 = evaluate_arc_answer("<answer_1>\n1 2\n3 4\n</answer_1>", gt4)
    assert not r6["correct"]; print("  [PASS] 6: Missing answer_2 = fail")

    # 7: Three tests all correct
    gt7 = [[[1, 1]], [[2, 2]], [[3, 3]]]
    r7 = evaluate_arc_answer("<answer_1>\n1 1\n</answer_1>\n<answer_2>\n2 2\n</answer_2>\n<answer_3>\n3 3\n</answer_3>", gt7)
    assert r7["correct"]; print("  [PASS] 7: Three tests correct")

    # 8: Case insensitive
    r8 = evaluate_arc_answer("<ANSWER_1>\n0 0 2\n0 1 0\n2 2 2\n</ANSWER_1>", gt1)
    assert r8["correct"]; print("  [PASS] 8: Case insensitive")

    # 9: Legacy single grid
    r9 = evaluate_arc_answer("<answer_1>\n1 2\n3 4\n</answer_1>", [[1, 2], [3, 4]])
    assert r9["correct"]; print("  [PASS] 9: Legacy format")

    # 10: Empty tags
    r10 = evaluate_arc_answer("<answer_1>\n\n</answer_1>", gt1)
    assert not r10["correct"]; print("  [PASS] 10: Empty tags")

    # 11: Tags out of order still work
    r11 = evaluate_arc_answer("<answer_2>\n5 6\n7 8\n</answer_2>\n<answer_1>\n1 2\n3 4\n</answer_1>", gt4)
    assert r11["correct"]; print("  [PASS] 11: Tags out of order")

    print("\nAll 11 tests passed!")

    try:
        with open("arc_problems.json") as f:
            problems = json.load(f)
        print(f"\nValidating {len(problems)} problems...")
        for p in problems:
            gt = p["ground_truth"]
            blocks = []
            for i, grid in enumerate(gt, 1):
                grid_text = "\n".join(" ".join(str(c) for c in row) for row in grid)
                blocks.append(f"<answer_{i}>\n{grid_text}\n</answer_{i}>")
            result = evaluate_arc_answer("\n".join(blocks), gt)
            assert result["correct"], f"Failed: {p['problem_id']}"
        print(f"  All {len(problems)} problems validate ✅")
    except FileNotFoundError:
        print("\n  (arc_problems.json not found)")
