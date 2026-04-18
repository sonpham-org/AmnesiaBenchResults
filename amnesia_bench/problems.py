# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026 (updated 30-March-2026)
# PURPOSE: Problem loading utilities for AmnesiaBench v3. Loads problem JSON files from
#   the problems/ directory adjacent to the package. Supports exact ID match and
#   substring match for convenience. Also provides load_arc_problem() for loading
#   ARC-AGI problems from the arc-explainer repo on disk.
#   Integration points: imported by predict.py, evaluate.py, cli.py, arc_evaluate.py.
# SRP/DRY check: Pass — single source of problem I/O; no duplication with result I/O.

import json
from pathlib import Path
from typing import List, Optional

# Default problems directory: one level up from this file (the package root), then problems/
_PACKAGE_DIR = Path(__file__).parent
PROBLEMS_DIR = _PACKAGE_DIR.parent / "problems"


def set_problems_dir(path: Path) -> None:
    """Override the problems directory (e.g. for testing)."""
    global PROBLEMS_DIR
    PROBLEMS_DIR = Path(path)


def load_problem(problem_id: str) -> dict:
    """
    Load a single problem JSON file from PROBLEMS_DIR.
    Matches on exact stem first, then substring.
    Raises FileNotFoundError if no match found.

    Expected problem JSON schema:
    {
        "problem_id": str,
        "problem_text": str,
        "ground_truth": str,          # expected answer string
        ... (any additional metadata)
    }
    """
    exact = PROBLEMS_DIR / f"{problem_id}.json"
    if exact.exists():
        return json.loads(exact.read_text())

    # Substring fallback
    for p in sorted(PROBLEMS_DIR.glob("*.json")):
        if problem_id in p.stem:
            return json.loads(p.read_text())

    raise FileNotFoundError(
        f"No problem matching '{problem_id}' found in {PROBLEMS_DIR}"
    )


def load_all_problems() -> List[dict]:
    """Load every .json file in PROBLEMS_DIR, sorted by filename."""
    if not PROBLEMS_DIR.exists():
        raise FileNotFoundError(
            f"Problems directory not found: {PROBLEMS_DIR}\n"
            "Create it and add problem JSON files."
        )
    files = sorted(PROBLEMS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No problem files found in {PROBLEMS_DIR}")
    return [json.loads(p.read_text()) for p in files]


def list_problem_ids() -> List[str]:
    """Return a sorted list of all problem IDs (stems) available."""
    if not PROBLEMS_DIR.exists():
        return []
    return sorted(p.stem for p in PROBLEMS_DIR.glob("*.json"))


# ─── ARC Problem Loading ──────────────────────────────────────────────────────

# ARC dataset roots on this machine
_ARC2_DIR = Path("/Users/macmini/Documents/GitHub/arc-explainer/data/evaluation2")
_ARC1_DIR = Path("/Users/macmini/Documents/GitHub/arc-explainer/data/evaluation")


def load_arc_problem(problem_id: str) -> dict:
    """
    Load a single ARC problem JSON by ID.

    Search order:
      1. ARC-AGI-2 (evaluation2/) — checked first
      2. ARC-AGI-1 (evaluation/)  — fallback

    Exact stem match first, then substring match within each directory.

    The returned dict contains all original ARC fields (train, test) plus:
      - problem_id (str): the matched stem
      - source (str): "arc2" or "arc1"
      - problem_text (str): LLM-ready formatted string showing all training
        examples and the test input grid

    Raises FileNotFoundError if the problem is not found in either directory.
    """
    from .utils import grid_to_text

    def _search_dir(directory: Path) -> Optional[Path]:
        """Try exact then substring match in directory."""
        exact = directory / f"{problem_id}.json"
        if exact.exists():
            return exact
        for p in sorted(directory.glob("*.json")):
            if problem_id in p.stem:
                return p
        return None

    matched_path: Optional[Path] = None
    source: str = ""

    for arc_dir, src_label in ((_ARC2_DIR, "arc2"), (_ARC1_DIR, "arc1")):
        if arc_dir.exists():
            matched_path = _search_dir(arc_dir)
            if matched_path:
                source = src_label
                break

    if matched_path is None:
        searched = []
        for d in (_ARC2_DIR, _ARC1_DIR):
            if d.exists():
                searched.append(str(d))
        raise FileNotFoundError(
            f"ARC problem '{problem_id}' not found.\n"
            f"Searched: {', '.join(searched) if searched else '(no ARC directories found)'}"
        )

    data = json.loads(matched_path.read_text())
    pid = matched_path.stem

    # Build LLM-friendly problem_text
    problem_text = _format_arc_problem_text(data, grid_to_text)

    data["problem_id"] = pid
    data["source"] = source
    data["problem_text"] = problem_text
    return data


def _format_arc_problem_text(problem: dict, grid_to_text_fn) -> str:
    """
    Format an ARC problem dict as a text block suitable for an LLM prompt.

    Shows all training examples (input + output) then the test input.
    Only the first test case is shown (ARC evaluation puzzles always have one).
    """
    lines = ["<training_examples>"]
    for i, example in enumerate(problem.get("train", []), start=1):
        lines.append(f"Example {i}:")
        lines.append("Input:")
        lines.append(grid_to_text_fn(example["input"]))
        lines.append("Output:")
        lines.append(grid_to_text_fn(example["output"]))
        lines.append("")

    lines.append("</training_examples>")
    lines.append("")
    lines.append("<test_input>")
    test_cases = problem.get("test", [])
    if test_cases:
        lines.append(grid_to_text_fn(test_cases[0]["input"]))
    lines.append("</test_input>")
    return '\n'.join(lines)


def list_arc_problem_ids(source: str = "both") -> List[str]:
    """
    Return sorted list of ARC problem IDs available on disk.

    source: "arc1", "arc2", or "both" (default)
    """
    ids = set()
    if source in ("arc2", "both") and _ARC2_DIR.exists():
        ids.update(p.stem for p in _ARC2_DIR.glob("*.json"))
    if source in ("arc1", "both") and _ARC1_DIR.exists():
        ids.update(p.stem for p in _ARC1_DIR.glob("*.json"))
    return sorted(ids)
