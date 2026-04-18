# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026 (updated 30-March-2026)
# PURPOSE: Shared utility functions for AmnesiaBench v3. Covers answer extraction,
#   result file path helpers, model name sanitization, and ARC grid utilities.
#   Imported by predict.py, evaluate.py, score.py, cli.py, and arc_evaluate.py.
#   Integration points: no circular imports — this module imports nothing from the package.
# SRP/DRY check: Pass — every utility here is used in >=2 modules; nothing is duplicated.

import re
from pathlib import Path
from typing import Optional


# ─── Answer Extraction ────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the model's final answer from {final_answer: "..."} format.
    Returns the raw string inside the quotes, or None if not found.
    Strips leading/trailing whitespace from the captured value.
    """
    # Match {final_answer: "VALUE"} with or without surrounding braces in text
    pattern = r'\{final_answer:\s*"([^"]+)"\s*\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    # Fallback: bare format without outer braces (defensive)
    pattern2 = r'final_answer:\s*"([^"]+)"'
    match2 = re.search(pattern2, text)
    if match2:
        return match2.group(1).strip()
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""
    Extract the last \boxed{...} answer from text, ignoring <think> blocks.
    Returns the raw boxed content as a string, or None if not found.
    Used for legacy v2 problem sets that use \boxed{} format.
    """
    # Strip think blocks first
    non_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    target = non_think if non_think.strip() else text
    matches = re.findall(r"\\boxed\{([^{}]+)\}", target)
    if not matches:
        matches = re.findall(r"\\boxed\{(.+?)\}", target)
    if not matches:
        return None
    return matches[-1].strip()


# ─── Result File Helpers ──────────────────────────────────────────────────────

def sanitize_model_name(model_name: str) -> str:
    """Replace non-alphanumeric chars (except dash/underscore) with underscores."""
    return re.sub(r"[^\w\-]", "_", model_name)


def prediction_filename(results_dir: Path, model_name: str, problem_id: str) -> Path:
    """Return path for a prediction result file."""
    safe = sanitize_model_name(model_name)
    return Path(results_dir) / f"{safe}_{problem_id}_prediction.json"


def evaluation_filename(results_dir: Path, model_name: str, problem_id: str) -> Path:
    """Return path for an evaluation result file."""
    safe = sanitize_model_name(model_name)
    return Path(results_dir) / f"{safe}_{problem_id}_evaluation.json"


# ─── ARC Grid Utilities ───────────────────────────────────────────────────────

def grid_to_text(grid: list) -> str:
    """Format a 2D integer grid as a readable text block.

    Each row becomes a space-separated string of integers.
    Example: [[0,1],[2,3]] → '0 1\\n2 3'
    """
    return '\n'.join(' '.join(str(c) for c in row) for row in grid)


def grids_match(predicted: list, expected: list) -> bool:
    """Exact grid match — every cell must be identical.

    Returns True iff predicted and expected have the same dimensions and
    every corresponding cell value is equal.
    """
    if len(predicted) != len(expected):
        return False
    for r1, r2 in zip(predicted, expected):
        if list(r1) != list(r2):
            return False
    return True


def extract_arc_answers(text: str) -> list:
    """Extract up to 2 grid answers from a model response.

    Tries multiple patterns in order of decreasing strictness:
      1. {attempt_1: [[...]]}  and  {attempt_2: [[...]]}
      2. attempt_1 = [[...]]   and  attempt_2 = [[...]]
      3. Any [[...]] blocks (first two, in order)

    Returns a list of up to 2 grids, each a list[list[int]].
    Returns an empty list if nothing parseable is found.
    Silently skips any extracted block that cannot be parsed as
    a 2D integer array.
    """
    import json

    def _parse_grid(raw: str) -> Optional[list]:
        """Try to parse a raw bracket block as a 2D int list."""
        raw = raw.strip()
        # Ensure it starts/ends with outer brackets
        if not (raw.startswith('[') and raw.endswith(']')):
            raw = f"[{raw}]"
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # JSON is strict — try ast.literal_eval as fallback
            try:
                import ast
                parsed = ast.literal_eval(raw)
            except Exception:
                return None
        # Validate: must be a list of lists of ints
        if not isinstance(parsed, list) or not parsed:
            return None
        rows = []
        for row in parsed:
            if not isinstance(row, list):
                return None
            int_row = []
            for cell in row:
                try:
                    int_row.append(int(cell))
                except (TypeError, ValueError):
                    return None
            rows.append(int_row)
        return rows

    grids = []

    # Strategy 1: {attempt_1: [[...]]} and {attempt_2: [[...]]}
    for key in ("attempt_1", "attempt_2"):
        # Match {attempt_1: [[...]]} — allow trailing content in the outer braces
        pat = r'\{' + key + r'\s*:\s*(\[\[.*?\]\])\s*\}'
        m = re.search(pat, text, re.DOTALL)
        if m:
            g = _parse_grid(m.group(1))
            if g is not None:
                grids.append(g)
                if len(grids) == 2:
                    return grids

    if len(grids) == 2:
        return grids

    # Strategy 2: attempt_1 = [[...]] and attempt_2 = [[...]]
    for key in ("attempt_1", "attempt_2"):
        pat = key + r'\s*=\s*(\[\[.*?\]\])'
        m = re.search(pat, text, re.DOTALL)
        if m:
            g = _parse_grid(m.group(1))
            if g is not None and g not in grids:
                grids.append(g)
                if len(grids) == 2:
                    return grids

    if len(grids) == 2:
        return grids

    # Strategy 3: find all [[...]] blocks and take the first two unique parseable ones
    all_blocks = re.findall(r'(\[\[.*?\]\])', text, re.DOTALL)
    for block in all_blocks:
        g = _parse_grid(block)
        if g is not None and g not in grids:
            grids.append(g)
        if len(grids) == 2:
            break

    return grids


def arc_evaluation_filename(results_dir, model_name: str, problem_id: str):
    """Return path for an ARC evaluation result file."""
    safe = sanitize_model_name(model_name)
    return Path(results_dir) / f"{safe}_{problem_id}_arc_evaluation.json"


def arc_prediction_filename(results_dir, model_name: str, problem_id: str):
    """Return path for an ARC prediction result file."""
    safe = sanitize_model_name(model_name)
    return Path(results_dir) / f"{safe}_{problem_id}_arc_prediction.json"


def derive_model_name(url: str) -> str:
    """
    Derive a short human-readable model name from a backend URL.
    Examples:
      anthropic://claude-sonnet-4-6   → claude-sonnet-4-6
      gemini://gemini-2.0-flash-lite  → gemini-2.0-flash-lite
      openrouter://openai/gpt-4o      → openai_gpt-4o
      http://localhost:8080           → localhost_8080
    """
    for scheme in ("anthropic://", "gemini://", "google://", "openrouter://"):
        if url.startswith(scheme):
            remainder = url[len(scheme):].strip("/")
            return re.sub(r"[^\w\-.]", "_", remainder) or scheme.rstrip("://")
    # http/https: strip scheme, replace non-word chars
    host_part = re.sub(r"^https?://", "", url).rstrip("/")
    return re.sub(r"[^\w\-]", "_", host_part) or "local"
