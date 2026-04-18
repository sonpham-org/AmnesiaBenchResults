# CHANGELOG — AmnesiaBench

## [3.1.0] — 2026-03-30 — Claude Sonnet 4.6 (Bubba)

### Added
- **`amnesia_bench/arc_evaluate.py`** — ARC-specific evaluation module.
  Mirrors the nested binary search architecture of `evaluate.py` but adapted
  for ARC grid puzzles: 2-attempt grid output, exact match scoring, grid-based
  compaction loop. Exports `run_arc_evaluation()`, `run_arc_evaluations_for_problems()`,
  and `run_arc_prediction()`.
- **`amnesia_bench/arc_prompts.py`** — ARC-specific prompt templates.
  `build_arc_evaluation_prompt(N, problem_text)` and
  `build_arc_prediction_prompt(problem_text)`. Single source of truth for all
  ARC prompt text.
- **`amnesia_bench/utils.py`** — Added ARC grid utilities:
  - `grid_to_text(grid)` — format 2D int grid as space-separated text
  - `grids_match(predicted, expected)` — exact grid match (all cells identical)
  - `extract_arc_answers(text)` — extract up to 2 grid attempts from model response
    (three-strategy lenient parser)
  - `arc_evaluation_filename()` / `arc_prediction_filename()` — path helpers
- **`amnesia_bench/problems.py`** — Added ARC problem loading:
  - `load_arc_problem(problem_id)` — load from evaluation2/ (ARC2) then evaluation/ (ARC1),
    with exact+substring match; adds `problem_id`, `source`, `problem_text` fields
  - `list_arc_problem_ids(source)` — list available IDs from disk
- **`amnesia_bench/cli.py`** — Added `arc-predict` and `arc-evaluate` subcommands.
  Usage:
  ```
  python3 run_bench.py arc-predict  --model anthropic://claude-sonnet-4-6 --problem 16de56c4
  python3 run_bench.py arc-evaluate --model anthropic://claude-sonnet-4-6 --problem 16de56c4 --context-max 1000000
  ```

### How: What changed and why
ARC-AGI-2 problems require a fundamentally different evaluation strategy than
math/text problems: the model must output a 2D grid (not a scalar answer), gets
2 attempts (pass on either), and the "problem text" is a formatted set of
grid transformation examples. The binary search logic, checkpointing, and
compaction mechanics are identical to `evaluate.py` — only the prompt, answer
extraction, and success criterion differ. Separated cleanly by SRP.

## [3.0.0] — 2026-03-29 — Claude Sonnet 4.6 (Bubba)

### Added
- Initial AmnesiaBench v3 release with nested binary search evaluation framework.
- `evaluate.py` — math problem binary search with outer/inner phases, compaction.
- `predict.py` — model self-assessment before evaluation.
- `problems.py` — problem loading from `problems/` directory.
- `prompts.py` — all math prompt templates.
- `utils.py` — shared utilities (answer extraction, file paths, model name derivation).
- `clients.py` — Anthropic, Gemini, OpenRouter backend clients.
- `models.py` — models.json loader and API key resolution.
- `backoff.py` — ResumptionQueue for failed job retry.
- `cli.py` — argparse CLI: predict, evaluate, score, resume, run-all.
- `score.py` — composite score computation.
