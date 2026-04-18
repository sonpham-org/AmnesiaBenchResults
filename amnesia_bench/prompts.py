# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# Updated: 05-April-2026
# PURPOSE: Backward-compatible prompt helpers for AmnesiaBench v3.
#   All prompt text now lives in compaction_prompts.py (the single source of truth).
#   This module re-exports the builder functions so existing imports still work.
#   Integration points: predict.py uses build_prediction_prompt; evaluate.py uses
#   build_evaluation_prompt.

from .compaction_prompts import (
    build_instructions,
    build_evaluation_prompt,
    build_prediction_prompt,
    build_unbounded_system,
    build_system_prompt,
    build_user_message,
    build_compact_prompt,
    build_resume_prompt,
    get_variant,
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
)

__all__ = [
    "build_instructions",
    "build_evaluation_prompt",
    "build_prediction_prompt",
    "build_unbounded_system",
    "build_system_prompt",
    "build_user_message",
    "build_compact_prompt",
    "build_resume_prompt",
    "get_variant",
    "DEFAULT_PROMPT_VARIANT",
    "PROMPT_VARIANTS",
]
