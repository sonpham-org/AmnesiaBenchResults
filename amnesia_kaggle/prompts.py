"""Canonical prompts for AmnesiaBench v1 (Kaggle).

System prompt is EMPTY. Everything in the user message.
No XML tags. Just the problem text + answer format.

Prompt caching works because the user message is byte-identical
across all trials for the same problem.
"""

# ── User message template ────────────────────────────────────────────────
# This is the ONLY prompt the model sees. No system prompt.

USER_MESSAGE_TEMPLATE = (
    'Solve the following problem without using tool calls. '
    'Format your answer like this: {{final_answer: "YOUR_NUMBER_HERE"}}\n\n'
    '{problem}'
)

# ── Prediction prompt ────────────────────────────────────────────────────

PREDICTION_PROMPT_TEMPLATE = (
    "You are about to be evaluated on the below evaluation_prompt.\n\n"
    "<evaluation_prompt>\n"
    "{evaluation_prompt_preview}\n"
    "</evaluation_prompt>\n\n"
    "N is the total context window (input + output tokens combined) you will "
    "be given to solve the problem. When your context usage reaches 50% of N, "
    "you will be asked to compact your work so far into a summary; your context "
    "then resets to that summary and you continue solving. You may compact up "
    "to 5 times, or until your total tokens across all compactions reaches "
    "5x your natural (unbounded) token usage.\n\n"
    "Do you want to attempt the problem? You should opt out from solving "
    "the above problem if you don't think you can solve it reliably.\n\n"
    "What value of N do you want to use? Choose the lowest value for which "
    "you feel confident that you can still solve the problem correctly.\n\n"
    "Make your determinations in 300 tokens or less, and output them as "
    "strings in the format:\n"
    '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
)

# ── Compaction (injected by harness) ─────────────────────────────────────

COMPACTION_PROMPT = (
    "Compact everything above into <compact>...</compact> tags. "
    "Your compacted context will replace the current context."
)

POST_COMPACTION_PROMPT_TEMPLATE = (
    "Your previous compaction:\n"
    "\n"
    "---\n"
    "{compacted_context}\n"
    "---\n"
    "\n"
    "Continue solving.  Tokens left until next compaction: "
    "{tokens_left}"
)


# ── Builder functions ────────────────────────────────────────────────────

def build_problem_message(problem: str) -> str:
    """Return the user message — problem + answer format. No system prompt."""
    return USER_MESSAGE_TEMPLATE.format(problem=problem)


def build_evaluation_prompt(problem: str) -> str:
    """Return the full evaluation prompt (same as user message).

    Used for token estimation and prediction prompt preview.
    """
    return build_problem_message(problem)


def build_prediction_prompt(problem: str) -> str:
    """Return the prediction prompt."""
    eval_preview = build_evaluation_prompt(problem)
    return PREDICTION_PROMPT_TEMPLATE.format(evaluation_prompt_preview=eval_preview)


def build_post_compaction_prompt(compacted_context: str, tokens_left: int) -> str:
    """Return the post-compaction resume prompt with remaining budget."""
    return POST_COMPACTION_PROMPT_TEMPLATE.format(
        compacted_context=compacted_context,
        tokens_left=tokens_left,
    )
