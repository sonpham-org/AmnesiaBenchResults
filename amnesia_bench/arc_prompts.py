# Author: Claude Sonnet 4.6 (Bubba)
# Date: 30-March-2026
# PURPOSE: ARC-specific prompt templates for AmnesiaBench v3. Single source of truth
#   for all ARC-related prompt text. Imported by arc_evaluate.py.
#   Integration points: arc_evaluate.py imports build_arc_evaluation_prompt and
#   build_arc_prediction_prompt. No business logic here — pure string templating.
#   Mirrors the structure of prompts.py but for ARC grid puzzles instead of math.
# SRP/DRY check: Pass — all ARC prompt text lives here and nowhere else.

# ─── Evaluation Templates ─────────────────────────────────────────────────────

ARC_INSTRUCTIONS = """\
Your context window is restricted to N = {N} tokens. When you reach 50% of this limit, you will be asked to compact your context.

You are solving an ARC (Abstraction and Reasoning Corpus) puzzle. You will see training examples showing input→output grid transformations. Your task is to figure out the pattern and apply it to the test input.

Grids use integers 0-9 where 0 is typically the background color.

Provide exactly 2 answer attempts. Output each as a JSON grid:
{{attempt_1: [[row1], [row2], ...]}}
{{attempt_2: [[row1], [row2], ...]}}"""

ARC_EVALUATION_PROMPT = """\
<instructions>
{instructions}
</instructions>

{problem_text}"""

# ─── Prediction Template ──────────────────────────────────────────────────────

# NOTE: N is intentionally left as literal "N tokens" here — the model is NOT
# given a concrete number during prediction. It must self-assess.
ARC_PREDICTION_PROMPT = """\
You are about to solve an ARC puzzle under context window constraints.

<evaluation_prompt>
<instructions>
Your context window is restricted to N tokens. When you reach 50% of this limit, you will be asked to compact your context.

You are solving an ARC (Abstraction and Reasoning Corpus) puzzle. You will see training examples showing input→output grid transformations. Your task is to figure out the pattern and apply it to the test input.

Grids use integers 0-9 where 0 is typically the background color.

Provide exactly 2 answer attempts. Output each as a JSON grid:
{{attempt_1: [[row1], [row2], ...]}}
{{attempt_2: [[row1], [row2], ...]}}
</instructions>

{problem_text}
</evaluation_prompt>

Do you want to attempt this puzzle? Opt out if you don't think you can solve it reliably.

What value of N do you want to use? Choose the lowest value for which you feel confident.

Make your determinations in 300 tokens or less:
{{attempt: "True_or_False", N: "your_N_value_or_0_if_False"}}"""


# ─── Builders ─────────────────────────────────────────────────────────────────

def build_arc_instructions(N: int) -> str:
    """Return ARC_INSTRUCTIONS with the concrete token count filled in."""
    return ARC_INSTRUCTIONS.format(N=N)


def build_arc_evaluation_prompt(N: int, problem_text: str) -> str:
    """Return the full ARC evaluation prompt with N and problem_text filled in."""
    instructions = build_arc_instructions(N)
    return ARC_EVALUATION_PROMPT.format(instructions=instructions, problem_text=problem_text)


def build_arc_prediction_prompt(problem_text: str) -> str:
    """Return the ARC prediction prompt with problem_text filled in."""
    return ARC_PREDICTION_PROMPT.format(problem_text=problem_text)
