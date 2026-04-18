"""
Unified prompt variants for AmnesiaBench.

Single source of truth for ALL prompt text used by both local (ollama_runner)
and commercial (evaluate.py, predict.py) pipelines.

Each variant defines the complete set of prompts:
  - instructions:        System-level instructions with {N} placeholder
  - evaluation_prompt:   Wraps instructions + problem for bounded trials
  - prediction_prompt:   Asks model to self-assess before evaluation
  - unbounded_system:    System prompt for unbounded test (no N mention)
  - compact_prompt:      Compaction trigger (when context hits 50%)
  - resume_prompt:       Post-compaction resume with summary

Available placeholders:
  instructions:
    {N}               — context window size in tokens
  evaluation_prompt:
    {instructions}    — filled instructions text
    {problem}         — problem text
  prediction_prompt:
    {evaluation_prompt_preview} — the evaluation prompt with N left as literal "N"
  unbounded_system:
    (none)
  compact_prompt:
    {n}               — compaction number (1-indexed, before increment)
    {prev_output}     — the model's previous output wrapped in tags
  resume_prompt:
    {user_msg}        — original problem text
    {summary}         — extracted summary from compaction
    {n_done}          — compaction number (after increment)
"""

# ─── Variant Definitions ─────────────────────────────────────────────────────

PROMPT_VARIANTS = {

    # ── Original prompt — minimal guidance ────────────────────────────────
    "vanilla": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem.\n\n"
            'When you finish the problem, output your answer as a string in this exact format: '
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        "unbounded_system": (
            "Solve the following problem without using tool calls. "
            'When you finish the problem, output your answer as a string in this format: '
            '{final_answer: "YOUR_ANSWER_HERE"}'
        ),
        "compact_prompt": (
            "[COMPACTION #{n} TRIGGERED]\n\n"
            "Here is what you wrote so far:\n"
            "{prev_output}\n\n"
            "Summarize ALL your progress (including your thinking/reasoning) in a compact form. "
            "Include any intermediate results, equations, key insights, and partial answers. "
            "This summary will replace your entire context — anything not in the summary is LOST. "
            "Write your summary inside <compact>...</compact> tags."
        ),
        "resume_prompt": (
            "{user_msg}\n\n"
            "Your previous progress (compaction #{n_done}):\n"
            "---\n{summary}\n---\n"
            "Continue solving from where you left off. "
            'Output your answer as: {{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
    },

    # ── Structured: forces categorized extraction ─────────────────────────
    "structured": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem.\n\n"
            'When you finish the problem, output your answer as a string in this exact format: '
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        "unbounded_system": (
            "Solve the following problem without using tool calls. "
            'When you finish the problem, output your answer as a string in this format: '
            '{final_answer: "YOUR_ANSWER_HERE"}'
        ),
        "compact_prompt": (
            "[COMPACTION #{n} TRIGGERED — your context is about to reset]\n\n"
            "Here is what you wrote so far:\n"
            "{prev_output}\n\n"
            "Your context will be wiped. Extract ONLY what matters into this structure:\n\n"
            "<compact>\n"
            "PROBLEM RESTATEMENT: (1-2 sentences — what are we solving?)\n"
            "APPROACH: (what method/strategy are you using?)\n"
            "VERIFIED RESULTS: (intermediate values you computed and checked)\n"
            "FAILED APPROACHES: (what you tried that didn't work — don't repeat these)\n"
            "CURRENT STATE: (where exactly did you stop? what's the next step?)\n"
            "PARTIAL ANSWER: (your best guess so far, if any)\n"
            "</compact>"
        ),
        "resume_prompt": (
            "{user_msg}\n\n"
            "═══ SAVED CONTEXT (compaction #{n_done}) ═══\n"
            "The following is your own verified work from a previous attempt. "
            "These results are CORRECT — do not re-derive them. "
            "Pick up exactly where you left off.\n\n"
            "<prior_work>\n{summary}\n</prior_work>\n\n"
            'Continue solving. Output your answer as: {{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
    },

    # ── Surgical: bullet points only, no prose ────────────────────────────
    "surgical": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem.\n\n"
            'When you finish the problem, output your answer as a string in this exact format: '
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        "unbounded_system": (
            "Solve the following problem without using tool calls. "
            'When you finish the problem, output your answer as a string in this format: '
            '{final_answer: "YOUR_ANSWER_HERE"}'
        ),
        "compact_prompt": (
            "[COMPACTION #{n} — context reset imminent]\n\n"
            "Here is your work so far:\n"
            "{prev_output}\n\n"
            "Extract ONLY concrete facts. No prose, no reasoning, no explanations.\n"
            "Format:\n"
            "<compact>\n"
            "- [fact/equation/value]\n"
            "- [fact/equation/value]\n"
            "- NEXT: [exact next step to take]\n"
            "</compact>"
        ),
        "resume_prompt": (
            "{user_msg}\n\n"
            "══════════════════════════════════════\n"
            "ESTABLISHED FACTS (from your prior work, compaction #{n_done}):\n"
            "These are verified — do NOT re-derive.\n"
            "══════════════════════════════════════\n"
            "{summary}\n"
            "══════════════════════════════════════\n\n"
            'Continue from the NEXT step above. Output your answer as: {{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
    },

    # ── Notebook: framed as lab notebook for future-self ──────────────────
    "notebook": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem.\n\n"
            'When you finish the problem, output your answer as a string in this exact format: '
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        "unbounded_system": (
            "Solve the following problem without using tool calls. "
            'When you finish the problem, output your answer as a string in this format: '
            '{final_answer: "YOUR_ANSWER_HERE"}'
        ),
        "compact_prompt": (
            "[COMPACTION #{n} — saving to notebook]\n\n"
            "Here is your work:\n"
            "{prev_output}\n\n"
            "You are writing a lab notebook entry. Future-you will read ONLY this entry "
            "and the original problem — nothing else survives.\n\n"
            "Write the entry inside <compact>...</compact>. Include:\n"
            "- Key equations and computed values (with enough context to understand them)\n"
            "- Dead ends (so you don't retry them)\n"
            "- Your current best answer or leading candidate\n"
            "- Exactly what to do next\n"
            "Be precise. Every token counts."
        ),
        "resume_prompt": (
            "{user_msg}\n\n"
            "╔══ LAB NOTEBOOK — Entry #{n_done} (your own prior notes) ══╗\n"
            "{summary}\n"
            "╚═══════════════════════════════════════════════════════════╝\n\n"
            "You wrote the notes above during a previous attempt at this problem. "
            "Trust your notes — the computations are correct. "
            "Continue from where your notes left off.\n"
            'Output your answer as: {{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
    },

    # ── Refined: minimal compaction + clean resume ────────────────────────
    "refined": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem.\n\n"
            'When you finish the problem, output your answer as a string in this format: '
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        "unbounded_system": (
            "Solve the following problem without using tool calls. "
            'When you finish the problem, output your answer as a string in this format: '
            '{final_answer: "YOUR_ANSWER_HERE"}'
        ),
        "compact_prompt": (
            "You have reached 50% of your context limit. "
            "Compact your previous outputs (including the problem statement) to proceed. "
            "Write your compacted context inside <compact>...</compact> tags, "
            "and it will replace your current context."
        ),
        "resume_prompt": (
            "Your previous progress (from compaction):\n"
            "---\n{summary}\n---\n\n"
            "Continue solving."
        ),
    },

    # ── Refined3: empty system, everything in user message ────────────────
    # Design notes:
    #   - System prompt is EMPTY. All guidance (N restriction, "solve",
    #     answer format) lives in the FIRST user message.
    #   - Reason: post-compaction resume is a brand-new session with a fresh
    #     KV cache. We can't rely on any system prompt surviving.
    #   - The compact prompt is maximally universal ("compact everything")
    #     — no schema, no labels, no problem-specific framing.
    #   - Resume contains ONLY the compact summary. The model is expected to
    #     have folded the problem statement, progress, and answer-format
    #     directive into its own summary during the compact step.
    #   - Two user-message templates: `user_message_bounded` (with N for the
    #     compaction path) and `user_message_unbounded` (no N).
    "refined3": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem."
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        # Empty system for both bounded and unbounded paths.
        "system_prompt": "",
        "unbounded_system": "",
        "user_message_bounded": (
            "Your context window is restricted to N = {N} tokens "
            "(including this message). When you reach 50% of this limit, "
            "you will be asked to compact your context so that you have "
            "room to continue working on the problem.\n\n"
            "Solve the following problem without using tool calls.\n\n"
            "<problem>\n{problem}\n</problem>\n\n"
            "When you finish the problem, output your answer as a string in this format: "
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "user_message_unbounded": (
            "Solve the following problem without using tool calls.\n\n"
            "<problem>\n{problem}\n</problem>\n\n"
            "When you finish the problem, output your answer as a string in this format: "
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "compact_prompt": (
            "Compact everything above into <compact>...</compact> tags. "
            "Your compacted context will replace the current context."
        ),
        "resume_prompt": "{summary}",
    },

    # ── Refined4: refined3 + "Tokens left until compaction" budget hint ──
    # Design notes:
    #   - Same philosophy as refined3 (empty system, self-contained user msgs,
    #     problem statement folded into the compact summary by the model).
    #   - Adds an explicit "Tokens left until compaction: {tokens_left}" line
    #     on BOTH the initial user message and every resume message.
    #   - Rationale: after a compact the model starts in a fresh KV cache and
    #     has no idea how much budget it has for the next chunk. refined3 told
    #     it N at t=0 but never updated that number on resume. refined4 does.
    #   - tokens_left is filled in by the runner with half_budget (the per-
    #     chunk ceiling before another compact is forced).
    "refined4": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem."
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        "system_prompt": "",
        "unbounded_system": "",
        "user_message_bounded": (
            "Tokens left until compaction: {tokens_left}\n\n"
            "Your context window is restricted to N = {N} tokens "
            "(including this message). When you reach 50% of this limit, "
            "you will be asked to compact your context so that you have "
            "room to continue working on the problem.\n\n"
            "Solve the following problem without using tool calls.\n\n"
            "<problem>\n{problem}\n</problem>\n\n"
            "When you finish the problem, output your answer as a string in this format: "
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "user_message_unbounded": (
            "Solve the following problem without using tool calls.\n\n"
            "<problem>\n{problem}\n</problem>\n\n"
            "When you finish the problem, output your answer as a string in this format: "
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "compact_prompt": (
            "Compact everything above into <compact>...</compact> tags. "
            "Your compacted context will replace the current context."
        ),
        "resume_prompt": (
            "Tokens left until compaction: {tokens_left}\n\n"
            "{summary}"
        ),
    },

    # ── Refined2: exclude problem from compaction, re-inject on resume ───
    "refined2": {
        "instructions": (
            "Your context window is restricted to N = {N} tokens "
            "(including these instructions and the problem statement). "
            "When you reach 50% of this limit, you will be asked to compact "
            "your context so that you have room to continue working on the problem.\n\n"
            'When you finish the problem, output your answer as a string in this format: '
            '{{final_answer: "YOUR_ANSWER_HERE"}}'
        ),
        "evaluation_prompt": (
            "<instructions>\n"
            "{instructions}\n"
            "</instructions>\n\n"
            "<problem>\n"
            "{problem}\n"
            "</problem>"
        ),
        "prediction_prompt": (
            "You are about to be evaluated on the below evaluation_prompt.\n\n"
            "<evaluation_prompt>\n"
            "{evaluation_prompt_preview}\n"
            "</evaluation_prompt>\n\n"
            "Do you want to attempt the problem? You should opt out from solving "
            "the above problem if you don't think you can solve it reliably.\n\n"
            "What value of N do you want to use? Choose the lowest value for which "
            "you feel confident that you can still solve the problem correctly.\n\n"
            "Make your determinations in 300 tokens or less, and output them as "
            "strings in the format:\n"
            '{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}'
        ),
        "unbounded_system": (
            "Solve the following problem without using tool calls. "
            'When you finish the problem, output your answer as a string in this format: '
            '{final_answer: "YOUR_ANSWER_HERE"}'
        ),
        "compact_prompt": (
            "You have reached 50% of your context limit. "
            "Compact your previous outputs (excluding the problem statement) to proceed. "
            "Write your compacted context inside <compact>...</compact> tags, "
            "and it will replace your current context."
        ),
        "resume_prompt": (
            "{instructions}\n\n"
            "{user_msg}\n\n"
            "Your previous progress (from compaction):\n"
            "---\n{summary}\n---\n\n"
            "Continue solving."
        ),
    },
}

DEFAULT_PROMPT_VARIANT = "vanilla"


# ─── Builder Functions ───────────────────────────────────────────────────────

def get_variant(name: str = DEFAULT_PROMPT_VARIANT) -> dict:
    """Return a prompt variant by name, falling back to default."""
    return PROMPT_VARIANTS.get(name, PROMPT_VARIANTS[DEFAULT_PROMPT_VARIANT])


def build_instructions(N: int, variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Return instructions with the concrete token count filled in."""
    v = get_variant(variant)
    return v["instructions"].format(N=N)


def build_evaluation_prompt(N: int, problem: str, variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Return the full evaluation prompt with N and problem text filled in."""
    v = get_variant(variant)
    instructions = v["instructions"].format(N=N)
    return v["evaluation_prompt"].format(instructions=instructions, problem=problem)


def build_prediction_prompt(problem: str, variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Return the prediction prompt with problem text filled in.

    N is left as the literal text "N" in the evaluation preview so the model
    is not primed with a concrete number.
    """
    v = get_variant(variant)
    # Build the evaluation prompt preview with N left as literal "N"
    instructions_preview = v["instructions"].replace("{N}", "N")
    # Need to double-brace any remaining braces for the format call
    eval_preview = v["evaluation_prompt"].format(
        instructions=instructions_preview, problem=problem,
    )
    return v["prediction_prompt"].format(evaluation_prompt_preview=eval_preview)


def build_unbounded_system(variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Return the unbounded system prompt (no N mention).

    Variants may set this to "" (e.g. refined3) to keep all guidance in the
    user message instead.
    """
    v = get_variant(variant)
    return v.get("unbounded_system", "")


def build_system_prompt(N: int, variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Return the system prompt for bounded evaluation trials.

    This is the instructions text used as the system message. For pipelines
    that use system + user message separation (Ollama, Anthropic), use this
    as the system message and the problem text as the user message.

    Variants may override with an explicit `system_prompt` field — for
    example refined3 uses an empty system and puts everything in the user
    message so resumed sessions are self-contained.
    """
    v = get_variant(variant)
    if "system_prompt" in v:
        return v["system_prompt"]
    return build_instructions(N, variant)


def build_user_message(
    problem_text: str,
    variant: str = DEFAULT_PROMPT_VARIANT,
    N: int | None = None,
    tokens_left: int = 0,
) -> str:
    """Return the user message for a given variant.

    For most variants this is just the raw problem text. Variants that move
    guidance out of the system prompt (e.g. refined3) define
    `user_message_bounded` (uses {N}) and/or `user_message_unbounded`
    templates that wrap the problem.

    Pass N for the bounded/compaction path; leave None for the unbounded path.
    `tokens_left` is consumed by variants that surface a per-chunk budget
    hint to the model (e.g. refined4); other variants ignore it.
    """
    v = get_variant(variant)
    if N is not None:
        tmpl = v.get("user_message_bounded")
        if tmpl:
            return tmpl.format(problem=problem_text, N=N, tokens_left=tokens_left)
    else:
        tmpl = v.get("user_message_unbounded")
        if tmpl:
            return tmpl.format(problem=problem_text, tokens_left=tokens_left)
    return problem_text


def build_compact_prompt(
    n: int,
    prev_output: str = "",
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> str:
    """Return the compaction prompt with placeholders filled in."""
    v = get_variant(variant)
    return v["compact_prompt"].format(n=n, prev_output=prev_output)


def build_resume_prompt(
    user_msg: str,
    summary: str,
    n_done: int,
    variant: str = DEFAULT_PROMPT_VARIANT,
    N: int = 0,
    tokens_left: int = 0,
) -> str:
    """Return the resume prompt after compaction.

    N is needed for variants (like refined2) that re-inject instructions on resume.
    tokens_left is the per-chunk budget (half_budget) the model has before
    another compact is forced — surfaced by variants like refined4.
    """
    v = get_variant(variant)
    instructions = v["instructions"].format(N=N) if N > 0 else ""
    return v["resume_prompt"].format(
        user_msg=user_msg, summary=summary, n_done=n_done,
        instructions=instructions, tokens_left=tokens_left,
    )
