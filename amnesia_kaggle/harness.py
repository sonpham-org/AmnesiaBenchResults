"""AmnesiaBench harness — unbounded, prediction, and bounded trials against kbench LLMs.

Ports amnesia_bench/evaluate.py (`_test_unbounded`, `_run_trial`, no-compact +
compact branches) to the kaggle-benchmarks framework primitives:

  kbench.chats.new("name") as chat    — isolated chat context
  kbench.actors.user.send(text)       — append user message to chat
  llm.respond(max_tokens=..., ...)    — bounded response generation

Budget enforcement (plan.md General Notes):
  - N applies to ALL tokens including the system prompt.
  - Enforced by setting max_tokens appropriately on each llm.respond() call.
  - For Google Gemini backends, the kwarg is `max_output_tokens`; for
    OpenAI-compatible backends (Claude, Llama, etc. via Kaggle proxy),
    the kwarg is `max_tokens`. We dispatch based on the LLM class.

Abort conditions (plan.md General Notes):
  - cost cap: 3× the unbounded cost for this problem
  - compaction insufficient: reset context alone exceeds N
  - no answer, no compact: trial produced neither a final_answer tag nor a
    compact tag → hard failure
  - context_exceeded during compaction: abort, record as failure
  - final_answer detected during any turn (including compaction) → success
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .log_search import (
    INNER_CHECKS_PER_N,
    OUTER_CHECKS_PER_N,
    SearchLog,
    TrialOutcome,
    find_n_reliable as log_find_n_reliable,
)
from .parsers import (
    check_answer,
    extract_compact_tag,
    extract_final_answer,
    parse_prediction,
)
from .prompts import (
    COMPACTION_PROMPT,
    build_evaluation_prompt,
    build_problem_message,
    build_post_compaction_prompt,
    build_prediction_prompt,
)


# ── Config ──────────────────────────────────────────────────────────────────

UNBOUNDED_TRIALS = 3
UNBOUNDED_PASS_THRESHOLD = 2           # ≥2/3 must succeed → n_while_unbounded valid
COMPACTION_TRIGGER = 0.50              # trigger compaction at 50% of N
MAX_COMPACT_TURNS = 5                  # max compaction cycles per trial
PREDICTION_MAX_OUTPUT_TOKENS = 1024  # generous buffer; some models truncate short responses


# ── Trace collection ──────────────────────────────────────────────────────

def _capture_chat_trace(chat) -> list[dict]:
    """Extract all messages from a kbench chat as serializable dicts."""
    trace = []
    messages = list(chat.messages) if hasattr(chat, "messages") else []
    for m in messages:
        sender = getattr(m, "sender", None)
        role = getattr(sender, "role", "unknown") if sender else "unknown"
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        usage = getattr(m, "usage", None)
        entry = {"role": role, "content": content}
        if usage is not None:
            entry["input_tokens"] = getattr(usage, "input_tokens", None)
            entry["output_tokens"] = getattr(usage, "output_tokens", None)
            entry["cost_nanodollars"] = (
                (getattr(usage, "input_tokens_cost_nanodollars", 0) or 0)
                + (getattr(usage, "output_tokens_cost_nanodollars", 0) or 0)
            )
        meta = getattr(m, "_meta", None) or {}
        if isinstance(meta, dict):
            for key in ("finish_reason", "stop_reason", "finishReason"):
                if key in meta:
                    entry["finish_reason"] = str(meta[key])
                    break
        trace.append(entry)
    return trace


# ── Tokenizer fallback (tiktoken if available, else len//4) ────────────────

def _token_count(text: str) -> int:
    """Estimate tokens in text. Uses tiktoken (cl100k_base) if available.

    Fallback: 4 characters ≈ 1 token. Conservative enough for pre-call budget
    estimation; post-call measurements use chat.messages usage metadata.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = _get_tiktoken_encoding()
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


_tiktoken_enc = None


def _get_tiktoken_encoding():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


# ── Budget measurement from a chat ──────────────────────────────────────────

def _chat_context_size(chat) -> int:
    """Total tokens currently in the chat (input + output, cumulative).

    Strategy:
      1. If the last assistant message has valid usage metadata, use
         `input_tokens + output_tokens` — this equals the full context size
         after that turn (input includes all prior history).
      2. Otherwise, tiktoken-estimate every message's content.
    """
    messages = list(chat.messages) if hasattr(chat, "messages") else []
    if not messages:
        return 0
    # Prefer the last assistant's usage.input + output (exact if Kaggle proxy)
    for m in reversed(messages):
        sender = getattr(m, "sender", None)
        role = getattr(sender, "role", None) if sender else None
        usage = getattr(m, "usage", None)
        if role == "assistant" and usage is not None:
            input_t = getattr(usage, "input_tokens", None)
            output_t = getattr(usage, "output_tokens", None)
            if input_t is not None and output_t is not None:
                return int(input_t) + int(output_t)
            break  # fall through to tiktoken if missing
    # Tiktoken fallback
    return sum(_token_count(getattr(m, "content", "") or "") for m in messages)


def _chat_usage_totals(chat) -> dict:
    """Pull cumulative usage totals from chat.usage (aggregate across all turns).

    Used for cost-cap checks and scoring. Returns zeros if Kaggle proxy didn't
    populate usage (local dev without MODEL_PROXY_API_KEY).
    """
    usage = getattr(chat, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "cost_nanodollars": 0}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cost_nanodollars": int(
            (getattr(usage, "input_tokens_cost_nanodollars", 0) or 0)
            + (getattr(usage, "output_tokens_cost_nanodollars", 0) or 0)
        ),
    }


# ── LLM adapter: max_tokens vs max_output_tokens ────────────────────────────

# Gemini caps max_output_tokens at 8192 per call (regardless of context window).
# Anthropic caps at 8192 or 64K depending on model. For budgets larger than
# this, we auto-chain calls with "Continue." prompts so the model can
# effectively use the full context window.
GEMINI_OUTPUT_CAP = 8192
OPENAI_OUTPUT_CAP = 8192  # conservative default for Claude/Llama via proxy


def _respond_bounded(
    llm,
    max_output_tokens: Optional[int],
    temperature: float,
    disable_thinking: bool = False,
) -> Any:
    """Call llm.respond() with a hard output-token cap.

    For large budgets that exceed the API's per-call cap, auto-chain calls:
    after each chunk, send "Continue." and keep generating until we reach
    the budget or the model stops naturally.

    Dispatches the correct kwarg name based on the LLM backend:
      - Google Gemini (`GoogleGenAI`) → `max_output_tokens=...`
      - OpenAI-compatible (Claude/Llama/etc.) → `max_tokens=...`

    If `max_output_tokens` is None, no cap is applied (unbounded call).
    """
    import kaggle_benchmarks as kbench

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    # On Kaggle's Model Proxy, ALL models (including Gemini) are routed
    # through the OpenAI-compatible class — `isinstance(llm, GoogleGenAI)` is
    # always False on Kaggle. Detect Gemini by the model string instead.
    # Local dev with a direct google.genai client is the only time
    # isinstance(GoogleGenAI) is True.
    is_google_genai_client = False
    try:
        from kaggle_benchmarks.actors.llms import GoogleGenAI
        is_google_genai_client = isinstance(llm, GoogleGenAI)
    except ImportError:
        pass
    _model_name = str(getattr(llm, "model", "")).lower()
    is_gemini_model = "gemini" in _model_name

    # Thinking control for Gemini via the direct google.genai client (local dev only).
    # On Kaggle's OpenAI-compat proxy, `thinking_config` is not accepted;
    # controlling Gemini thinking there requires proxy-specific kwargs which
    # we don't currently support — the model thinks by default and those
    # thinking tokens count against max_tokens invisibly (they show up only
    # as `total_tokens - prompt_tokens - completion_tokens`).
    if is_google_genai_client:
        try:
            from google.genai import types
            if disable_thinking:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            else:
                kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        except Exception:
            pass

    # Per-call output cap: Gemini caps at 8192 regardless of client; use same
    # floor for OpenAI-compat so behavior is consistent.
    per_call_cap = GEMINI_OUTPUT_CAP if (is_google_genai_client or is_gemini_model) else OPENAI_OUTPUT_CAP
    max_tok_key = "max_output_tokens" if is_google_genai_client else "max_tokens"

    if max_output_tokens is None or max_output_tokens <= 0:
        # Unbounded: single call with per-call cap (models with thinking may
        # need chaining to reach full context, but for unbounded we accept
        # the cap as "natural stop").
        if is_google_genai_client or is_gemini_model:
            kwargs[max_tok_key] = per_call_cap
        return llm.respond(**kwargs)

    # Bounded: chain if budget exceeds per-call cap.
    # Gemini rejects max_output_tokens < some minimum — use a safe floor.
    budget = max(int(max_output_tokens), 16)
    if budget <= per_call_cap:
        kwargs[max_tok_key] = budget
        return llm.respond(**kwargs)

    # Chain calls. Each call: max_tokens=per_call_cap, then "Continue."
    generated = 0
    while generated < budget:
        chunk = min(per_call_cap, budget - generated)
        call_kwargs = dict(kwargs)
        call_kwargs[max_tok_key] = chunk
        llm.respond(**call_kwargs)
        # Measure what was actually generated
        chat = kbench.chats.get_current_chat()
        last = chat.messages[-1] if chat.messages else None
        actual_out = 0
        if last is not None:
            usage = getattr(last, "usage", None)
            if usage:
                actual_out = int(getattr(usage, "output_tokens", 0) or 0)
        # If model generated less than the chunk, it stopped naturally → done
        if actual_out < chunk:
            return
        generated += actual_out
        if generated >= budget:
            return
        # Still have budget and model hit the cap — send continuation
        kbench.actors.user.send("Continue.")


# ── Phase 1: unbounded test (3 trials) ──────────────────────────────────────

@dataclass
class UnboundedResult:
    n_while_unbounded: float           # mean tokens across successful trials, or math.inf
    solved: bool                       # True iff ≥2/3 trials succeeded
    context_exceeded: bool             # True iff any failure was due to budget
    cost_nanodollars: int              # cumulative across the 3 trials
    input_tokens: int
    output_tokens: int
    trial_logs: list[dict] = field(default_factory=list)
    traces: list[list[dict]] = field(default_factory=list)  # per-trial chat traces


def run_unbounded(
    llm,
    problem_text: str,
    ground_truth,
    temperature: float = 0.7,
) -> UnboundedResult:
    """Run 3 unbounded trials (no max_tokens cap).

    Plan.md §5 rules:
      - Run 3 times, take mean tokens across successful attempts.
      - If ≥2 of 3 fail, n_while_unbounded = inf.
      - If failures were due to context exceeded → still proceed to compaction
        sweep (flagged via `context_exceeded=True`).
      - If failures were NOT due to context → stop further testing.
    """
    import kaggle_benchmarks as kbench

    successes: list[int] = []  # tokens used on successful trials
    any_context_exceeded = False
    total_cost = 0
    total_input = 0
    total_output = 0
    trial_logs: list[dict] = []
    all_traces: list[list[dict]] = []

    for trial_idx in range(UNBOUNDED_TRIALS):
        t0 = time.time()
        trial_ok = False
        trial_tokens = 0
        finish = "no_answer"
        error = None
        trial_trace: list[dict] = []

        try:
            with kbench.chats.new(f"unbounded_t{trial_idx}") as chat:
                problem_msg = build_problem_message(problem_text)
                kbench.actors.user.send(problem_msg)
                _respond_bounded(llm, max_output_tokens=None, temperature=temperature)

                last_content = _last_assistant_content(chat)
                answer = extract_final_answer(last_content)
                if answer is not None and check_answer(answer, ground_truth):
                    trial_ok = True
                    finish = "solved"

                trial_tokens = _chat_context_size(chat)
                totals = _chat_usage_totals(chat)
                total_cost += totals["cost_nanodollars"]
                total_input += totals["input_tokens"]
                total_output += totals["output_tokens"]

                if not trial_ok:
                    last_msg = chat.messages[-1] if chat.messages else None
                    reason = _finish_reason(last_msg)
                    if reason in ("length", "max_tokens", "MAX_TOKENS"):
                        any_context_exceeded = True
                        finish = "context_exceeded"
                    else:
                        finish = reason or "wrong_answer"

                trial_trace = _capture_chat_trace(chat)
        except Exception as e:
            error = str(e)
            finish = "error"

        if trial_ok:
            successes.append(trial_tokens)

        trial_logs.append({
            "trial_idx": trial_idx,
            "success": trial_ok,
            "finish_reason": finish,
            "tokens_used": trial_tokens,
            "wall_time_s": round(time.time() - t0, 2),
            "error": error,
        })
        all_traces.append(trial_trace)

    if len(successes) >= UNBOUNDED_PASS_THRESHOLD:
        n_while_unbounded: float = sum(successes) / len(successes)
        solved = True
    else:
        n_while_unbounded = math.inf
        solved = False

    return UnboundedResult(
        n_while_unbounded=n_while_unbounded,
        solved=solved,
        context_exceeded=any_context_exceeded,
        cost_nanodollars=total_cost,
        input_tokens=total_input,
        output_tokens=total_output,
        trial_logs=trial_logs,
        traces=all_traces,
    )


# ── Phase 2: prediction ─────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    attempt: bool
    n_predicted: float
    cost_nanodollars: int
    input_tokens: int
    output_tokens: int
    raw_response: str = ""
    trace: list[dict] = field(default_factory=list)


def run_prediction(
    llm,
    problem_text: str,
    temperature: float = 0.7,
) -> PredictionResult:
    """Ask the model to self-assess (attempt + N prediction).

    Plan.md §6: 300 output token cap, parse `{attempt: "...", N: "..."}`.
    On parse failure, fallback is (True, math.inf). **Advisory only** —
    the caller runs the full sweep regardless.
    """
    import kaggle_benchmarks as kbench

    prompt = build_prediction_prompt(problem_text)
    raw = ""
    trace: list[dict] = []
    totals = {"cost_nanodollars": 0, "input_tokens": 0, "output_tokens": 0}
    try:
        with kbench.chats.new("predict") as chat:
            kbench.actors.user.send(prompt)
            _respond_bounded(
                llm,
                max_output_tokens=PREDICTION_MAX_OUTPUT_TOKENS,
                temperature=temperature,
            )
            raw = _last_assistant_content(chat) or ""
            totals = _chat_usage_totals(chat)
            trace = _capture_chat_trace(chat)
    except Exception:
        return PredictionResult(
            attempt=True, n_predicted=math.inf,
            cost_nanodollars=0, input_tokens=0, output_tokens=0,
            raw_response="",
        )

    attempt, n_predicted = parse_prediction(raw)
    return PredictionResult(
        attempt=attempt,
        n_predicted=n_predicted,
        cost_nanodollars=totals["cost_nanodollars"],
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        raw_response=raw,
        trace=trace,
    )


# ── Phase 3: bounded no-compact trial ───────────────────────────────────────

def run_trial_no_compact(
    llm,
    problem_text: str,
    ground_truth,
    N: int,
    temperature: float = 0.7,
) -> TrialOutcome:
    """Single-turn bounded trial. No compaction.

    The model has N total tokens (input + output). We pre-compute the input
    token count for the evaluation prompt and set `max_tokens = N - input`.
    If the input alone already exceeds N, the trial fails without an LLM call.
    """
    import kaggle_benchmarks as kbench

    problem_msg = build_problem_message(problem_text)
    input_est = _token_count(problem_msg)
    remaining = N - input_est
    if remaining <= 0:
        return TrialOutcome(
            success=False, aborted=False,
            cost_nanodollars=0, input_tokens=input_est, output_tokens=0,
            finish_reason="input_exceeds_N",
        )

    trace: list[dict] = []
    try:
        with kbench.chats.new(f"trial_nocompact_N{N}") as chat:
            kbench.actors.user.send(problem_msg)
            _respond_bounded(llm, max_output_tokens=remaining, temperature=temperature)
            last = _last_assistant_content(chat) or ""
            answer = extract_final_answer(last)
            success = answer is not None and check_answer(answer, ground_truth)
            totals = _chat_usage_totals(chat)

            if success:
                finish = "solved"
            else:
                last_msg = chat.messages[-1] if chat.messages else None
                reason = _finish_reason(last_msg)
                if reason in ("length", "max_tokens", "MAX_TOKENS"):
                    finish = "budget_exceeded"
                elif answer is None:
                    finish = "no_answer"
                else:
                    finish = "wrong_answer"

            trace = _capture_chat_trace(chat)
    except Exception as e:
        return TrialOutcome(
            success=False, aborted=True,
            cost_nanodollars=0, input_tokens=0, output_tokens=0,
            finish_reason=f"error:{type(e).__name__}",
        )

    return TrialOutcome(
        success=success,
        aborted=False,
        cost_nanodollars=totals["cost_nanodollars"],
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        finish_reason=finish,
        traces=trace,
    )


# ── Phase 4: bounded compact trial ──────────────────────────────────────────

def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens. Uses tiktoken when available."""
    if max_tokens <= 0 or not text:
        return ""
    try:
        enc = _get_tiktoken_encoding()
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return enc.decode(ids[:max_tokens])
    except Exception:
        # char-based fallback (~4 chars/token)
        approx = max_tokens * 4
        return text[:approx]


def run_trial_compact(
    llm,
    problem_text: str,
    ground_truth,
    N: int,
    temperature: float = 0.7,
    cost_cap_nanodollars: Optional[int] = None,
    token_cap: Optional[int] = None,
    unbounded_assistant_text: Optional[str] = None,
) -> TrialOutcome:
    """Bounded trial with multi-turn compaction.

    The model works within budget N per chunk. When context hits 50% of N,
    compaction is triggered: the model produces a <compact> summary, context
    resets, and work continues in a fresh session. This repeats until:

      - final_answer found → success
      - total tokens (input + output) exceed token_cap → fail
      - cost exceeds cost_cap_nanodollars → fail
      - no <compact> tag produced → fail
      - MAX_COMPACT_TURNS reached → fail

    token_cap: max total tokens across all compaction cycles. Typically
        int(5 * n_while_unbounded). Combined with MAX_COMPACT_TURNS=5,
        whichever limit is hit first stops the trial.

    unbounded_assistant_text: If provided, Cycle 0's thinking turn is NOT
        regenerated — we inject a truncated prefix of this text (fit to
        N/2 - input_tokens) as the assistant's pre-compaction output. This
        reuses the model's prior reasoning so compaction operates on real
        long-form thinking rather than a rushed short response at N/2 budget.
    """
    import kaggle_benchmarks as kbench
    from kaggle_benchmarks.actors.base import Actor
    from kaggle_benchmarks.messages import Message

    problem_msg = build_problem_message(problem_text)
    input_est = _token_count(problem_msg)
    if input_est >= N:
        return TrialOutcome(
            success=False, aborted=False,
            cost_nanodollars=0, input_tokens=input_est, output_tokens=0,
            finish_reason="input_exceeds_N",
        )

    total_cost = 0
    total_input = 0
    total_output = 0
    n_compactions = 0
    summary: Optional[str] = None  # None on first turn, set after compaction
    cycle_traces: list[dict] = []   # traces from each chat cycle

    def _absorb(chat) -> None:
        nonlocal total_cost, total_input, total_output
        t = _chat_usage_totals(chat)
        total_cost += t["cost_nanodollars"]
        total_input += t["input_tokens"]
        total_output += t["output_tokens"]

    def _over_token_cap() -> bool:
        return token_cap is not None and (total_input + total_output) > token_cap

    def _over_both_limits() -> bool:
        """Stop only when BOTH token cap AND compaction limit are exceeded."""
        over_tokens = token_cap is not None and (total_input + total_output) > token_cap
        over_compactions = n_compactions >= MAX_COMPACT_TURNS
        return over_tokens and over_compactions

    def _over_cost_cap() -> bool:
        return cost_cap_nanodollars is not None and total_cost > cost_cap_nanodollars

    def _result(success, finish):
        return _compact_result(success, total_cost, total_input, total_output,
                               finish, traces=cycle_traces)

    try:
        for cycle in range(50):  # hard safety cap; real limit is _over_both_limits()
            # ── Thinking turn ────────────────────────────────────────────
            chat_name = f"trial_compact_N{N}_c{cycle}"
            with kbench.chats.new(chat_name) as chat:
                if summary is None:
                    # First cycle: send the problem
                    kbench.actors.user.send(problem_msg)
                    prompt_est = input_est
                else:
                    # After compaction: ONLY the resume with summary, no problem.
                    # tokens_left = N/2 - resume_prompt_length (actual remaining
                    # budget before the next compaction fires).
                    stub_post = build_post_compaction_prompt(summary, 0)
                    resume_overhead = _token_count(stub_post)
                    tokens_left = max(16, N // 2 - resume_overhead - 5)  # -5 for digits
                    post = build_post_compaction_prompt(summary, tokens_left)
                    kbench.actors.user.send(post)
                    prompt_est = _token_count(post)

                # Cap thinking turn at 50% of N — compaction fires after
                half_budget = max(1, int(N * COMPACTION_TRIGGER) - prompt_est)

                if cycle == 0 and unbounded_assistant_text:
                    # Inject unbounded assistant text cut at N/2 tokens.
                    # Compaction prompt is then appended in the same session.
                    truncated = _truncate_to_tokens(unbounded_assistant_text, N // 2)
                    fake_asst = Actor(name="unbounded_prefill", role="assistant")
                    chat.append(Message(content=truncated, sender=fake_asst))
                else:
                    _respond_bounded(llm, max_output_tokens=half_budget, temperature=temperature)

                last_content = _last_assistant_content(chat) or ""
                last_msg = chat.messages[-1] if chat.messages else None

                # Check for final answer — BUT skip this early-return when we
                # injected a prefill at Cycle 0. A prefill that already contains
                # {final_answer: ...} would short-circuit the whole trial,
                # meaning compaction never fires and we're not actually testing
                # context-compression behavior. Force compaction to run at
                # least once so the benchmark exercises what it's supposed to.
                prefill_injected = (cycle == 0 and unbounded_assistant_text is not None)
                answer = extract_final_answer(last_content)
                if answer is not None and not prefill_injected:
                    _absorb(chat)
                    cycle_traces.append(_capture_chat_trace(chat))
                    finish = "solved" if cycle == 0 else f"solved_post_reset_{cycle}"
                    return _result(check_answer(answer, ground_truth), finish)

                # No valid final_answer yet — always proceed to compaction.
                # This gives the model a recovery path even when it stopped
                # early with an invalid/partial response. Compaction fires
                # regardless of whether the output limit was hit.
                _absorb(chat)

                if _over_cost_cap():
                    cycle_traces.append(_capture_chat_trace(chat))
                    return _result(False, "cost_cap_exceeded")
                if _over_both_limits():
                    cycle_traces.append(_capture_chat_trace(chat))
                    return _result(False, f"limits_exceeded_{n_compactions}_compactions")

                # ── Compaction turn (same chat) ────
                # Compaction gets N/2 tokens total (prompt + response). This
                # enforces that the full cycle stays within N: thinking gets
                # N/2, compaction gets N/2.
                # Caveat: for Gemini Pro on Kaggle's OpenAI-compat proxy,
                # thinking tokens count against this cap invisibly — the
                # compact response may truncate. Uncap manually if needed.
                kbench.actors.user.send(COMPACTION_PROMPT)
                _compact_prompt_tokens = _token_count(COMPACTION_PROMPT)
                compact_budget = max(16, N // 2 - _compact_prompt_tokens)
                _respond_bounded(llm, max_output_tokens=compact_budget, temperature=temperature,
                                 disable_thinking=True)

                last_content = _last_assistant_content(chat) or ""

                # Detect final_answer during compaction
                answer = extract_final_answer(last_content)
                if answer is not None:
                    _absorb(chat)
                    cycle_traces.append(_capture_chat_trace(chat))
                    return _result(check_answer(answer, ground_truth), "solved_during_compaction")

                summary = extract_compact_tag(last_content)
                _absorb(chat)
                cycle_traces.append(_capture_chat_trace(chat))
                if summary is None:
                    return _result(False, "compact_parse_fail")

                n_compactions += 1

                if _over_cost_cap():
                    return _result(False, "cost_cap_exceeded")
                if _over_both_limits():
                    return _result(False, f"limits_exceeded_{n_compactions}_compactions")

            # Chat closes — context dropped. Check resume prompt fits in N.
            # tokens_left = remaining budget before next compaction fires.
            stub_post = build_post_compaction_prompt(summary, 0)
            resume_overhead = _token_count(stub_post)
            tokens_left = max(16, N // 2 - resume_overhead - 5)
            post = build_post_compaction_prompt(summary, tokens_left)
            reset_input_est = _token_count(post)
            if reset_input_est >= N:
                return _result(False, "compaction_insufficient")

        # Exhausted hard safety cap
        return _result(False, f"safety_cap_exceeded_{n_compactions}_compactions")

    except Exception as e:
        return TrialOutcome(
            success=False, aborted=True,
            cost_nanodollars=total_cost,
            input_tokens=total_input, output_tokens=total_output,
            finish_reason=f"error:{type(e).__name__}",
            traces=cycle_traces,
        )


def _compact_result(
    success: bool, cost: int, inp: int, out: int, finish: str,
    traces: Optional[list] = None,
) -> TrialOutcome:
    return TrialOutcome(
        success=success,
        aborted=finish in ("cost_cap_exceeded", "compaction_insufficient"),
        cost_nanodollars=cost,
        input_tokens=inp,
        output_tokens=out,
        finish_reason=finish,
        traces=traces or [],
    )


# ── Phase 3+4 driver: find n_reliable ───────────────────────────────────────

def find_n_reliable(
    llm,
    problem_text: str,
    ground_truth,
    n_max: int,
    compaction_enabled: bool,
    temperature: float = 0.7,
    cost_cap_nanodollars: Optional[int] = None,
    token_cap: Optional[int] = None,
) -> tuple[float, SearchLog, int, int, int]:
    """Run the nested log-scale binary search.

    Returns (n_reliable, log, total_cost, total_input, total_output) — so the
    caller can aggregate costs across the sweep for the scoring step.

    token_cap: per-trial token cap for compact trials (5 * n_while_unbounded).
    cost_cap_nanodollars: cumulative cost cap across the entire sweep.
    """
    if n_max < 1:
        return math.inf, SearchLog(), 0, 0, 0

    cumulative_cost = 0
    cumulative_input = 0
    cumulative_output = 0
    aborted_due_to_cost = False

    def trial_fn(N: int, n_trials: int) -> list[TrialOutcome]:
        nonlocal cumulative_cost, cumulative_input, cumulative_output, aborted_due_to_cost
        results: list[TrialOutcome] = []
        for i in range(n_trials):
            if aborted_due_to_cost:
                results.append(TrialOutcome(
                    success=False, aborted=True,
                    cost_nanodollars=0, input_tokens=0, output_tokens=0,
                    finish_reason="cost_cap_exceeded_in_sweep",
                ))
                continue
            if compaction_enabled:
                outcome = run_trial_compact(
                    llm, problem_text, ground_truth, N,
                    temperature=temperature,
                    cost_cap_nanodollars=cost_cap_nanodollars,
                    token_cap=token_cap,
                )
            else:
                outcome = run_trial_no_compact(
                    llm, problem_text, ground_truth, N,
                    temperature=temperature,
                )
            cumulative_cost += outcome.cost_nanodollars
            cumulative_input += outcome.input_tokens
            cumulative_output += outcome.output_tokens
            results.append(outcome)
            if cost_cap_nanodollars is not None and cumulative_cost > cost_cap_nanodollars:
                aborted_due_to_cost = True
        return results

    n_reliable, log = log_find_n_reliable(trial_fn, n_min=1, n_max=n_max)
    return n_reliable, log, cumulative_cost, cumulative_input, cumulative_output


# ── Helpers ─────────────────────────────────────────────────────────────────

def _last_assistant_content(chat) -> Optional[str]:
    """Return concatenation of all assistant messages since the last user message.

    This handles chained calls correctly: when a response was split across
    multiple `llm.respond()` calls (via "Continue." prompts), we need the full
    concatenated content to find a `{final_answer}` that may be in any chunk.
    """
    messages = list(chat.messages) if hasattr(chat, "messages") else []
    # Walk backward, collect consecutive assistant messages, stop at user/system
    chunks: list[str] = []
    for m in reversed(messages):
        sender = getattr(m, "sender", None)
        role = getattr(sender, "role", None) if sender else None
        if role == "assistant":
            content = getattr(m, "content", None)
            if isinstance(content, str):
                chunks.append(content)
            elif content is not None:
                chunks.append(str(content))
        else:
            break  # hit the user/system message that preceded this assistant run
    if not chunks:
        return None
    # Reverse to get chronological order, join with newlines
    return "\n".join(reversed(chunks))


def _finish_reason(msg) -> Optional[str]:
    """Extract the finish_reason from a kbench Message, if present in meta."""
    if msg is None:
        return None
    meta = getattr(msg, "_meta", None) or {}
    # Different backends store finish_reason under different keys
    for key in ("finish_reason", "stop_reason", "finishReason"):
        v = meta.get(key) if isinstance(meta, dict) else None
        if v:
            return str(v)
    return None


def _close_chat(chat_cm) -> None:
    """Idempotently close a chat context manager."""
    try:
        chat_cm.__exit__(None, None, None)
    except Exception:
        pass
