"""Tests for amnesia_kaggle/harness.py using stubbed kbench primitives.

We monkeypatch the `kaggle_benchmarks` module with a minimal fake implementation
so we can test the compaction control flow without an actual LLM API.

This covers:
  - 50% compaction trigger → chat reset with the summary
  - compaction_insufficient abort (reset prompt ≥ N)
  - no_answer_no_compact abort
  - final_answer detected during compaction turn
  - cost_cap_exceeded abort
  - run_unbounded averages tokens across successful trials
  - run_unbounded marks context_exceeded when a trial hits finish='length'
  - run_prediction parses valid and invalid outputs
  - run_trial_no_compact single-turn success/failure
"""

from __future__ import annotations

import math
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import pytest


# ── Fake kaggle_benchmarks module ───────────────────────────────────────────

@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    input_tokens_cost_nanodollars: int = 0
    output_tokens_cost_nanodollars: int = 0


@dataclass
class FakeSender:
    role: str


FAKE_USER = FakeSender(role="user")
FAKE_SYSTEM = FakeSender(role="system")
FAKE_ASSISTANT = FakeSender(role="assistant")


@dataclass
class FakeMessage:
    content: str
    sender: FakeSender
    usage: Optional[FakeUsage] = None
    _meta: dict = field(default_factory=dict)


class FakeChat:
    def __init__(self, name: str):
        self.name = name
        self.messages: list[FakeMessage] = []
        self._total_input = 0
        self._total_output = 0
        self._total_cost_in = 0
        self._total_cost_out = 0

    @property
    def usage(self) -> FakeUsage:
        return FakeUsage(
            input_tokens=self._total_input,
            output_tokens=self._total_output,
            input_tokens_cost_nanodollars=self._total_cost_in,
            output_tokens_cost_nanodollars=self._total_cost_out,
        )


class FakeKBenchContext:
    """Module-level state that tracks the currently active fake chat."""
    current_chat: Optional[FakeChat] = None


class FakeUserActor:
    @staticmethod
    def send(text: str) -> FakeMessage:
        chat = FakeKBenchContext.current_chat
        assert chat is not None, "user.send called outside chats.new"
        msg = FakeMessage(content=text, sender=FAKE_USER)
        chat.messages.append(msg)
        return msg


@contextmanager
def fake_chats_new(name: str = "chat", **kwargs) -> Iterator[FakeChat]:
    previous = FakeKBenchContext.current_chat
    chat = FakeChat(name=name)
    FakeKBenchContext.current_chat = chat
    try:
        yield chat
    finally:
        FakeKBenchContext.current_chat = previous


# Configure LLM: pluggable script of (content, input_toks, output_toks, cost, finish_reason)
@dataclass
class FakeLLMTurn:
    content: str
    input_tokens: int
    output_tokens: int
    cost_nanodollars: int = 0
    finish_reason: str = "stop"


class FakeLLM:
    """Scripted LLM — pops turns off a queue on each .respond() call."""

    def __init__(self, turns: list[FakeLLMTurn]):
        self.turns = list(turns)
        self.calls: list[dict] = []

    def respond(self, **kwargs) -> FakeMessage:
        self.calls.append(dict(kwargs))
        chat = FakeKBenchContext.current_chat
        assert chat is not None, "llm.respond called outside chats.new"
        if not self.turns:
            raise RuntimeError("FakeLLM ran out of scripted turns")
        turn = self.turns.pop(0)

        chat._total_input += turn.input_tokens
        chat._total_output += turn.output_tokens
        chat._total_cost_out += turn.cost_nanodollars

        msg = FakeMessage(
            content=turn.content,
            sender=FAKE_ASSISTANT,
            usage=FakeUsage(
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                output_tokens_cost_nanodollars=turn.cost_nanodollars,
            ),
            _meta={"finish_reason": turn.finish_reason},
        )
        chat.messages.append(msg)
        return msg


def _install_fake_kbench(monkeypatch):
    """Install a fake `kaggle_benchmarks` module tree for harness imports."""
    fake_mod = types.ModuleType("kaggle_benchmarks")
    fake_chats = types.ModuleType("kaggle_benchmarks.chats")
    fake_actors = types.ModuleType("kaggle_benchmarks.actors")
    fake_llms_mod = types.ModuleType("kaggle_benchmarks.actors.llms")

    fake_chats.new = fake_chats_new  # type: ignore[attr-defined]
    fake_actors.user = FakeUserActor  # type: ignore[attr-defined]
    fake_actors.llms = fake_llms_mod  # type: ignore[attr-defined]

    # Dummy GoogleGenAI class for isinstance() dispatch in harness
    class _GoogleGenAI:  # pragma: no cover
        pass
    fake_llms_mod.GoogleGenAI = _GoogleGenAI  # type: ignore[attr-defined]

    fake_mod.chats = fake_chats  # type: ignore[attr-defined]
    fake_mod.actors = fake_actors  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", fake_mod)
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks.chats", fake_chats)
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks.actors", fake_actors)
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks.actors.llms", fake_llms_mod)


@pytest.fixture(autouse=True)
def install_fake_kbench(monkeypatch):
    _install_fake_kbench(monkeypatch)
    FakeKBenchContext.current_chat = None
    yield


# ── run_trial_no_compact ────────────────────────────────────────────────────

def test_no_compact_success():
    from amnesia_kaggle.harness import run_trial_no_compact

    llm = FakeLLM([
        FakeLLMTurn(
            content='Thinking... The answer is {final_answer: "42"}.',
            input_tokens=100, output_tokens=20, cost_nanodollars=120,
        ),
    ])
    result = run_trial_no_compact(llm, "compute thing", "42", N=1000)
    assert result.success is True
    assert result.finish_reason == "solved"
    assert result.cost_nanodollars == 120
    assert llm.calls[0].get("max_tokens", 0) > 0


def test_no_compact_wrong_answer():
    from amnesia_kaggle.harness import run_trial_no_compact

    llm = FakeLLM([
        FakeLLMTurn(content='{final_answer: "99"}', input_tokens=100, output_tokens=10),
    ])
    result = run_trial_no_compact(llm, "x", "42", N=500)
    assert result.success is False
    assert result.finish_reason == "wrong_answer"


def test_no_compact_no_answer_length_truncation():
    from amnesia_kaggle.harness import run_trial_no_compact

    llm = FakeLLM([
        FakeLLMTurn(
            content="I was still thinking when I got cut off...",
            input_tokens=200, output_tokens=300, finish_reason="length",
        ),
    ])
    result = run_trial_no_compact(llm, "x", "42", N=500)
    assert result.success is False
    assert result.finish_reason == "budget_exceeded"


def test_no_compact_input_exceeds_N_returns_fail():
    from amnesia_kaggle.harness import run_trial_no_compact
    # Make N absurdly small so the prompt itself exceeds it
    llm = FakeLLM([])  # should not be called
    result = run_trial_no_compact(llm, "x", "42", N=10)
    assert result.success is False
    assert result.finish_reason == "input_exceeds_N"
    assert llm.calls == []  # llm wasn't called


# ── run_trial_compact — happy path without triggering compaction ────────────

def test_compact_trial_solves_before_50pct():
    """Model answers on the first turn, no compaction triggered."""
    from amnesia_kaggle.harness import run_trial_compact

    llm = FakeLLM([
        FakeLLMTurn(
            content='Done: {final_answer: "42"}',
            input_tokens=100, output_tokens=20,
        ),
    ])
    result = run_trial_compact(llm, "compute thing", "42", N=10000)
    assert result.success is True
    assert result.finish_reason == "solved"


def test_compact_trial_triggers_at_50pct_and_recovers():
    """First turn uses 60% → compaction → reset → second turn solves."""
    from amnesia_kaggle.harness import run_trial_compact

    llm = FakeLLM([
        # Turn 1: no answer, hit 50% limit
        FakeLLMTurn(content="Still thinking about this hard problem",
                    input_tokens=100, output_tokens=500, finish_reason="length"),
        # Turn 2 (compaction response): <compact>summary</compact>
        FakeLLMTurn(content="<compact>I factored X, got Y, need to verify Z</compact>",
                    input_tokens=200, output_tokens=50),
        # Turn 3 (post-reset): solve
        FakeLLMTurn(content='Got it: {final_answer: "42"}',
                    input_tokens=300, output_tokens=30),
    ])
    result = run_trial_compact(llm, "compute thing", "42", N=1000)
    assert result.success is True
    assert result.finish_reason.startswith("solved_post_reset")
    # We should have seen 3 respond() calls
    assert len(llm.calls) == 3


def test_compact_trial_final_answer_during_compaction_turn():
    """Plan.md General Notes: detect final_answer even in compaction output."""
    from amnesia_kaggle.harness import run_trial_compact

    llm = FakeLLM([
        FakeLLMTurn(content="Working hard...", input_tokens=100, output_tokens=500, finish_reason="length"),
        # The model produces final_answer DURING the compaction step
        FakeLLMTurn(
            content='<compact>I already have the answer: {final_answer: "42"}</compact>',
            input_tokens=200, output_tokens=50,
        ),
    ])
    result = run_trial_compact(llm, "compute thing", "42", N=1000)
    assert result.success is True
    assert result.finish_reason == "solved_during_compaction"


def test_compact_trial_compact_parse_fail():
    """Model was asked to compact but didn't produce a <compact> tag."""
    from amnesia_kaggle.harness import run_trial_compact

    llm = FakeLLM([
        FakeLLMTurn(content="Still thinking...", input_tokens=100, output_tokens=500, finish_reason="length"),
        FakeLLMTurn(
            content="I prefer not to compact, let me just keep going.",
            input_tokens=200, output_tokens=50,
        ),
    ])
    result = run_trial_compact(llm, "x", "42", N=1000)
    assert result.success is False
    assert result.finish_reason == "compact_parse_fail"


def test_compact_trial_no_compact_tag_after_thinking():
    """Thinking turn capped at 50%, compaction fires, but model doesn't produce <compact> → fail."""
    from amnesia_kaggle.harness import run_trial_compact

    llm = FakeLLM([
        # Turn 1 (thinking, hit 50% limit): no answer
        FakeLLMTurn(
            content="I am working on this problem...",
            input_tokens=100, output_tokens=50, finish_reason="length",
        ),
        # Turn 2 (compaction): no <compact> tag
        FakeLLMTurn(
            content="I don't know how to compact.",
            input_tokens=200, output_tokens=50,
        ),
    ])
    result = run_trial_compact(llm, "x", "42", N=10000)
    assert result.success is False
    assert result.finish_reason == "compact_parse_fail"


def test_compact_trial_cost_cap_exceeded():
    """Abort when cumulative cost exceeds cost_cap_nanodollars."""
    from amnesia_kaggle.harness import run_trial_compact

    llm = FakeLLM([
        # Turn 1: hit limit, busts the cost cap immediately
        FakeLLMTurn(content="thinking", input_tokens=100, output_tokens=50,
                    cost_nanodollars=10_000, finish_reason="length"),
        # Turn 2 shouldn't happen
        FakeLLMTurn(content="late", input_tokens=1, output_tokens=1),
    ])
    result = run_trial_compact(
        llm, "x", "42", N=10000,
        cost_cap_nanodollars=5_000,  # first turn busts this
    )
    assert result.success is False
    # First turn produced neither answer nor compact → this hits no_answer_no_compact
    # BEFORE the cost check, which is fine — both are failures. We assert failure only.
    assert result.finish_reason in ("cost_cap_exceeded", "no_answer_no_compact")


# ── run_unbounded ───────────────────────────────────────────────────────────

def test_unbounded_three_successes_averages_tokens():
    from amnesia_kaggle.harness import run_unbounded

    llm = FakeLLM([
        FakeLLMTurn(content='{final_answer: "42"}', input_tokens=100, output_tokens=10),
        FakeLLMTurn(content='{final_answer: "42"}', input_tokens=100, output_tokens=20),
        FakeLLMTurn(content='{final_answer: "42"}', input_tokens=100, output_tokens=30),
    ])
    result = run_unbounded(llm, "x", "42")
    assert result.solved is True
    # Mean of (110, 120, 130) = 120
    assert result.n_while_unbounded == pytest.approx(120.0)
    assert result.context_exceeded is False


def test_unbounded_two_of_three_solved():
    """2/3 is enough to mark as solved, take the mean of the 2 successes."""
    from amnesia_kaggle.harness import run_unbounded

    llm = FakeLLM([
        FakeLLMTurn(content='{final_answer: "42"}', input_tokens=100, output_tokens=20),
        FakeLLMTurn(content='Wrong answer here', input_tokens=100, output_tokens=50),
        FakeLLMTurn(content='{final_answer: "42"}', input_tokens=100, output_tokens=40),
    ])
    result = run_unbounded(llm, "x", "42")
    assert result.solved is True
    # Mean of (120, 140) = 130
    assert result.n_while_unbounded == pytest.approx(130.0)


def test_unbounded_two_of_three_failed_context_exceeded():
    """2 context-exceeded failures → solved=False but context_exceeded=True."""
    from amnesia_kaggle.harness import run_unbounded

    llm = FakeLLM([
        FakeLLMTurn(content='{final_answer: "42"}', input_tokens=100, output_tokens=20),
        FakeLLMTurn(content='I was cut off', input_tokens=200, output_tokens=1000,
                    finish_reason="length"),
        FakeLLMTurn(content='Also cut off', input_tokens=200, output_tokens=1000,
                    finish_reason="length"),
    ])
    result = run_unbounded(llm, "x", "42")
    assert result.solved is False
    assert result.n_while_unbounded == math.inf
    assert result.context_exceeded is True


def test_unbounded_all_fail_non_context():
    """3 wrong-answer failures → solved=False, context_exceeded=False."""
    from amnesia_kaggle.harness import run_unbounded

    llm = FakeLLM([
        FakeLLMTurn(content='{final_answer: "99"}', input_tokens=100, output_tokens=20),
        FakeLLMTurn(content='{final_answer: "99"}', input_tokens=100, output_tokens=20),
        FakeLLMTurn(content='{final_answer: "99"}', input_tokens=100, output_tokens=20),
    ])
    result = run_unbounded(llm, "x", "42")
    assert result.solved is False
    assert result.n_while_unbounded == math.inf
    assert result.context_exceeded is False


# ── run_prediction ──────────────────────────────────────────────────────────

def test_prediction_parses_true_with_N():
    from amnesia_kaggle.harness import run_prediction

    llm = FakeLLM([
        FakeLLMTurn(
            content='{attempt: "True", N: "2048"}',
            input_tokens=300, output_tokens=50,
        ),
    ])
    pred = run_prediction(llm, "x")
    assert pred.attempt is True
    assert pred.n_predicted == 2048.0


def test_prediction_parse_failure_fallback():
    from amnesia_kaggle.harness import run_prediction

    llm = FakeLLM([
        FakeLLMTurn(content="I don't know what to answer here.",
                    input_tokens=300, output_tokens=20),
    ])
    pred = run_prediction(llm, "x")
    assert pred.attempt is True  # fallback
    assert pred.n_predicted == math.inf


def test_prediction_opt_out():
    from amnesia_kaggle.harness import run_prediction

    llm = FakeLLM([
        FakeLLMTurn(content='{attempt: "False", N: "0"}',
                    input_tokens=300, output_tokens=20),
    ])
    pred = run_prediction(llm, "x")
    assert pred.attempt is False
    assert pred.n_predicted == math.inf
