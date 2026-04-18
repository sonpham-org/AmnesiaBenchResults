"""Unit tests for amnesia_kaggle/parsers.py."""

from __future__ import annotations

import math

import pytest

from amnesia_kaggle.parsers import (
    check_answer,
    extract_compact_tag,
    extract_final_answer,
    parse_prediction,
)


# ── extract_final_answer ────────────────────────────────────────────────────

class TestExtractFinalAnswer:
    def test_canonical_format(self):
        assert extract_final_answer('{final_answer: "42"}') == "42"

    def test_no_space_after_colon(self):
        assert extract_final_answer('{final_answer:"42"}') == "42"

    def test_with_surrounding_text(self):
        text = (
            "Let me work through this step by step. After careful analysis, "
            'the answer is {final_answer: "554"}. I am confident in this result.'
        )
        assert extract_final_answer(text) == "554"

    def test_strips_whitespace(self):
        assert extract_final_answer('{final_answer: "  42  "}') == "42"

    def test_answer_with_spaces(self):
        assert extract_final_answer('{final_answer: "twenty four"}') == "twenty four"

    def test_answer_with_special_chars(self):
        assert extract_final_answer('{final_answer: "3.14e-10"}') == "3.14e-10"

    def test_negative_number(self):
        assert extract_final_answer('{final_answer: "-17"}') == "-17"

    def test_fraction_answer(self):
        assert extract_final_answer('{final_answer: "1/3"}') == "1/3"

    def test_no_final_answer(self):
        assert extract_final_answer("I cannot solve this problem.") is None

    def test_empty_string(self):
        assert extract_final_answer("") is None

    def test_none_input(self):
        assert extract_final_answer(None) is None

    def test_found_inside_compaction_turn(self):
        """Plan.md General Notes: detect final_answer even in compaction output."""
        text = (
            "<compact>\n"
            "I computed the integral and got 554.\n"
            'Actually I\'m confident — {final_answer: "554"}\n'
            "</compact>"
        )
        assert extract_final_answer(text) == "554"

    def test_found_on_truncated_output(self):
        """Plan.md General Notes: detect even if model was truncated."""
        text = 'After working through... the answer is {final_answer: "7"}. Now let me'
        assert extract_final_answer(text) == "7"

    def test_bare_fallback(self):
        """Defensive: accept `final_answer: "X"` without outer braces."""
        assert extract_final_answer('my final_answer: "42" is here') == "42"

    def test_first_match_wins(self):
        """If there are multiple, take the first one (most models answer once)."""
        text = '{final_answer: "first"} ... {final_answer: "second"}'
        assert extract_final_answer(text) == "first"


# ── check_answer ────────────────────────────────────────────────────────────

class TestCheckAnswer:
    def test_exact_match(self):
        assert check_answer("42", "42") is True

    def test_int_ground_truth(self):
        assert check_answer("42", 42) is True

    def test_whitespace_tolerance(self):
        assert check_answer("  42  ", "42") is True

    def test_mismatch(self):
        assert check_answer("42", "43") is False

    def test_none_answer(self):
        assert check_answer(None, "42") is False

    def test_case_sensitive(self):
        """For math problems these are numeric; we don't lowercase."""
        assert check_answer("Yes", "yes") is False


# ── extract_compact_tag ─────────────────────────────────────────────────────

class TestExtractCompactTag:
    def test_simple(self):
        assert extract_compact_tag("<compact>summary here</compact>") == "summary here"

    def test_multiline(self):
        text = (
            "<compact>\n"
            "Problem: compute X\n"
            "Progress: got Y\n"
            "Next: verify Z\n"
            "</compact>"
        )
        result = extract_compact_tag(text)
        assert result is not None
        assert "Problem: compute X" in result
        assert "Next: verify Z" in result

    def test_with_surrounding_text(self):
        text = "Here is my summary:\n<compact>key facts</compact>\nDone."
        assert extract_compact_tag(text) == "key facts"

    def test_empty_tag(self):
        assert extract_compact_tag("<compact></compact>") == ""

    def test_missing_tag(self):
        assert extract_compact_tag("no compact here") is None

    def test_none_input(self):
        assert extract_compact_tag(None) is None

    def test_strips_whitespace(self):
        assert extract_compact_tag("<compact>  hello  </compact>") == "hello"


# ── parse_prediction ────────────────────────────────────────────────────────

class TestParsePrediction:
    def test_canonical_true(self):
        attempt, n = parse_prediction('{attempt: "True", N: "1024"}')
        assert attempt is True
        assert n == 1024.0

    def test_canonical_false(self):
        attempt, n = parse_prediction('{attempt: "False", N: "0"}')
        assert attempt is False
        assert n == math.inf

    def test_lowercase_true(self):
        attempt, n = parse_prediction('{attempt: "true", N: "512"}')
        assert attempt is True
        assert n == 512.0

    def test_without_quotes(self):
        """Some models drop the quotes — accept gracefully."""
        attempt, n = parse_prediction("{attempt: True, N: 2048}")
        assert attempt is True
        assert n == 2048.0

    def test_with_surrounding_prose(self):
        raw = (
            "I think I can solve this. My determination is:\n"
            '{attempt: "True", N: "4096"}\n'
            "That should be enough."
        )
        attempt, n = parse_prediction(raw)
        assert attempt is True
        assert n == 4096.0

    def test_n_infinity(self):
        attempt, n = parse_prediction('{attempt: "True", N: "inf"}')
        assert attempt is True
        assert n == math.inf

    def test_n_zero_means_inf(self):
        attempt, n = parse_prediction('{attempt: "False", N: "0"}')
        assert n == math.inf

    def test_parse_failure_fallback(self):
        """Plan.md §6: 'If the answers can't be extracted, set attempt=True, N=inf.'"""
        attempt, n = parse_prediction("I cannot decide whether to attempt.")
        assert attempt is True
        assert n == math.inf

    def test_empty_string(self):
        attempt, n = parse_prediction("")
        assert attempt is True
        assert n == math.inf

    def test_none_input(self):
        attempt, n = parse_prediction(None)
        assert attempt is True
        assert n == math.inf

    def test_only_attempt_no_n(self):
        attempt, n = parse_prediction('{attempt: "True"}')
        assert attempt is True
        assert n == math.inf  # N missing → fallback

    def test_negative_n_treated_as_inf(self):
        # Our regex only matches \d+ so negative won't parse as N; fallback to inf.
        attempt, n = parse_prediction('{attempt: "True", N: "-5"}')
        assert attempt is True
        assert n == math.inf
