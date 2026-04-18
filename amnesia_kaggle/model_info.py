"""Context windows for models available via Kaggle Model Proxy.

Context windows are TOTAL context (input + output). Values as of 2026-04-12
verified from official model documentation. For unknown models, default
to 131072 (128K).

Sources (see each family below).
"""

from __future__ import annotations

DEFAULT_CTX_WINDOW = 131072  # safe default if model not in dict

# Keys match Kaggle Model Proxy model identifiers.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # ── Anthropic Claude ──
    # docs.claude.com: Opus 4.6 / Sonnet 4.6 = 1M standard. Older = 200K.
    # Sonnet 4.5 / Sonnet 4 have a 1M beta but it's being retired 2026-04-30.
    "anthropic/claude-opus-4.6":         1_000_000,
    "anthropic/claude-sonnet-4.6":       1_000_000,
    "anthropic/claude-opus-4.5":           200_000,
    "anthropic/claude-sonnet-4.5":         200_000,
    "anthropic/claude-haiku-4.5":          200_000,
    "anthropic/claude-opus-4.1":           200_000,
    "anthropic/claude-sonnet-4":           200_000,

    # ── DeepSeek ──
    # V3.2: 128K (reasoner) / 163K (config); V3.1: 128K; R1: 64K reasoner
    "deepseek/deepseek-v3.2":              128_000,
    "deepseek/deepseek-v3.1":              128_000,
    "deepseek/deepseek-r1":                 64_000,

    # ── Google Gemini ──
    # ai.google.dev / Vertex docs: all Gemini 2.5/3/3.1 models = 1M input tokens.
    # No 2M models as of Apr 2026.
    "google/gemini-3.1-pro":             1_048_576,
    "google/gemini-3.1-flash-lite":      1_048_576,
    "google/gemini-3-flash":             1_048_576,
    "google/gemini-2.5-pro":             1_048_576,
    "google/gemini-2.5-flash":           1_048_576,
    "google/gemini-2.0-flash":           1_048_576,
    "google/gemini-2.0-flash-lite":      1_048_576,

    # ── Google Gemma (open) ──
    # deepmind.google/models/gemma: 4 26B A4B / 31B = 256K; 3 27B / 12B / 4B = 128K; 3 1B = 32K
    "google/gemma-4-26b":                  262_144,
    "google/gemma-4-26b-a4b":              262_144,
    "google/gemma-4-31b":                  262_144,
    "google/gemma-3-27b":                  131_072,
    "google/gemma-3-12b":                  131_072,
    "google/gemma-3-4b":                   131_072,
    "google/gemma-3-1b":                    32_768,

    # ── OpenAI GPT ──
    # OpenAI docs: GPT-5.4 / mini / nano = 400K. gpt-oss 120b/20b = 128K.
    "openai/gpt-5.4":                      400_000,
    "openai/gpt-5.4-mini":                 400_000,
    "openai/gpt-5.4-nano":                 400_000,
    "openai/gpt-oss-120b":                 131_072,
    "openai/gpt-oss-20b":                  131_072,

    # ── Qwen (open) ──
    # HF model cards: Qwen3-Next-80B / Qwen3-235B-A22B / Qwen3-Coder-480B = 262144 (256K)
    "qwenlm/qwen-3-next-80b-thinking":     262_144,
    "qwenlm/qwen-3-next-80b-instruct":     262_144,
    "qwenlm/qwen-3-coder-480b-a35b":       262_144,
    "qwenlm/qwen-3-235b-a22b-instruct":    262_144,

    # ── Z.ai GLM ──
    # GLM-5 (Zhipu) = 204,800 tokens (200K). 128K output.
    "zai/glm-5":                           204_800,

    # ── Ollama local (for parity) ──
    "gemma4:26b":                          262_144,
    "gemma4:31b":                          262_144,
    "gemma4:latest":                       131_072,
    "gemma3:27b":                          131_072,
}


def get_context_window(model_name: str) -> int:
    """Return the context window for a model name. Falls back to DEFAULT_CTX_WINDOW.

    Tries exact match first, then prefix match (longest prefix wins), so
    "google/gemini-2.5-flash-001" still matches "google/gemini-2.5-flash".
    """
    if not model_name:
        return DEFAULT_CTX_WINDOW
    if model_name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_name]
    best = None
    for key in MODEL_CONTEXT_WINDOWS:
        if model_name.startswith(key):
            if best is None or len(key) > len(best):
                best = key
    if best:
        return MODEL_CONTEXT_WINDOWS[best]
    return DEFAULT_CTX_WINDOW
