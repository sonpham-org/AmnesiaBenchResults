# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: All LLM backend clients for AmnesiaBench v3. Provides LLMClient (OpenAI-compat),
#   GeminiClient (Google Gemini API), AnthropicClient (direct Anthropic API via OAuth),
#   and create_client() factory. Each client returns the same dict shape from generate().
#   Integration points: imported by predict.py and evaluate.py.
#   Exponential backoff is applied inside each client via backoff.with_exponential_backoff().
#   v3.1: Added stream=False support to LLMClient for exact token counts + llama-server
#   timings block (prefill/decode speeds, KV cache hits). GeminiClient and AnthropicClient
#   accept stream param for API consistency but handle streaming internally.
# SRP/DRY check: Pass — one class per backend; factory is the only routing logic; no
#   prompt construction here (that lives in prompts.py).

import json
import os
import sys
from typing import Optional, Union

import requests

from .backoff import with_exponential_backoff

TEMPERATURE = 0.7


# ─── LLMClient ───────────────────────────────────────────────────────────────

class LLMClient:
    """
    Client for any OpenAI-compatible /v1/chat/completions endpoint.
    Covers llama.cpp local servers and OpenRouter (via model_name passthrough).

    stream=True  → SSE streaming (for predict / interactive use)
    stream=False → single JSON response; llama-server returns timings block with
                   exact prefill/decode speeds and KV cache hit counts.
    """

    def __init__(
        self,
        server_url: str,
        temperature: float = TEMPERATURE,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.model_name = model_name
        self._auth_header = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def generate(self, messages: list, max_tokens: int, stream: bool = True) -> dict:
        """
        Send messages to the model. Returns:
        {
            content: str,           # full content (including <think> wrapper if reasoning)
            reasoning_content: str, # raw reasoning/thinking tokens (DeepSeek R1, etc.)
            final_content: str,     # non-reasoning response text only
            input_tokens: int,      # prompt token count
            output_tokens: int,     # completion token count (all tokens incl. reasoning)
            thinking_tokens: int,   # reasoning-only tokens (0 for most models)
            total_tokens: int,
            finish_reason: str,
            timings: dict,          # llama-server timings (non-streaming only; {} otherwise)
        }

        timings keys (when stream=False against llama-server):
            cache_n, prompt_n, prompt_ms, prompt_per_second,
            predicted_n, predicted_ms, predicted_per_second
        """
        max_tokens = max(1, max_tokens)
        url = f"{self.server_url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if self.model_name:
            payload["model"] = self.model_name

        def _do_request():
            headers = dict(self._auth_header)
            if "openrouter.ai" in self.server_url:
                headers["X-OpenRouter-Cache"] = "true"
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=3600,
                stream=stream,
            )
            resp.raise_for_status()
            return resp

        resp = with_exponential_backoff(_do_request)

        # ── Non-streaming path ────────────────────────────────────────────────
        if not stream:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "") or ""
            finish_reason = choice.get("finish_reason", "stop") or "stop"

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

            timings = data.get("timings", {})
            prefill_tps = timings.get("prompt_per_second", 0)
            decode_tps = timings.get("predicted_per_second", 0)
            cache_n = timings.get("cache_n", 0)

            if reasoning:
                full_content = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                full_content = content

            # Count thinking tokens — reasoning_content from API or <think> blocks in content
            thinking_tokens = 0
            if reasoning:
                thinking_tokens = len(reasoning) // 4  # rough estimate: 4 chars/token

            print(
                f"    [llm] in={input_tokens} out={output_tokens} think={thinking_tokens} | {finish_reason} | "
                f"prefill={prefill_tps:.0f}t/s decode={decode_tps:.0f}t/s cache={cache_n}"
            )

            return {
                "content": full_content,
                "reasoning_content": reasoning,
                "final_content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_tokens,
                "finish_reason": finish_reason,
                "timings": timings,
            }

        # ── Streaming path (predict / interactive) ────────────────────────────
        reasoning = ""
        content = ""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        finish_reason = "unknown"

        print("    [stream] ", end="", flush=True)
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            if text.startswith("data: "):
                text = text[6:]
            if text == "[DONE]":
                break
            try:
                chunk = json.loads(text)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            r_piece = delta.get("reasoning_content", "") or ""
            c_piece = delta.get("content", "") or ""
            if r_piece:
                reasoning += r_piece
                sys.stdout.write(r_piece)
                sys.stdout.flush()
            if c_piece:
                content += c_piece
                sys.stdout.write(c_piece)
                sys.stdout.flush()
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            usage = chunk.get("usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
        print()

        if reasoning:
            full_content = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            full_content = content

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        thinking_tokens = len(reasoning) // 4 if reasoning else 0

        return {
            "content": full_content,
            "reasoning_content": reasoning,
            "final_content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "timings": {},
        }

    def ping(self) -> bool:
        """Check if the server is reachable. Skipped for remote cloud endpoints."""
        if "openrouter.ai" in self.server_url:
            return True
        if "localhost" not in self.server_url and not self.server_url.startswith("http://192."):
            return True
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ─── GeminiClient ─────────────────────────────────────────────────────────────

class GeminiClient:
    """
    Client for Google Gemini generateContent API.
    Accepts OpenAI-style message lists; converts to Gemini format internally.
    Returns same dict shape as LLMClient.generate().
    stream parameter accepted for API consistency but ignored (Gemini uses REST).
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-lite",
        temperature: float = TEMPERATURE,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def _convert_messages(self, messages: list) -> tuple:
        """Convert OpenAI-style messages to (system_instruction, contents) for Gemini."""
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = {"parts": [{"text": text}]}
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})
        return system_instruction, contents

    def generate(self, messages: list, max_tokens: int, stream: bool = True) -> dict:
        max_tokens = max(1, max_tokens)
        system_instruction, contents = self._convert_messages(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": self.temperature,
            },
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction

        url = f"{self.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"

        def _do_request():
            resp = requests.post(url, json=payload, timeout=3600)
            resp.raise_for_status()
            return resp

        resp = with_exponential_backoff(_do_request)
        data = resp.json()

        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)
            fr_raw = candidate.get("finishReason", "STOP")
            fr_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "stop",
                      "RECITATION": "stop", "OTHER": "stop"}
            finish_reason = fr_map.get(fr_raw, "stop")
        else:
            content = ""
            finish_reason = "stop"

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", input_tokens + output_tokens)

        print(f"    [gemini] {output_tokens} tokens | finish={finish_reason}")
        print(content[:120].replace("\n", " ") + ("..." if len(content) > 120 else ""))

        return {
            "content": content,
            "reasoning_content": "",
            "final_content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": 0,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "timings": {},
        }

    def ping(self) -> bool:
        try:
            resp = self.generate([{"role": "user", "content": "Say OK."}], max_tokens=10)
            return bool(resp.get("content"))
        except Exception:
            return False


# ─── AnthropicClient ──────────────────────────────────────────────────────────

class AnthropicClient:
    """
    Direct client for Anthropic's Messages API using an OAuth bearer token.

    Auth requirements (BOTH headers required — missing either = 401):
        Authorization: Bearer {ANTHROPIC_OAUTHTOKEN}
        anthropic-beta: oauth-2025-04-20
        anthropic-version: 2023-06-01

    Features:
        - Prompt caching on system message via cache_control: {"type": "ephemeral"}
        - Streaming via SSE (stream param accepted for API consistency but ignored)
        - Extended thinking tokens tracked in thinking_tokens field
        - Supported models: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5

    Returns same dict shape as LLMClient.generate().
    timings is always {} (Anthropic API does not return llama-style timings).
    """

    ENDPOINT = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        model: str,
        temperature: float = TEMPERATURE,
        oauth_token: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self._token = (oauth_token
                       or os.environ.get("ANTHROPIC_OAUTHTOKEN", "")
                       or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""))
        if not self._token:
            raise ValueError(
                "AnthropicClient requires ANTHROPIC_OAUTHTOKEN env var "
                "or explicit oauth_token argument."
            )

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "anthropic-beta": "oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: list) -> tuple:
        """
        Split OpenAI-style messages into (system_blocks, anthropic_messages).
        System messages are converted to Anthropic's system block format with
        cache_control for prompt caching.
        """
        system_blocks = []
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_blocks.append({
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                })
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                anthropic_messages.append({"role": "user", "content": content})

        return system_blocks, anthropic_messages

    def generate(self, messages: list, max_tokens: int, stream: bool = True) -> dict:
        # Anthropic always uses SSE internally; stream param accepted for API consistency
        max_tokens = max(1, max_tokens)
        system_blocks, anthropic_messages = self._convert_messages(messages)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "messages": anthropic_messages,
        }
        if system_blocks:
            payload["system"] = system_blocks

        def _do_request():
            resp = requests.post(
                self.ENDPOINT,
                headers=self._build_headers(),
                json=payload,
                timeout=3600,
                stream=True,
            )
            resp.raise_for_status()
            return resp

        resp = with_exponential_backoff(_do_request)

        content_text = ""
        thinking_text = ""
        input_tokens = 0
        output_tokens = 0
        finish_reason = "unknown"
        current_block_type = None  # noqa: F841

        print("    [anthropic] ", end="", flush=True)

        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8") if isinstance(line, bytes) else line

            if text.startswith("event: "):
                continue

            if not text.startswith("data: "):
                continue
            data_str = text[6:]
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "content_block_start":
                block = event.get("content_block", {})
                current_block_type = block.get("type", "text")

            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                dtype = delta.get("type", "")
                if dtype == "text_delta":
                    piece = delta.get("text", "")
                    content_text += piece
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                elif dtype == "thinking_delta":
                    piece = delta.get("thinking", "")
                    thinking_text += piece
                    sys.stdout.write(piece)
                    sys.stdout.flush()

            elif etype == "message_delta":
                delta = event.get("delta", {})
                fr = delta.get("stop_reason")
                if fr:
                    finish_reason = fr
                usage = event.get("usage", {})
                if usage:
                    output_tokens = usage.get("output_tokens", output_tokens)

            elif etype == "message_start":
                msg_obj = event.get("message", {})
                usage = msg_obj.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

            elif etype == "message_stop":
                break

        print()

        if thinking_text:
            full_content = f"<think>\n{thinking_text}\n</think>\n{content_text}"
        else:
            full_content = content_text

        total_tokens = input_tokens + output_tokens
        # Rough thinking token estimate — Anthropic includes thinking in output_tokens
        thinking_tokens = len(thinking_text) // 4 if thinking_text else 0

        return {
            "content": full_content,
            "reasoning_content": thinking_text,
            "final_content": content_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "timings": {},
        }

    def ping(self) -> bool:
        """Attempt a minimal generation to verify credentials."""
        try:
            resp = self.generate(
                [{"role": "user", "content": "Say OK."}],
                max_tokens=10,
            )
            return bool(resp.get("content"))
        except Exception:
            return False


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_client(
    server_url: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = TEMPERATURE,
) -> Union[LLMClient, GeminiClient, AnthropicClient]:
    """
    Create the appropriate client based on the URL scheme:

      anthropic://MODEL   → AnthropicClient (OAuth, direct Anthropic API)
      gemini://MODEL      → GeminiClient (Google Gemini API, requires api_key)
      google://MODEL      → GeminiClient (alias)
      openrouter://MODEL  → LLMClient pointed at openrouter.ai
      http://...          → LLMClient (llama.cpp or any OAI-compat server)
      https://...         → LLMClient

    api_key is required for gemini:// and openrouter://.
    For anthropic://, token is read from ANTHROPIC_OAUTHTOKEN env var.
    """
    if server_url.startswith("anthropic://"):
        model = server_url[len("anthropic://"):].strip("/") or "claude-sonnet-4-6"
        return AnthropicClient(model=model, temperature=temperature)

    if server_url.startswith("gemini://") or server_url.startswith("google://"):
        scheme = "gemini://" if server_url.startswith("gemini://") else "google://"
        gemini_model = server_url[len(scheme):].strip("/") or "gemini-2.0-flash-lite"
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GeminiClient requires an API key. Pass --api-key or set GEMINI_API_KEY."
            )
        return GeminiClient(api_key=api_key, model=gemini_model, temperature=temperature)

    if server_url.startswith("openrouter://"):
        or_model = server_url[len("openrouter://"):].strip("/")
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter requires an API key. Pass --api-key or set OPENROUTER_API_KEY."
            )
        return LLMClient(
            server_url="https://openrouter.ai/api",
            temperature=temperature,
            api_key=api_key,
            model_name=or_model or model_name,
        )

    if server_url.startswith("http://") or server_url.startswith("https://"):
        return LLMClient(
            server_url=server_url,
            temperature=temperature,
            api_key=api_key,
            model_name=model_name,
        )

    raise ValueError(
        f"Unrecognised URL scheme: '{server_url}'. "
        "Use http://, https://, anthropic://, openrouter://, gemini://, or google://"
    )
