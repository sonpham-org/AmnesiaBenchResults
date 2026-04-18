"""Ollama adapter — wraps a local Ollama model as a kbench LLMChat.

Usage:
    from amnesia_kaggle.ollama_adapter import OllamaChat
    llm = OllamaChat(model="qwen3.5:4b")

    import kaggle_benchmarks as kbench
    with kbench.chats.new("test") as chat:
        kbench.actors.user.send("What is 2+2?")
        llm.respond(max_tokens=100)
        print(chat.messages[-1].content)
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from kaggle_benchmarks.actors.llms import LLMChat, LLMResponse


class OllamaChat(LLMChat):
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 0,
        think: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("name", model)
        super().__init__(support_temperature=True, **kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think
        # num_ctx=0 means use Ollama's default (typically 32K).
        # Don't auto-detect the model's theoretical max — it may be 262K
        # which blows out KV cache on consumer GPUs.
        self.num_ctx = num_ctx if num_ctx > 0 else 0  # 0 = let Ollama decide

    def invoke(self, messages, system=None, **kwargs) -> LLMResponse:
        max_tokens = (
            kwargs.get("max_tokens")
            or kwargs.get("max_output_tokens")
        )
        # None means unbounded — use a large finite cap (context window size).
        # Ollama's -1 can hang, so use num_ctx as the practical max.
        max_tokens = int(max_tokens) if max_tokens else self.num_ctx or 32768
        temperature = float(kwargs.get("temperature") or 0.7)

        # Build messages in Ollama chat format
        ollama_msgs = []
        if system:
            ollama_msgs.append({"role": "system", "content": system})
        for msg in messages:
            content = getattr(msg, "content", None)
            if content is None or not isinstance(content, str):
                continue
            sender = getattr(msg, "sender", None)
            role = getattr(sender, "role", "user") if sender else "user"
            if role == "assistant":
                ollama_msgs.append({"role": "assistant", "content": content})
            elif role == "system":
                ollama_msgs.append({"role": "system", "content": content})
            else:
                ollama_msgs.append({"role": "user", "content": content})

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        if self.num_ctx > 0:
            options["num_ctx"] = self.num_ctx
        payload = json.dumps({
            "model": self.model,
            "messages": ollama_msgs,
            "stream": False,
            "think": self.think,
            "options": options,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=None) as resp:
            data = json.loads(resp.read())

        msg = data.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        # Prepend thinking to content so extract_final_answer can find answers
        # that may be in either field, and the trace preserves both.
        full_content = content
        if thinking:
            full_content = f"<thinking>{thinking}</thinking>\n\n{content}"
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        done_reason = data.get("done_reason", "stop")
        finish = "length" if done_reason == "length" else "stop"

        return LLMResponse(
            content=full_content,
            meta={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_tokens_cost_nanodollars": 0,
                "output_tokens_cost_nanodollars": 0,
                "finish_reason": finish,
                "thinking": thinking,
            },
        )
