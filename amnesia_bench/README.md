# AmnesiaBench

**How much context does a model *actually* need to solve a math problem?**

Binary-searches (on log scale) for the minimum context window at which an LLM can
solve competition-math problems at a 20% success rate, across 4 configurations:

| Config | Tools | On budget exceeded |
|--------|-------|--------------------|
| No-TIR + Hard Cutoff | none | Generation truncated, last `\boxed{}` extracted |
| TIR + Hard Cutoff | `python` | Generation truncated, last `\boxed{}` extracted |
| No-TIR + Compaction | `compact` | **HARD FAIL** — no answer |
| TIR + Compaction | `python` + `compact` | **HARD FAIL** — no answer |

## Key Design Decisions

### The model knows its budget — once
Every prompt includes `Your context window is {token_limit} tokens total`.
No ongoing token counter is provided. The model must estimate its own usage.
If it misjudges and runs out of context without compacting, that's a real failure
of self-awareness.

### Compaction is a tool call, not an injection
In compaction modes, the model has a `compact` tool. It must call it *before* running
out of tokens, or the run is a hard fail. The compact call itself costs tokens.

When `compact(summary=...)` is called:
1. Conversation resets to: `[system prompt] + [problem] + [summary]`
2. Token budget resets to a fresh `{token_limit}` window
3. Python sandbox state is **preserved** (variables survive across compactions)
4. Max 5 compactions per run

### Tool calling implementation
We use **text-based tool calling** (not the OpenAI tools API) for maximum
compatibility across llama.cpp versions and model architectures:

- **Python execution**: Model writes ` ```python ` code blocks. We detect them,
  execute code, and inject output as a user message.
- **Compact tool**: Model writes `<compact>summary text here</compact>`.
  We detect the tag, extract the summary, and reset the conversation.

Rationale: llama.cpp's native tool calling support varies by model chat template
and version. Text-based parsing is universally reliable and makes prompts
self-documenting. The model doesn't need special tool-call tokens — just a text
convention.

### Binary search on log scale
The search space is `[MIN_WINDOW, MAX_WINDOW]` in tokens.
Midpoints are **geometric means**: `mid = exp((ln(lo) + ln(hi)) / 2)`.
This gives equal weight to doubling vs. halving, which is more natural for
token budgets (the difference between 1K and 2K matters more than 31K vs 32K).

Convergence criterion: `hi / lo < 1.05` (within 5% on log scale).

### TIR (Tool-Integrated Reasoning)
When enabled, the model can write Python code blocks. Code is executed in an
in-process sandbox (`exec()` with a persistent namespace). Variables survive
across turns *and* across compactions.

Code output is injected as a user message and **counts toward the token budget**.

## Prompts

### System Prompt Templates

**Hard Cutoff — No-TIR:**
```
You are a mathematical problem solver.
Your context window is {token_limit} tokens total (this prompt + your output).
If you run out, generation stops and I take your last \boxed{} answer.
Plan your reasoning to fit. Give your final answer as \boxed{integer}.
```

**Hard Cutoff — TIR:**
```
You are a mathematical problem solver.
Your context window is {token_limit} tokens total (this prompt + your output + code outputs).
If you run out, generation stops and I take your last \boxed{} answer.
You can execute Python by writing ```python blocks. I will run them and show output.
Code output counts toward your token budget. Plan accordingly.
Give your final answer as \boxed{integer}.
```

**Compaction — No-TIR:**
```
You are a mathematical problem solver.
Your context window is {token_limit} tokens total. If you exceed it without compacting, you FAIL with score 0.

You have one tool: compact. To call it, write:
<compact>your summary here</compact>

When you call compact, the conversation resets to:
  [this system prompt] + [the problem] + [your summary]
You get a fresh {token_limit} budget, but the reset prompt eats into it.
The compact call itself costs tokens. You may compact at most 5 times.

Give your final answer as \boxed{integer}.
```

**Compaction — TIR:**
```
(same as Compaction — No-TIR, plus:)
You can also execute Python by writing ```python blocks. I will run them and show output.
Code output counts toward your token budget.
Python variables persist across compactions — your computed values survive.
```

### Post-Compaction Restart
```
User: {problem_text}

Your previous progress (from compact call):
---
{model's summary}
---
Continue solving. Give your final answer as \boxed{integer}.
```

## Problems

Three problems from GPT-OSS-120B curated traces (16 samples each), chosen for
low pass rates (hard but solvable):

| Problem | Source | Pass Rate (120B) | Correct Tokens (avg) | Topic |
|---------|--------|-------------------|----------------------|-------|
| `ab507a9f` — Tic-tac-toe probability | aimo3_hard | 1/16 (6%) | 3,345 | Probability/Combinatorics |
| `limo_25dbca34` — Square ABCD geometry | LIMO | 1/16 (6%) | 8,071 | Geometry |
| `46dd3688` — Floor of arccot integral | aimo3_hard | 4/16 (25%) | 7,532 | Analysis |

## Running

```bash
# 1. Start llama.cpp server with Qwen3.5-35B-A3B
/home/son/llama.cpp/build/bin/llama-server \
  --model /home/son/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 99 --ctx-size 65536 --parallel 1 --threads 16

# 2. Run experiment (one problem, all 4 configs)
python3 amnesia_bench.py --problem ab507a9f

# 3. Run everything
python3 amnesia_bench.py --all

# 4. Analyze results
python3 amnesia_bench.py --analyze
```

Results are saved to `results/` as JSON files, one per (problem, config) combination.

## Output Format

Each JSON result file:
```json
{
  "problem_id": "ab507a9f",
  "model": "Qwen3.5-35B-A3B-Q4_K_M",
  "config": {"tir": true, "compaction": true},
  "binary_search": [
    {
      "window": 8192,
      "trials": [
        {
          "trial_idx": 0,
          "success": true,
          "answer": 554,
          "correct_answer": 554,
          "total_tokens_used": 6847,
          "n_turns": 7,
          "n_compactions": 1,
          "n_code_calls": 3,
          "wall_time_s": 142.3,
          "error": null,
          "conversation": [...]
        }
      ],
      "pass_rate": 0.4,
      "passed": true
    }
  ],
  "minimum_window": 4096,
  "search_range_final": [3891, 4096]
}
```

## Hardware

Tested on:
- **CPU**: AMD Ryzen AI MAX+ 395
- **GPU**: Radeon 8060S (gfx1151), 103 GB unified VRAM
- **Backend**: llama.cpp (HIP/ROCm or Vulkan)
- **Model**: Qwen3.5-35B-A3B Q4_K_M (21 GB)
- **Speed**: ~50 tok/s single stream, ~85 tok/s aggregate with 8 parallel slots
