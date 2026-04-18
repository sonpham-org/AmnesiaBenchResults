# Local LLM Benchmark Results — Strix Halo (96GB)

**Date:** 2026-03-22
**Hardware:** AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 96GB VRAM (LPDDR5X-8000, ~215 GB/s)
**Software:** ROCm 7.2, llama.cpp (build 5f4cdac38, ROCm+HIP, rocWMMA FATTN)

## Infrastructure Built
- **ROCm 7.2** installed (fixed repo pinning, purged 6.x conflicts)
- **llama.cpp** built from source with ROCm HIP + rocWMMA for gfx1151
- **vLLM abandoned** — fundamentally broken on gfx1151 (5 tok/s vs 57 tok/s llama.cpp, 11x slower)
- **local_worker.py** updated to use llama-server instead of vLLM
- **API secured** with `ARENA_LOCAL_KEY` + code dedup via hash + program.md fetch endpoint
- Python 3.12 venv with all dependencies

## Benchmark Results — Qwen 3.5

| Model | Quant | Size | pp512 tok/s | tg128 tok/s | Serves via llama-server? |
|-------|-------|------|-------------|-------------|--------------------------|
| **Qwen3.5-35B-A3B** | **Q4_K_M** | **20 GB** | **929** | **57.4** | **Yes** |
| Qwen3.5-9B | Q4_K_M | 5 GB | 564 | 34.9 | Yes |
| Qwen3.5-122B-A10B | Q4_K_M | 69 GB | 297 | 24.3 | No (KV cache OOM) |
| Qwen3.5-122B-A10B | Q3_K_M | 53 GB | 128 | 20.6 | Yes (with `--no-warmup`) |
| Qwen3.5-35B-A3B | Q3_K_M | 15 GB | 440 | 50.3 | Yes (but slower than Q4) |

## Optimal Config (35B-A3B Q4_K_M)
```bash
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -fa 0 -t 4 -c 8192 --no-mmap
```
- **No HIPBLASLT** (hurts pp by 47%)
- **Flash attention OFF** (hurts pp by 11%)
- **4 threads** (minimal CPU involvement, GPU-bound)
- **Speculative decoding** with 0.8B draft: 64 tok/s (+11%)

## Tuning Matrix (35B-A3B Q4_K_M)

| Setting | pp512 | tg128 | Verdict |
|---------|-------|-------|---------|
| **Optimal: t=4, FA off, no HIPBLASLT** | **929** | **57.4** | **Best** |
| HIPBLASLT=1 | 496 | 57.7 | -47% pp, avoid |
| Flash attention on | 830 | 57.5 | -11% pp, avoid |
| KV cache Q4/Q8 | 495 | 56.5 | Worse |
| UBatch 128 | 502 | 57 | -46% pp |
| UBatch 512 (default) | 920 | 57 | Good |
| Threads 4-32 | ~900 | ~57 | Minimal diff |
| Batch 512-8192 | ~890 | ~57 | No effect |

## Context Length Impact

| Prompt | tok/s |
|--------|-------|
| pp128 | 491 |
| pp512 | 908 |
| pp1024 | 886 |
| pp2048 | 849 |
| pp4096 | 808 |

## Speculative Decoding

| Draft Model | Effective tok/s | vs Baseline |
|-------------|----------------|-------------|
| None (baseline) | 57.4 | — |
| **0.8B Q4_K_M** | **63.9** | **+11%** |
| 2B Q4_K_M | 51.7 | -10% (draft too slow) |

## Code Quality (35B-A3B Think Mode)
- 47.9 raw tok/s, generates ~8K tokens per request (thinking + code)
- Produces 53-line BFS snake agents with edge case handling
- 6/6 quality tests passed (simple_agent, tool_calling, code_analysis)
- ~2.7 minutes per agent evolution cycle
- Thinking mode required — `/no_think` doesn't work with llama-server

## vLLM vs llama.cpp Comparison

| Engine | pp512 tok/s | tg128 tok/s | Notes |
|--------|-------------|-------------|-------|
| **llama.cpp** | **929** | **57.4** | Winner |
| vLLM 0.18.0+rocm700 | ~500 | 5.2 | 92% time in hipMemcpyWithStream |

vLLM is fundamentally broken on gfx1151 due to:
- Missing AITER optimized kernels for RDNA
- HIP Graph capture crashes → forced `--enforce-eager`
- 92% of decode time spent on memory copies

## What's Left
- 122B via llama-cli (no server) could work but needs more testing
- Snake evolution run hasn't started yet
- `ARENA_LOCAL_KEY` env var needs to be set on the server
- All benchmarks committed and pushed to remote

## Key Learnings
- **vLLM is broken on RDNA/gfx1151** — use llama.cpp
- **Q4_K_M is the sweet spot** — Q3 dequant overhead kills performance
- **tg is memory-bandwidth locked at ~57 tok/s** on LPDDR5X-8000
- **MoE models (3B active) are 2x faster** than dense models (9B) on tg
- **35B-A3B is the winner** — fastest, fits easily, reportedly matches 122B quality on code
- **`--no-mmap` required** for llama-server on Strix Halo (mmap loading is extremely slow)
- **`--no-mmap` OOMs for 122B** since system RAM is only 30GB — only works for models < 30GB
- **GTT expanded** from 15GB to 128GB (`ttm.pages_limit=33554432`) for better memory management
