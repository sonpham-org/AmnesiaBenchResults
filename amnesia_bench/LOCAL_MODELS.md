# Local LLM Agent Evolution — Setup & Usage

## Hardware
- **GPU**: AMD Strix Halo (Radeon 8060S) — 96GB unified VRAM
- **CPU**: AMD Ryzen AI MAX+ 395 — 32 cores
- **ROCm**: 7.2 (up to 5x faster than ROCm 6.x on Strix Halo)

## Quick Start

```bash
# 1. Install ROCm 7.2 + vLLM (one-time, needs sudo)
chmod +x setup_rocm.sh && sudo ./setup_rocm.sh

# 2. Activate venv
source .venv/bin/activate

# 3. List available models
python3 local_worker.py --list-models

# 4. Benchmark ALL models (comparison report)
python3 benchmark_models.py --skip-large

# 5. Benchmark specific models
python3 benchmark_models.py --models qwen3-235b,kimi-k2-70b,minimax-m1-80b

# 6. Compare existing benchmark results
python3 benchmark_models.py --compare

# 7. Run evolution (snake game, 10 gens)
python3 local_worker.py --model qwen3-235b --game snake --generations 10

# 8. Run + auto-upload to remote server
python3 local_worker.py --model qwen3-235b --upload
```

## Supported Models

| Model ID | Type | Active | Total | VRAM (FP16) | Notes |
|----------|------|--------|-------|-------------|-------|
| `qwen3-235b` | MoE | 22B | 235B | ~48GB | Best overall. 128 experts |
| `qwen3-30b` | MoE | 3B | 30B | ~16GB | Fastest. Good for rapid iteration |
| `qwen3-32b` | Dense | 32B | 32B | ~64GB | Strong reasoning |
| `qwen3-14b` | Dense | 14B | 14B | ~28GB | Good speed/quality balance |
| `qwen3-8b` | Dense | 8B | 8B | ~16GB | Fast. Pipeline testing |
| `minimax-m1-80b` | MoE | 45B | 80B | ~90GB | Strong reasoning, tight 96GB fit |
| `kimi-k2-70b` | MoE | 12B | 70B | ~35GB | Moonshot, strong code gen |
| `deepseek-v3-671b` | MoE | 37B | 671B | Needs offload | Too large without quantization |

## Benchmarking

Every model is benchmarked on 3 arena-relevant tasks before use:

1. **simple_agent** — Can it write a valid `get_move(state)` with BFS?
2. **tool_calling** — Does it correctly call `create_agent()` tool?
3. **code_analysis** — Can it analyze and improve existing snake agents?

Results are logged to `benchmarks/` for comparison:
```
benchmarks/
  bench_Qwen_Qwen3-235B-A22B_20260321_143052.json
  bench_moonshotai_Kimi-K2-Instruct_20260321_144512.json
  comparison_20260321_150000.json     # side-by-side comparison
```

### Running Benchmarks

```bash
# Benchmark all models (takes a while — each loads into GPU)
python3 benchmark_models.py

# Skip models too large for 96GB
python3 benchmark_models.py --skip-large

# Benchmark specific models with 5 runs each
python3 benchmark_models.py --models qwen3-235b,kimi-k2-70b --runs 5

# View latest comparison report
python3 benchmark_models.py --compare
```

### Sample Comparison Output

```
  MODEL BENCHMARK COMPARISON — 2026-03-21 15:00:00
  GPU: AMD Strix Halo Radeon 8060S (96GB VRAM) | ROCm: 7.2
  ========================================================================================
  Model                Type   Active   Tok/s    Quality    Est Evo    VRAM
  ----------------------------------------------------------------------------------------
  qwen3-30b            MoE    3B       85.2     9/9        ~45s       ~16GB FP16
  qwen3-8b             Dense  8B       62.1     7/9        ~65s       ~16GB FP16
  kimi-k2-70b          MoE    12B      38.5     8/9        ~110s      ~35GB FP16
  qwen3-235b           MoE    22B      24.3     9/9        ~180s      ~48GB FP16
  qwen3-32b            Dense  32B      18.7     9/9        ~220s      ~64GB FP16
  minimax-m1-80b       MoE    45B      12.1     8/9        ~350s      ~90GB FP16

  Recommendations:
    Fastest:        qwen3-30b (85.2 tok/s)
    Best quality:   qwen3-235b (9/9 passed)
    Best overall:   qwen3-235b
```

## Architecture

```
local_worker.py          ← Evolution worker (uses vLLM)
benchmark_models.py      ← Multi-model benchmark + comparison
setup_rocm.sh            ← ROCm 7.2 + vLLM installer
local_agents/            ← File-based agent store
evolution_logs/          ← Full LLM conversation logs
benchmarks/              ← Benchmark results + comparisons
```

## Agent Naming

Local agents get `local/` prefix: `local/gen1_flood_fill`, `local/gen5_aggro_cutter`

On the leaderboard: muted yellow `LOCAL` badge (#C8A84E).

## Cost Comparison

| Provider | Model | Cost per evolution cycle |
|----------|-------|-------------------------|
| Anthropic | Claude Haiku 4.5 | ~$0.02 |
| Google | Gemini 3.1 Flash Lite | ~$0.005 |
| **Local** | **Qwen3-235B-A22B** | **~$0.001 (electricity)** |
| **Local** | **Qwen3-30B-A3B** | **~$0.0003 (electricity)** |

## ROCm 7.2 Performance

ROCm 7 delivers massive improvements on Strix Halo vs ROCm 6:
- **Qwen2-72B**: 3.4x faster
- **DeepSeek R1**: 3.8x faster
- **SDXL**: 2.6x faster
- **Flux**: 5.2x faster
- **Overall LLM inference**: up to 5x uplift

Note: Kernel 6.18.4+ recommended for gfx1151 stability.

## Troubleshooting

**vLLM won't start:** Check `rocminfo | grep GPU`. Try `--model qwen3-8b` first.

**Kernel too old:** gfx1151 needs 6.18.4+. Run `uname -r` — if < 6.18, consider upgrading.

**Model too large:** Use `--skip-large` with benchmark_models.py, or try Q4 quantized variants.
