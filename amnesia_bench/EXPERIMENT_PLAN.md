# AmnesiaBench Experiment Plan — 30-March-2026

**Authors:** Son Pham, Scott Weiss, Mark (82deutschmark)
**Operator:** Bubba (Claude Sonnet 4.6)
**Hardware:** Strix Halo (AMD Ryzen AI MAX+ 395, 96GB shared, 262K ctx) + Mac Mini M4 Pro

---

## 1. Research Question

How much context does a model actually need to solve a problem reliably?
And can the model accurately predict its own context requirements?

## 2. Problem Sets

### 2a. AIMO Hard Math (10 problems) — ALL models

| # | Problem ID | Source | GPT-OSS-120B Pass Rate | Avg Tokens |
|---|------------|--------|----------------------|------------|
| 1 | `aimo3_hard_271f3da5` | AIMO3 | 0.50 | 3,162 |
| 2 | `aimo3_hard_6111f603` | AIMO3 | 0.44 | 3,226 |
| 3 | `aimo3_hard_e2001d21` | AIMO3 | 0.38 | 1,841 |
| 4 | `crt_three_congruences` | Competition Math | 0.44 | 2,500 |
| 5 | `digit_sum_ten` | Competition Math | 0.56 | 1,800 |
| 6 | `handshakes_10` | Competition Math | 0.63 | 800 |
| 7 | `milly_grid_walk` | Competition Math | 1.00 | 500 |
| 8 | `modular_power_2_1000` | Competition Math | 0.50 | 1,500 |
| 9 | `sum_divisors_720` | Competition Math | 0.50 | 1,600 |
| 10 | `essay_questions` | Custom | 1.00 | 300 |

### 2b. ARC2 Barely-Solved (4 problems) — COMMERCIAL models only

Source: arc-explainer difficulty data (the only ARC2 eval problems with <100% solve rate)

| # | Problem ID | Score | Tokens Used | Attempts | Grid Size |
|---|------------|-------|-------------|----------|-----------|
| 1 | `16de56c4` | 0.50 | 2,271 | 4 | 12×9 → 9×21 |
| 2 | `65b59efc` | 0.50 | 4,095 | 4 | TBD |
| 3 | `8e5c0c38` | 0.50 | 5,811 | 4 | TBD |
| 4 | `1ae2feb7` | 0.67 | 2,640 | 6 | TBD |

### 2c. ARC Unsolved (15 problems) — COMMERCIAL models only

| # | Problem ID | Source |
|---|------------|--------|
| 1 | `50f325b5` | ARC1 Eval (last unsolved) |
| 2 | `62593bfd` | ARC2 Eval |
| 3 | `2b83f449` | ARC2 Eval |
| 4 | `88bcf3b4` | ARC2 Eval |
| 5 | `8b7bacbf` | ARC2 Eval |
| 6 | `faa9f03d` | ARC2 Eval |
| 7 | `269e22fb` | ARC2 Eval |
| 8 | `4e34c42c` | ARC2 Eval |
| 9 | `21897d95` | ARC2 Eval |
| 10 | `abc82100` | ARC2 Eval |
| 11 | `9bbf930d` | ARC2 Eval |
| 12 | `a32d8b75` | ARC2 Eval |
| 13 | `e12f9a14` | ARC2 Eval |
| 14 | `13e47133` | ARC2 Eval |
| 15 | `88e364bc` | ARC2 Eval |

**Total: 29 problems**

---

## 3. Models (10)

### Frontier (3) — Commercial APIs

| # | Model | Context Max | Access | Cost (in/out per M) |
|---|-------|-------------|--------|---------------------|
| 1 | Claude Opus 4.6 | 1,000,000 | Anthropic OAuth (subscription) | $5 / $25 ref |
| 2 | Claude Sonnet 4.6 | 1,000,000 | Anthropic OAuth (subscription) | $3 / $15 ref |
| 3 | Gemini 3.1 Pro Preview | 1,048,576 | Gemini API key | $2 / $12 |

### Open Source — Local on Strix Halo (4)

| # | Model | Params | Active | Context Max | File |
|---|-------|--------|--------|-------------|------|
| 4 | Qwen3.5-35B-A3B | 35B MoE | 3.5B | 262,144 | Qwen3.5-35B-A3B-Q4_K_M.gguf |
| 5 | DeepSeek-R1-Distill-Qwen-32B | 32B dense | 32B | 131,072 | DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf |
| 6 | Qwen3-32B | 32B dense | 32B | 131,072 | Qwen3-32B-Q4_K_M.gguf |
| 7 | Qwen3.5-9B | 9B dense | 9B | 262,144 | Qwen3.5-9B-Q4_K_M.gguf |

### Open Source — via OpenRouter API (3)

| # | Model | Context Max | Cost (in/out per M) |
|---|-------|-------------|---------------------|
| 8 | Llama 4 Maverick | 1,048,576 | $0.15 / $0.60 |
| 9 | Llama 4 Scout | 327,680 | $0.08 / $0.30 |
| 10 | Qwen3.5-122B-A10B | 262,144 | $0.26 / $2.08 |

---

## 4. Methodology

### 4a. Per (model, problem) pair — two jobs:

**Prediction Job:**
- Model sees the problem but NOT the context limit (N is literal "N tokens")
- Model predicts: attempt (True/False) and N (predicted minimum tokens needed)
- 300 token output limit
- One call per (model, problem)

**Evaluation Job:**
- Unbounded test: model solves with full context, no restriction → measure actual tokens used = `n_while_unbounded`
- Binary search WITHOUT compaction → `n_reliable_no_compact`
  - Outer: 1 trial per N, stop at 5% threshold
  - Inner: 3 trials per N, 2/3 pass required (66.7%), 1-token precision
- Binary search WITH 50% compaction → `n_reliable_compact`
  - Same nested search, but at 50% of N tokens used → inject compaction prompt
  - Model writes `<compact>summary</compact>`, conversation resets
  - If compaction can't bring tokens below 50% → FAIL at this N
  - No limit on number of compactions

### 4b. ARC-specific evaluation
- Same binary search methodology
- Model gets 2 answer attempts per trial
- Correct if either attempt matches expected grid exactly
- Grids formatted as text (space-separated integers, one row per line)

### 4c. Data captured per trial
- Full conversation trace (all messages, all turns)
- Per-turn token counts (input, output, thinking)
- Timing metrics (prefill t/s, decode t/s, KV cache hits — local models)
- Compaction summaries (what the model chose to preserve)
- Cost in USD
- Wall time

---

## 5. Scoring Formula

```
final_score = (ctx_eff)^2.0 × (eff_pred)^0.5 × (suc_pred)^1.5 × (accuracy)^1.0 / (cost_per_token)^1.0
```

Where:
- `ctx_eff` = mean(baseline_n_reliable / model_n_reliable) — context efficiency vs best model
- `eff_pred` = mean(n_reliable / n_predicted) capped at 1.0, 0.0 if overconfident — prediction calibration
- `suc_pred` = mean(success_prediction_score) — 1.0 correct, 0.0 false positive, 0.8 false negative
- `accuracy` = problems_solved_unbounded / eligible_problems
- `cost_per_token` = weighted average USD per token

---

## 6. Pair Counts & Time Estimates

| Problem Set | Models | Pairs | Est. Time per Pair | Total Time |
|-------------|--------|-------|--------------------|------------|
| AIMO (10) × Local (4) | 4 | 40 | ~35 min | ~23 hrs |
| AIMO (10) × Commercial (3) | 3 | 30 | ~10 min | ~5 hrs |
| AIMO (10) × OpenRouter (3) | 3 | 30 | ~15 min | ~7.5 hrs |
| ARC (19) × Commercial (3) | 3 | 57 | ~15 min | ~14 hrs |
| **Total** | | **157** | | |

**Wall time estimate:** ~3 days (local serial, API parallel)
**Budget estimate:** ~$39 API cost (well within $500)

---

## 7. Execution Order

1. ✅ Smoke test — Qwen3.5 / digit_sum_ten (completed)
2. Build ARC evaluation module (in progress)
3. Create all problem JSON files (ARC + remaining AIMO)
4. Create models.json with all 10 models
5. Run local models first (cheapest, slowest — start ASAP)
6. Run commercial models in parallel
7. Run scoring
8. Generate report with traces

---

## 8. Deliverables

- Per-model scoring table (all 5 components + final score)
- n_reliable curves: how context efficiency degrades across models
- Prediction calibration analysis: which models know their own limits?
- Compaction analysis: does compaction help? By how much? For which problem types?
- Full conversation traces for every trial (traces/ directory)
- ARC trace analysis: how do frontier models reason about visual patterns under context pressure?

---

## 9. Banned Models

- **GPT-5.4-Pro** (`openai/gpt-5.4-pro`) — $30/$180 per M tokens. Explicitly banned by Mark.
- Any OpenAI model until OAuth is sorted out.

## 10. Infrastructure

- **Strix Halo:** llama-server with `--ctx-size 262144 --parallel 1` for local models
- **Anthropic:** OAuth subscription (no per-token cost to us)
- **Gemini:** API key (GEMINI_API_KEY on Strix Halo)
- **OpenRouter:** Paid key (OPENROUTER_API_KEY on Strix Halo)
- **Monitoring:** Cron job every 10 min for active runs
- **Resume:** Checkpoint files written after every binary search step
