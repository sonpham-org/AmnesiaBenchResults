# amnesia_kaggle — AmnesiaBench v1 as a Kaggle Benchmark

Port of [sonpham-org/AmnesiaBench](https://github.com/sonpham-org/AmnesiaBench) to the [kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks) SDK, following Scott's [canonical `plan.md`](../amnesia_bench/EXPERIMENT_PLAN.md).

**The question:** how much context does a model actually need to solve a competition-math problem reliably?

**The answer:** a composite score per model, published to Kaggle with cost-per-token surfaced on the x-axis (ARC-Prize-style chart).

## Files

```
amnesia_kaggle/
├── prompts.py                  Canonical prompts from plan.md §1-7 (verbatim)
├── parsers.py                  {final_answer: "..."}, <compact>, {attempt, N}
├── log_search.py               Log-scale nested binary search (outer + inner)
├── scoring.py                  The 5 metrics from plan.md §8
├── harness.py                  LLM-calling logic: unbounded, predict, trial, find_n_reliable
├── amnesia_bench_notebook.py   Kaggle notebook in jupytext percent format
├── run_local.py                Local dev smoke runner
├── data/
│   ├── problems.json           10 AIMO Hard Math problems (Phase 1 set)
│   └── baselines.json          Frozen n_reliable per problem (PLACEHOLDER — see below)
└── tests/
    ├── test_parsers.py         35 tests for output parsing
    ├── test_log_search.py      15 tests for log-scale convergence
    ├── test_scoring.py         23 tests for the 5 metrics
    └── test_harness_compact.py 17 tests for compaction control flow (uses mock LLM)
```

## Running the tests

```bash
cd /home/son/GitHub/autoresearch-arena
python -m pytest amnesia_kaggle/tests/ -v
# Expected: 98 passed in ~0.05s
```

All tests pass without any LLM API — the harness tests use a `FakeLLM` stub via monkeypatched `kaggle_benchmarks` module tree.

## Running locally against Kaggle Model Proxy

Follow [`local_development.md`](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/local_development.md):

```bash
# 1. Clone kaggle-benchmarks next to autoresearch-arena
cd /home/son/GitHub
git clone https://github.com/Kaggle/kaggle-benchmarks.git
cd kaggle-benchmarks
uv venv && source .venv/bin/activate
uv pip install -e .

# 2. Add .env with your proxy credentials
cat > .env <<EOF
MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi
MODEL_PROXY_API_KEY={your_token}
LLM_DEFAULT=google/gemini-2.5-flash
LLM_DEFAULT_EVAL=google/gemini-2.5-pro
LLMS_AVAILABLE=anthropic/claude-sonnet-4,google/gemini-2.5-flash,meta/llama-3.1-70b,google/gemini-2.5-pro
PYTHONPATH=src
EOF

# 3. Run the smoke test (1 problem, full pipeline) — ~15-20 min
cd /home/son/GitHub/autoresearch-arena/amnesia_kaggle
python run_local.py --problem digit_sum_ten

# Or just the unbounded phase (~1 min) for a quick check
python run_local.py --problem digit_sum_ten --phase unbounded
```

## Baseline generation (REQUIRED before first Kaggle publish)

The `data/baselines.json` file currently ships **placeholder values**. Before publishing to Kaggle, run the full pipeline against a reference model and write real baselines:

```bash
cd amnesia_kaggle
python run_local.py --all --model google/gemini-2.5-pro \
    --out data/baselines_gemini25pro.json
# Then extract n_reliable from each problem entry into a flat dict
# and copy into data/baselines.json, replacing the placeholders.
```

Document the reference model, date, and commit SHA in the `__reference_model__`, `__generated_at__`, and `__version__` keys. Baselines are **frozen per benchmark version** — bump `@kbench.task(version=N)` if you regenerate them.

**Reference model candidates:**
- `google/gemini-2.5-pro` (default — 1M context, strong at competition math)
- `anthropic/claude-sonnet-4` (alternative)
- Scott's plan.md says "best performing model on each problem" — pick whichever has the smallest `n_reliable` on most problems.

## Publishing to Kaggle

1. **Convert the notebook** from `.py` percent format to `.ipynb`:
   ```bash
   pip install jupytext
   jupytext --to ipynb amnesia_bench_notebook.py
   ```
2. **Upload the `amnesia_kaggle/` directory** as a Kaggle Dataset named e.g. `amnesia-kaggle-v1` (so the notebook can load `data/problems.json` and `data/baselines.json`).
3. Navigate to https://www.kaggle.com/benchmarks/tasks/new → this creates a new notebook with `kaggle-benchmarks` pre-installed.
4. Attach your `amnesia-kaggle-v1` dataset to the notebook.
5. Paste in the cells from `amnesia_bench_notebook.ipynb`.
6. **Save Version** — runs once in the Kaggle sandbox (~60-120 min).
7. Once the run completes and the leaderboard entry appears, use **"Evaluate More Models"** to add `anthropic/claude-sonnet-4`, `meta/llama-3.1-70b`, `google/gemini-2.5-flash`, etc.

## Pipeline summary (from `plan.md`)

### Phase 1 — Unbounded (3 trials, no `max_tokens` cap)

Solve with unlimited output. Take the mean tokens across successful trials. If ≥2 of 3 fail → `n_while_unbounded = inf`. If failures were due to context-exceeded, still run the compaction sweep. If failures weren't due to context, skip further testing.

### Phase 2 — Prediction (1 call, 300 output tokens)

Ask the model to self-assess: `{attempt: "True|False", N: "<int>|0"}`. **Advisory only** — the sweep runs regardless.

### Phase 3 & 4 — Sweeps (no-compact + compact, log-scale binary search)

```
Outer: 1 trial per N, log-scale midpoint, stop at (hi - lo) < 5% * mid
Inner: 3 trials per N, pass iff ≥2/3 succeed, stop at (hi - lo) ≤ 1
```

For the compact sweep, each trial is a 3-turn linear sequence:
1. **Thinking turn** — respond to eval prompt. If `{final_answer}` → success. If `finish=length` → fail. If context ≥50% → compact. Otherwise → `no_answer_no_compact` fail.
2. **Compaction turn** — send compaction prompt, respond. If `{final_answer}` → success (even during compaction). If `<compact>` tag → reset. Otherwise → `compact_parse_fail`.
3. **Post-reset turn** — fresh chat with `[eval_prompt + POST_COMPACTION_PROMPT(summary)]`. If `{final_answer}` → success. If reset prompt alone ≥ N → `compaction_insufficient`.

Cost cap: abort any trial that exceeds 3× the unbounded cost.

### Scoring (plan.md §8)

```
ctx_eff     = mean( baseline / n_reliable )                           primary (leaderboard y-axis)
window_pred = mean( ratio if ratio<=1 else (1/ratio)^2 )              n_reliable / n_predicted
suc_pred    = mean( 1.0 TP / 0.0 FP / 0.8 FN / 1.0 TN )
accuracy    = (n_unbounded<inf OR n_reliable<inf) / eligible_problems
cost/tok    = total_cost_nanodollars / total_tokens                   leaderboard x-axis
```

The leaderboard primary is `composite_context_efficiency_score`. Other metrics surface via `kbench.assertions.assert_true(...)` with descriptive `expectation=` strings on the task detail page. Cost is rendered natively by Kaggle from `chat.usage.input_tokens_cost_nanodollars + output_tokens_cost_nanodollars`.

## Scope & non-goals

This is **v1**, Phase 1 only:
- ✅ 10 AIMO Hard Math problems
- ✅ Unbounded + predict + no-compact sweep + compact sweep
- ✅ 5 metrics per plan.md §8
- ❌ ARC visual problems (v2)
- ❌ OpenRouter pricing lookup (v2 — for now we use Kaggle's `chat.usage`)
- ❌ Multi-compaction loops (v1 does 1 compaction per trial, 3-turn linear)
- ❌ Thinking-token handling (submitter-controlled per plan.md)
