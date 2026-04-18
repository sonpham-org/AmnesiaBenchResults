# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %% [markdown]
# # AmnesiaBench v1 — Kaggle Benchmark
#
# **How much context does a model actually need to solve a competition-math
# problem reliably?**
#
# This notebook runs the full AmnesiaBench pipeline against `kbench.llm`:
#
# 1. **Unbounded** — solve with no budget cap (3 trials, average successful tokens)
# 2. **Predict** — model self-assesses: `{attempt, N}`
# 3. **Sweep (no-compact)** — halving search for `n_reliable_no_compact`
# 4. **Sweep (compact)** — same, with 50% compaction trigger → `n_reliable_compact`
#
# Then it computes the 5 metrics from plan.md §8:
# - `composite_context_efficiency_score` ← **leaderboard primary**
# - `composite_window_prediction_score`
# - `composite_success_prediction_score`
# - `accuracy_score`
# - `average_cost_per_token_nanodollars`
#
# Reference spec: https://github.com/sonpham-org/AmnesiaBench/ — `plan.md`

# %% [markdown]
# ## Setup

# %%
!pip install -q 'protobuf>=5.29.6' kaggle-benchmarks tiktoken 2>&1 | tail -3

# %%
import json
import math
import sys
import time
from pathlib import Path

import kaggle_benchmarks as kbench

# ── Load amnesia_kaggle package from the attached dataset ────────────────
DS_ROOT = Path("/kaggle/input/datasets/sonphamorg/amnesia-kaggle-v1")
if not DS_ROOT.exists():
    # Fallback paths for local dev
    for candidate in [Path.cwd() / "amnesia_kaggle", Path.cwd().parent]:
        if (candidate / "data" if candidate.name == "amnesia_kaggle" else candidate / "amnesia_kaggle" / "data").exists():
            DS_ROOT = candidate if candidate.name != "amnesia_kaggle" else candidate.parent
            break

# Copy package files into /kaggle/working so imports work
pkg = Path("/kaggle/working/amnesia_kaggle")
pkg.mkdir(exist_ok=True)
src_dir = DS_ROOT if DS_ROOT.name == "amnesia_kaggle" else DS_ROOT
for name in ["__init__.py", "prompts.py", "parsers.py", "log_search.py",
             "halving_search.py", "scoring.py", "harness.py"]:
    src = src_dir / name
    if src.exists():
        (pkg / name).write_bytes(src.read_bytes())
if str(pkg.parent) not in sys.path:
    sys.path.insert(0, str(pkg.parent))

from amnesia_kaggle.harness import (  # noqa: E402
    run_unbounded, run_prediction, run_trial_no_compact, run_trial_compact,
)
from amnesia_kaggle.halving_search import halving_search  # noqa: E402
from amnesia_kaggle.scoring import compute_scores  # noqa: E402

# %% [markdown]
# ## Load problems and frozen baselines

# %%
_data_dir = DS_ROOT / "data" if (DS_ROOT / "data").exists() else pkg / "data"
problems = json.loads((_data_dir / "problems.json").read_text())
_raw_baselines = json.loads((_data_dir / "baselines.json").read_text())
BASELINES = {k: v for k, v in _raw_baselines.items() if not k.startswith("__")}

print(f"Loaded {len(problems)} problems and {len(BASELINES)} baselines")
for p in problems:
    print(f"  {p['problem_id']} → gt={p['ground_truth']}")

# %% [markdown]
# ## Define the benchmark task

# %%
CTX_WINDOW = 8192  # model context window for halving search


@kbench.task(
    name="AmnesiaBench v1",
    description=(
        "How much context does a model actually need to solve competition math "
        "reliably? Runs unbounded → predict → halving sweep (no-compact + compact) "
        "across 10 AIMO problems. Returns composite_context_efficiency_score as the "
        "leaderboard primary. Spec: github.com/sonpham-org/AmnesiaBench"
    ),
)
def amnesia_bench(llm) -> float:
    """Full AmnesiaBench pipeline. Returns composite_context_efficiency_score."""
    t0_global = time.time()
    all_results = []

    for prob in problems:
        pid = prob["problem_id"]
        gt = prob["ground_truth"]
        text = prob["problem_text"]
        print(f'\n{"="*60}')
        print(f"PROBLEM: {pid} (gt={gt})")
        print(f'{"="*60}')
        t0 = time.time()

        # ── Phase 1: Unbounded (3 trials) ──
        print("\n[1/4] Unbounded (3 trials)...")
        unb = run_unbounded(llm, text, gt, temperature=0.7)
        print(f"  solved={unb.solved}  n_while_unbounded={unb.n_while_unbounded}")
        for log in unb.trial_logs:
            print(f'    trial {log["trial_idx"]}: {log["finish_reason"]} '
                  f'tokens={log["tokens_used"]} {log["wall_time_s"]}s')

        # ── Phase 2: Prediction ──
        print("\n[2/4] Prediction...")
        pred = run_prediction(llm, text, temperature=0.7)
        print(f"  attempt={pred.attempt}  n_predicted={pred.n_predicted}")
        print(f"  raw: {pred.raw_response[:200]}")

        total_inp = unb.input_tokens + pred.input_tokens
        total_out = unb.output_tokens + pred.output_tokens
        total_cost = unb.cost_nanodollars + pred.cost_nanodollars

        # Token cap for compact trials: 2.5x unbounded tokens
        compact_token_cap = (
            int(2.5 * unb.n_while_unbounded)
            if unb.n_while_unbounded < math.inf else None
        )

        # Build trial functions for halving search
        def _make_trial_fns(llm_, text_, gt_, tcap):
            def _nc_fn(N, n_trials):
                return [run_trial_no_compact(llm_, text_, gt_, N=N, temperature=0.7)
                        for _ in range(n_trials)]
            def _c_fn(N, n_trials):
                return [run_trial_compact(llm_, text_, gt_, N=N, temperature=0.7,
                                          token_cap=tcap)
                        for _ in range(n_trials)]
            return _nc_fn, _c_fn

        nc_fn, c_fn = _make_trial_fns(llm, text, gt, compact_token_cap)

        # ── Phase 3: Halving sweep no-compact ──
        print(f"\n[3/4] Halving sweep no-compact (ctx={CTX_WINDOW})...")
        nr_nc, log_nc = halving_search(
            nc_fn, context_window=CTX_WINDOW,
            n_while_unbounded=unb.n_while_unbounded,
        )
        n_nc = f"{nr_nc:.0f}" if nr_nc < math.inf else "inf"
        print(f"  n_reliable_no_compact={n_nc}  steps={len(log_nc.entries)}")
        for e in log_nc.entries:
            print(f'    N={e["N"]:>6} {"PASS" if e["passed"] else "FAIL"}')

        # ── Phase 4: Halving sweep compact ──
        print(f"\n[4/4] Halving sweep compact (ctx={CTX_WINDOW})...")
        nr_c, log_c = halving_search(
            c_fn, context_window=CTX_WINDOW,
            n_while_unbounded=unb.n_while_unbounded,
        )
        n_c = f"{nr_c:.0f}" if nr_c < math.inf else "inf"
        print(f"  n_reliable_compact={n_c}  steps={len(log_c.entries)}")
        for e in log_c.entries:
            print(f'    N={e["N"]:>6} {"PASS" if e["passed"] else "FAIL"}')

        n_reliable = min(nr_nc, nr_c)
        elapsed = time.time() - t0
        print(f"\n  n_reliable = min({n_nc}, {n_c}) = "
              f"{n_reliable if n_reliable < math.inf else 'inf'}")
        print(f"  elapsed: {elapsed:.1f}s")

        all_results.append({
            "problem_id": pid,
            "n_while_unbounded": unb.n_while_unbounded,
            "n_reliable_no_compact": nr_nc,
            "n_reliable_compact": nr_c,
            "n_reliable": n_reliable,
            "n_predicted": pred.n_predicted,
            "attempt": pred.attempt,
            "unbounded_solved": unb.solved,
            "input_tokens": total_inp,
            "output_tokens": total_out,
            "cost_nanodollars": total_cost,
        })

    # ── Scoring ──
    scores = compute_scores(all_results, BASELINES, model_ctx_window=CTX_WINDOW)

    print(f'\n{"="*60}')
    print("SCORING — all 5 metrics from plan.md §8")
    print(f'{"="*60}')
    print(f"composite_context_efficiency_score = {scores['composite_context_efficiency_score']:.4f}")
    print(f"composite_window_prediction_score  = {scores['composite_window_prediction_score']:.4f}")
    print(f"composite_success_prediction_score = {scores['composite_success_prediction_score']:.4f}")
    print(f"accuracy_score                     = {scores['accuracy_score']:.4f}")
    print(f"average_cost_per_token_nanodollars = {scores['average_cost_per_token_nanodollars']:.2f}")
    print(f"\nTotal wall time: {time.time()-t0_global:.1f}s")

    # Surface all 5 metrics on the leaderboard detail page
    kbench.assertions.assert_true(
        scores["n_problems_scored"] > 0,
        expectation=f"Scored {scores['n_problems_scored']} / {len(problems)} problems.",
    )
    kbench.assertions.assert_true(
        scores["accuracy_score"] >= 0,
        expectation=f"accuracy = {scores['accuracy_score']:.4f}",
    )
    kbench.assertions.assert_true(
        scores["composite_context_efficiency_score"] >= 0,
        expectation=(
            f"composite_context_efficiency_score = "
            f"{scores['composite_context_efficiency_score']:.4f}"
        ),
    )
    kbench.assertions.assert_true(
        scores["composite_window_prediction_score"] >= 0,
        expectation=(
            f"composite_window_prediction_score = "
            f"{scores['composite_window_prediction_score']:.4f}"
        ),
    )
    kbench.assertions.assert_true(
        scores["composite_success_prediction_score"] >= 0,
        expectation=(
            f"composite_success_prediction_score = "
            f"{scores['composite_success_prediction_score']:.4f}"
        ),
    )
    kbench.assertions.assert_true(
        scores["average_cost_per_token_nanodollars"] >= 0,
        expectation=(
            f"avg cost per token = "
            f"{scores['average_cost_per_token_nanodollars']:.2f} nanodollars"
        ),
    )

    # Per-problem detail
    for r in all_results:
        nr = r.get("n_reliable", math.inf)
        np_ = r.get("n_predicted", math.inf)
        print(f"  {r['problem_id']:30s}  n_unbounded={r['n_while_unbounded']!s:>8s}  "
              f"n_reliable={nr!s:>8s}  n_predicted={np_!s:>8s}  attempt={r['attempt']}")

    return scores["composite_context_efficiency_score"]

# %% [markdown]
# ## Run against the default model

# %%
amnesia_bench.run(kbench.llm)

# %% [markdown]
# ## Publish to the leaderboard

# %%
# %choose amnesia_bench
