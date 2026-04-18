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
# # AmnesiaBench — 25 Problems, Full Benchmark + Traces
#
# Runs ALL 25 `op_*` problems through the full benchmark pipeline
# (unbounded -> predict -> sweep -> compact) as a proper `@kbench.task`.
# Per-problem logic is identical to the debug notebook — the only difference
# is the outer loop and aggregate scoring across all 25.
#
# Prints full traces inline (compaction injection, <compact> responses,
# resumption prompts) and saves traces + results to `/kaggle/working/`.
# Main score returned: `compaction_ratio = n_reliable / n_while_unbounded`
# (lower = better).

# %%
# !pip install -q 'protobuf>=5.29.6' kaggle-benchmarks tiktoken 2>&1 | tail -3

# %%
import json
import math
import sys
import time
from pathlib import Path

import kaggle_benchmarks as kbench

# Find the dataset root by searching /kaggle/input for one that has harness.py
DS_ROOT = None
for p in [
    Path("/kaggle/input/amnesiabench-harness-2026-04-13"),
    Path("/kaggle/input/amnesiabench-harness-2026-04-12"),
    Path("/kaggle/input/amnesia-bench-scott25-v2"),
    Path("/kaggle/input/amnesia-kaggle-v1"),
    Path("/kaggle/input/datasets/sonphamorg/amnesiabench-harness-2026-04-13"),
    Path("/kaggle/input/datasets/sonphamorg/amnesiabench-harness-2026-04-12"),
]:
    if p.exists() and (p / "harness.py").exists():
        DS_ROOT = p
        break
if DS_ROOT is None:
    # Last-ditch: glob anything in /kaggle/input that has harness.py
    for kaggle_in in Path("/kaggle/input").glob("*") if Path("/kaggle/input").exists() else []:
        if (kaggle_in / "harness.py").exists():
            DS_ROOT = kaggle_in
            break
if DS_ROOT is None:
    # Local dev fallback
    for candidate in [Path.cwd() / "amnesia_kaggle", Path.cwd().parent / "amnesia_kaggle", Path.cwd()]:
        if (candidate / "harness.py").exists():
            DS_ROOT = candidate
            break

print(f"Dataset root: {DS_ROOT}")
assert DS_ROOT is not None, "Could not find amnesia_kaggle dataset"

pkg = Path("/kaggle/working/amnesia_kaggle")
pkg.mkdir(exist_ok=True)
copied = []
for name in ["__init__.py", "prompts.py", "parsers.py", "log_search.py",
             "halving_search.py", "scoring.py", "harness.py", "model_info.py"]:
    src = DS_ROOT / name
    if src.exists():
        (pkg / name).write_bytes(src.read_bytes())
        copied.append(name)
    else:
        print(f"WARNING: {name} not found at {src}")
print(f"Copied {len(copied)} files: {copied}")

if str(pkg.parent) not in sys.path:
    sys.path.insert(0, str(pkg.parent))

from amnesia_kaggle.harness import (
    run_unbounded, run_prediction, run_trial_no_compact, run_trial_compact,
)
from amnesia_kaggle.halving_search import halving_search
from amnesia_kaggle.scoring import compute_scores
from amnesia_kaggle.prompts import (
    build_problem_message, build_prediction_prompt,
    COMPACTION_PROMPT, build_post_compaction_prompt,
)
from amnesia_kaggle.model_info import get_context_window

# %% [markdown]
# ## Probe: do Gemini Pro thinking tokens leak past `thinking_budget=0`?
#
# Monkey-patches `GoogleGenAI._get_usage_meta` to capture the full
# `usage_metadata` object, then makes three identical probe calls with
# different thinking configs. Prints `thoughts_token_count` alongside the
# visible `candidates_token_count` so we can see whether `thinking_budget=0`
# actually disables thinking, and whether `include_thoughts=True` exposes it.

# %%
try:
    from kaggle_benchmarks.actors.llms import GoogleGenAI, OpenAI
    from google.genai import types as _gtypes

    # Diagnose: what class is kbench.llm actually?
    print(f"kbench.llm class: {type(kbench.llm).__module__}.{type(kbench.llm).__name__}")
    print(f"kbench.llm MRO: {[c.__name__ for c in type(kbench.llm).__mro__]}")
    print(f"model attr: {getattr(kbench.llm, 'model', '?')}")
    print(f"attrs: {[a for a in dir(kbench.llm) if not a.startswith('_')]}")
    print()

    _captured_usage = []
    # Patch BOTH class hooks so whichever path is taken gets captured
    _orig_google = GoogleGenAI._get_usage_meta
    def _probe_g(self, usage):
        if usage is not None: _captured_usage.append(("google", usage))
        return _orig_google(self, usage)
    GoogleGenAI._get_usage_meta = _probe_g

    if hasattr(OpenAI, "_get_usage_meta"):
        _orig_openai = OpenAI._get_usage_meta
        def _probe_o(self, usage):
            if usage is not None: _captured_usage.append(("openai", usage))
            return _orig_openai(self, usage)
        OpenAI._get_usage_meta = _probe_o

    PROBE_PROMPT = ("What is the final value of N? Start N=10. For primes 2,3,5,7 "
                    "add to N if N%3!=0 else subtract. Answer with "
                    "{final_answer: \"VALUE\"}.")
    _is_gemini_class = isinstance(kbench.llm, GoogleGenAI)
    _is_openai_class = isinstance(kbench.llm, OpenAI)
    _model_name = str(getattr(kbench.llm, "model", "")).lower()
    _looks_gemini = "gemini" in _model_name
    print(f"isinstance GoogleGenAI: {_is_gemini_class}")
    print(f"isinstance OpenAI: {_is_openai_class}")
    print(f"model string looks Gemini: {_looks_gemini}")

    if _is_gemini_class:
        _configs = [
            ("thinking_budget=0 (should disable)", _gtypes.ThinkingConfig(thinking_budget=0)),
            ("include_thoughts=True (should expose)", _gtypes.ThinkingConfig(include_thoughts=True)),
            ("no thinking_config (default)", None),
        ]
        for _label, _cfg in _configs:
            _captured_usage.clear()
            with kbench.chats.new(f"probe_{_label[:10]}") as _c:
                kbench.actors.user.send(PROBE_PROMPT)
                _kw = {"temperature": 0.0, "max_output_tokens": 512}
                if _cfg is not None:
                    _kw["thinking_config"] = _cfg
                kbench.llm.respond(**_kw)
                _text = _c.messages[-1].content if _c.messages else ""
            _u = _captured_usage[-1] if _captured_usage else None
            print(f"\n--- {_label} ---")
            if _u is None:
                print("  (no usage captured)")
            else:
                for _k in ("prompt_token_count", "candidates_token_count",
                           "thoughts_token_count", "total_token_count"):
                    print(f"  {_k}: {getattr(_u, _k, 'N/A')}")
                print(f"  visible content length: {len(_text)} chars")
                print(f"  visible content preview: {_text[:200]!r}")

    # Also: one neutral call (no thinking_config at all) to see what the
    # default path does regardless of class.
    print("\n--- Neutral call (no extra kwargs) ---")
    _captured_usage.clear()
    with kbench.chats.new("probe_neutral") as _c:
        kbench.actors.user.send(PROBE_PROMPT)
        kbench.llm.respond(temperature=0.0, max_tokens=512)
        _text = _c.messages[-1].content if _c.messages else ""
    for _tag, _u in _captured_usage:
        print(f"  via: {_tag}")
        for _k in ("prompt_token_count", "candidates_token_count",
                   "thoughts_token_count", "total_token_count",
                   "prompt_tokens", "completion_tokens", "total_tokens"):
            _v = getattr(_u, _k, None)
            if _v is not None:
                print(f"  {_k}: {_v}")
    print(f"  visible len: {len(_text)} chars, preview: {_text[:200]!r}")

    # Restore originals
    GoogleGenAI._get_usage_meta = _orig_google
    if hasattr(OpenAI, "_get_usage_meta") and "_orig_openai" in dir():
        OpenAI._get_usage_meta = _orig_openai
except Exception as _probe_err:
    print(f"Probe skipped or failed: {_probe_err}")

# %% [markdown]
# ## Load the 25 problems

# %%
_data_dir = DS_ROOT / "data" if (DS_ROOT / "data").exists() else pkg / "data"
all_problems = json.loads((_data_dir / "problems.json").read_text())
baselines_raw = json.loads((_data_dir / "baselines.json").read_text())
BASELINES = {k: v for k, v in baselines_raw.items() if not k.startswith("__")}

# Scott's 25 "op_*" problems (the set with frozen baselines)
problems = [p for p in all_problems if p["problem_id"].startswith("op_")]
print(f"Loaded {len(problems)} problems")
for p in problems:
    print(f"  {p['problem_id']}  (gt={p['ground_truth']})")

# %% [markdown]
# ## Inspect prompts (sanity check on the first problem)

# %%
_first = problems[0]
print("="*70)
print("USER MESSAGE:")
print("="*70)
print(build_problem_message(_first["problem_text"]))
print()
print("="*70)
print("COMPACTION PROMPT:")
print("="*70)
print(COMPACTION_PROMPT)
print()
print("="*70)
print("POST-COMPACTION PROMPT (example):")
print("="*70)
print(build_post_compaction_prompt("<summary>", 2048))
print()
print("="*70)
print("PREDICTION PROMPT:")
print("="*70)
print(build_prediction_prompt(_first["problem_text"]))

# %% [markdown]
# ## Define the benchmark task (all 25 problems)

# %%
MODEL_NAME = getattr(kbench.llm, 'name', '') or getattr(kbench.llm, 'model', '') or ''
CTX_WINDOW = get_context_window(MODEL_NAME)
print(f"Model: {MODEL_NAME} → context_window = {CTX_WINDOW}")
TRACE_PATH = Path("/kaggle/working/traces_scott25.json")
RESULT_PATH = Path("/kaggle/working/results_scott25.json")


def _json_default(o):
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    raise TypeError(f"not serializable: {type(o)}")


def _run_one_problem(llm, prob, idx, total):
    """Run the full debug-notebook pipeline for a single problem.

    Returns (result_dict, full_trace_dict). Prints every trace inline —
    same output format as the debug notebook, just labeled with the
    problem index so you can tell them apart.
    """
    pid = prob["problem_id"]
    gt = prob["ground_truth"]
    text = prob["problem_text"]
    t0 = time.time()

    print(f"\n{'#'*70}")
    print(f"# [{idx+1}/{total}] PROBLEM: {pid} (gt={gt})")
    print(f"{'#'*70}")

    full_trace = {"problem_id": pid, "ground_truth": gt, "phases": {}}

    # ── Phase 1: Unbounded ──
    print(f"\n{'='*70}\nPHASE 1: UNBOUNDED (3 trials)\n{'='*70}")
    unb = run_unbounded(llm, text, gt, temperature=0.7)
    print(f"solved={unb.solved}  n_while_unbounded={unb.n_while_unbounded}")
    print(f"total: in={unb.input_tokens} out={unb.output_tokens} cost={unb.cost_nanodollars}")
    for i, (log, conv) in enumerate(zip(unb.trial_logs, unb.traces)):
        print(f"\n--- Trial {i}: {log['finish_reason']} ({log['tokens_used']} tok, {log['wall_time_s']}s) ---")
        for msg in conv:
            role = msg.get('role', '?').upper()
            content = msg.get('content', '')
            print(f"\n[{role}]:")
            print(content)
    full_trace["phases"]["unbounded"] = {
        "n_while_unbounded": unb.n_while_unbounded if unb.n_while_unbounded < math.inf else "inf",
        "solved": unb.solved,
        "trial_logs": unb.trial_logs,
        "conversations": unb.traces,
    }

    # ── Phase 2: Prediction ──
    print(f"\n{'='*70}\nPHASE 2: PREDICTION\n{'='*70}")
    pred = run_prediction(llm, text, temperature=0.7)
    print(f"attempt={pred.attempt}  n_predicted={pred.n_predicted}")
    print(f"raw: {pred.raw_response}")
    for msg in pred.trace:
        role = msg.get('role', '?').upper()
        print(f"\n[{role}]:")
        print(msg.get('content', ''))
    full_trace["phases"]["prediction"] = {
        "attempt": pred.attempt,
        "n_predicted": pred.n_predicted if pred.n_predicted < math.inf else "inf",
        "raw_response": pred.raw_response,
        "conversation": pred.trace,
    }

    # ── Phase 3+4: Sweep (halving) + Binary Search, all with compaction ──
    compact_token_cap = int(5 * unb.n_while_unbounded) if unb.n_while_unbounded < math.inf else None

    # Pick the SHORTEST (fewest tokens_used) successful unbounded trial for
    # prefill. Smaller prefill → faster compact trials, real reasoning.
    _unb_text = None
    _best_tokens = math.inf
    for _log, _conv in zip(unb.trial_logs, unb.traces):
        if not _log.get("success"):
            continue
        _asst_chunks = [m.get("content", "") for m in _conv if m.get("role") == "assistant"]
        if not _asst_chunks:
            continue
        _toks = _log.get("tokens_used", math.inf)
        if _toks < _best_tokens:
            _best_tokens = _toks
            _unb_text = "\n".join(_asst_chunks)
    print(f"Unbounded text reused as Cycle-0 prefill: "
          f"{'yes (' + str(len(_unb_text)) + ' chars, ' + str(_best_tokens) + ' tok)' if _unb_text else 'no (no successful trial)'}")

    def _c(N, n):
        return [run_trial_compact(llm, text, gt, N=N, temperature=0.7,
                                   token_cap=compact_token_cap,
                                   unbounded_assistant_text=_unb_text)
                for _ in range(n)]

    print(f"\n{'='*70}\nSWEEP + BINARY SEARCH (compact, token_cap={compact_token_cap})\n{'='*70}")
    n_reliable, search_log = halving_search(
        _c, context_window=CTX_WINDOW,
        n_while_unbounded=unb.n_while_unbounded,
        unbounded_trials=unb.trial_logs,
        compact_mode=True,
    )

    # Split entries by phase: halving levels vs binary search refinement
    halving_levels = set()
    n_pow = CTX_WINDOW
    while n_pow >= 64:
        halving_levels.add(n_pow)
        n_pow //= 2

    sweep_entries = [e for e in search_log.entries if e['N'] in halving_levels]
    binary_entries = [e for e in search_log.entries if e['N'] not in halving_levels]
    sweep_entries.sort(key=lambda e: -e['N'])
    binary_entries.sort(key=lambda e: -e['N'])

    def _print_entry_trace(e, label_prefix=""):
        """Print the full trace for every trial at this N level."""
        reasons = [t.get('finish_reason','?') for t in e.get('trials', [])]
        print(f"\n{label_prefix}N={e['N']:>7} {'PASS' if e['passed'] else 'FAIL'}  trials={reasons}")
        for ti, trial in enumerate(e.get('trials', [])):
            fr = trial.get('finish_reason','?')
            succ = trial.get('success', False)
            in_tok = trial.get('input_tokens', 0)
            out_tok = trial.get('output_tokens', 0)
            cost = trial.get('cost_nanodollars', 0)
            print(f"\n  ---- Trial {ti}: {'SOLVED' if succ else fr}  in={in_tok} out={out_tok} cost={cost} ----")
            traces = trial.get('traces', [])
            if not traces:
                print(f"    (no trace — likely reused_unbounded)")
                continue
            if isinstance(traces[0], list):
                cycles = traces
            else:
                cycles = [traces]
            for ci, cycle in enumerate(cycles):
                cycle_header = ("CYCLE 0 — PROBLEM → THINKING → COMPACTION INJECTION → <compact> RESPONSE"
                                if ci == 0 else
                                f"CYCLE {ci} — FRESH SESSION: RESUMPTION PROMPT (summary only) → THINKING → COMPACTION INJECTION → <compact> RESPONSE")
                print(f"\n    {'#'*60}\n    # {cycle_header}\n    {'#'*60}")
                for mi, msg in enumerate(cycle):
                    role = msg.get('role', '?').upper()
                    content = msg.get('content', '')
                    if mi == 0 and role == 'USER':
                        label = 'USER (PROBLEM)' if ci == 0 else 'USER (RESUMPTION PROMPT — fresh session)'
                    elif role == 'USER' and 'Compact everything above' in content:
                        label = 'USER (COMPACTION INJECTION PROMPT)'
                    elif role == 'ASSISTANT' and '<compact>' in content:
                        label = 'ASSISTANT (<compact> RESPONSE)'
                    else:
                        label = role
                    print(f"\n    [{label}]:")
                    print('    ' + content.replace('\n', '\n    '))

    print(f"\n{'='*70}\nPHASE 3: SWEEP (halving N/2^k) — full traces\n{'='*70}")
    for e in sweep_entries:
        _print_entry_trace(e)

    print(f"\n{'='*70}\nPHASE 4: BINARY SEARCH (refinement) — full traces\n{'='*70}")
    if binary_entries:
        for e in binary_entries:
            _print_entry_trace(e)
    else:
        print("  (no refinement needed)")

    # Aggregate tokens/cost from sweep+binary entries
    sweep_in = sweep_out = sweep_cost = 0
    for e in search_log.entries:
        for t in e.get('trials', []):
            sweep_in += t.get('input_tokens', 0) or 0
            sweep_out += t.get('output_tokens', 0) or 0
            sweep_cost += t.get('cost_nanodollars', 0) or 0

    full_trace["phases"]["sweep"] = {
        "entries": sweep_entries,
        "input_tokens": sweep_in,
        "output_tokens": sweep_out,
        "cost_nanodollars": sweep_cost,
    }
    full_trace["phases"]["binary_search"] = {"entries": binary_entries}
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"RESULT: n_reliable = {n_reliable if n_reliable < math.inf else 'inf'}  ({elapsed:.1f}s)")
    print(f"{'='*70}")

    total_in = unb.input_tokens + pred.input_tokens + sweep_in
    total_out = unb.output_tokens + pred.output_tokens + sweep_out
    total_cost = unb.cost_nanodollars + pred.cost_nanodollars + sweep_cost
    result = {
        "problem_id": pid,
        "ground_truth": gt,
        "model": MODEL_NAME,
        "context_window": CTX_WINDOW,
        "n_while_unbounded": unb.n_while_unbounded,
        "n_reliable": n_reliable,
        "n_predicted": pred.n_predicted,
        "attempt": pred.attempt,
        "unbounded_solved": unb.solved,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_nanodollars": total_cost,
        "wall_time_s": round(elapsed, 2),
        "phase_breakdown": {
            "unbounded": {"input_tokens": unb.input_tokens, "output_tokens": unb.output_tokens, "cost_nanodollars": unb.cost_nanodollars},
            "prediction": {"input_tokens": pred.input_tokens, "output_tokens": pred.output_tokens, "cost_nanodollars": pred.cost_nanodollars},
            "sweep_and_binary": {"input_tokens": sweep_in, "output_tokens": sweep_out, "cost_nanodollars": sweep_cost},
        },
    }
    return result, full_trace


@kbench.task(
    name="AmnesiaBench — Scott 25",
    description=(
        "Runs the full AmnesiaBench pipeline on all 25 op_* problems: "
        "unbounded -> predict -> sweep (halving, compact w/ prefill reuse) "
        "-> binary search. Returns compaction_ratio = mean(n_reliable / "
        "n_while_unbounded) across problems. Lower is better."
    ),
)
def amnesia_scott25(llm) -> float:
    t0_global = time.time()
    all_results = []
    all_traces = []

    for idx, prob in enumerate(problems):
        result, trace = _run_one_problem(llm, prob, idx, len(problems))
        all_results.append(result)
        all_traces.append(trace)

        # Crash-safe save after each problem
        TRACE_PATH.write_text(json.dumps(all_traces, indent=2, default=_json_default))
        RESULT_PATH.write_text(json.dumps(all_results, indent=2, default=_json_default))

    # ── Scoring across all 25 ──
    scores = compute_scores(all_results, BASELINES, model_ctx_window=CTX_WINDOW)

    print(f"\n{'='*70}")
    print("FINAL SCORES (across all 25 problems)")
    print(f"{'='*70}")
    _ratio = scores['compaction_ratio']
    print(f"compaction_ratio (main)            = "
          f"{_ratio:.4f}" if _ratio < math.inf else "compaction_ratio (main)            = inf")
    print(f"composite_context_efficiency_score = {scores['composite_context_efficiency_score']:.4f}")
    print(f"composite_window_prediction_score  = {scores['composite_window_prediction_score']:.4f}")
    print(f"composite_success_prediction_score = {scores['composite_success_prediction_score']:.4f}")
    print(f"accuracy_score                     = {scores['accuracy_score']:.4f}")
    print(f"average_cost_per_token_nanodollars = {scores['average_cost_per_token_nanodollars']:.2f}")
    print(f"n_problems_scored / solved         = {scores['n_problems_scored']} / {scores['n_problems_solved']}")
    print(f"\nTotal wall time: {time.time()-t0_global:.1f}s")
    print(f"Traces saved to {TRACE_PATH}")
    print(f"Results saved to {RESULT_PATH}")

    # Soft assertion — surfaces summary in Kaggle UI, never fails the run.
    kbench.assertions.assert_true(
        True,
        expectation=(
            f"scored {scores['n_problems_scored']}/{len(problems)}, "
            f"solved {scores['n_problems_solved']}, "
            f"compaction_ratio={_ratio if _ratio < math.inf else 'inf'}, "
            f"accuracy={scores['accuracy_score']:.4f}, "
            f"ctx_eff={scores['composite_context_efficiency_score']:.4f}"
        ),
    )

    # Return the main score. Lower = better.
    return scores["compaction_ratio"]

# %% [markdown]
# ## Run against the default model

# %%
_run = amnesia_scott25.run(kbench.llm)
print(f"Done. compaction_ratio: {_run.result}")
None  # suppress Bokeh Run widget rendering

# %% [markdown]
# ## Publish

# %%
# %choose amnesia_scott25
