# AmnesiaBench — Experiment Plan

**Date:** 28-March-2026
**Goal:** Establish the experiment roadmap for AmnesiaBench context window benchmarking — method improvements, scoring fixes, model coverage, and compaction strategy.
**Contributors:** Son Pham, Scott Weiss, the boss, Bubba
**Status:** Phase 1 baseline in progress. Design proposals under discussion — not approved for implementation.

---

## Background

AmnesiaBench binary-searches for the minimum context window (in tokens) at which an LLM can reliably solve a problem (≥2/3 trials pass). Results feed a leaderboard scoring models on context window efficiency.

The benchmark answers: **how little context does a model actually need?**

### Current Setup
- **Location:** Son's Strix Halo (`ssh -p 10075 bubba-bench@0.tcp.ngrok.io`)
- **Code:** `/home/bubba-bench/bench-work/AmnesiaBench/amnesia_bench.py`
- **Problems:** 10 (in `problems/` — math, combinatorics, logic)
- **Models:** Qwen3.5-35B-A3B-Q4 (local), gemini-3.1-flash-lite (API)
- **Configs per problem:** `NoTIR_HardCut` (hard truncation), `NoTIR_Compact` (compaction)

---

## Known Bugs (Found 28-March-2026)

### Bug 1: MIN_WINDOW floor artifact
`MIN_WINDOW` was 512 with a 64-token snap (`(mid // 64) * 64`). This created a floor at 640 — 7 of 9 problems reported `minimum_window = 640`, which is clearly an artifact.

**Fix applied:** `MIN_WINDOW = 1`, snap floor protected with `max(1, ...)`.
**Action:** Rerun all problems that reported 640 with `--min-window 1 --max-window 640`.

### Bug 2: Coverage > 1.0
`coverage = problems_attempted / problems_eligible` could exceed 1.0 when a model attempted problems that had no baseline from other models.

**Fix applied:** Capped coverage at 1.0.

### Bug 3: Dynamic baseline contamination
Baseline was `min(n_reliable)` across all models including the model being scored. A model could become its own baseline, inflating its score.

**Fix applied:** Baseline now uses `gptoss_120b_correct_token_avg` from problem JSON (fixed GPT-4o 120B reference).

**Son's position:** The metric should depend on model + problem alone. No external reference necessary. Raw `minimum_window` is the primary metric — lower is better. Prediction accuracy is a separate, decoupled score.

### Bug 4: Prediction score rewards overconfidence
`pred_composite = mean(baseline_n_pred / model_n_pred)` rewarded models that predicted tiny windows (overconfident). Qwen predicted 50-150 tokens but needed 640+ → pred_composite was inflated to 17×.

**Status:** Formula needs redesign. Metacognition scores should be decoupled from performance scores (Scott's recommendation, Son agrees).

---

## Phase 1 — Baseline (In Progress)

**Goal:** Get real minimum_window values for all problems with the MIN_WINDOW bug fixed.

- Qwen rerun: 7 problems at [1, 640] range (launched 28-March-2026 14:20 EDT)
- aimo3_hard results kept (above 640 floor, unaffected by bug)
- Gemini: 3/10 problems complete; remaining need `GEMINI_API_KEY` in env

**Deliverable:** Corrected `--scores` table for all 10 problems × 2 models.

---

## Phase 2 — Method Improvements (Proposals, Pending Approval)

### 2a. Nested Binary Search
**Proposer:** Scott Weiss

- Phase 1 scan: 1 trial per window — find transition cheaply
- Phase 2 confirm: 3 trials per window, 3× safety range around Phase 1 result
- Estimated 40-50% fewer trials. Risk: noisy Phase 1 for borderline problems.

### 2b. Cost-Adjusted Scoring
**Proposer:** Scott Weiss

Add `avg_cost_per_token` to the formula to reward cheaper models:
```
final_score = composite × pred_composite × coverage × accuracy × pred_accuracy / avg_cost_per_token × 10000
```

Cost per token: weighted by actual input/output counts (input dominates ~85-90%). For local models, use API-equivalent pricing. Track actual token counts per trial.

### 2c. 1-Token Snap
**Proposer:** Scott Weiss

Snap window sizes to nearest token (not 64-token boundaries). Low impact for large windows, correct for small N. Needs discussion on pros/cons.

### 2d. Decouple Metacognition from Performance
**Proposers:** Scott Weiss, Son Pham

Prediction accuracy (how well the model predicts its own context needs) should be a separate leaderboard dimension, not multiplied into the main score. Otherwise poor metacognition zeroes out otherwise good solving models.

---

## Phase 3 — Model Coverage (Post-Method Lock)

**Budget:** $500 (Son Pham)

| Tier | Models | $/problem | Strategy |
|------|--------|-----------|----------|
| Cheap | gemini-3.1-flash-lite, gemini-2.5-flash, gpt-4o-mini | $0.03-0.06 | All problems |
| Mid | claude-haiku-3.5, o4-mini | $0.37-0.43 | All problems |
| Expensive | gpt-4o, gemini-2.5-pro | $0.82-0.99 | Curated subset |
| Frontier | o3, claude-sonnet-4.5 | $1.38-3.95 | Curated subset |

**Parallelism rule:** Local models serialize (one GPU). Commercial API models parallelize (one process per problem).

---

## Phase 4 — Compaction Strategy (v2)

### 4a. Three Compaction Configs

| Config | Trigger | Measures |
|--------|---------|----------|
| `NoTIR_HardCut` | No compaction, hard truncation at N | Raw context limit tolerance |
| `NoTIR_EarlyCompact` | Fires at ~50% of N | Safe default (Scott's suggestion for testing) |
| `NoTIR_LateCompact` | Fires at 100% of N | N reported = N tested; overflow = failure |

### 4b. Model-Directed Compaction
**Approved in principle by Son (28-March-2026)**

Notify model at 60% context fill, then every 5%. Model decides whether to compact at each notification. Prompt asks: "Do you have enough remaining context to solve this problem?"

- Model must gauge its own remaining solvability
- Compaction decision stored in result trace
- Separate leaderboard column: `NoTIR_MetacogCompact`
- Implementation complexity: high (multi-turn interrupts, token tracking mid-generation)

---

## Open Questions

1. **Scoring formula:** What replaces the current multiplicative formula? Son wants raw minimum_window as primary metric. How does prediction accuracy factor in separately?
2. **Compaction trigger for Phase 1:** Hardcode 50% (Scott) vs current behavior vs defer to Phase 4?
3. **n_while_unbounded:** Need to run unbounded comparison to verify compaction is helping at all (Scott's diagnostic)
4. **Problem count:** 10 for proof-of-concept. How many for full competition submission?

---

## Operational Notes

- `amnesia_bench.py` has exponential backoff on all API calls (429/503, 5 retries, 1-60s with jitter)
- ngrok tunnel can die — if SSH connection refused, Son restarts it
- All result traces stored in JSON — metrics can be back-calculated without rerunning
- Old results (512-floor artifacts) archived in `results_old_512floor/`
