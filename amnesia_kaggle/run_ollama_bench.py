#!/usr/bin/env python3
"""Run AmnesiaBench against a local Ollama model.

Writes per-problem results JSON to data/runs/{model_slug}.json.
"""
from __future__ import annotations
import argparse, json, math, sys, time, os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from amnesia_kaggle.harness import run_unbounded, run_prediction, run_trial_no_compact, run_trial_compact
from amnesia_kaggle.halving_search import halving_search
from amnesia_kaggle.scoring import compute_scores
from amnesia_kaggle.ollama_adapter import OllamaChat


def _json_default(obj):
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    raise TypeError(f"not serializable: {type(obj)}")


def run_model(model: str, base_url: str, ctx_window: int, problems_filter: list[str] | None, out_path: Path, think: bool = True):
    llm = OllamaChat(model=model, base_url=base_url, think=think)
    problems = json.loads((_HERE / "data" / "problems.json").read_text())
    if problems_filter:
        problems = [p for p in problems if p["problem_id"] in problems_filter]

    # Resume: load existing results and skip already-done problems
    all_results = []
    done_ids = set()
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text().replace('Infinity','1e309').replace('NaN','null'))
            all_results = prior.get('results', [])
            done_ids = {r['problem_id'] for r in all_results}
            for r in all_results:
                for k in ('n_while_unbounded','n_reliable_no_compact','n_reliable_compact','n_reliable','n_predicted'):
                    if r.get(k) == 1e309:
                        r[k] = math.inf
            print(f'Resuming: {len(done_ids)} problems already done')
        except Exception as e:
            print(f'Could not resume ({e}), starting fresh')
    problems = [p for p in problems if p['problem_id'] not in done_ids]

    print(f"Model: {model}  Base URL: {base_url}  Problems: {len(problems)} remaining  Ctx: {ctx_window}")
    t0_global = time.time()

    for i, prob in enumerate(problems):
        pid = prob["problem_id"]
        gt = prob["ground_truth"]
        text = prob["problem_text"]
        print(f"\n[{i+1}/{len(problems)}] {pid} (gt={gt})")
        t0 = time.time()

        try:
            unb = run_unbounded(llm, text, gt, temperature=0.7)
            print(f"  unbounded: solved={unb.solved} n_while_unbounded={unb.n_while_unbounded}")

            pred = run_prediction(llm, text, temperature=0.7)
            print(f"  predict: attempt={pred.attempt} n_predicted={pred.n_predicted}")

            # Token cap for compact trials: 5x unbounded tokens
            tcap = int(5 * unb.n_while_unbounded) if unb.n_while_unbounded < math.inf else None

            def _mk(llm_, text_, gt_, tcap_):
                def _nc(N, n): return [run_trial_no_compact(llm_, text_, gt_, N=N, temperature=0.7) for _ in range(n)]
                def _c(N, n): return [run_trial_compact(llm_, text_, gt_, N=N, temperature=0.7, token_cap=tcap_) for _ in range(n)]
                return _nc, _c

            nc_fn, c_fn = _mk(llm, text, gt, tcap)

            nr_nc, log_nc = halving_search(nc_fn, context_window=ctx_window, n_while_unbounded=unb.n_while_unbounded, unbounded_trials=unb.trial_logs, compact_mode=False)
            print(f"  no-compact: n_reliable={nr_nc if nr_nc < math.inf else 'inf'}")

            nr_c, log_c = halving_search(c_fn, context_window=ctx_window, n_while_unbounded=unb.n_while_unbounded, unbounded_trials=unb.trial_logs, compact_mode=True)
            print(f"  compact: n_reliable={nr_c if nr_c < math.inf else 'inf'}")

            n_reliable = min(nr_nc, nr_c)
            print(f"  n_reliable = {n_reliable if n_reliable < math.inf else 'inf'}  ({time.time()-t0:.1f}s)")

            result_entry = {
                "problem_id": pid,
                "n_while_unbounded": unb.n_while_unbounded,
                "n_reliable_no_compact": nr_nc,
                "n_reliable_compact": nr_c,
                "n_reliable": n_reliable,
                "n_predicted": pred.n_predicted,
                "attempt": pred.attempt,
                "unbounded_solved": unb.solved,
                "input_tokens": unb.input_tokens + pred.input_tokens,
                "output_tokens": unb.output_tokens + pred.output_tokens,
                "cost_nanodollars": 0,
                "wall_time_s": round(time.time() - t0, 2),
                "traces": {
                    "unbounded": {"trial_logs": unb.trial_logs, "conversations": unb.traces},
                    "prediction": {"raw": pred.raw_response, "conversation": pred.trace},
                    "sweep_no_compact": {"entries": log_nc.entries},
                    "sweep_compact": {"entries": log_c.entries},
                },
            }
            all_results.append(result_entry)

            # Save after each problem (crash-safe)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "model": model,
                "base_url": base_url,
                "ctx_window": ctx_window,
                "wall_time_s": round(time.time() - t0_global, 2),
                "results": all_results,
            }, indent=2, default=_json_default))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # Compute scores
    baselines = json.loads((_HERE / "data" / "baselines.json").read_text())
    baselines_clean = {k: v for k, v in baselines.items() if not k.startswith("__")}
    scores = compute_scores(all_results, baselines_clean, model_ctx_window=ctx_window)
    print("\n── Scores ──")
    for k, v in scores.items():
        print(f"  {k:40s} {v}")
    print(f"\nTotal wall time: {time.time() - t0_global:.1f}s")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--ctx-window", type=int, default=8192)
    parser.add_argument("--problems", nargs="*", help="Filter to specific problem IDs")
    parser.add_argument("--out", help="Output JSON path (default: data/runs/{model_slug}.json)")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        slug = args.model.replace("/", "_").replace(":", "_")
        out_path = _HERE / "data" / "runs" / f"{slug}.json"

    run_model(args.model, args.base_url, args.ctx_window, args.problems, out_path, think=not args.no_thinking)
