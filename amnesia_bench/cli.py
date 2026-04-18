# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026 (updated 30-March-2026)
# PURPOSE: CLI entry point for AmnesiaBench v3. Provides subcommands: predict, evaluate,
#   score, resume, run-all, arc-predict, arc-evaluate. Wires argparse to job modules.
#   No business logic here — all real work delegated to predict.py, evaluate.py,
#   arc_evaluate.py, score.py.
#   Integration points: run_bench.py calls main(); imports all job modules.
# SRP/DRY check: Pass — routing only; zero duplication with job logic.

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .backoff import ResumptionQueue
from .clients import create_client
from .models import load_models_json, resolve_api_key
from .problems import load_problem, load_all_problems, load_arc_problem, list_arc_problem_ids
from .utils import derive_model_name

# Default directories resolved relative to the package parent
_PACKAGE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = _PACKAGE_DIR.parent / "results"
DEFAULT_TEMPERATURE = 0.7


# ─── Subcommand Handlers ──────────────────────────────────────────────────────

def cmd_predict(args):
    from .predict import run_prediction, run_predictions_for_problems

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_url = args.model
    model_name = args.model_name or derive_model_name(model_url)
    api_key = _resolve_key(model_url, getattr(args, "api_key", None))

    client = _make_client(model_url, api_key, model_name, args.temperature)
    queue = ResumptionQueue(results_dir)

    problems = _load_problems(args)

    print(f"[predict] model={model_name} | problems={len(problems)}")
    from .predict import run_predictions_for_problems
    run_predictions_for_problems(
        client, model_name, problems,
        results_dir=results_dir,
        queue=queue,
        force=getattr(args, "force", False),
    )


def cmd_evaluate(args):
    from .evaluate import run_evaluations_for_problems

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_url = args.model
    model_name = args.model_name or derive_model_name(model_url)
    api_key = _resolve_key(model_url, getattr(args, "api_key", None))

    context_max = getattr(args, "context_max", None) or None
    if context_max is None:
        context_max = _get_context_max(model_name, model_url)
    # Ensure it's an int, not argparse default
    context_max = int(context_max)

    client = _make_client(model_url, api_key, model_name, args.temperature)
    queue = ResumptionQueue(results_dir)

    problems = _load_problems(args)

    print(f"[evaluate] model={model_name} | context_max={context_max} | problems={len(problems)}")
    run_evaluations_for_problems(
        client, model_name, problems,
        context_max=context_max,
        results_dir=results_dir,
        queue=queue,
        force=getattr(args, "force", False),
    )


def cmd_score(args):
    from .score import compute_scores
    results_dir = Path(args.results_dir)
    compute_scores(results_dir)


def cmd_resume(args):
    """Retry all queued failed jobs from queue.json."""
    from .predict import run_prediction
    from .evaluate import run_evaluation
    from .problems import load_problem

    results_dir = Path(args.results_dir)
    queue = ResumptionQueue(results_dir)
    entries = queue.entries()

    if not entries:
        print("[resume] Queue is empty — nothing to retry.")
        return

    print(f"[resume] {len(entries)} queued job(s) to retry ...")

    for entry in entries:
        model_name = entry["model_name"]
        problem_id = entry["problem_id"]
        job_type = entry["job_type"]
        retry_count = entry.get("retry_count", 0)

        print(f"\n[resume] {job_type} | {model_name} / {problem_id} (retry #{retry_count + 1})")

        try:
            problem = load_problem(problem_id)
        except FileNotFoundError as e:
            print(f"  Cannot find problem: {e} — skipping")
            continue

        # Resolve client from models.json by name
        try:
            models = load_models_json()
            model_entry = next((m for m in models if m["name"] == model_name), None)
            if model_entry is None:
                print(f"  Model '{model_name}' not found in models.json — skipping")
                continue
            api_key = resolve_api_key(model_entry)
            client = _make_client(
                model_entry["url"], api_key, model_name,
                DEFAULT_TEMPERATURE,
            )
        except Exception as e:
            print(f"  Cannot create client: {e} — skipping")
            continue

        try:
            if job_type == "prediction":
                run_prediction(
                    client, model_name, problem,
                    results_dir=results_dir,
                    queue=None,  # don't re-queue on resume failure
                    force=True,
                )
            elif job_type == "evaluation":
                context_max = model_entry.get("context_max", 32768)
                run_evaluation(
                    client, model_name, problem, context_max,
                    results_dir=results_dir,
                    queue=None,
                    force=True,
                )
            queue.remove(model_name, problem_id, job_type)
            print(f"  SUCCESS — removed from queue")
        except Exception as e:
            print(f"  FAILED: {e}")
            queue.push(model_name, problem_id, job_type, str(e), retry_count + 1)


def cmd_arc_predict(args):
    """Run ARC prediction job: ask model if it can solve the puzzle and what N it needs."""
    from .arc_evaluate import run_arc_prediction

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_url = args.model
    model_name = args.model_name or derive_model_name(model_url)
    api_key = _resolve_key(model_url, getattr(args, "api_key", None))

    client = _make_client(model_url, api_key, model_name, args.temperature)
    queue = ResumptionQueue(results_dir)

    problems = _load_arc_problems(args)

    print(f"[arc-predict] model={model_name} | problems={len(problems)}")
    for problem in problems:
        run_arc_prediction(
            client, model_name, problem,
            results_dir=results_dir,
            queue=queue,
            force=getattr(args, "force", False),
        )


def cmd_arc_evaluate(args):
    """Run ARC nested binary search evaluation."""
    from .arc_evaluate import run_arc_evaluations_for_problems

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_url = args.model
    model_name = args.model_name or derive_model_name(model_url)
    api_key = _resolve_key(model_url, getattr(args, "api_key", None))

    context_max = getattr(args, "context_max", None) or None
    if context_max is None:
        context_max = _get_context_max(model_name, model_url)
    context_max = int(context_max)

    client = _make_client(model_url, api_key, model_name, args.temperature)
    queue = ResumptionQueue(results_dir)

    problems = _load_arc_problems(args)

    print(f"[arc-evaluate] model={model_name} | context_max={context_max} | problems={len(problems)}")
    run_arc_evaluations_for_problems(
        client, model_name, problems,
        context_max=context_max,
        results_dir=results_dir,
        queue=queue,
        force=getattr(args, "force", False),
    )


def cmd_run_all(args):
    """Run predict then evaluate for every model in models.json × every problem."""
    from .predict import run_predictions_for_problems
    from .evaluate import run_evaluations_for_problems

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    queue = ResumptionQueue(results_dir)

    problems = _load_problems(args)

    try:
        models = load_models_json()
    except Exception as e:
        print(f"[run-all] Cannot load models.json: {e}")
        sys.exit(1)

    print(f"[run-all] {len(models)} model(s) × {len(problems)} problem(s)")

    for model_entry in models:
        model_name = model_entry["name"]
        model_url = model_entry["url"]
        context_max = model_entry.get("context_max", 32768)
        api_key = resolve_api_key(model_entry)

        print(f"\n{'#' * 60}")
        print(f"  MODEL: {model_name}  ({model_url})")
        print(f"{'#' * 60}")

        try:
            client = _make_client(model_url, api_key, model_name, DEFAULT_TEMPERATURE)
        except ValueError as e:
            print(f"  Cannot create client: {e} — skipping")
            continue

        run_predictions_for_problems(
            client, model_name, problems,
            results_dir=results_dir, queue=queue,
        )
        run_evaluations_for_problems(
            client, model_name, problems,
            context_max=context_max,
            results_dir=results_dir, queue=queue,
        )

    print("\n[run-all] Done. Run `score` subcommand for composite results.")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_problems(args) -> list:
    if getattr(args, "all", False):
        return load_all_problems()
    problem_id = getattr(args, "problem", None)
    if problem_id:
        return [load_problem(problem_id)]
    print("ERROR: specify --problem ID or --all")
    sys.exit(1)


def _load_arc_problems(args) -> list:
    """Load ARC problems from the arc-explainer dataset on disk."""
    if getattr(args, "all", False):
        ids = list_arc_problem_ids()
        if not ids:
            print("ERROR: no ARC problems found on disk")
            sys.exit(1)
        print(f"[arc] loading {len(ids)} problem(s) from disk ...")
        problems = []
        for pid in ids:
            try:
                problems.append(load_arc_problem(pid))
            except Exception as e:
                print(f"  WARNING: failed to load ARC problem {pid}: {e}")
        if not problems:
            print("ERROR: could not load any ARC problems")
            sys.exit(1)
        return problems
    problem_id = getattr(args, "problem", None)
    if problem_id:
        try:
            return [load_arc_problem(problem_id)]
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    print("ERROR: specify --problem ID or --all")
    sys.exit(1)


def _make_client(model_url: str, api_key, model_name: str, temperature: float):
    try:
        return create_client(
            server_url=model_url,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _resolve_key(model_url: str, cli_key=None):
    if cli_key:
        return cli_key
    if model_url.startswith("gemini://") or model_url.startswith("google://"):
        return os.environ.get("GEMINI_API_KEY")
    if model_url.startswith("openrouter://"):
        return os.environ.get("OPENROUTER_API_KEY")
    return None


def _get_context_max(model_name: str, model_url: str) -> int:
    """Try to read context_max from models.json; fall back to 32768."""
    try:
        models = load_models_json()
        for m in models:
            if m["name"] == model_name or m["url"] == model_url:
                return int(m.get("context_max", 32768))
    except Exception:
        pass
    return 32768


# ─── Argument Parser ──────────────────────────────────────────────────────────

def _add_model_args(p):
    p.add_argument("--model", required=True,
                   help="Backend URL: anthropic://MODEL, gemini://MODEL, openrouter://MODEL, http://host:port")
    p.add_argument("--model-name", default=None,
                   help="Human label for this model (default: derived from --model URL)")
    p.add_argument("--context-max", type=int, default=None,
                   help="Model's full context window in tokens (overrides models.json)")
    p.add_argument("--api-key", default=None,
                   help="API key for Gemini/OpenRouter backends")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)


def _add_problem_args(p):
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", type=str, help="Problem ID (or substring)")
    group.add_argument("--all", action="store_true", help="Run all problems")


def _add_results_arg(p):
    p.add_argument(
        "--results-dir", default=str(DEFAULT_RESULTS_DIR),
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})"
    )


def _add_arc_problem_args(p):
    """Problem args for ARC subcommands — sources from arc-explainer dataset."""
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--problem", type=str,
        help="ARC problem ID (or substring) from arc-explainer evaluation/ or evaluation2/",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run all available ARC problems from disk",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amnesia_bench",
        description=f"AmnesiaBench v{__version__} — context window benchmark",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p_predict = sub.add_parser("predict", help="Run prediction job")
    _add_model_args(p_predict)
    _add_problem_args(p_predict)
    _add_results_arg(p_predict)
    p_predict.add_argument("--force", action="store_true", help="Re-run even if result exists")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run nested binary search evaluation")
    _add_model_args(p_eval)
    _add_problem_args(p_eval)
    _add_results_arg(p_eval)
    # --context-max already added by _add_model_args
    p_eval.add_argument("--force", action="store_true", help="Re-run even if result exists")

    # score
    p_score = sub.add_parser("score", help="Compute and print composite scores")
    _add_results_arg(p_score)

    # resume
    p_resume = sub.add_parser("resume", help="Retry failed jobs from queue.json")
    _add_results_arg(p_resume)

    # run-all
    p_all = sub.add_parser("run-all", help="Run predict + evaluate for all models in models.json")
    _add_problem_args(p_all)
    _add_results_arg(p_all)

    # arc-predict
    p_arc_pred = sub.add_parser(
        "arc-predict",
        help="Run ARC prediction job (model self-assesses whether it can solve the puzzle)",
    )
    _add_model_args(p_arc_pred)
    _add_arc_problem_args(p_arc_pred)
    _add_results_arg(p_arc_pred)
    p_arc_pred.add_argument("--force", action="store_true", help="Re-run even if result exists")

    # arc-evaluate
    p_arc_eval = sub.add_parser(
        "arc-evaluate",
        help="Run nested binary search ARC evaluation (grid puzzle answer matching)",
    )
    _add_model_args(p_arc_eval)
    _add_arc_problem_args(p_arc_eval)
    _add_results_arg(p_arc_eval)
    # --context-max already added by _add_model_args
    p_arc_eval.add_argument("--force", action="store_true", help="Re-run even if result exists")

    return parser


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "score": cmd_score,
        "resume": cmd_resume,
        "run-all": cmd_run_all,
        "arc-predict": cmd_arc_predict,
        "arc-evaluate": cmd_arc_evaluate,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
