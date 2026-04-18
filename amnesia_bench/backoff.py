# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Exponential backoff wrapper for all API calls, and ResumptionQueue for
#   persisting failed jobs across sessions. Used by every client and every job runner.
#   Integration points: imported by clients.py, predict.py, evaluate.py.
#   ResumptionQueue backed by queue.json in the results directory.
# SRP/DRY check: Pass — single retry engine; ResumptionQueue is the single queue impl.

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests


def with_exponential_backoff(
    fn: Callable,
    max_retries: int = 20,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
):
    """
    Wrap any API call with exponential backoff on 429/503 errors.
    Respects Retry-After header when present.
    Uses full jitter: delay = min(base * 2^attempt + uniform(0,2), max_delay).
    Raises immediately on non-retriable errors or when retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 503) and attempt < max_retries - 1:
                retry_after = None
                if e.response is not None:
                    retry_after = (
                        e.response.headers.get("Retry-After")
                        or e.response.headers.get("x-ratelimit-reset-requests")
                    )
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except (ValueError, TypeError):
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 2), max_delay)
                else:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 2), max_delay)
                print(
                    f"    [backoff] HTTP {status} — retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            else:
                raise
    # Should not reach here, but raise on exhaustion
    raise RuntimeError(f"with_exponential_backoff: exhausted {max_retries} retries")


class ResumptionQueue:
    """
    Persistent queue for failed jobs. Backed by queue.json in results_dir.

    Entry schema:
    {
        "model_name": str,
        "problem_id": str,
        "job_type": "prediction" | "evaluation",
        "error": str,
        "timestamp": ISO-8601 str,
        "retry_count": int
    }

    Usage:
        q = ResumptionQueue(results_dir)
        q.push(model_name, problem_id, "evaluation", error_str)
        for entry in q.entries():
            ...
        q.remove(model_name, problem_id, "evaluation")
    """

    def __init__(self, results_dir: Path):
        self.path = Path(results_dir) / "queue.json"

    def _load(self) -> list:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []

    def _save(self, entries: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2))

    def push(
        self,
        model_name: str,
        problem_id: str,
        job_type: str,
        error: str,
        retry_count: int = 0,
    ) -> None:
        """Append or update a failed-job entry."""
        entries = self._load()
        # Update if already present
        for e in entries:
            if (
                e["model_name"] == model_name
                and e["problem_id"] == problem_id
                and e["job_type"] == job_type
            ):
                e["error"] = error
                e["timestamp"] = datetime.now(timezone.utc).isoformat()
                e["retry_count"] = retry_count
                self._save(entries)
                return
        entries.append(
            {
                "model_name": model_name,
                "problem_id": problem_id,
                "job_type": job_type,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retry_count": retry_count,
            }
        )
        self._save(entries)

    def remove(self, model_name: str, problem_id: str, job_type: str) -> None:
        """Remove a successfully-retried entry."""
        entries = self._load()
        entries = [
            e
            for e in entries
            if not (
                e["model_name"] == model_name
                and e["problem_id"] == problem_id
                and e["job_type"] == job_type
            )
        ]
        self._save(entries)

    def entries(self) -> list:
        """Return all queued entries."""
        return self._load()

    def is_empty(self) -> bool:
        return len(self._load()) == 0
