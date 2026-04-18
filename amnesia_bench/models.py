# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Model configuration loading for AmnesiaBench v3. Reads models.json from the
#   project root and validates required fields. Also exposes cost-per-token lookup used
#   by score.py.
#   Integration points: imported by cli.py, score.py, and evaluate.py (for context_max).
# SRP/DRY check: Pass — single source of model config I/O; no model routing logic here
#   (that belongs in clients.py).

import json
import os
from pathlib import Path
from typing import List, Optional

_PACKAGE_DIR = Path(__file__).parent
MODELS_JSON = _PACKAGE_DIR.parent / "models.json"


def set_models_json(path: Path) -> None:
    """Override the models.json path (e.g. for testing)."""
    global MODELS_JSON
    MODELS_JSON = Path(path)


def load_models_json() -> List[dict]:
    """
    Load and validate models.json.

    Expected schema per entry:
    {
        "name": str,                        # human label, used in result filenames
        "url": str,                         # backend URL (scheme determines client type)
        "context_max": int,                 # max context window in tokens
        "cost_per_input_token": float,      # USD per input token (e.g. 3e-6)
        "cost_per_output_token": float,     # USD per output token (e.g. 15e-6)
        "api_key_env": str (optional),      # env var name for API key
    }

    Raises FileNotFoundError or ValueError on bad config.
    """
    if not MODELS_JSON.exists():
        raise FileNotFoundError(
            f"models.json not found at {MODELS_JSON}. "
            "Create it with a list of model config objects."
        )
    models = json.loads(MODELS_JSON.read_text())
    if not isinstance(models, list) or not models:
        raise ValueError("models.json must be a non-empty JSON array.")

    for m in models:
        for field in ("name", "url", "context_max"):
            if field not in m:
                raise ValueError(
                    f"Model entry missing required field '{field}': {m}"
                )
        # Cost fields default to 0 if absent (local models often have no cost)
        m.setdefault("cost_per_input_token", 0.0)
        m.setdefault("cost_per_output_token", 0.0)

    return models


def get_model_config(name: str) -> Optional[dict]:
    """Return the config for a model by name, or None if not found."""
    try:
        models = load_models_json()
    except Exception:
        return None
    for m in models:
        if m["name"] == name:
            return m
    return None


def resolve_api_key(model_entry: dict, cli_api_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve the API key for a model entry in priority order:
      1. Explicit cli_api_key argument
      2. api_key_env field in model entry → env var lookup
      3. Scheme-specific fallback env vars (GEMINI_API_KEY, OPENROUTER_API_KEY)
    Returns None if no key found (valid for local/anthropic:// backends).
    """
    if cli_api_key:
        return cli_api_key

    env_var = model_entry.get("api_key_env")
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key

    url = model_entry.get("url", "")
    if url.startswith("gemini://") or url.startswith("google://"):
        return os.environ.get("GEMINI_API_KEY")
    if url.startswith("openrouter://"):
        return os.environ.get("OPENROUTER_API_KEY")

    return None
