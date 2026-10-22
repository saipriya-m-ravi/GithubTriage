"""LangSmith tracing setup and per-run trace metadata.

pydantic-settings loads `.env` into the `settings` object, not into `os.environ`,
but the LangSmith SDK only reads `os.environ`. `configure_tracing` bridges the two,
so `config.py` stays the single source of configuration.

Call `configure_tracing` before building or running the graph: the SDK caches
environment values the first time it reads them.
"""

import os

from githubtriage.config import Settings
from githubtriage.nodes.classify import PROMPT_VERSION as CLASSIFY_PROMPT_VERSION
from githubtriage.nodes.completeness import (
    PROMPT_VERSION as COMPLETENESS_PROMPT_VERSION,
)


class TracingConfigError(RuntimeError):
    """Tracing is enabled but not configured correctly."""


def configure_tracing(settings: Settings) -> bool:
    """Apply LangSmith settings to the environment. Returns True if tracing is on."""
    if not settings.langsmith_tracing:
        # Explicitly off, so a stray variable in the shell can't turn tracing on.
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    key = (
        settings.langsmith_api_key.get_secret_value()
        if settings.langsmith_api_key
        else ""
    )
    if not key:
        raise TracingConfigError(
            "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is not set."
        )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return True


def run_config(
    repo: str,
    issue_number: int,
    *,
    model: str,
    source: str,
    reasoning_effort: str | None = None,
) -> dict:
    """Name, tags and metadata attached to one triage run's trace.

    Shared by every adapter (CLI now, FastAPI in slice 7) so traces look the same
    whatever started them. Never put secrets here — metadata is uploaded.
    """
    return {
        "run_name": "triage",
        "tags": [source],
        "metadata": {
            "repo": repo,
            "issue_number": issue_number,
            "model": model,
            "reasoning_effort": reasoning_effort or "default",
            "prompt_versions": {
                "classify": CLASSIFY_PROMPT_VERSION,
                "completeness": COMPLETENESS_PROMPT_VERSION,
            },
        },
    }
