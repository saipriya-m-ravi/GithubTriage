import os

import pytest

from githubtriage.config import Settings
from githubtriage.nodes.classify import PROMPT_VERSION as CLASSIFY_VERSION
from githubtriage.nodes.completeness import PROMPT_VERSION as COMPLETENESS_VERSION
from githubtriage.observability import (
    TracingConfigError,
    configure_tracing,
    run_config,
)


def make_settings(**overrides) -> Settings:
    # _env_file=None: build settings only from these values, never from the real .env.
    values = {"groq_api_key": "test-key", "groq_model": "test-model"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_tracing_off_by_default():
    assert configure_tracing(make_settings()) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_tracing_on_without_key_fails_fast():
    with pytest.raises(TracingConfigError, match="LANGSMITH_API_KEY"):
        configure_tracing(make_settings(langsmith_tracing=True))


def test_tracing_on_with_empty_key_fails_fast():
    # .env.example ships `LANGSMITH_API_KEY=` (empty) — must count as missing.
    with pytest.raises(TracingConfigError):
        configure_tracing(make_settings(langsmith_tracing=True, langsmith_api_key=""))


def test_tracing_on_sets_environment():
    settings = make_settings(
        langsmith_tracing=True, langsmith_api_key="ls-key", langsmith_project="proj"
    )

    assert configure_tracing(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"


def test_run_config_carries_governance_metadata():
    config = run_config("owner/repo", 7, model="m-1", source="cli")

    assert config["run_name"] == "triage"
    assert config["tags"] == ["cli"]
    assert config["metadata"]["model"] == "m-1"
    assert config["metadata"]["reasoning_effort"] == "default"
    assert config["metadata"]["prompt_versions"] == {
        "classify": CLASSIFY_VERSION,
        "completeness": COMPLETENESS_VERSION,
    }
