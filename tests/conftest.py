"""Shared test fixtures. pytest loads this file automatically for every test in tests/."""

import pytest

from githubtriage.config import settings

LANGSMITH_VARS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT")


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """Every test runs with tracing off, so tests never upload traces.

    delenv first so monkeypatch records the original values and restores them
    after the test, even if code under test writes to os.environ.
    """
    for var in LANGSMITH_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(settings, "langsmith_tracing", False)


class FakeStructured:
    """Stands in for llm.with_structured_output(...): returns a result or raises."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []  # every `messages` list passed to invoke()

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.result


class FakeChatModel:
    """Stands in for ChatGroq: only the method classify() uses."""

    def __init__(self, structured):
        self.structured = structured

    def with_structured_output(self, schema, **kwargs):
        return self.structured


@pytest.fixture
def use_fake_llm(monkeypatch):
    """Install a fake LLM in the classify module; returns the fake for inspection."""

    def install(result=None, error=None):
        structured = FakeStructured(result=result, error=error)
        # Patch where it's USED (classify's own name), not where it's defined (llm.py).
        monkeypatch.setattr(
            "githubtriage.nodes.classify.get_chat_groq_model",
            lambda: FakeChatModel(structured),
        )
        return structured

    return install
