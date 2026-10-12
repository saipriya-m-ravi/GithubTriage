"""Shared test fixtures. pytest loads this file automatically for every test in tests/."""

import pytest


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
