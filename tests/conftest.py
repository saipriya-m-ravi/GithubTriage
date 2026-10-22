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
    """Stands in for ChatGroq: returns the fake registered for each output schema."""

    def __init__(self, by_schema):
        self.by_schema = by_schema

    def with_structured_output(self, schema, **kwargs):
        if schema not in self.by_schema:
            # Surfaces unexpected LLM calls (e.g. a node that should have been skipped).
            raise AssertionError(f"unexpected LLM call for {schema.__name__}")
        return self.by_schema[schema]


# Every module that calls the LLM. Patch where the name is USED, not where it's
# defined (llm.py): each module holds its own reference after `from ... import`.
LLM_USERS = (
    "githubtriage.nodes.classify.get_chat_groq_model",
    "githubtriage.nodes.completeness.get_chat_groq_model",
)


@pytest.fixture
def use_fake_llm(monkeypatch):
    """Install fake LLM responses, one per output schema; returns the fake.

    use_fake_llm(result=Classification(...))                  # classify returns this
    use_fake_llm(result=Completeness(...))                    # completeness returns this
    use_fake_llm(error=RuntimeError("x"))                     # classify raises
    use_fake_llm(error=RuntimeError("x"), schema=Completeness)  # completeness raises
    Calls accumulate within a test, so classify and completeness can both be faked.
    """
    from githubtriage.models import Classification

    by_schema = {}
    for target in LLM_USERS:
        monkeypatch.setattr(target, lambda: FakeChatModel(by_schema))

    def install(result=None, error=None, schema=None):
        schema = schema or (type(result) if result is not None else Classification)
        structured = FakeStructured(result=result, error=error)
        by_schema[schema] = structured
        return structured

    return install
