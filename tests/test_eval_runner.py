"""The eval runner: caching, pacing, stopping, failures. No real LLM calls."""

import pytest

from githubtriage.config import settings
from githubtriage.evaluation.dataset import IssueText
from githubtriage.evaluation.runner import (
    Prediction,
    PredictionStore,
    TokenBudget,
    config_id,
    predict,
    run_predictions,
)
from githubtriage.models import Classification


def text(n: int) -> IssueText:
    return IssueText(repo="o/r", number=n, title=f"t{n}", body="b")


def fake_prediction(text: IssueText, run: int, predicted="bug") -> Prediction:
    return Prediction(
        repo=text.repo,
        number=text.number,
        run=run,
        predicted=predicted,
        confidence=0.9,
        input_tokens=1000,
        output_tokens=100,
        latency_s=1.0,
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def no_pacing() -> TokenBudget:
    clock = FakeClock()
    return TokenBudget(10**9, clock=clock, sleep=clock.sleep)


class RateLimitError(Exception):  # same name as groq's
    pass


def test_cached_predictions_are_not_requested_again(tmp_path):
    calls = []

    def predict_fn(t, run):
        calls.append(t.number)
        return fake_prediction(t, run)

    path = tmp_path / "p.jsonl"
    texts = [text(1), text(2)]
    run_predictions(
        texts, 1, PredictionStore(path), budget=no_pacing(), predict_fn=predict_fn
    )
    report = run_predictions(
        texts, 1, PredictionStore(path), budget=no_pacing(), predict_fn=predict_fn
    )

    assert calls == [1, 2]  # the second invocation only re-read the cache
    assert (report.called, report.cached) == (0, 2)


def test_a_damaged_last_line_is_skipped_and_redone(tmp_path):
    # A crash mid-write leaves a half line; it must not stop the next run.
    path = tmp_path / "p.jsonl"
    store = PredictionStore(path)
    store.add(fake_prediction(text(1), 1))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"repo": "o/r", "numb')

    reloaded = PredictionStore(path)
    assert reloaded.get("o/r", 1, 1) is not None
    report = run_predictions(
        [text(1), text(2)], 1, reloaded, budget=no_pacing(), predict_fn=fake_prediction
    )
    assert report.called == 1  # only #2


def test_failures_are_reported_not_scored_as_unclear(tmp_path):
    def predict_fn(t, run):
        if t.number == 2:
            raise ValueError("bad output")
        return fake_prediction(t, run)

    store = PredictionStore(tmp_path / "p.jsonl")
    report = run_predictions(
        [text(1), text(2), text(3)], 1, store, budget=no_pacing(), predict_fn=predict_fn
    )

    assert report.failures == ["o/r#2 run 1: ValueError"]
    assert store.get("o/r", 2, 1) is None  # not cached: retried next time
    assert report.called == 2


def test_rate_limit_stops_the_run_cleanly(tmp_path):
    def predict_fn(t, run):
        if t.number == 2:
            raise RateLimitError("tokens per day (TPD) exceeded")
        return fake_prediction(t, run)

    store = PredictionStore(tmp_path / "p.jsonl")
    report = run_predictions(
        [text(1), text(2), text(3)], 1, store, budget=no_pacing(), predict_fn=predict_fn
    )

    assert report.stopped.startswith("rate limited")
    assert report.called == 1  # #3 never attempted: no point hammering the limit
    assert store.get("o/r", 1, 1) is not None  # done work is kept


def test_max_calls_caps_new_calls(tmp_path):
    store = PredictionStore(tmp_path / "p.jsonl")
    report = run_predictions(
        [text(n) for n in range(5)],
        1,
        store,
        budget=no_pacing(),
        max_calls=2,
        predict_fn=fake_prediction,
    )
    assert report.called == 2
    assert "max-calls" in report.stopped


def test_runs_are_cached_separately(tmp_path):
    store = PredictionStore(tmp_path / "p.jsonl")
    run_predictions([text(1)], 1, store, budget=no_pacing(), predict_fn=fake_prediction)
    report = run_predictions(
        [text(1)], 2, store, budget=no_pacing(), predict_fn=fake_prediction
    )
    assert (report.cached, report.called) == (1, 1)  # run 1 reused, run 2 new


def test_budget_waits_when_the_minute_is_full():
    clock = FakeClock()
    budget = TokenBudget(2500, clock=clock, sleep=clock.sleep)

    budget.wait_for(1100)
    budget.record(1100)  # t=0
    clock.now = 10
    budget.wait_for(1100)
    budget.record(1100)  # t=10: 2,200 used
    clock.now = 20
    budget.wait_for(1100)  # 3,300 > 2,500: wait until the t=0 call leaves

    assert clock.slept == [40]
    assert clock.now == 60


def test_budget_does_not_wait_when_there_is_room():
    clock = FakeClock()
    budget = TokenBudget(8000, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        budget.wait_for(1100)
        budget.record(1100)
    assert clock.slept == []


def test_config_id_changes_with_what_predictions_depend_on(monkeypatch):
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "groq_reasoning_effort", "low")
    first = config_id()
    monkeypatch.setattr(settings, "groq_reasoning_effort", None)
    assert config_id() != first
    monkeypatch.setattr(settings, "classify_prompt_version", "classify-v2")
    assert "classify-v2" in config_id()
    assert "/" not in config_id()  # safe as a file name


def test_predict_uses_the_production_classify_call(use_fake_llm):
    # Same prompt and code path as the graph node, but a failure RAISES here
    # instead of becoming "unclear".
    fake = use_fake_llm(
        result=Classification(type="docs", confidence=0.8, reasoning="r")
    )
    prediction = predict(text(1), run=1)
    assert (prediction.predicted, prediction.run) == ("docs", 1)
    assert "<issue>" in fake.calls[0][1][1]  # the same delimited issue block

    use_fake_llm(error=RuntimeError("network down"))
    with pytest.raises(RuntimeError):
        predict(text(1), run=1)
