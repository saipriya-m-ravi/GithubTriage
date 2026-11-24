"""Run the classifier over eval issues (slice 4 module 11).

Built around Groq's limits on this account (2026-11-11): 30 requests and 8,000
tokens per MINUTE, 1,000 requests and 200,000 tokens per DAY — at ~1,150 tokens
per call that's ~7 calls a minute and ~170 a day. So:
- every prediction is CACHED on disk, keyed by (configuration, issue, run):
  a stopped run resumes where it stopped, re-scoring costs no calls, and a
  configuration already measured is never paid for twice;
- calls are PACED to stay under the per-minute token budget instead of
  colliding with it (parallel calls would only turn into 429s);
- a rate-limit error after the client's own retries STOPS the run cleanly
  (e.g. the daily budget is spent) — it's resumed later, not scored;
- other failures are reported separately and never scored as "unclear":
  the graph node's fallback to "unclear" is right for the product, but in an
  eval it would count a network error as the model choosing to abstain.
"""

import logging
import re
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from langchain_core.callbacks import UsageMetadataCallbackHandler
from pydantic import BaseModel, ValidationError

from githubtriage.config import settings
from githubtriage.evaluation.dataset import IssueText
from githubtriage.nodes import classify as classify_node

logger = logging.getLogger(__name__)

Predicted = Literal["bug", "feature", "question", "docs", "unclear"]


class Prediction(BaseModel):
    repo: str
    number: int
    run: int
    predicted: Predicted
    confidence: float
    input_tokens: int
    output_tokens: int
    latency_s: float


class RateLimited(Exception):
    """The provider refused after its own retries — stop now, resume later."""


def config_id() -> str:
    """What a prediction depends on: model, reasoning effort, prompt version.

    Changing any of them is a new configuration with its own cache, so results
    from different setups can never be mixed up.
    """
    effort = settings.groq_reasoning_effort or "default"
    raw = f"{settings.groq_model}__{effort}__{classify_node.prompt_version()}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw)


class PredictionStore:
    """Append-only JSONL cache of predictions for one configuration.

    One complete line is written per prediction; a crash can at worst leave a
    half line at the end, which is skipped on load (and that call is redone).
    """

    def __init__(self, path: Path, model: type[BaseModel] = Prediction):
        # `model`: any prediction type with repo, number and run (R1's Prediction
        # by default; R3's eval stores its own).
        self.path = path
        self._by_key: dict[tuple[str, int, int], Prediction] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    p = model.model_validate_json(line)
                except ValidationError:
                    logger.warning("Skipping a damaged line in %s", path)
                    continue
                self._by_key[(p.repo, p.number, p.run)] = p

    def get(self, repo: str, number: int, run: int) -> Prediction | None:
        return self._by_key.get((repo, number, run))

    def add(self, prediction: Prediction) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(prediction.model_dump_json() + "\n")
        self._by_key[(prediction.repo, prediction.number, prediction.run)] = prediction

    def all(self) -> list[Prediction]:
        return list(self._by_key.values())


class TokenBudget:
    """Keeps the tokens used in any 60-second window under `per_minute`.

    Before a call we wait until the window has room for an ESTIMATE of the
    call; afterwards the estimate is replaced by the real count.
    """

    def __init__(
        self,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.per_minute = per_minute
        self._clock = clock
        self._sleep = sleep
        self._window: deque[tuple[float, int]] = deque()  # (time, tokens)

    def _used(self) -> int:
        cutoff = self._clock() - 60
        while self._window and self._window[0][0] <= cutoff:
            self._window.popleft()
        return sum(tokens for _, tokens in self._window)

    def wait_for(self, estimate: int) -> None:
        while self._window and self._used() + estimate > self.per_minute:
            # Sleep until the oldest call leaves the window.
            self._sleep(max(self._window[0][0] + 60 - self._clock(), 0.1))

    def record(self, tokens: int) -> None:
        self._window.append((self._clock(), tokens))


def estimate_tokens(text: IssueText) -> int:
    """Rough upper estimate: prompt + issue (~3 chars/token for code-heavy
    text, capped like build_issue_block) + output."""
    chars = len(text.title) + min(len(text.body), 8000)
    return 450 + chars // 3 + 250


def predict(text: IssueText, run: int) -> Prediction:
    """One real classification of one issue, with tokens and latency measured."""
    usage = UsageMetadataCallbackHandler()
    started = time.monotonic()
    result = classify_node.classify_issue(text.title, text.body, callbacks=[usage])
    latency = time.monotonic() - started
    totals = list(usage.usage_metadata.values())
    return Prediction(
        repo=text.repo,
        number=text.number,
        run=run,
        predicted=result.type,
        confidence=result.confidence,
        input_tokens=sum(u.get("input_tokens", 0) for u in totals),
        output_tokens=sum(u.get("output_tokens", 0) for u in totals),
        latency_s=round(latency, 3),
    )


def is_rate_limit(error: Exception) -> bool:
    return (
        type(error).__name__ == "RateLimitError" or "rate limit" in str(error).lower()
    )


class RunReport(BaseModel):
    called: int = 0
    cached: int = 0
    failures: list[str] = []  # "repo#number run k: ErrorType" — not scored
    stopped: str | None = None  # why the run stopped early, if it did


def run_predictions(
    texts: Iterable[IssueText],
    runs: int,
    store: PredictionStore,
    *,
    budget: TokenBudget,
    max_calls: int | None = None,
    predict_fn: Callable[[IssueText, int], Prediction] = predict,
) -> RunReport:
    """Make sure every (issue, run) has a prediction in `store`, within limits."""
    report = RunReport()
    texts = list(texts)
    for run in range(1, runs + 1):
        for text in texts:
            if store.get(text.repo, text.number, run):
                report.cached += 1
                continue
            if max_calls is not None and report.called >= max_calls:
                report.stopped = f"--max-calls {max_calls} reached"
                return report
            budget.wait_for(estimate_tokens(text))
            try:
                prediction = predict_fn(text, run)
            # Any failure is reported (or stops the run), never crashes the eval
            # or gets scored: catching everything is the point here.
            except Exception as e:  # noqa: BLE001
                budget.record(estimate_tokens(text))  # it may still have counted
                if is_rate_limit(e):
                    report.stopped = f"rate limited ({type(e).__name__}): resume later"
                    return report
                logger.warning(
                    "Prediction failed for %s#%s: %s", text.repo, text.number, e
                )
                report.failures.append(
                    f"{text.repo}#{text.number} run {run}: {type(e).__name__}"
                )
                continue
            budget.record(prediction.input_tokens + prediction.output_tokens)
            store.add(prediction)
            report.called += 1
    return report
