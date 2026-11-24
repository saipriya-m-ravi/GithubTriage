"""Evaluating R3 completeness against the hand labels (slice 4 module 12d).

Two views per field (version, repro_steps, logs):
- EXACT: does R3's status equal the label (missing / vague / ok / not_needed)?
- ASK: what the reply actually does. The draft asks the reporter for a field
  when its status is missing or vague (draft.REQUEST_STATUSES). So:
    false nag  = R3 asks, but the label says it's there (ok / not_needed)
    missed ask = R3 doesn't ask, but the label says it's missing or vague
  Both are costly in different ways: a false nag annoys a reporter who did their
  part (and the maintainer edits it out); a missed ask leaves the issue stuck.

Rows the labeller marked `unsure` are scored, but also reported separately, so
a borderline label can't silently decide a result.
"""

import re
import time
from collections import Counter

from langchain_core.callbacks import UsageMetadataCallbackHandler
from pydantic import BaseModel

from githubtriage.config import settings
from githubtriage.evaluation.completeness_labels import FIELDS, CompletenessLabel
from githubtriage.evaluation.dataset import IssueText
from githubtriage.models import FieldStatus
from githubtriage.nodes import completeness as completeness_node
from githubtriage.nodes.draft import REQUEST_STATUSES


class CompletenessPrediction(BaseModel):
    repo: str
    number: int
    run: int
    version: FieldStatus
    repro_steps: FieldStatus
    logs: FieldStatus
    input_tokens: int
    output_tokens: int
    latency_s: float


def config_id() -> str:
    """model__effort__completeness-prompt-version — its own cache per setup."""
    effort = settings.groq_reasoning_effort or "default"
    raw = f"{settings.groq_model}__{effort}__{completeness_node.prompt_version()}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw)


def predict(text: IssueText, run: int) -> CompletenessPrediction:
    """One real R3 call on one issue (the production prompt), tokens measured."""
    usage = UsageMetadataCallbackHandler()
    started = time.monotonic()
    result = completeness_node.check_completeness(
        text.title, text.body, callbacks=[usage]
    )
    latency = time.monotonic() - started
    totals = list(usage.usage_metadata.values())
    return CompletenessPrediction(
        repo=text.repo,
        number=text.number,
        run=run,
        version=result.version.status,
        repro_steps=result.repro_steps.status,
        logs=result.logs.status,
        input_tokens=sum(u.get("input_tokens", 0) for u in totals),
        output_tokens=sum(u.get("output_tokens", 0) for u in totals),
        latency_s=round(latency, 3),
    )


def asks(status: str) -> bool:
    return status in REQUEST_STATUSES


class FieldScores(BaseModel):
    n: int
    exact: int  # status identical to the label
    should_ask: int  # label is missing / vague
    asked: int  # R3 said missing / vague
    correct_asks: int
    false_nags: int  # asked, but the label says it's there
    missed_asks: int  # didn't ask, but the label says it's missing / vague
    confusion: dict[str, dict[str, int]]  # label -> R3 status -> count

    @property
    def ask_precision(self) -> float | None:
        return self.correct_asks / self.asked if self.asked else None

    @property
    def ask_recall(self) -> float | None:
        return self.correct_asks / self.should_ask if self.should_ask else None


STATUSES = ("missing", "vague", "ok", "not_needed")


def score_field(
    pairs: list[tuple[CompletenessLabel, CompletenessPrediction]], field: str
) -> FieldScores:
    labels = [getattr(label, field) for label, _ in pairs]
    preds = [getattr(pred, field) for _, pred in pairs]
    counts = Counter(zip(labels, preds, strict=True))
    return FieldScores(
        n=len(pairs),
        exact=sum(lab == pred for lab, pred in zip(labels, preds, strict=True)),
        should_ask=sum(asks(lab) for lab in labels),
        asked=sum(asks(pred) for pred in preds),
        correct_asks=sum(
            asks(lab) and asks(pred) for lab, pred in zip(labels, preds, strict=True)
        ),
        false_nags=sum(
            asks(pred) and not asks(lab)
            for lab, pred in zip(labels, preds, strict=True)
        ),
        missed_asks=sum(
            asks(lab) and not asks(pred)
            for lab, pred in zip(labels, preds, strict=True)
        ),
        confusion={lab: {p: counts[(lab, p)] for p in STATUSES} for lab in STATUSES},
    )


def score_all(
    pairs: list[tuple[CompletenessLabel, CompletenessPrediction]],
) -> dict[str, FieldScores]:
    return {field: score_field(pairs, field) for field in FIELDS}


def reports_with_a_false_nag(
    pairs: list[tuple[CompletenessLabel, CompletenessPrediction]],
) -> int:
    """Issues whose reply would ask for at least one thing that's already there —
    what a reporter actually experiences."""
    return sum(
        any(asks(getattr(pred, f)) and not asks(getattr(label, f)) for f in FIELDS)
        for label, pred in pairs
    )
