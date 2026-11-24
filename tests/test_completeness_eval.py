"""R3 eval scoring, checked by hand, plus the production-call path."""

import pytest

from githubtriage.evaluation.completeness_eval import (
    CompletenessPrediction,
    predict,
    reports_with_a_false_nag,
    score_all,
    score_field,
)
from githubtriage.evaluation.completeness_labels import CompletenessLabel
from githubtriage.evaluation.dataset import IssueText
from githubtriage.models import Completeness, FieldCheck


def label(n, version="ok", repro_steps="ok", logs="not_needed", unsure=False):
    return CompletenessLabel(
        repo="o/r",
        number=n,
        split="dev",
        version=version,
        repro_steps=repro_steps,
        logs=logs,
        unsure=unsure,
    )


def pred(n, version="ok", repro_steps="ok", logs="not_needed"):
    return CompletenessPrediction(
        repo="o/r",
        number=n,
        run=1,
        version=version,
        repro_steps=repro_steps,
        logs=logs,
        input_tokens=1,
        output_tokens=1,
        latency_s=0.1,
    )


PAIRS = [
    (label(1), pred(1)),  # all agree
    (label(2), pred(2, logs="missing")),  # false nag: logs not needed, R3 asks
    (label(3, version="missing"), pred(3, version="missing")),  # correct ask
    (label(4, version="vague"), pred(4, version="ok")),  # missed ask
    (label(5, logs="ok"), pred(5, logs="vague")),  # false nag: logs given, R3 asks
]


def test_ask_view_counts_false_nags_and_missed_asks():
    logs = score_field(PAIRS, "logs")
    assert (logs.false_nags, logs.missed_asks, logs.should_ask) == (2, 0, 0)
    assert logs.ask_precision == 0  # asked twice, never rightly
    assert logs.ask_recall is None  # nothing should have been asked

    version = score_field(PAIRS, "version")
    assert (version.correct_asks, version.missed_asks, version.false_nags) == (1, 1, 0)
    assert version.ask_precision == 1.0
    assert version.ask_recall == pytest.approx(0.5)


def test_exact_view_and_confusion():
    logs = score_field(PAIRS, "logs")
    assert logs.exact == 3  # 1, 3, 4 agree on not_needed
    assert logs.confusion["not_needed"]["missing"] == 1
    assert logs.confusion["ok"]["vague"] == 1


def test_vague_counts_as_asking_both_ways():
    # The draft asks for missing AND vague: missing↔vague disagree exactly,
    # but the reply is the same, so it's neither a false nag nor a missed ask.
    s = score_field(
        [(label(1, version="vague"), pred(1, version="missing"))], "version"
    )
    assert (s.exact, s.correct_asks, s.false_nags, s.missed_asks) == (0, 1, 0, 0)


def test_reports_with_a_false_nag_counts_issues_not_fields():
    nagged = [(label(1), pred(1, version="missing", logs="missing"))]  # 2 fields
    assert reports_with_a_false_nag(nagged + PAIRS) == 3  # #1 here, #2, #5


def test_score_all_covers_every_field():
    assert set(score_all(PAIRS)) == {"version", "repro_steps", "logs"}


def test_predict_uses_the_production_completeness_call(use_fake_llm):
    ok = FieldCheck(status="ok", note="")
    fake = use_fake_llm(
        result=Completeness(
            version=ok,
            repro_steps=FieldCheck(status="vague", note="add code"),
            logs=FieldCheck(status="not_needed", note=""),
        )
    )
    p = predict(IssueText(repo="o/r", number=7, title="t", body="b"), run=1)
    assert (p.version, p.repro_steps, p.logs) == ("ok", "vague", "not_needed")
    assert "<issue>" in fake.calls[0][1][1]  # the same delimited issue block
