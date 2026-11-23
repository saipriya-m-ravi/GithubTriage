"""Scoring, checked against numbers computed by hand."""

import pytest

from githubtriage.evaluation.metrics import (
    agreement,
    confusion,
    ece,
    reliability,
    score,
    separation,
    spread,
)

# 10 issues: 4 bug, 2 feature, 2 question, 2 docs.
PAIRS = [
    ("bug", "bug"),
    ("bug", "bug"),
    ("bug", "bug"),
    ("bug", "unclear"),  # abstained
    ("feature", "feature"),
    ("feature", "bug"),  # wrong
    ("question", "question"),
    ("question", "unclear"),  # abstained
    ("docs", "bug"),  # wrong
    ("docs", "docs"),
]


def test_coverage_and_accuracy_on_answered():
    s = score(PAIRS)
    assert s.answered == 8
    assert s.coverage == pytest.approx(0.8)
    assert s.accuracy_answered == pytest.approx(6 / 8)


def test_per_class_precision_recall():
    bug = score(PAIRS).per_class["bug"]
    # predicted bug 5 times (3 right) ; 3 answered bugs, all right
    assert bug.precision == pytest.approx(3 / 5)
    assert bug.recall == pytest.approx(1.0)
    assert bug.coverage == pytest.approx(3 / 4)
    assert bug.f1 == pytest.approx(2 * 0.6 * 1.0 / 1.6)


def test_macro_f1_weights_classes_equally():
    s = score(PAIRS)
    f1 = {c: s.per_class[c].f1 for c in s.per_class}
    assert f1["feature"] == pytest.approx(2 * 1 * 0.5 / 1.5)  # P=1/1, R=1/2
    assert f1["question"] == pytest.approx(1.0)
    assert s.macro_f1 == pytest.approx(sum(f1.values()) / 4)


def test_always_unclear_has_no_coverage_and_no_accuracy():
    # The reason coverage is always reported next to accuracy.
    s = score([(t, "unclear") for t, _ in PAIRS])
    assert s.coverage == 0
    assert s.accuracy_answered is None
    assert s.macro_f1 is None


def test_frequency_weighted_accuracy_uses_real_mix():
    s = score(PAIRS, frequencies={"bug": 7, "feature": 1, "question": 1, "docs": 1})
    # per-class recall: bug 1.0, feature 0.5, question 1.0, docs 0.5
    assert s.weighted_accuracy == pytest.approx((7 * 1 + 0.5 + 1 + 0.5) / 10)


def test_a_never_predicted_class_scores_zero_not_missing():
    s = score([("docs", "bug"), ("bug", "bug")])
    assert s.per_class["docs"].recall == 0
    assert s.per_class["docs"].precision is None  # never predicted
    assert s.per_class["docs"].f1 is None


def test_confusion_includes_abstentions():
    m = confusion(PAIRS)
    assert m["bug"] == {"bug": 3, "feature": 0, "question": 0, "docs": 0, "unclear": 1}
    assert m["docs"]["bug"] == 1


# --- Calibration ---------------------------------------------------------------

# What dev showed (2026-11-20): almost every answer ~0.97, right or wrong.
OVERCONFIDENT = [(0.97, True)] * 13 + [(0.97, False)] * 7


def test_reliability_groups_by_confidence():
    items = [(0.6, True), (0.6, False), (0.92, True), (0.97, True), (1.0, False)]
    bins = {(b.low, b.high): b for b in reliability(items)}

    assert bins[(0.5, 0.7)].n == 2
    assert bins[(0.5, 0.7)].accuracy == pytest.approx(0.5)
    assert bins[(0.9, 0.95)].n == 1
    assert bins[(0.95, 1.0)].n == 2  # the top bin includes 1.0
    assert sum(b.n for b in bins.values()) == len(items)


def test_ece_is_zero_when_confidence_matches_accuracy():
    calibrated = [(0.75, True)] * 3 + [(0.75, False)]  # says 0.75, right 75%
    assert ece(calibrated) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    # says 0.97, right 65% → gap 0.32
    assert ece(OVERCONFIDENT) == pytest.approx(0.97 - 0.65)


def test_separation_shows_when_no_threshold_can_help():
    right, wrong = separation(OVERCONFIDENT)
    assert right == wrong  # same confidence whether right or wrong
    assert separation([(0.9, True), (0.4, False)]) == (0.9, 0.4)


def test_calibration_of_nothing_is_none():
    assert ece([]) is None
    assert separation([]) == (None, None)


def test_spread_and_agreement():
    assert spread([0.5, 0.7, None]).model_dump() == pytest.approx(
        {"mean": 0.6, "low": 0.5, "high": 0.7}
    )
    assert agreement({1: ["bug", "bug"], 2: ["bug", "docs"], 3: ["x"]}) == 0.5
    assert agreement({1: ["bug"]}) is None  # one run: stability unknown
