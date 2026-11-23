"""Scoring classifications against the answer key (slice 4 module 11). Pure functions.

`unclear` is an ABSTENTION, not an error (D4: by design a human decides), so
every score comes as two numbers that must be read together:
- coverage: the share of issues the classifier committed to a type;
- accuracy on answered: how often those committed answers were right.
"Always unclear" would have perfect accuracy on answered and zero coverage —
reading only one of the two numbers would hide that.

Per-class precision/recall/F1 and the confusion matrix use answered issues
(the matrix also shows where the abstentions came from). Macro-F1 averages the
four classes equally, so a classifier that ignores a rare class (docs) can't
hide behind the common one (bug).
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from statistics import mean

from pydantic import BaseModel

TYPES = ("bug", "feature", "question", "docs")
Pair = tuple[str, str]  # (true type, predicted type or "unclear")


class ClassScores(BaseModel):
    support: int  # issues of this type
    coverage: float | None
    precision: float | None
    recall: float | None  # among answered issues of this type
    f1: float | None


class Scores(BaseModel):
    n: int
    answered: int
    coverage: float | None
    accuracy_answered: float | None
    macro_f1: float | None
    weighted_accuracy: float | None  # per-class accuracy weighted by real frequency
    per_class: dict[str, ClassScores]


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def score(pairs: Sequence[Pair], frequencies: dict[str, int] | None = None) -> Scores:
    answered = [(t, p) for t, p in pairs if p != "unclear"]
    correct = sum(t == p for t, p in answered)
    per_class = {}
    for c in TYPES:
        support = sum(t == c for t, _ in pairs)
        answered_c = sum(t == c for t, _ in answered)
        hits = sum(t == p == c for t, p in answered)
        precision = _ratio(hits, sum(p == c for _, p in answered))
        recall = _ratio(hits, answered_c)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else (0.0 if precision is not None and recall is not None else None)
        )
        per_class[c] = ClassScores(
            support=support,
            coverage=_ratio(answered_c, support),
            precision=precision,
            recall=recall,
            f1=f1,
        )
    f1s = [s.f1 for s in per_class.values() if s.f1 is not None]
    weighted = None
    if frequencies:
        parts = [
            (frequencies[c], per_class[c].recall)
            for c in TYPES
            if per_class[c].recall is not None and frequencies.get(c)
        ]
        total = sum(w for w, _ in parts)
        weighted = sum(w * r for w, r in parts) / total if total else None
    return Scores(
        n=len(pairs),
        answered=len(answered),
        coverage=_ratio(len(answered), len(pairs)),
        accuracy_answered=_ratio(correct, len(answered)),
        macro_f1=mean(f1s) if f1s else None,
        weighted_accuracy=weighted,
        per_class=per_class,
    )


def confusion(pairs: Iterable[Pair]) -> dict[str, dict[str, int]]:
    """counts[true][predicted], predicted including "unclear"."""
    counts = Counter(pairs)
    return {t: {p: counts[(t, p)] for p in (*TYPES, "unclear")} for t in TYPES}


class Spread(BaseModel):
    mean: float | None
    low: float | None
    high: float | None


def spread(values: Iterable[float | None]) -> Spread:
    """Mean and range of one metric over repeated runs (LLM output varies)."""
    present = [v for v in values if v is not None]
    if not present:
        return Spread(mean=None, low=None, high=None)
    return Spread(mean=mean(present), low=min(present), high=max(present))


class Bin(BaseModel):
    low: float
    high: float
    n: int
    mean_confidence: float
    accuracy: float


# Fine bins near the top, where LLM confidences cluster.
EDGES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


def reliability(
    items: Sequence[tuple[float, bool]], edges: Sequence[float] = EDGES
) -> list[Bin]:
    """Group (confidence, correct) by confidence: does 0.9 mean 90% right?

    Answered predictions only — "unclear" carries no claim to be right.
    """
    bins = []
    for i, (low, high) in enumerate(pairwise(edges)):
        last = i == len(edges) - 2  # the top bin includes 1.0
        group = [(c, ok) for c, ok in items if low <= c < high or (last and c == high)]
        if group:
            bins.append(
                Bin(
                    low=low,
                    high=high,
                    n=len(group),
                    mean_confidence=mean(c for c, _ in group),
                    accuracy=mean(ok for _, ok in group),
                )
            )
    return bins


def ece(
    items: Sequence[tuple[float, bool]], edges: Sequence[float] = EDGES
) -> float | None:
    """Expected calibration error: the average |confidence - accuracy| gap over
    the bins, weighted by how many answers each holds. 0 = perfectly calibrated."""
    if not items:
        return None
    return sum(
        b.n / len(items) * abs(b.mean_confidence - b.accuracy)
        for b in reliability(items, edges)
    )


def separation(
    items: Sequence[tuple[float, bool]],
) -> tuple[float | None, float | None]:
    """Mean confidence when right vs when wrong. Calibration can be re-mapped
    afterwards; if these two are equal, no threshold can tell right from wrong."""
    right = [c for c, ok in items if ok]
    wrong = [c for c, ok in items if not ok]
    return (mean(right) if right else None, mean(wrong) if wrong else None)


def agreement(predictions_by_issue: dict[object, list[str]]) -> float | None:
    """Share of issues that got the SAME answer in every run: how stable it is."""
    multi = [ps for ps in predictions_by_issue.values() if len(ps) > 1]
    return _ratio(sum(len(set(ps)) == 1 for ps in multi), len(multi))
