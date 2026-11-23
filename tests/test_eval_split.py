"""The dev/test split: stratified, deterministic, and frozen once assigned."""

import random
from collections import Counter
from datetime import UTC, datetime

from githubtriage.evaluation.dataset import ManifestRow
from githubtriage.evaluation.labels import GroundTruth
from githubtriage.evaluation.split import with_splits

CREATED = datetime(2025, 1, 1, tzinfo=UTC)


def row(number, type_="bug", trust="gold", relabelled=False, split=None):
    return ManifestRow(
        repo="o/r",
        number=number,
        created_at=CREATED,
        truth=GroundTruth(type=type_, label="x", trust=trust, relabelled=relabelled),
        split=split,
    )


def splits(rows):
    return {r.number: r.split for r in rows}


def test_every_stratum_is_cut_exactly_in_half():
    rows = (
        [row(n, "docs") for n in range(57)]  # odd: the extra issue goes to test
        + [row(100 + n, "question", relabelled=True) for n in range(56)]
        + [row(200 + n, "question") for n in range(4)]
        + [row(300 + n, "feature", trust="silver") for n in range(41)]
    )
    counts = Counter(
        (r.truth.type, r.truth.trust, r.truth.relabelled, r.split)
        for r in with_splits(rows)
    )

    assert counts[("docs", "gold", False, "dev")] == 28
    assert counts[("docs", "gold", False, "test")] == 29
    assert counts[("question", "gold", True, "dev")] == 28
    assert counts[("question", "gold", False, "dev")] == 2  # small strata too
    assert counts[("feature", "silver", False, "dev")] == 20


def test_split_does_not_depend_on_input_order():
    # The API's newest-first order must not leak in (a hidden time split).
    rows = [row(n) for n in range(40)]
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)

    assert splits(with_splits(rows)) == splits(with_splits(shuffled))


def test_same_issues_always_get_the_same_split():
    rows = [row(n) for n in range(40)]
    assert splits(with_splits(rows)) == splits(with_splits(rows))


def test_existing_splits_are_frozen_when_issues_are_added():
    # Re-cutting the group after adding issues would move the cut point and
    # slide test issues into dev; assigned issues must never move.
    first = with_splits([row(n) for n in range(10)])
    grown = with_splits(
        [row(n) for n in range(10)] + [row(n) for n in range(10, 30)], first
    )

    assert {n: s for n, s in splits(grown).items() if n < 10} == splits(first)
    new = Counter(s for n, s in splits(grown).items() if n >= 10)
    assert new == {"dev": 10, "test": 10}  # new issues stratified among themselves


def test_a_split_already_on_the_row_is_kept():
    rows = [row(1, split="test"), row(2, split="test")]
    assert splits(with_splits(rows)) == {1: "test", 2: "test"}


def test_an_issue_that_left_the_dataset_is_dropped():
    first = with_splits([row(n) for n in range(4)])
    assert splits(with_splits([row(0), row(1)], first)).keys() == {0, 1}
