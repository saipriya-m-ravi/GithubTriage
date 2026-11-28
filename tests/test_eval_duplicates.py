"""Duplicate answer key and retrieval scoring — no network, no database."""

from collections import Counter
from datetime import UTC, datetime

import pytest

from githubtriage.evaluation.duplicates import fetch_targets, usable_pairs, with_splits
from githubtriage.evaluation.retrieval import recall_at, target_ranks

JAN, MAR, MAY = (datetime(2025, m, 1, tzinfo=UTC) for m in (1, 3, 5))


def test_usable_pairs_apply_every_rule_and_count_exclusions():
    duplicates = {10: MAR, 11: MAR, 12: MAR, 13: MAR, 14: MAR}
    targets = {
        10: {"type": "Issue", "number": 1},  # usable: earlier, indexed
        11: None,  # no target recorded
        12: {"type": "PullRequest", "number": 2},
        13: {"type": "Issue", "number": 999},  # not in the index
        14: {"type": "Issue", "number": 4},  # created after the duplicate
    }
    indexed = {1: JAN, 2: JAN, 4: MAY, 10: MAR, 11: MAR, 12: MAR, 13: MAR, 14: MAR}

    pairs, excluded = usable_pairs("o/r", duplicates, targets, indexed)

    assert [(p.duplicate, p.target) for p in pairs] == [(10, 1)]
    assert excluded == Counter(
        {
            "no target recorded": 1,
            "target is a PullRequest": 1,
            "target not in the index (older than the V2 era)": 1,
            "target created after the duplicate": 1,
        }
    )


def pair_list(n):
    pairs, _ = usable_pairs(
        "o/r",
        {i: MAR for i in range(100, 100 + n)},
        {i: {"type": "Issue", "number": 1} for i in range(100, 100 + n)},
        {1: JAN},
    )
    return pairs


def test_split_is_half_and_half_and_frozen():
    first = with_splits(pair_list(9))
    assert Counter(p.split for p in first) == {"dev": 4, "test": 5}  # odd → test
    grown = with_splits(pair_list(15), first)
    kept = {p.duplicate: p.split for p in grown if p.duplicate < 109}
    assert kept == {p.duplicate: p.split for p in first}  # never moved


class FakeGraphQL:
    def __init__(self):
        self.queries = []

    def graphql_query(self, query, variables):
        self.queries.append((query, variables))
        names = [
            part.split(":")[0].strip() for part in query.split("i")[1:] if ":" in part
        ]
        repo = {}
        for name in names:
            if name.isdigit():
                n = int(name)
                repo[f"i{n}"] = {
                    "duplicateOf": {"__typename": "Issue", "number": n - 100}
                }
        return {"repository": repo}


def test_fetch_targets_batches_and_uses_variables():
    fake = FakeGraphQL()
    found = fetch_targets(fake, "o/r", range(200, 290))  # 90 issues → 3 batches of 40
    assert len(fake.queries) == 3
    assert fake.queries[0][1] == {"owner": "o", "name": "r"}  # never formatted in
    assert found[200] == {"type": "Issue", "number": 100}


def test_target_ranks_and_recall():
    keyword = [5, 9, 1]  # target 1 is 3rd here
    vector = [1, 7]  # and 1st here
    r = target_ranks(50, 1, keyword, vector)
    assert (r.keyword, r.vector) == (3, 1)
    assert r.hybrid == 1  # in both lists: RRF puts it on top
    misses = target_ranks(51, 42, keyword, vector)
    assert (misses.keyword, misses.vector, misses.hybrid) == (None, None, None)
    assert recall_at([r, misses], "hybrid", 5) == pytest.approx(0.5)
    assert recall_at([r, misses], "keyword", 2) == 0.0
    assert recall_at([], "vector", 5) is None
