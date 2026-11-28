"""Hybrid search: RRF by hand, and the time-travel rule against a real database.

The database tests are opt-in (GITHUBTRIAGE_DB_TESTS=1) and run inside a
transaction that always rolls back.
"""

import os
from datetime import UTC, datetime

import pytest

from githubtriage.issue_index import EMBED_DIMS, IssueRecord, index_issues
from githubtriage.search import rrf, search_similar


def test_rrf_by_hand():
    scores = rrf([[10, 20, 30], [20, 40]], k=60)
    assert scores[20] == pytest.approx(1 / 62 + 1 / 61)  # 2nd in one, 1st in other
    assert scores[10] == pytest.approx(1 / 61)
    assert scores[40] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == 20  # good in both beats top in one


def test_rrf_favours_consensus_over_a_single_top_rank():
    # A property to KNOW, not a goal: with k=60, rank gaps are small (1/61 vs
    # 1/82), so an issue in BOTH lists — even mid-list (22nd and 31st) — beats
    # one that is 1st in a single list. (My first version of this test claimed
    # the opposite; the arithmetic disagreed.) An exact-error-name match found
    # only by keyword search can therefore lose: recall@5 (module 8) shows
    # whether that happens on real data; tune k or weights on dev only.
    one_list_top = [1] + list(range(100, 120)) + [7]  # 7 is 22nd here
    other_list = list(range(200, 230)) + [7]  # and 31st here
    scores = rrf([one_list_top, other_list])
    assert scores[7] > scores[1]


def test_rrf_ignores_score_scales():
    # The reason RRF exists: only positions count, so an unbounded keyword
    # score can't swamp a 0-1 similarity — there are no scores to compare.
    assert rrf([[5, 6]]) == rrf([[5, 6]], k=60)
    assert rrf([[5, 6], [6, 5]])[5] == pytest.approx(rrf([[5, 6], [6, 5]])[6])


def test_naive_before_is_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        search_similar(
            None,
            None,
            repo="o/r",
            text="x",
            before=datetime(2025, 1, 1),  # noqa: DTZ001 (naive on purpose)
            exclude_number=1,
        )


# --- Against a real database (opt-in) ---------------------------------------------

needs_db = pytest.mark.skipif(
    os.environ.get("GITHUBTRIAGE_DB_TESTS") != "1",
    reason="needs a running database: docker compose up -d, GITHUBTRIAGE_DB_TESTS=1",
)

TOPICS = ("discriminator", "serializer", "datetime")


class TopicEmbedder:
    """Fake embeddings: one axis per topic word, so 'similar meaning' is controlled."""

    model = "fake-topics"

    def embed(self, texts):
        out = []
        for text in texts:
            v = [0.0] * EMBED_DIMS
            for axis, topic in enumerate(TOPICS):
                if topic in text.lower():
                    v[axis] = 1.0
            v[-1] = 0.01  # never an all-zero vector
            out.append(v)
        return out


def issue(number, title, body, created):
    return IssueRecord(
        repo="t/search",
        number=number,
        title=title,
        body=body,
        state="closed",
        state_reason=None,
        labels=[],
        html_url=f"https://github.com/t/search/issues/{number}",
        created_at=created,
        updated_at=created,
        closed_at=None,
    )


JAN, MAR, MAY = (datetime(2025, m, 1, tzinfo=UTC) for m in (1, 3, 5))


@pytest.fixture
def indexed_conn():
    from githubtriage.db import connect, migrate

    migrate()
    with connect() as conn, conn.transaction(force_rollback=True):
        index_issues(
            conn,
            [
                issue(
                    1,
                    "Discriminator fails on nested union",
                    "KeyError when parsing",
                    JAN,
                ),
                issue(
                    2, "Custom serializer ignored", "model_serializer not called", JAN
                ),
                issue(3, "Datetime parsing off by one day", "timezone issue", JAN),
                # Same topic as #1 but created LATER than the query below:
                issue(
                    4, "Discriminator broken again", "KeyError with nested union", MAY
                ),
            ],
            TopicEmbedder(),
        )
        yield conn


@needs_db
def test_finds_the_similar_issue_first(indexed_conn):
    found = search_similar(
        indexed_conn,
        TopicEmbedder(),
        repo="t/search",
        text="Discriminator raises KeyError on a nested union",
        before=MAR,
        exclude_number=99,
    )
    assert found[0].number == 1
    assert found[0].keyword_rank == 1 and found[0].vector_rank == 1


@needs_db
def test_never_returns_issues_from_the_future(indexed_conn):
    # #4 is the best match of all — but it was created after the query issue.
    found = search_similar(
        indexed_conn,
        TopicEmbedder(),
        repo="t/search",
        text="Discriminator broken again: KeyError with nested union",
        before=MAR,
        exclude_number=99,
    )
    assert 4 not in [c.number for c in found]


@needs_db
def test_never_returns_the_issue_itself(indexed_conn):
    found = search_similar(
        indexed_conn,
        TopicEmbedder(),
        repo="t/search",
        text="Discriminator broken again: KeyError with nested union",
        before=datetime(2026, 1, 1, tzinfo=UTC),  # everything is in the past
        exclude_number=4,
    )
    assert 4 not in [c.number for c in found]
    assert found[0].number == 1


@needs_db
def test_common_words_are_measured_and_left_out(indexed_conn):
    from githubtriage.search import common_words, ranked_lists

    # "nest" (nested) appears in 2 of the 4 issues; "union" in 2; "custom" in 1.
    common = common_words(indexed_conn, "t/search", max_df=0.4)
    assert "nest" in common and "custom" not in common

    # A query made ONLY of common words now has no keyword query at all.
    keyword, _ = ranked_lists(
        indexed_conn,
        TopicEmbedder(),
        repo="t/search",
        text="nested union",
        before=datetime(2026, 1, 1, tzinfo=UTC),
        exclude_number=99,
        common=common,
    )
    assert keyword == []


@needs_db
def test_a_long_issue_still_matches_by_its_words(indexed_conn):
    # Plain query parsers AND every word: a long issue would match nothing.
    long_text = "Custom serializer ignored. " + "Lots of unrelated words here. " * 50
    found = search_similar(
        indexed_conn,
        TopicEmbedder(),
        repo="t/search",
        text=long_text,
        before=MAR,
        exclude_number=99,
    )
    assert found[0].number == 2
    assert found[0].keyword_rank is not None
