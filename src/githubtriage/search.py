"""Stage 1 of the duplicate finder: hybrid search over past issues (slice 5 module 4).

Keyword search and vector search each return their top `per_list` issues; the
two ranked lists are merged with Reciprocal Rank Fusion; the top `k` are the
candidates Stage 2 (the LLM verdict) looks at.

The time-travel rule lives INSIDE both SQL queries, before ranking:
    created_at < before  AND  number <> exclude_number
`before` is a required argument with no default — a caller can't forget it —
and is passed as a query parameter, never formatted into the SQL. Filtering
the top-k afterwards would leave fewer than k candidates and hide true
duplicates ranked just below the "future" ones.

Two subtleties:
- The query is a whole issue, not a few keywords. Postgres's usual query
  parsers AND every word, so a long issue would match nothing; the keyword
  query is the OR of the issue's stemmed words instead, ranked by which and
  how many match (title words weigh more: they were weighted 'A').
- The HNSW vector index is approximate and, by default, applies the WHERE
  clause only to the ~40 neighbours it visits — for an old query most of them
  are "in the future" and you silently get fewer results. pgvector 0.8's
  iterative scan keeps searching until enough rows pass the filter.
"""

from collections.abc import Sequence
from datetime import datetime

from pgvector.psycopg import register_vector
from pydantic import AwareDatetime, BaseModel

from githubtriage.issue_index import EMBED_MAX_CHARS, Embedder

RRF_K = 60  # the usual constant: dampens the gap between rank 1 and rank 2


class Candidate(BaseModel):
    repo: str
    number: int
    title: str
    state: str
    state_reason: str | None
    labels: list[str]
    html_url: str
    created_at: AwareDatetime
    keyword_rank: int | None  # 1-based position in the keyword list, if present
    vector_rank: int | None
    score: float  # RRF score


def rrf(ranked_lists: Sequence[Sequence[int]], *, k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: each list adds 1 / (k + rank) for each item.

    Uses positions only, so scores from different methods (keyword ranks are
    unbounded, cosine distances aren't) are never compared directly.

    Trade-off to know: with k=60 rank gaps are small (1/61 vs 1/82), so being
    in BOTH lists counts for a lot — an issue 22nd and 31st beats one that is
    1st in a single list. An exact-error-name match found only by keyword
    search can lose; recall@5 shows whether it does (tune k/weights on dev).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


KEYWORD_SQL = """
WITH q AS (
    SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS query
    FROM unnest(tsvector_to_array(to_tsvector('english', %(text)s))) AS lexeme
    WHERE lexeme <> ALL(%(common)s)  -- too common to tell issues apart
)
SELECT i.number
FROM issues i, q
WHERE i.repo = %(repo)s
  AND i.created_at < %(before)s
  AND i.number <> %(exclude)s
  AND q.query IS NOT NULL
  AND i.search @@ q.query
ORDER BY ts_rank_cd(i.search, q.query) DESC, i.number DESC
LIMIT %(limit)s
"""

VECTOR_SQL = """
SELECT i.number
FROM issue_embeddings e
JOIN issues i USING (repo, number)
WHERE e.repo = %(repo)s
  AND e.model = %(model)s
  AND i.created_at < %(before)s
  AND i.number <> %(exclude)s
ORDER BY e.embedding <=> %(vector)s::vector
LIMIT %(limit)s
"""

DETAILS_SQL = """
SELECT repo, number, title, state, state_reason, labels, html_url, created_at
FROM issues WHERE repo = %(repo)s AND number = ANY(%(numbers)s)
"""


COMMON_WORDS_SQL = """
SELECT word FROM ts_stat(
    'SELECT search FROM issues WHERE repo = ' || quote_literal(%(repo)s)
)
WHERE ndoc > %(max_df)s * (SELECT count(*) FROM issues WHERE repo = %(repo)s)
"""


def common_words(conn, repo: str, max_df: float) -> list[str]:
    """Stemmed words that appear in more than `max_df` of the repo's issues.

    Postgres's ts_rank has no IDF (it weighs every word the same, however
    common), so an OR query over a whole issue is dominated by words every
    issue shares — template headings, "version", "confirm"… — and finds
    nothing useful (dev recall@5: 0%). Leaving out the most common words is a
    data-driven stand-in for IDF; `max_df` is tuned on dev.

    Computed over the whole index, so for an old query it includes later
    issues' statistics — corpus statistics, not answers; in production the
    index only holds past issues.
    """
    rows = conn.execute(COMMON_WORDS_SQL, {"repo": repo, "max_df": max_df}).fetchall()
    return [word for (word,) in rows]


def ranked_lists(
    conn,
    embedder: Embedder,
    *,
    repo: str,
    text: str,
    before: datetime,
    exclude_number: int | None,
    per_list: int = 50,
    common: Sequence[str] = (),
) -> tuple[list[int], list[int]]:
    """(keyword ranking, vector ranking): issue numbers, best first, each list
    already filtered by the time-travel rule. The eval scores each list on its
    own and fused; search_similar fuses them."""
    if before.tzinfo is None:
        raise ValueError("`before` must be timezone-aware (created_at is timestamptz)")
    register_vector(conn)
    text = text[:EMBED_MAX_CHARS]  # what the vector side can read; same for keywords
    params = {
        "repo": repo,
        "before": before,
        "exclude": exclude_number if exclude_number is not None else -1,
        "limit": per_list,
        "text": text,
        "model": embedder.model,
        "common": list(common),
    }
    with conn.transaction():
        keyword = [n for (n,) in conn.execute(KEYWORD_SQL, params).fetchall()]
        # Keep scanning the HNSW graph until `limit` rows pass the WHERE clause.
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        params["vector"] = embedder.embed([text])[0]
        vector = [n for (n,) in conn.execute(VECTOR_SQL, params).fetchall()]
    return keyword, vector


def fuse(keyword: list[int], vector: list[int], k: int) -> list[int]:
    """The top k of the RRF merge (ties: the newer issue first)."""
    scores = rrf([keyword, vector])
    return sorted(scores, key=lambda n: (-scores[n], -n))[:k]


def search_similar(
    conn,
    embedder: Embedder,
    *,
    repo: str,
    text: str,
    before: datetime,
    exclude_number: int | None,
    k: int = 5,
    per_list: int = 50,
) -> list[Candidate]:
    """The top-k past issues most similar to `text`, created before `before`."""
    keyword, vector = ranked_lists(
        conn,
        embedder,
        repo=repo,
        text=text,
        before=before,
        exclude_number=exclude_number,
        per_list=per_list,
    )
    scores = rrf([keyword, vector])
    top = fuse(keyword, vector, k)
    if not top:
        return []
    rows = conn.execute(DETAILS_SQL, {"repo": repo, "numbers": top}).fetchall()
    by_number = {row[1]: row for row in rows}
    kw_rank = {n: i for i, n in enumerate(keyword, start=1)}
    vec_rank = {n: i for i, n in enumerate(vector, start=1)}
    return [
        Candidate(
            repo=by_number[n][0],
            number=n,
            title=by_number[n][2],
            state=by_number[n][3],
            state_reason=by_number[n][4],
            labels=by_number[n][5],
            html_url=by_number[n][6],
            created_at=by_number[n][7],
            keyword_rank=kw_rank.get(n),
            vector_rank=vec_rank.get(n),
            score=scores[n],
        )
        for n in top
    ]
