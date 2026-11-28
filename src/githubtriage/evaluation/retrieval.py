"""Scoring Stage 1 (retrieval) on duplicate pairs: recall@k per variant.

recall@k = the share of duplicates whose true target is among the top k
candidates. If the target isn't retrieved, Stage 2 can never pick it — so this
bounds the whole duplicate finder.

Three variants from the same two ranked lists, so each half's contribution is
visible: keyword only, vector only, and the RRF fusion actually used.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from githubtriage.search import fuse

VARIANTS = ("keyword", "vector", "hybrid")


class Ranks(BaseModel):
    """Where the true target landed in each list (1-based), or None if absent."""

    duplicate: int
    target: int
    keyword: int | None
    vector: int | None
    hybrid: int | None


def target_ranks(
    duplicate: int,
    target: int,
    keyword: list[int],
    vector: list[int],
    *,
    depth: int = 50,
) -> Ranks:
    def rank(ranked: list[int]) -> int | None:
        return ranked.index(target) + 1 if target in ranked else None

    return Ranks(
        duplicate=duplicate,
        target=target,
        keyword=rank(keyword),
        vector=rank(vector),
        hybrid=rank(fuse(keyword, vector, depth)),
    )


def recall_at(ranks: Sequence[Ranks], variant: str, k: int) -> float | None:
    if not ranks:
        return None
    hits = sum(
        1 for r in ranks if (pos := getattr(r, variant)) is not None and pos <= k
    )
    return hits / len(ranks)
