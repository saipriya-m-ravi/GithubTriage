"""Answer key for the duplicate finder (slice 5, retrieval eval).

Source of truth: GitHub's `duplicateOf`, set when a maintainer uses "Close as
duplicate of…" — a structured record of a human decision. (Investigated
2026-11-28: the REST API only says state_reason="duplicate", without a target;
comments name targets inconsistently and sometimes point at PRs. Only the
GraphQL API exposes duplicateOf.)

A pair is USABLE only if all hold:
- a target was recorded, and it is an Issue (not a pull request);
- the target is in our index (pydantic V2 era, created since 2023-07-01);
- the target was created BEFORE the duplicate — maintainers sometimes close the
  older issue as a duplicate of a newer one (the discussion moved there); with
  the time-travel rule no search could find it, so it can't count as a miss.
Every exclusion is counted and reported, never dropped silently.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel

Split = Literal["dev", "test"]
BATCH = 40  # issues per GraphQL query (aliases), well under GitHub's limits


class DuplicatePair(BaseModel):
    repo: str
    duplicate: int  # the issue that was closed as a duplicate (the query)
    target: int  # the issue it duplicates (the right answer)
    duplicate_created_at: AwareDatetime
    split: Split | None = None


def fetch_targets(gh, repo: str, numbers: Iterable[int]) -> dict[int, dict | None]:
    """number -> {"type": ..., "number": ...} of its duplicateOf, or None."""
    owner, name = repo.split("/")
    numbers = list(numbers)
    out: dict[int, dict | None] = {}
    for start in range(0, len(numbers), BATCH):
        chunk = numbers[start : start + BATCH]
        fields = " ".join(
            f"i{n}: issue(number: {int(n)}) {{ duplicateOf {{ __typename number }} }}"
            for n in chunk
        )
        data = gh.graphql_query(
            "query($owner: String!, $name: String!) {"
            f" repository(owner: $owner, name: $name) {{ {fields} }} }}",
            {"owner": owner, "name": name},
        )
        repo_data = data["repository"]
        for n in chunk:
            dup_of = (repo_data.get(f"i{n}") or {}).get("duplicateOf")
            out[n] = (
                {"type": dup_of["__typename"], "number": dup_of["number"]}
                if dup_of
                else None
            )
    return out


def usable_pairs(
    repo: str,
    duplicates: dict[int, datetime],  # duplicate-closed issue -> created_at
    targets: dict[int, dict | None],
    indexed: dict[int, datetime],  # every issue in the index -> created_at
) -> tuple[list[DuplicatePair], Counter]:
    pairs, excluded = [], Counter()
    for number, created in sorted(duplicates.items()):
        t = targets.get(number)
        if t is None:
            excluded["no target recorded"] += 1
        elif t["type"] != "Issue":
            excluded[f"target is a {t['type']}"] += 1
        elif t["number"] not in indexed:
            excluded["target not in the index (older than the V2 era)"] += 1
        elif indexed[t["number"]] >= created:
            excluded["target created after the duplicate"] += 1
        else:
            pairs.append(
                DuplicatePair(
                    repo=repo,
                    duplicate=number,
                    target=t["number"],
                    duplicate_created_at=created,
                )
            )
    return pairs, excluded


def _order(pair: DuplicatePair) -> str:
    # Its own salt: never reuse another split's hash order (see completeness_labels).
    key = f"duplicates:{pair.repo}#{pair.duplicate}"
    return hashlib.sha256(key.encode()).hexdigest()


def with_splits(
    pairs: list[DuplicatePair], previous: Iterable[DuplicatePair] = ()
) -> list[DuplicatePair]:
    """Frozen 50/50 split: known pairs keep theirs; new pairs are split among
    themselves in a fixed random order (an odd one out goes to test)."""
    known = {(p.repo, p.duplicate): p.split for p in previous if p.split}
    kept, new = [], []
    for p in pairs:
        split = known.get((p.repo, p.duplicate))
        (kept if split else new).append(p.model_copy(update={"split": split}))
    new.sort(key=_order)
    half = len(new) // 2
    out = kept + [p.model_copy(update={"split": "dev"}) for p in new[:half]]
    out += [p.model_copy(update={"split": "test"}) for p in new[half:]]
    return sorted(out, key=lambda p: p.duplicate)
