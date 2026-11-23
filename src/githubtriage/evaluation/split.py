"""Dev/test split of the eval set (slice 4 module 10).

dev:  tune prompts, compare models, read individual errors — as often as needed.
test: the final measurement only. No decision may be based on it; otherwise it
      measures how well we fitted these issues, not how well we'll do on new ones.

How an issue's split is decided (decisions 2026-11-11):
- Stratified: issues are grouped by (type, trust, relabelled) and EACH group is
  cut in half, so dev and test have the same mix — a plain "hash < 0.5" rule
  is only 50/50 on average, and a 57-issue class can easily come out 22/35.
- Within a group, the order is a hash of the issue id: fixed and effectively
  random. (The API's order would turn the cut into a hidden time split.)
- An odd group gives its extra issue to test.
- FROZEN: an issue that already has a split keeps it. Cutting a group in half
  again after adding issues would move the cut point and slide test issues into
  dev; so only new issues are assigned, stratified among themselves.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from githubtriage.evaluation.dataset import ManifestRow

Split = Literal["dev", "test"]


def split_key(repo: str, number: int) -> str:
    """A fixed, effectively random sort key for one issue."""
    return hashlib.sha256(f"{repo}#{number}".encode()).hexdigest()


def stratum(row: ManifestRow) -> tuple:
    return (row.truth.type, row.truth.trust, row.truth.relabelled)


def with_splits(
    rows: Iterable[ManifestRow], previous: Iterable[ManifestRow] = ()
) -> list[ManifestRow]:
    """`rows` with a split on every row: kept from `previous`, else newly assigned."""
    known = {(r.repo, r.number): r.split for r in previous if r.split is not None}
    out, unassigned = [], defaultdict(list)
    for row in rows:
        split = row.split or known.get((row.repo, row.number))
        if split is None:
            unassigned[stratum(row)].append(row)
        else:
            out.append(row.model_copy(update={"split": split}))
    for group in unassigned.values():
        group.sort(key=lambda r: split_key(r.repo, r.number))
        half = len(group) // 2  # odd group: the extra issue goes to test
        out += [r.model_copy(update={"split": "dev"}) for r in group[:half]]
        out += [r.model_copy(update={"split": "test"}) for r in group[half:]]
    return sorted(out, key=lambda r: (r.truth.type, r.number))
