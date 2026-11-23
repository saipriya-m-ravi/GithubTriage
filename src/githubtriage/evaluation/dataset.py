"""The eval dataset: which issues, their answer key, and their text (slice 4 module 9).

Two files, kept apart on purpose (D3):
- the MANIFEST (committed): issue numbers + answer key, no text. Small,
  reviewable in git, and anyone can rebuild the texts from it.
- the TEXTS (local, gitignored under DATA_DIR): titles and bodies. They're other
  people's writing — public on GitHub, but not ours to republish in bulk.
Authors are in neither: the classifier doesn't need them.
"""

import json
import logging
import os
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel

from githubtriage.evaluation.labels import (
    LABEL_TO_TYPE,
    Excluded,
    GroundTruth,
    IssueType,
    ground_truth,
)
from githubtriage.github_client import GitHubClient, TruncatedError

logger = logging.getLogger(__name__)


class ManifestRow(BaseModel):
    """One eval issue and its answer key. Never shown to the model."""

    repo: str
    number: int
    created_at: AwareDatetime
    truth: GroundTruth
    split: Literal["dev", "test"] | None = None  # see evaluation/split.py; frozen


class IssueText(BaseModel):
    """What the classifier sees — the same fields it gets at triage time."""

    repo: str
    number: int
    title: str
    body: str


class Collected(BaseModel):
    rows: list[ManifestRow]
    texts: list[IssueText]
    excluded: dict[str, int]  # "label: reason" -> count, for the summary


def cached_events(
    gh: GitHubClient, repo: str, number: int, cache_dir: Path | None
) -> list[dict]:
    """An issue's events, from the on-disk cache if fetched before.

    A backfill makes hundreds of requests; without this, a network drop near
    the end loses everything (it happened on the first real run). Closed issues'
    histories are settled, so a cached copy stays valid. Written atomically, so
    a crash mid-write never leaves a half file that would be trusted later.
    """
    if cache_dir is None:
        return list(gh.list_issue_events(repo, number))
    path = cache_dir / f"{repo.replace('/', '__')}#{number}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    events = list(gh.list_issue_events(repo, number))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(events), encoding="utf-8")
    os.replace(tmp, path)
    return events


def collect(
    gh: GitHubClient,
    repo: str,
    *,
    created_since: datetime,
    per_type: int,
    labels: Iterable[str] = LABEL_TO_TYPE,
    cache_dir: Path | None = None,
) -> Collected:
    """The newest `per_type` usable issues per label, created since `created_since`.

    Closed issues only: their labels are settled. GitHub lists newest-created
    first, so paging stops at the first issue older than `created_since`.
    cache_dir: keep fetched event histories, so a failed run resumes cheaply.
    """
    rows, texts, excluded = [], [], Counter()
    for label in labels:
        kept = 0
        for issue in gh.list_issues(repo, state="closed", labels=label, max_pages=20):
            created = datetime.fromisoformat(issue["created_at"])
            if created < created_since:
                break
            try:
                events = cached_events(gh, repo, issue["number"], cache_dir)
                truth = ground_truth(issue, events)
            except TruncatedError:  # can't see the whole history: don't guess
                truth = Excluded(reason="event history too long")
            if isinstance(truth, Excluded):
                excluded[f"{label}: {truth.reason}"] += 1
                continue
            rows.append(
                ManifestRow(
                    repo=repo, number=issue["number"], created_at=created, truth=truth
                )
            )
            texts.append(
                IssueText(
                    repo=repo,
                    number=issue["number"],
                    title=issue["title"],
                    body=issue.get("body") or "",
                )
            )
            kept += 1
            if kept == per_type:
                break
        if kept < per_type:
            logger.warning(
                "%s: only %s usable issues since %s (wanted %s)",
                label,
                kept,
                created_since.date(),
                per_type,
            )
    return Collected(rows=rows, texts=texts, excluded=dict(excluded))


def natural_frequencies(
    gh: GitHubClient, repo: str, *, created_since: datetime
) -> dict[IssueType, int]:
    """How often each type really occurs (closed issues with each label), for a
    frequency-weighted accuracy next to the balanced one (D2)."""
    day = created_since.date().isoformat()
    return {
        LABEL_TO_TYPE[label]: gh.count_issues(
            f'repo:{repo} is:issue is:closed label:"{label}" created:>={day}'
        )
        for label in LABEL_TO_TYPE
    }
