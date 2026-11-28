"""Build / refresh the duplicate finder's index (needs GITHUB_TOKEN and the database).

    docker compose up -d
    uv run python debug_helper_scripts/index_issues.py
    uv run python debug_helper_scripts/index_issues.py --repo pydantic/pydantic --since 2023-07-01

Fetches every issue (open and closed, no PRs) created since --since, upserts it
into `issues`, and embeds it locally if its text is new or changed. Re-running
only re-embeds what changed. The first run downloads the embedding model
(fastembed caches it). Read-only on GitHub.
"""

import argparse
import sys
import time
from datetime import UTC, datetime

from githubtriage.config import settings
from githubtriage.db import connect, migrate
from githubtriage.github_client import GitHubClient
from githubtriage.issue_index import (
    EMBED_MODEL,
    FastEmbedder,
    fetch_issue_records,
    index_issues,
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="pydantic/pydantic")
    parser.add_argument("--since", default="2023-07-01", help="created on or after")
    args = parser.parse_args()

    if not (settings.github_token and settings.github_token.get_secret_value()):
        print(
            "GITHUB_TOKEN is required (thousands of issues to page through).",
            file=sys.stderr,
        )
        return 1
    migrate()
    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)

    started = time.monotonic()
    with GitHubClient(settings.github_token, settings.github_api_url) as gh:
        records = list(fetch_issue_records(gh, args.repo, created_since=since))
    fetched = time.monotonic() - started
    print(
        f"fetched {len(records)} issues from {args.repo} since {args.since} "
        f"in {fetched:.0f}s"
    )

    print(f"loading {EMBED_MODEL} (downloaded on first use)…", flush=True)
    embedder = FastEmbedder()
    started = time.monotonic()
    # autocommit: each batch is committed, so a crash keeps finished batches.
    with connect(autocommit=True) as conn:
        report = index_issues(conn, records, embedder)
        total = conn.execute(
            "SELECT count(*) FROM issues WHERE repo = %s", (args.repo,)
        ).fetchone()[0]
        vectors = conn.execute(
            "SELECT count(*) FROM issue_embeddings WHERE repo = %s AND model = %s",
            (args.repo, embedder.model),
        ).fetchone()[0]
    took = time.monotonic() - started
    print(
        f"upserted {report.issues_upserted}, embedded {report.embedded}, "
        f"unchanged {report.unchanged} in {took:.0f}s"
    )
    print(
        f"index for {args.repo}: {total} issues, {vectors} embeddings ({embedder.model})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
