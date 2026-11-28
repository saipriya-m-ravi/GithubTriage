"""Show the hybrid search's top candidates for an issue already in the index.

    uv run python debug_helper_scripts/similar_issues.py 11023
    uv run python debug_helper_scripts/similar_issues.py 11023 --k 10 --repo pydantic/pydantic

Searches as if the issue had just been opened: only issues created BEFORE it,
never the issue itself. Read-only; no LLM, no GitHub calls.
"""

import argparse
import sys

from githubtriage.db import connect
from githubtriage.issue_index import FastEmbedder, IssueRecord, embed_text
from githubtriage.search import search_similar


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("number", type=int)
    parser.add_argument("--repo", default="pydantic/pydantic")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with connect() as conn:
        row = conn.execute(
            "SELECT repo, number, title, body, state, state_reason, labels, html_url,"
            " created_at, updated_at, closed_at FROM issues WHERE repo = %s AND number = %s",
            (args.repo, args.number),
        ).fetchone()
        if row is None:
            print(
                f"#{args.number} is not in the index for {args.repo}", file=sys.stderr
            )
            return 1
        cols = (
            "repo",
            "number",
            "title",
            "body",
            "state",
            "state_reason",
            "labels",
            "html_url",
            "created_at",
            "updated_at",
            "closed_at",
        )
        issue = IssueRecord(**dict(zip(cols, row, strict=True)))
        print(f"#{issue.number} ({issue.created_at:%Y-%m-%d})  {issue.title}\n")
        embedder = FastEmbedder()
        found = search_similar(
            conn,
            embedder,
            repo=issue.repo,
            text=embed_text(issue),
            before=issue.created_at,
            exclude_number=issue.number,
            k=args.k,
        )
    print(f"{'#':>7} {'created':10} {'kw':>4} {'vec':>4} {'rrf':>7}  title")
    for c in found:
        kw = c.keyword_rank if c.keyword_rank is not None else "-"
        vec = c.vector_rank if c.vector_rank is not None else "-"
        reason = f" [{c.state_reason}]" if c.state_reason else ""
        print(
            f"{c.number:>7} {c.created_at:%Y-%m-%d} {kw:>4} {vec:>4} {c.score:7.4f}"
            f"  {c.title[:70]}{reason}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
