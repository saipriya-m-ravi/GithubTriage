"""Duplicate finder, Stage 1: build the answer key, then measure recall@k.

    uv run python debug_helper_scripts/duplicate_eval.py --build          # once (GraphQL)
    uv run python debug_helper_scripts/duplicate_eval.py --split dev
    uv run python debug_helper_scripts/duplicate_eval.py --split dev --show-misses
    uv run python debug_helper_scripts/duplicate_eval.py --split test     # final, logged

--build reads every duplicate-closed issue's `duplicateOf` (GitHub GraphQL,
read-only), keeps the usable pairs and writes evals/pydantic/duplicates.jsonl
(numbers only — committed). Frozen: refuses to overwrite without --force, and
keeps existing splits. Needs GITHUB_TOKEN and the database.

--split searches each duplicate as if it had just been opened (only issues
created before it, never itself) and reports recall@1/5/10 for keyword-only,
vector-only and hybrid. No LLM; the embedding model runs locally.
--show-misses works on dev only; test runs are appended to
evals/pydantic/retrieval_test_runs.jsonl.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from githubtriage.config import settings
from githubtriage.db import connect
from githubtriage.evaluation.duplicates import (
    DuplicatePair,
    fetch_targets,
    usable_pairs,
    with_splits,
)
from githubtriage.evaluation.retrieval import VARIANTS, recall_at, target_ranks
from githubtriage.github_client import GitHubClient
from githubtriage.issue_index import FastEmbedder, embed_text
from githubtriage.search import common_words, ranked_lists

REPO = "pydantic/pydantic"
EVALS = Path("evals/pydantic")
PAIRS = EVALS / "duplicates.jsonl"
TEST_LOG = EVALS / "retrieval_test_runs.jsonl"


def read_pairs() -> list[DuplicatePair]:
    if not PAIRS.exists():
        return []
    return [
        DuplicatePair.model_validate_json(x)
        for x in PAIRS.read_text("utf-8").splitlines()
    ]


def build(force: bool) -> int:
    previous = read_pairs()
    if previous and not force:
        print(
            f"{PAIRS} exists: the answer key is frozen (use --force to rebuild; "
            "existing splits are kept).",
            file=sys.stderr,
        )
        return 1
    if not (settings.github_token and settings.github_token.get_secret_value()):
        print("GITHUB_TOKEN is required (GraphQL needs a token).", file=sys.stderr)
        return 1
    with connect() as conn:
        duplicates = dict(
            conn.execute(
                "SELECT number, created_at FROM issues WHERE repo = %s"
                " AND state_reason = 'duplicate'",
                (REPO,),
            ).fetchall()
        )
        indexed = dict(
            conn.execute(
                "SELECT number, created_at FROM issues WHERE repo = %s", (REPO,)
            ).fetchall()
        )
    with GitHubClient(settings.github_token, settings.github_api_url) as gh:
        targets = fetch_targets(gh, REPO, duplicates)
    pairs, excluded = usable_pairs(REPO, duplicates, targets, indexed)
    pairs = with_splits(pairs, previous)
    PAIRS.write_text(
        "".join(p.model_dump_json() + "\n" for p in pairs), encoding="utf-8"
    )
    splits = Counter(p.split for p in pairs)
    print(
        f"{len(duplicates)} duplicate-closed issues → {len(pairs)} usable pairs "
        f"({splits['dev']} dev / {splits['test']} test) → {PAIRS}"
    )
    for reason, n in excluded.most_common():
        print(f"  excluded {n:3}: {reason}")
    return 0


def evaluate(split: str, show_misses: bool, max_df: float | None) -> int:
    pairs = [p for p in read_pairs() if p.split == split]
    if not pairs:
        print(f"no {split} pairs — run --build first", file=sys.stderr)
        return 1
    embedder = FastEmbedder()
    ranks = []
    with connect() as conn:
        common = common_words(conn, REPO, max_df) if max_df is not None else []
        print(
            f"keyword query leaves out {len(common)} words in > {max_df:.0%} of issues"
            if max_df is not None
            else "keyword query uses every word (no common-word filter)"
        )
        for p in pairs:
            row = conn.execute(
                "SELECT title, body FROM issues WHERE repo = %s AND number = %s",
                (p.repo, p.duplicate),
            ).fetchone()
            text = embed_text_from(row)
            keyword, vector = ranked_lists(
                conn,
                embedder,
                repo=p.repo,
                text=text,
                before=p.duplicate_created_at,
                exclude_number=p.duplicate,
                common=common,
            )
            ranks.append(target_ranks(p.duplicate, p.target, keyword, vector))

    print(
        f"\n=== {split}: {len(ranks)} duplicate pairs — recall@k (target among the top k) ==="
    )
    print(f"{'variant':9}" + "".join(f"{f'@{k}':>8}" for k in (1, 5, 10, 50)))
    results = {}
    for v in VARIANTS:
        row = {k: recall_at(ranks, v, k) for k in (1, 5, 10, 50)}
        results[v] = row
        print(f"{v:9}" + "".join(f"{100 * row[k]:7.1f}%" for k in row))
    print("(@50 = found anywhere in that list: the ceiling for any re-ranking)")

    if show_misses:
        print("\nmisses at @5 for hybrid (dev only):")
        for r in ranks:
            if r.hybrid is None or r.hybrid > 5:
                print(
                    f"  #{r.duplicate} → #{r.target}   keyword {r.keyword}, "
                    f"vector {r.vector}, hybrid {r.hybrid}   "
                    f"https://github.com/{REPO}/issues/{r.duplicate}"
                )
    if split == "test":
        entry = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n": len(ranks),
            "max_df": max_df,
            "recall": results,
        }
        with TEST_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\ntest run logged in {TEST_LOG}")
    return 0


def embed_text_from(row) -> str:
    # Same text the index embedded: title + start of the body.
    from githubtriage.issue_index import IssueRecord

    stub = IssueRecord(
        repo=REPO,
        number=0,
        title=row[0],
        body=row[1],
        state="open",
        state_reason=None,
        labels=[],
        html_url="x",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        closed_at=None,
    )
    return embed_text(stub)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument(
        "--max-df",
        type=float,
        help="leave words in more than this share of issues out of the keyword query",
    )
    args = parser.parse_args()
    if args.build:
        return build(args.force)
    if not args.split:
        parser.error("use --build or --split dev|test")
    if args.show_misses and args.split == "test":
        print("--show-misses is dev-only.", file=sys.stderr)
        return 1
    return evaluate(args.split, args.show_misses, args.max_df)


if __name__ == "__main__":
    sys.exit(main())
