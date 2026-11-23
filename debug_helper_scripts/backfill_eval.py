"""Build the pydantic eval set (read-only; needs GITHUB_TOKEN for the rate limit).

    uv run python debug_helper_scripts/backfill_eval.py               # collect
    uv run python debug_helper_scripts/backfill_eval.py --rebuild     # texts only
    uv run python debug_helper_scripts/backfill_eval.py --split-only  # assign splits

The dataset is FROZEN once built: collecting again a week later would pick a
different "newest 60" and silently change what we measure on. So collect refuses
to overwrite an existing manifest without --force, and with --force every issue
still present keeps its dev/test split (evaluation/split.py).

Collect: closed issues created since --since, up to --per-type usable issues per
label, each judged by evaluation.labels.ground_truth. Writes:
- evals/pydantic/manifest.jsonl   answer key, no text (commit this)
- evals/pydantic/frequencies.json real counts per type (commit this)
- DATA_DIR/eval/pydantic_issues.jsonl  titles + bodies (gitignored, local only)

--rebuild: re-fetch the texts for the issues already in the manifest, without
re-deciding the answer key — how anyone recreates the local file from git.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from githubtriage.config import settings
from githubtriage.evaluation.dataset import (
    IssueText,
    ManifestRow,
    collect,
    natural_frequencies,
)
from githubtriage.evaluation.split import with_splits
from githubtriage.github_client import GitHubClient

REPO = "pydantic/pydantic"
OUT_DIR = Path("evals/pydantic")
MANIFEST = OUT_DIR / "manifest.jsonl"
FREQUENCIES = OUT_DIR / "frequencies.json"
TEXTS = settings.data_dir / "eval" / "pydantic_issues.jsonl"
EVENTS_CACHE = settings.data_dir / "eval" / "events_cache"  # gitignored


def write_jsonl(path: Path, models) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [m.model_dump_json() for m in models]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest() -> list[ManifestRow]:
    if not MANIFEST.exists():
        return []
    return [
        ManifestRow.model_validate_json(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]


def print_splits(rows: list[ManifestRow]) -> None:
    counts = Counter(
        (r.truth.type, r.truth.trust, r.truth.relabelled, r.split) for r in rows
    )
    print(f"{'type':9} {'trust':7} {'relabelled':11} {'dev':>4} {'test':>5}")
    for t, tr, rel in sorted({k[:3] for k in counts}):
        dev, test = counts[(t, tr, rel, "dev")], counts[(t, tr, rel, "test")]
        print(f"{t:9} {tr:7} {rel!s:11} {dev:4} {test:5}")
    dev = sum(r.split == "dev" for r in rows)
    print(f"{'total':29} {dev:4} {len(rows) - dev:5}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", default="2024-01-01", help="created on or after")
    parser.add_argument("--per-type", type=int, default=60)
    parser.add_argument("--rebuild", action="store_true", help="texts only")
    parser.add_argument(
        "--split-only", action="store_true", help="assign splits to the manifest"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-collect over an existing manifest"
    )
    args = parser.parse_args()

    previous = read_manifest()
    if args.split_only:
        rows = with_splits(previous, previous)  # existing splits are kept
        write_jsonl(MANIFEST, rows)
        print_splits(rows)
        return 0
    if previous and not (args.rebuild or args.force):
        print(
            f"{MANIFEST} exists: the eval set is frozen. Use --rebuild for texts, "
            "or --force to re-collect (splits of remaining issues are kept).",
            file=sys.stderr,
        )
        return 1

    if not (settings.github_token and settings.github_token.get_secret_value()):
        # ~1 request per issue: far beyond the anonymous 60/hour.
        print("GITHUB_TOKEN is required for the backfill.", file=sys.stderr)
        return 1

    with GitHubClient(settings.github_token, settings.github_api_url) as gh:
        if args.rebuild:
            texts = []
            for row in previous:
                issue = gh.get_issue(row.repo, row.number)
                texts.append(
                    IssueText(
                        repo=row.repo,
                        number=row.number,
                        title=issue["title"],
                        body=issue.get("body") or "",
                    )
                )
            write_jsonl(TEXTS, texts)
            print(f"rebuilt {len(texts)} texts -> {TEXTS}")
            return 0

        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
        result = collect(
            gh,
            REPO,
            created_since=since,
            per_type=args.per_type,
            cache_dir=EVENTS_CACHE,  # a failed run resumes from here
        )
        frequencies = natural_frequencies(gh, REPO, created_since=since)

    rows = with_splits(result.rows, previous)  # sorted; old splits kept
    order = {r.number: i for i, r in enumerate(rows)}
    write_jsonl(MANIFEST, rows)
    write_jsonl(TEXTS, sorted(result.texts, key=lambda t: order[t.number]))
    FREQUENCIES.write_text(
        json.dumps(
            {
                "repo": REPO,
                "created_since": args.since,
                "counted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "closed_issues_per_type": frequencies,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Summary: what the answer key is made of.
    kinds = Counter((r.truth.type, r.truth.trust, r.truth.relabelled) for r in rows)
    print(f"{len(rows)} issues -> {MANIFEST}  (texts -> {TEXTS})\n")
    print(f"{'type':9} {'gold':>5} {'silver':>7} {'relabelled':>11}   real count")
    for t in ("bug", "feature", "question", "docs"):
        gold = sum(v for (ty, tr, _), v in kinds.items() if ty == t and tr == "gold")
        silver = sum(
            v for (ty, tr, _), v in kinds.items() if ty == t and tr == "silver"
        )
        rel = sum(v for (ty, _, re), v in kinds.items() if ty == t and re)
        print(f"{t:9} {gold:5} {silver:7} {rel:11}   {frequencies[t]:>10}")
    print("\nexcluded:")
    for reason, n in sorted(result.excluded.items()):
        print(f"  {n:4}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
