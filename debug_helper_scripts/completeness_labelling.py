"""Hand-labelling kit for R3 completeness (no LLM or GitHub calls).

    uv run python debug_helper_scripts/completeness_labelling.py --create
    uv run python debug_helper_scripts/completeness_labelling.py --check

--create writes, once:
- evals/pydantic/completeness_labels.csv   the sheet you fill in (committed:
  your labels, no issue text)
- DATA_DIR/eval/completeness_to_label.md   the 50 issues to read, in sheet order
  (local, gitignored: other people's writing)
It refuses to overwrite an existing sheet — that file holds your work.

--check validates what you've filled in so far and shows progress.
Rules for labelling: evals/pydantic/COMPLETENESS_LABELLING.md
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from githubtriage.config import settings
from githubtriage.evaluation.completeness_labels import (
    FIELDS,
    blank_sheet_rows,
    load_labels,
    select_for_labelling,
    write_sheet,
)
from githubtriage.evaluation.dataset import IssueText, ManifestRow

EVALS = Path("evals/pydantic")
MANIFEST = EVALS / "manifest.jsonl"
SHEET = EVALS / "completeness_labels.csv"
TEXTS = settings.data_dir / "eval" / "pydantic_issues.jsonl"
READING = settings.data_dir / "eval" / "completeness_to_label.md"


def create() -> int:
    if SHEET.exists():
        print(
            f"{SHEET} already exists — it holds your labels; not overwritten.",
            file=sys.stderr,
        )
        return 1
    rows = [
        ManifestRow.model_validate_json(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    texts = {
        (t.repo, t.number): t
        for t in (
            IssueText.model_validate_json(line)
            for line in TEXTS.read_text(encoding="utf-8").splitlines()
        )
    }
    chosen = select_for_labelling(rows, texts)
    write_sheet(SHEET, blank_sheet_rows(chosen))

    # Reading copy: what R3 sees at triage time — title and body only. No type,
    # labels or model output: labelling must be BLIND to what we'll compare with.
    parts = [
        "# Issues to label for completeness\n",
        (
            f"Rules: {EVALS / 'COMPLETENESS_LABELLING.md'}. "
            "Fill in the CSV row with the same number.\n"
        ),
    ]
    for i, r in enumerate(chosen, start=1):
        t = texts[(r.repo, r.number)]
        parts.append(
            f"\n---\n\n## {i}. #{r.number} — {t.title}\n\n"
            f"https://github.com/{r.repo}/issues/{r.number}\n\n"
            f"````text\n{t.body}\n````\n"
        )
    READING.parent.mkdir(parents=True, exist_ok=True)
    READING.write_text("".join(parts), encoding="utf-8")
    splits = Counter(r.split for r in chosen)
    print(
        f"wrote {SHEET} ({len(chosen)} rows: {splits['dev']} dev, {splits['test']} test)"
    )
    print(f"wrote {READING} (read the issues here)")
    return 0


def check() -> int:
    if not SHEET.exists():
        print(f"{SHEET} doesn't exist yet: run --create first.", file=sys.stderr)
        return 1
    report = load_labels(SHEET)
    done = len(report.labels)
    print(f"labelled: {done}   to do: {report.todo}   problems: {len(report.problems)}")
    for problem in report.problems:
        print(f"  ✗ {problem}")
    if report.labels:
        for field in FIELDS:
            counts = Counter(getattr(label, field) for label in report.labels)
            print(
                f"  {field:12} "
                + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            )
        unsure = sum(label.unsure for label in report.labels)
        print(f"  unsure: {unsure}")
    return 1 if report.problems else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return create() if args.create else check()


if __name__ == "__main__":
    sys.exit(main())
