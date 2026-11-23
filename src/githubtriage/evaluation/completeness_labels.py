"""Hand-labelled answer key for R3 completeness (slice 4 module 12d).

pydantic's labels say nothing about whether a bug report gives its version,
reproduction steps and error output, so this answer key is written by a human,
following evals/pydantic/COMPLETENESS_LABELLING.md.

Which issues (decided 2026-11-20):
- "bug-shaped" eval issues: answer key bug, or filed with the bug form (R1 sends
  most of these to R3, so this is what R3 really sees);
- 25 from dev and 25 from test; within each, ENRICHED: every bug-shaped issue
  without a "pydantic version:" line, then the rest in a fixed random order. On pydantic the bug template makes most reports
  complete; a random 50 would hold too few "missing" cases to measure whether R3
  asks for them. The sample is therefore not natural: report per-field results,
  never "x% of pydantic bug reports are complete".
- Each issue keeps its dev/test split: tune completeness-v2 on dev labels only.
"""

import csv
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from githubtriage.evaluation.dataset import IssueText, ManifestRow
from githubtriage.models import FieldStatus

FIELDS = ("version", "repro_steps", "logs")
COLUMNS = ("repo", "number", "split", "link", *FIELDS, "unsure", "note")


class CompletenessLabel(BaseModel):
    repo: str
    number: int
    split: Literal["dev", "test"]
    version: FieldStatus
    repro_steps: FieldStatus
    logs: FieldStatus
    unsure: bool = False  # the labeller wasn't sure: reported, not silently trusted
    note: str = ""


def bug_shaped(row: ManifestRow) -> bool:
    return row.truth.type == "bug" or row.truth.reporter_label == "bug V2"


def has_version_line(text: IssueText) -> bool:
    return "pydantic version:" in text.body.lower()


def _order(row: ManifestRow) -> str:
    """A fixed random order of its own. NOT split_key: the dev/test split put the
    lower half of each stratum (by split_key) in dev, so "the first 50 by
    split_key" picked 42 dev / 8 test. Reusing a random key for a second choice
    makes the two choices correlated; a salt makes them independent."""
    return hashlib.sha256(
        f"completeness-labels:{row.repo}#{row.number}".encode()
    ).hexdigest()


def select_for_labelling(
    rows: list[ManifestRow], texts: dict[tuple[str, int], IssueText], n: int = 50
) -> list[ManifestRow]:
    """Half from dev, half from test; within each, every bug-shaped issue without
    a version line first (enrichment), then the rest in a fixed random order.
    The sheet is shuffled the same way, so the labeller can't tell which issues
    were picked for enrichment or which split they're in."""
    chosen = []
    for split in ("dev", "test"):
        pool = sorted(
            (r for r in rows if bug_shaped(r) and r.split == split), key=_order
        )
        enriched = [r for r in pool if not has_version_line(texts[(r.repo, r.number)])]
        rest = [r for r in pool if r not in enriched]
        chosen += (enriched + rest)[: n // 2]
    return sorted(chosen, key=_order)


def blank_sheet_rows(chosen: list[ManifestRow]) -> list[dict[str, str]]:
    return [
        {
            "repo": r.repo,
            "number": str(r.number),
            "split": r.split or "",
            "link": f"https://github.com/{r.repo}/issues/{r.number}",
            **dict.fromkeys(FIELDS, ""),
            "unsure": "",
            "note": "",
        }
        for r in chosen
    ]


def write_sheet(path: Path, sheet: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sheet)


class LabelReport(BaseModel):
    labels: list[CompletenessLabel]  # complete, valid rows
    todo: int  # rows not filled in yet
    problems: list[str]  # "line N: ..." for invalid rows


def load_labels(path: Path) -> LabelReport:
    """Read the sheet, keeping valid rows and naming every problem by line.

    Unfilled rows are "to do", not errors. `unsure` accepts yes / y / true / 1.
    Values are trimmed and lower-cased, so " OK" from a spreadsheet still counts.
    """
    labels, problems, todo = [], [], 0
    # utf-8-sig: Excel on Windows adds a BOM when it saves as CSV UTF-8.
    with path.open(encoding="utf-8-sig", newline="") as f:
        for line, raw in enumerate(csv.DictReader(f), start=2):
            values = {k: (v or "").strip() for k, v in raw.items() if k}
            if not any(values.get(field) for field in FIELDS):
                todo += 1
                continue
            data = {k: values.get(k, "") for k in COLUMNS if k != "link"}
            for field in FIELDS:
                data[field] = data[field].lower()
            data["unsure"] = data["unsure"].lower() in ("yes", "y", "true", "1")
            try:
                labels.append(CompletenessLabel.model_validate(data))
            except ValidationError as e:
                fields = ", ".join(str(err["loc"][0]) for err in e.errors())
                problems.append(
                    f"line {line} (#{values.get('number')}): invalid {fields}"
                )
    return LabelReport(labels=labels, todo=todo, problems=problems)
