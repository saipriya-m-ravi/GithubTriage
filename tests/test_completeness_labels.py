"""The R3 labelling kit: issue selection and reading the hand-filled sheet."""

import csv
from datetime import UTC, datetime

from githubtriage.evaluation.completeness_labels import (
    COLUMNS,
    blank_sheet_rows,
    load_labels,
    select_for_labelling,
    write_sheet,
)
from githubtriage.evaluation.dataset import IssueText, ManifestRow
from githubtriage.evaluation.labels import GroundTruth

CREATED = datetime(2025, 1, 1, tzinfo=UTC)


def row(n, type_="bug", reporter_label="bug V2", split="dev"):
    return ManifestRow(
        repo="o/r",
        number=n,
        created_at=CREATED,
        truth=GroundTruth(
            type=type_,
            label="x",
            trust="gold",
            relabelled=False,
            reporter_label=reporter_label,
        ),
        split=split,
    )


def text(n, body):
    return IssueText(repo="o/r", number=n, title="t", body=body)


def test_selects_bug_shaped_issues_only():
    rows = [
        row(1),  # a bug
        row(2, "question"),  # filed with the bug form: R3 sees it
        row(3, "feature", "feature request"),  # not bug-shaped
    ]
    texts = {("o/r", n): text(n, "pydantic version: 2.9") for n in (1, 2, 3)}
    assert {r.number for r in select_for_labelling(rows, texts)} == {1, 2}


def test_issues_without_a_version_line_are_always_included():
    # Enrichment: with only 2 dev slots, the 2 dev issues lacking a version line
    # must be the ones chosen.
    rows = [row(n) for n in range(10)] + [row(n, split="test") for n in range(10, 20)]
    texts = {("o/r", n): text(n, "pydantic version: 2.9") for n in range(20)}
    texts[("o/r", 4)] = text(4, "no version here")
    texts[("o/r", 7)] = text(7, "also none")

    chosen = select_for_labelling(rows, texts, n=4)
    assert {r.number for r in chosen if r.split == "dev"} == {4, 7}


def test_half_from_each_split():
    # Found on the real data: selecting by split_key order gave 42 dev / 8 test,
    # because the split itself was cut by split_key order.
    rows = [row(n) for n in range(40)] + [row(n, split="test") for n in range(40, 80)]
    texts = {("o/r", n): text(n, "pydantic version: 2.9") for n in range(80)}
    chosen = select_for_labelling(rows, texts, n=20)
    assert sum(r.split == "dev" for r in chosen) == 10
    assert sum(r.split == "test" for r in chosen) == 10


def write(path, rows_):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows_)


def sheet_row(n, **values):
    base = blank_sheet_rows([row(n)])[0]
    base.update(values)
    return base


def test_blank_sheet_has_links_and_empty_labels(tmp_path):
    path = tmp_path / "s.csv"
    write_sheet(path, blank_sheet_rows([row(5, split="test")]))
    report = load_labels(path)
    assert (report.todo, report.labels, report.problems) == (1, [], [])
    assert "https://github.com/o/r/issues/5" in path.read_text(encoding="utf-8")


def test_filled_rows_are_read_with_forgiving_spelling(tmp_path):
    path = tmp_path / "s.csv"
    write(
        path,
        [
            sheet_row(
                1, version=" OK", repro_steps="vague", logs="Not_Needed", unsure="Yes"
            ),
            sheet_row(2),  # not done yet
        ],
    )
    report = load_labels(path)
    label = report.labels[0]
    assert (label.version, label.logs, label.unsure) == ("ok", "not_needed", True)
    assert report.todo == 1 and report.problems == []


def test_invalid_values_are_reported_by_line(tmp_path):
    path = tmp_path / "s.csv"
    write(
        path,
        [
            sheet_row(1, version="ok", repro_steps="ok", logs="ok"),
            sheet_row(2, version="okay", repro_steps="ok", logs=""),  # typo + blank
        ],
    )
    report = load_labels(path)
    assert len(report.labels) == 1
    assert report.problems == ["line 3 (#2): invalid version, logs"]


def test_excel_bom_is_accepted(tmp_path):
    # Excel's "CSV UTF-8" starts the file with a BOM; the header must still match.
    path = tmp_path / "s.csv"
    write(path, [sheet_row(1, version="ok", repro_steps="ok", logs="ok")])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert len(load_labels(path).labels) == 1
