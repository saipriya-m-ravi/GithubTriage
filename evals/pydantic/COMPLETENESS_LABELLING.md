# Labelling guide — R3 completeness (pydantic)

You are writing the **answer key** for R3, the completeness check. For each bug-shaped issue, decide whether the report gives the three things a maintainer needs to reproduce it. Later, R3's answers are compared with yours — so your labels define what "correct" means.

**Time:** about 2–3 minutes per issue → 50 issues ≈ 2 hours. Fine to split over several sittings; `--check` shows progress.

---

## How to work

1. Create the kit once:
   ```powershell
   uv run python debug_helper_scripts/completeness_labelling.py --create
   ```
2. Read the issues in `.githubtriage/eval/completeness_to_label.md` (local only).
3. Fill in `evals/pydantic/completeness_labels.csv` — one row per issue, same order. Columns to fill: `version`, `repro_steps`, `logs`, and optionally `unsure`, `note`. Don't change `repo`, `number`, `split`, `link`.
4. Check as you go:
   ```powershell
   uv run python debug_helper_scripts/completeness_labelling.py --check
   ```

**Editing the CSV:** in the IDE it's plain text — `missing`, `vague`, `ok` or `not_needed`, separated by commas. In Excel, save as *CSV UTF-8*. If a note needs a comma, keep it short or leave the note out.

---

## Three ground rules

1. **Judge only the title and body** — exactly what R3 sees when the issue is opened. Ignore comments, linked pages, gists and screenshots (R3 can't read them). Information that exists *only* behind a link or in an image counts as **not given**.
2. **Judge what is written, not whether the bug is real.** Many of these issues were later diagnosed as user error (question) or a docs gap. That doesn't matter here: does the *report* contain the information?
3. **Label blind.** Don't run R3 on these issues and don't look at their GitHub labels or comments before labelling. Seeing an answer first pulls your judgement towards it (*anchoring*) — and then the eval measures agreement with the model, not correctness.

---

## The four values

| Value | Meaning |
|---|---|
| `ok` | Given clearly enough for a maintainer to act on |
| `vague` | Mentioned, but not usable as given (see each field) |
| `missing` | Not given at all |
| `not_needed` | Doesn't apply to this issue (only possible for `logs`) |

---

## version — which pydantic version the reporter uses

| Value | When | Examples |
|---|---|---|
| `ok` | A specific **pydantic** version | `pydantic version: 2.9.2` (the template's `version_info()` block); "on pydantic 2.10.1" |
| `vague` | A version is hinted at but not specific | "latest", "main", "2.x", "since 2.10" / "after upgrading to 2.10" without the installed version; only `pydantic-core` given |
| `missing` | No pydantic version at all | only Python / OS versions given; nothing |

*Never `not_needed`.* The project is pydantic — a Python or OS version alone is **not** the version.

## repro_steps — can someone else see the problem?

| Value | When | Examples |
|---|---|---|
| `ok` | Code someone could run, or concrete steps | a self-contained snippet (small gaps like a missing `import` are fine); numbered steps with inputs |
| `vague` | Only a description, or code that can't run as given | "validation fails when I use a discriminated union"; a fragment using classes that are never defined; code only in a screenshot or behind a link |
| `missing` | Nothing about how to trigger it | a one-line complaint |

*Never `not_needed`* for a bug report.

**Docs issues filed with the bug form** (added 2026-11-24): "reproduce" means *can a maintainer find the problem*. `ok` = names the exact place (page, section, docstring) **and** what's wrong; `vague` = says the docs are wrong or confusing but not where or not what; `missing` = neither. Still never `not_needed` — R3 can't answer that for repro, so the label would always count as an R3 error. A link to the docs page is fine as the *location*, as long as what's wrong is described in the text.

## logs — the error output

| Value | When | Examples |
|---|---|---|
| `ok` | The error text is included | a traceback; the final error line with its message (`ValidationError: 1 validation error for …`); the error shown as a comment in the code (`# raises TypeError: …`) |
| `missing` | An error, exception or crash is described, but its text isn't given | "it raises a ValidationError" (no message); "it crashes" |
| `not_needed` | No error is described | wrong value returned; unexpected output; behaviour differs from docs; a schema looks wrong |
| `vague` | Rarely used: a fragment that can't identify the error | "some pydantic error appears" with a partial, unidentifiable line |

---

## unsure and note

- Put `yes` in **`unsure`** when a field could honestly go either way. Still choose your best value — `unsure` doesn't replace a value.
- Use **`note`** for the reason, briefly (e.g. "version only in issue title", "code needs a DB"). Unsure labels are reported separately, so they can't silently decide a result.
- **When you're sure, leave both blank.** Only `yes` (or `y`, `true`, `1`) marks a row as unsure; blank or `no` means sure. `note` is optional.

---

## Self-consistency check (optional, recommended)

A few days after finishing, re-label 5 issues **without looking at your first answers**, in a scratch copy. If you disagree with yourself on more than 1–2 fields out of 15, the rules above need sharpening before the labels are used — tell me which cases, and we'll adjust the guide (and re-check the affected rows).

---

## Why these rules exist (for later reference)

- *Title + body only* matches when R3 runs (issue opened). The module 6 idea of also reading the reporter's comments applies to re-triage, and would need its own labels.
- *Blind labelling* keeps the answer key independent of the system it measures.
- *Rules fixed before labelling* keep issue 1 and issue 50 judged the same way.
- *Enriched sample:* every bug-shaped issue without a `pydantic version:` line is included, so "missing" cases exist to measure; results are reported per field, never as "x% of pydantic reports are complete".
