"""R5: build the first-response draft from templates. No LLM, no I/O.

Only text we wrote goes into the reply: templates chosen by values we control (the
classification type, the completeness statuses) plus the author's username. Neither
the issue text nor any LLM-written text (e.g. FieldCheck.note) is ever quoted, so
attacker-influenced text (@mentions, links, images) can't be posted under the
project's name. See DESIGN.md §5.5.
"""

from githubtriage.models import Completeness, TriageState

GREETING = "Thanks for opening this issue, @{author}!"

TYPE_MESSAGES = {
    "bug": ("We've labelled this as a **bug**. A maintainer will look into it."),
    "feature": (
        "We've labelled this as a **feature request**. Maintainers will consider it. "
        "Describing the use case behind the request helps us understand and prioritise it."
    ),
    "question": (
        "This looks like a **question**. A maintainer or community member may be able "
        "to help here. For general usage questions, GitHub Discussions (if enabled for "
        "this project) is often the quickest place to get an answer."
    ),
    "docs": (
        "We've labelled this as a **documentation** issue. Documentation fixes are a "
        "great first contribution, so a pull request is very welcome."
    ),
    "unclear": (
        "We weren't able to tell what kind of issue this is. Could you clarify whether "
        "you're reporting something broken, requesting a new feature, or asking a question? "
        "A maintainer will review it either way."
    ),
}

# Statuses that mean "ask the reporter for this". ok / not_needed ask for nothing.
REQUEST_STATUSES = ("missing", "vague")

# Allowlist: the LLM only chooses the status; the wording is ours. The LLM's own
# `note` stays in the state for maintainers and never reaches the public comment.
REQUEST_PHRASES = {
    ("version", "missing"): (
        "the version of the project you're using (for example, 2.3.0)"
    ),
    ("version", "vague"): (
        'the exact version number, rather than "latest" or a branch name'
    ),
    ("repro_steps", "missing"): (
        "steps or a short code snippet that reproduce the problem"
    ),
    ("repro_steps", "vague"): (
        "the exact steps or code you ran, so someone else can see the same result"
    ),
    ("logs", "missing"): "the full error message or traceback you saw",
    ("logs", "vague"): (
        "the complete error message or traceback, rather than a summary"
    ),
}

REQUEST_INTRO = "To help us investigate, could you add:"

FOOTER = "_Drafted by GithubTriage — pending maintainer review._"


def _completeness_section(completeness: Completeness | None) -> str | None:
    """A "please add" list for missing/vague fields, or None if nothing to ask.

    None completeness means "unknown" (skipped, or the check failed): ask for nothing.
    """
    if completeness is None:
        return None
    requests = [
        REQUEST_PHRASES[(field, check.status)]
        for field, check in completeness  # fields in model order
        if check.status in REQUEST_STATUSES
    ]
    if not requests:
        return None
    return REQUEST_INTRO + "\n" + "\n".join(f"- {r}" for r in requests)


def draft(state: TriageState) -> dict:
    issue_type = state.classification.type if state.classification else "unclear"

    sections = [
        GREETING.format(author=state.issue.author),
        TYPE_MESSAGES.get(issue_type, TYPE_MESSAGES["unclear"]),
    ]
    if request := _completeness_section(state.completeness):
        sections.append(request)
    # Later slices append sections here: related issues (R2),
    # reproduction result (R4), grounded answer (R5 LLM).
    sections.append(FOOTER)

    return {"draft": "\n\n".join(sections)}
