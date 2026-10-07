"""R5 (slice 1): build the first-response draft from templates. No LLM, no I/O.

Only values we control (the classification type and the author's username) go into
the reply. The issue title and body are never quoted, so attacker-controlled text
(@mentions, links) can't be repeated under the project's name.
"""

from githubtriage.models import TriageState

GREETING = "Thanks for opening this issue, @{author}!"

TYPE_MESSAGES = {
    "bug": (
        "We've labelled this as a **bug**. A maintainer will look into it. "
        "If you haven't already, adding the version you're using and the steps to "
        "reproduce the problem will help us investigate."
    ),
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

FOOTER = "_Drafted by GithubTriage — pending maintainer review._"


def draft(state: TriageState) -> dict:
    issue_type = state.classification.type if state.classification else "unclear"

    sections = [
        GREETING.format(author=state.issue.author),
        TYPE_MESSAGES.get(issue_type, TYPE_MESSAGES["unclear"]),
    ]
    # Later slices append sections here: completeness requests (R3),
    # related issues (R2), reproduction result (R4), grounded answer (R5 LLM).
    sections.append(FOOTER)

    return {"draft": "\n\n".join(sections)}
