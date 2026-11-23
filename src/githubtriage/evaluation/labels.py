"""Ground truth for the eval set: pydantic's labels -> our types, and how far to trust them.

A label's meaning depends on who applied it. pydantic's issue templates add
labels automatically when the reporter picks a form, so those record the
REPORTER's choice; a maintainer's label is an expert judgement. GitHub's issue
events show who applied each label (the template acts as the reporter).

What the data showed (pydantic/pydantic, 2026-11-10):
- bug-v2.yml adds [bug V2, pending]; feature_request.yml adds [feature request];
  there are no question or documentation forms.
- "pending" means "not triaged yet": a maintainer removes it after looking.
- Reporters often pick the wrong form; maintainers relabel (bug V2 -> question,
  bug V2 -> documentation, feature request -> question...). Those issues are
  kept: they are exactly the hard cases.
- Template labels are recorded a second or so after the issue was created, so
  WHO applied a label is the reliable signal, not WHEN.

Trust rule (decisions D1–D3, 2026-11-10):
- gold: a maintainer applied or confirmed the label (applied it, removed
  "pending", or filed the issue themselves)
- silver: only the reporter applied it, e.g. a feature request nobody touched
- excluded: still "pending" (unreviewed), several of our labels (ambiguous),
  or none of them
"""

from typing import Literal

from pydantic import BaseModel

IssueType = Literal["bug", "feature", "question", "docs"]

LABEL_TO_TYPE: dict[str, IssueType] = {
    "bug V2": "bug",  # V2 only: V1-era issues describe a different codebase
    "feature request": "feature",
    "question": "question",
    "documentation": "docs",
}
PENDING = "pending"
# GitHub's author_association values for people with rights on the repo.
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


class GroundTruth(BaseModel):
    type: IssueType
    label: str
    trust: Literal["gold", "silver"]
    relabelled: bool  # a maintainer replaced the reporter's label: a hard case
    # The form the reporter picked (their first label of ours), or None. For the
    # "trust the form" BASELINE only — never shown to the model: it's
    # reporter-controlled, and wrong for most questions, so it would anchor the
    # model on the reporter's mistake (and the baseline comparison would no
    # longer be independent).
    reporter_label: str | None = None


class Excluded(BaseModel):
    reason: str


def ground_truth(issue: dict, events: list[dict]) -> GroundTruth | Excluded:
    """Decide the answer key for one closed issue from its labels and events."""
    names = [label["name"] for label in issue.get("labels") or []]
    if PENDING in names:
        return Excluded(reason="unreviewed: still pending")  # D1
    ours = [name for name in names if name in LABEL_TO_TYPE]
    if not ours:
        return Excluded(reason="none of our labels")
    if len(ours) > 1:
        return Excluded(
            reason=f"several of our labels: {', '.join(sorted(ours))}"
        )  # D3
    label = ours[0]

    reporter = (issue.get("user") or {}).get("login")
    label_events = [e for e in events if e.get("event") in ("labeled", "unlabeled")]

    def by_maintainer(event: dict) -> bool:
        # Not the reporter, and not a bot: a bot's label isn't a human judgement.
        actor = event.get("actor") or {}
        return actor.get("login") not in (None, reporter) and actor.get("type") != "Bot"

    confirmed = (
        issue.get("author_association") in MAINTAINER_ASSOCIATIONS
        or any(
            e["event"] == "labeled" and e["label"]["name"] == label and by_maintainer(e)
            for e in label_events
        )
        or any(
            e["event"] == "unlabeled"
            and e["label"]["name"] == PENDING
            and by_maintainer(e)
            for e in label_events
        )
    )
    relabelled = any(
        e["event"] == "unlabeled"
        and e["label"]["name"] in LABEL_TO_TYPE
        and e["label"]["name"] != label
        and by_maintainer(e)
        for e in label_events
    )
    reporter_label = next(
        (
            e["label"]["name"]
            for e in label_events
            if e["event"] == "labeled"
            and e["label"]["name"] in LABEL_TO_TYPE
            and (e.get("actor") or {}).get("login") == reporter
        ),
        None,
    )
    return GroundTruth(
        type=LABEL_TO_TYPE[label],
        label=label,
        trust="gold" if confirmed else "silver",  # D2: reporter-only → silver
        relabelled=relabelled,
        reporter_label=reporter_label,
    )
