"""R6: ask a maintainer to approve, edit or reject the draft — and wait.

notify -> approval (interrupt) -> route_after_approval

On resume, LangGraph re-runs the paused node FROM THE START, so nothing before
`interrupt()` in `approval` may have side effects. That's why notifying is its own
node: a completed node never re-runs, so the notification goes out exactly once.
"""

import logging
from typing import Literal

from langgraph.types import interrupt

from githubtriage.models import ApprovalRequest, Decision, FieldReview, TriageState

logger = logging.getLogger(__name__)


def build_approval_request(state: TriageState) -> ApprovalRequest:
    """The minimal view of the state a maintainer needs to decide. Pure function."""
    c = state.classification
    completeness = (
        {
            field: FieldReview(status=check.status, note=check.note)
            for field, check in state.completeness
        }
        if state.completeness is not None
        else None
    )
    return ApprovalRequest(
        repo=state.repo,
        issue_number=state.issue_number,
        issue_title=state.issue.title,
        issue_url=f"https://github.com/{state.repo}/issues/{state.issue_number}",
        issue_type=c.type if c else "unclear",
        confidence=c.confidence if c else 0.0,
        reasoning=c.reasoning if c else "",
        completeness=completeness,
        draft=state.draft or "",
        errors=list(state.errors),
    )


def notify(state: TriageState) -> dict:
    """Tell maintainers a draft is waiting. Slice 3: a log line (Slack/email later)."""
    logger.info("Draft ready for approval: %s#%s", state.repo, state.issue_number)
    return {}


def approval(state: TriageState) -> dict:
    """Pause until a maintainer decides; resume with their Decision.

    Everything before interrupt() runs again on resume, so it must be pure.
    """
    request = build_approval_request(state)
    # Plain JSON crosses the boundary: no class names to recreate from storage,
    # and the same payload works for the CLI now and an HTTP API later.
    answer = interrupt(request.model_dump(mode="json"))
    # Validate what came back in: the resume value is input from outside the graph.
    return {"decision": Decision.model_validate(answer)}


def route_after_approval(state: TriageState) -> Literal["stale_check", "__end__"]:
    """approve / edit -> stale_check (then post); reject -> end."""
    if state.decision is not None and state.decision.action in ("approve", "edit"):
        return "stale_check"
    return "__end__"  # reject, and the default if no decision is present
