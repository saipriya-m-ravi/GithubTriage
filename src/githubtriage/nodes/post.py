"""R6 (after approval): check the issue hasn't changed, then publish the reply.

Slice 3 "publishes" to a local outbox file; slice 4 replaces it with the GitHub API.

Posting must be safe to repeat. A checkpoint is only saved after a node returns,
so a crash after writing but before returning makes the retry run `post` again
with `posted_comment_id` still unset. The protection therefore lives in the side
effect itself: a deterministic file name per RUN (a second write overwrites the
same file) and an atomic write (temp file + rename, never a half-written file).

The idempotency key is the run (thread id), not the issue: a retry of the same
run must not duplicate, but a new triage of the same issue may post a new reply.
"""

import logging
import os
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from githubtriage.checkpointing import parse_thread_id
from githubtriage.config import settings
from githubtriage.models import TriageState

logger = logging.getLogger(__name__)


def stale_check(state: TriageState) -> dict:
    """Has the issue changed since the draft was made? Stub until slice 4.

    TODO(slice 4): fetch the issue's current updated_at from the GitHub API and
    compare it with state.issue.updated_at; if it changed, don't post — flag the
    draft as stale for the maintainer instead.
    """
    logger.info(
        "Stale check skipped for %s#%s (no GitHub API until slice 4)",
        state.repo,
        state.issue_number,
    )
    return {}


def marker(thread: str) -> str:
    """Hidden marker identifying this run's reply (GitHub doesn't display HTML
    comments). Slice 4 checks for it before posting, so a retry finds the
    existing comment instead of posting again.
    """
    return f"<!-- githubtriage:{thread} -->"


def outbox_path(thread: str) -> Path:
    """One deterministic file per run: writing again overwrites, never duplicates.

    The thread id is validated first (owner/name#N@timestamp), so nothing like
    "../.." can become part of the path (path traversal).
    """
    parse_thread_id(thread)  # raises ValueError if it isn't a valid run id
    return settings.outbox_dir / f"{thread.replace('/', '__')}.md"


def _write_atomically(path: Path, text: str) -> None:
    """Write to a temp file, then rename: the file is old or complete, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem


def post(state: TriageState, config: RunnableConfig) -> dict:
    """Publish the approved (or edited) reply. Safe to run more than once.

    LangGraph passes `config` to nodes that ask for it; it carries the thread id
    of this run, which is the idempotency key.
    """
    if state.posted_comment_id:
        return {}  # a shortcut for normal re-runs — not the duplicate protection

    decision = state.decision
    # Taking a public action: never guess. The router should make these impossible,
    # so if they happen it's a bug — fail loudly rather than post something.
    if decision is None or decision.action not in ("approve", "edit"):
        raise ValueError(f"post reached without approval: {decision!r}")
    body = decision.edited_body if decision.action == "edit" else state.draft
    if not body or not body.strip():
        raise ValueError("refusing to post an empty reply")

    thread = config["configurable"]["thread_id"]
    path = outbox_path(thread)
    _write_atomically(path, f"{marker(thread)}\n{body}\n")
    logger.info("Posted reply for %s to %s", thread, path)

    # Deterministic id: a retry after a crash produces exactly the same state.
    return {"posted_comment_id": f"outbox:{thread}"}
