"""R6 (after approval): check the issue hasn't changed, then publish the reply.

POST_MODE=outbox (default): write the reply to a local file — never touches GitHub.
POST_MODE=github: post a real comment, only to repositories in POST_ALLOWED_REPOS.

Posting must be safe to repeat. A checkpoint is only saved after a node returns,
so a crash after posting but before returning makes the retry run `post` again
with `posted_comment_id` still unset. The protection therefore lives in the side
effect itself:
- outbox: a deterministic file name per run + an atomic write;
- github: a hidden marker per run in the comment; before posting, look for it
  among comments created since the run started, and reuse it if found.

The idempotency key is the run (thread id), not the issue: a retry of the same
run must not duplicate, but a new triage of the same issue may post a new reply.
"""

import logging
import os
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from githubtriage.checkpointing import parse_thread_id, thread_started_at
from githubtriage.config import settings
from githubtriage.github_client import GitHubClient
from githubtriage.models import TriageState
from githubtriage.nodes.draft import FOOTER

logger = logging.getLogger(__name__)

# Bot disclosure on the public comment (DESIGN §5.3). No @mention of the approver:
# --approver is an OS username until OAuth (slice 7), and "@name" could notify an
# unrelated GitHub user.
POSTED_FOOTER = "_Drafted by GithubTriage, reviewed by a maintainer._"


class PostNotAllowedError(ValueError):
    """POST_MODE=github, but the repository isn't in POST_ALLOWED_REPOS."""


def stale_check(state: TriageState, config: RunnableConfig) -> dict:
    """Has the issue changed since the draft was made? If so, don't post it.

    "Changed" means what affects the reply — not GitHub's updated_at, which also
    moves when a maintainer adds a label or an assignee:
    - the title or body differs from the snapshot taken at triage, or
    - a comment other than this run's own reply was added since the run started
      (e.g. the reporter added the missing version in a comment). See
      is_this_runs_reply: the token may be a person's own account, so "same
      author" isn't enough.
    A stale draft is discarded (stale_reason set; the graph ends without posting)
    and the maintainer is asked to re-triage.

    GitHub errors propagate: the run stays at this step, so we never post
    unchecked, and `githubtriage retry` resumes it.
    """
    if state.issue.html_url is None:  # a local issue file: nothing to compare with
        logger.info(
            "Stale check skipped for %s#%s (not a GitHub issue)",
            state.repo,
            state.issue_number,
        )
        return {}

    thread = config["configurable"]["thread_id"]
    since = thread_started_at(thread).strftime("%Y-%m-%dT%H:%M:%SZ")
    has_token = bool(settings.github_token and settings.github_token.get_secret_value())
    reasons = []
    with GitHubClient(settings.github_token, settings.github_api_url) as gh:
        live = gh.get_issue(state.repo, state.issue_number)
        if (live.get("title") or "") != state.issue.title or (
            live.get("body") or ""
        ) != state.issue.body:
            reasons.append("the issue was edited")

        bot_login = gh.get_authenticated_login() if has_token else None
        others = [
            c
            for c in gh.list_comments(
                state.repo, state.issue_number, since=since, complete=True
            )
            if not is_this_runs_reply(c, bot_login, thread)
        ]
        if others:
            reasons.append(f"{len(others)} new comment(s) were added")

    if not reasons:
        return {}
    reason = " and ".join(reasons) + " since the draft was made"
    logger.warning("Draft for %s is stale: %s", thread, reason)
    return {"stale_reason": reason}


def marker(thread: str) -> str:
    """Hidden marker identifying this run's reply (GitHub doesn't display HTML
    comments). Checked before posting, so a retry finds the existing comment
    instead of posting again.
    """
    return f"<!-- githubtriage:{thread} -->"


def is_this_runs_reply(comment: dict, bot_login: str | None, thread: str) -> bool:
    """Is this comment the reply this run posted? Used by both stale_check and post.

    Both must hold:
    - written by the token's account — anyone can copy a marker into their own
      comment, so a marker alone could be forged (a suppression attack);
    - starts with this run's marker — the token may belong to a person (a
      personal access token posts as its owner), so the maintainer's own
      comments share the author and must still count as new information. A
      reply from another run is new information too.
    """
    author = (comment.get("user") or {}).get("login")
    body = comment.get("body") or ""
    return (
        bot_login is not None
        and author == bot_login
        and body.startswith(marker(thread))
    )


def with_disclosure(body: str) -> str:
    """The reply as published: the draft's "pending review" footer becomes the
    posted disclosure; an edited reply without a footer gets one added."""
    if FOOTER in body:
        return body.replace(FOOTER, POSTED_FOOTER)
    return f"{body.rstrip()}\n\n{POSTED_FOOTER}"


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


def _post_to_outbox(thread: str, text: str) -> str:
    path = outbox_path(thread)
    _write_atomically(path, text)
    logger.info("Posted reply for %s to %s", thread, path)
    return f"outbox:{thread}"


def _post_to_github(state: TriageState, thread: str, text: str) -> str:
    # Enforced here — the last point before the action — so every caller (the CLI
    # now, the webhook worker in slice 7) is covered.
    if state.repo.lower() not in settings.post_allowed_repos:
        raise PostNotAllowedError(
            f"posting to {state.repo} is not allowed (POST_ALLOWED_REPOS)"
        )

    # Our comment can only be newer than the run, so search from the run's start:
    # few comments, usually one request. complete=True: if the search can't see
    # everything, fail closed (don't post) rather than risk a duplicate.
    since = thread_started_at(thread).strftime("%Y-%m-%dT%H:%M:%SZ")
    with GitHubClient(settings.github_token, settings.github_api_url) as gh:
        # Only trust a marker in a comment by our own account (is_this_runs_reply).
        bot_login = gh.get_authenticated_login()
        for comment in gh.list_comments(
            state.repo, state.issue_number, since=since, complete=True
        ):
            if is_this_runs_reply(comment, bot_login, thread):
                logger.info(
                    "Reply for %s already on GitHub (comment %s); not posting again",
                    thread,
                    comment["id"],
                )
                return f"github:{comment['id']}"

        created = gh.create_comment(state.repo, state.issue_number, text)
    logger.info("Posted reply for %s as GitHub comment %s", thread, created["id"])
    return f"github:{created['id']}"


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
    text = f"{marker(thread)}\n{with_disclosure(body)}\n"
    if settings.post_mode == "github":
        posted_id = _post_to_github(state, thread, text)
    else:
        posted_id = _post_to_outbox(thread, text)

    # Deterministic per run: a retry after a crash produces the same state.
    return {"posted_comment_id": posted_id}
