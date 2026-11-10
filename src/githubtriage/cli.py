"""Command-line adapter: triage an issue, then approve, edit or reject the draft.

    githubtriage triage pydantic/pydantic#12580      # live issue (or its URL)
    githubtriage triage samples/issues/bug.json      # or a local issue file
    githubtriage show    owner/repo#7                # the waiting draft
    githubtriage approve owner/repo#7
    githubtriage edit    owner/repo#7 --body-file reply.md
    githubtriage reject  owner/repo#7 --reason "duplicate of #99"
    githubtriage retry   owner/repo#7                # approved, but posting failed

Approved replies go to the local outbox unless POST_MODE=github (then a real
comment, only on repositories in POST_ALLOWED_REPOS).

`triage` and the decision commands are separate processes: the paused run is
saved in a SQLite checkpoint (settings.checkpoint_path) and resumed from there.
A ref is an issue ("owner/repo#7" — its one waiting draft) or an exact run id.

Keep this module thin — no triage logic here. The same core (`build_graph`) will be
called by the FastAPI adapter in slice 7.

Exit codes: 0 = success (triage included even if degraded — errors are printed as
warnings), 1 = bad input, nothing waiting, or invalid configuration,
2 = wrong usage (set by argparse).
"""

import argparse
import getpass
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.types import Command
from pydantic import ValidationError

from githubtriage.checkpointing import (
    cancel_waiting_runs,
    issue_key,
    new_thread_id,
    parse_thread_id,
    sqlite_checkpointer,
    stuck_runs,
    thread_config,
    waiting_runs,
)
from githubtriage.config import settings
from githubtriage.github_client import (
    AuthError,
    GitHubClient,
    GitHubError,
    NotFoundError,
    RateLimitError,
)
from githubtriage.github_issues import (
    NotAnIssueError,
    parse_issue_ref,
    state_from_github,
)
from githubtriage.graph import build_graph
from githubtriage.models import Decision, TriageState
from githubtriage.nodes.approval import build_approval_request
from githubtriage.nodes.post import PostNotAllowedError
from githubtriage.observability import (
    TracingConfigError,
    configure_tracing,
    run_config,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_CONFIG_ERROR = 1
EXIT_NOT_POSTED = 1  # approved, but the draft was stale and discarded


class CliError(Exception):
    """An expected problem to report to the user (no traceback), exit code 1."""


# --- Arguments --------------------------------------------------------------


def _default_approver() -> str:
    # Slice 7 replaces this with the GitHub identity from OAuth login.
    try:
        return getpass.getuser()
    except OSError:
        return "unknown"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)  # options every command has
    common.add_argument("-v", "--verbose", action="store_true", help="show debug logs")

    decider = argparse.ArgumentParser(add_help=False)  # for approve / edit / reject
    decider.add_argument("ref", help="issue (owner/repo#7 or its URL) or exact run id")
    decider.add_argument(
        "--approver", default=_default_approver(), help="who is deciding"
    )

    parser = argparse.ArgumentParser(
        prog="githubtriage",
        description="Triage GitHub issues; approve, edit or reject the draft replies.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    triage = sub.add_parser(
        "triage", parents=[common], help="triage an issue; pause for approval"
    )
    triage.add_argument(
        "source",
        help="a live issue (owner/repo#7 or its GitHub URL) or an issue JSON file",
    )
    triage.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )

    show = sub.add_parser("show", parents=[common], help="show a waiting draft")
    show.add_argument("ref", help="issue (owner/repo#7 or its URL) or exact run id")

    sub.add_parser("approve", parents=[common, decider], help="post the draft as is")

    edit = sub.add_parser(
        "edit", parents=[common, decider], help="post your edited reply instead"
    )
    edit.add_argument(
        "--body-file", type=Path, required=True, help="file with the edited reply"
    )

    reject = sub.add_parser("reject", parents=[common, decider], help="post nothing")
    reject.add_argument("--reason", help="why (kept in the decision record)")

    retry = sub.add_parser(
        "retry", parents=[common], help="retry an approved run whose posting failed"
    )
    retry.add_argument("ref", help="issue (owner/repo#7 or its URL) or exact run id")

    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The HTTP client logs every request at INFO; keep it quiet unless debugging.
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


# --- Helpers ----------------------------------------------------------------


def _load_state(path: Path) -> TriageState:
    """Read and validate an issue file, as a CliError on expected problems."""
    try:
        # utf-8-sig = UTF-8 that also strips a leading BOM (many Windows tools add one).
        return TriageState.model_validate_json(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise CliError(f"file not found: {path}") from None
    except ValidationError as e:
        raise CliError(f"invalid issue file {path}:\n{e}") from None


def _fetch_state(repo: str, number: int) -> TriageState:
    """Fetch a live issue from GitHub, as a CliError on expected problems."""
    anonymous = (
        settings.github_token is None or not settings.github_token.get_secret_value()
    )
    if anonymous:
        logger.warning(
            "No GITHUB_TOKEN: anonymous access (60 requests/hour, public repos only)"
        )
    token_hint = " Set GITHUB_TOKEN" if anonymous else ""
    try:
        with GitHubClient(settings.github_token, settings.github_api_url) as gh:
            return state_from_github(gh.get_issue(repo, number))
    except NotAnIssueError as e:
        raise CliError(str(e)) from None
    except NotFoundError:
        hint = f"{token_hint} if the repository is private." if anonymous else ""
        raise CliError(f"{repo}#{number}: not found, or no access.{hint}") from None
    except RateLimitError as e:
        hint = f"{token_hint} for 5,000 requests/hour." if anonymous else ""
        raise CliError(
            f"GitHub rate limit reached; retry in {e.retry_after / 60:.0f} min.{hint}"
        ) from None
    except AuthError:
        raise CliError("GITHUB_TOKEN is invalid or lacks permission") from None
    except GitHubError as e:
        raise CliError(f"GitHub request failed: {e}") from None


def _triage_input(source: str) -> TriageState:
    """A file path if that file exists; otherwise an issue reference or URL."""
    path = Path(source)
    if path.is_file():
        return _load_state(path)
    try:
        ref = parse_issue_ref(source)
    except NotAnIssueError as e:  # a pull-request URL: no API call needed
        raise CliError(str(e)) from None
    if ref is None:
        raise CliError(
            f"{source}: no such file, and not an issue reference "
            "(owner/repo#7 or https://github.com/owner/repo/issues/7)"
        )
    return _fetch_state(*ref)


def _run_config(repo: str, issue_number: int, thread: str) -> dict:
    """Tracing metadata + which run to save/resume."""
    return {
        **run_config(
            repo,
            issue_number,
            model=settings.groq_model,
            source="cli",
            reasoning_effort=settings.groq_reasoning_effort,
        ),
        **thread_config(thread),
    }


def _issue_from_ref(ref: str) -> tuple[str, int]:
    """(repo, number) from "owner/repo#7" or an issue URL — the same forms
    `triage` accepts. ValueError (incl. NotAnIssueError for a PR URL) otherwise."""
    issue = parse_issue_ref(ref)
    if issue is None:
        raise ValueError(
            f"invalid issue reference: {ref!r} "
            "(expected owner/repo#7, an issue URL, or a run id)"
        )
    return issue


def _resolve(graph, saver, ref: str) -> str:
    """The waiting run a ref points to: an exact run id, or an issue's one draft."""
    try:
        if "@" in ref:
            parse_thread_id(ref)
            if graph.get_state(thread_config(ref)).next != ("approval",):
                raise CliError(
                    f"no draft waiting for run {ref} (already decided, or replaced "
                    "by a newer triage of the same issue)"
                )
            return ref
        repo, number = _issue_from_ref(ref)
    except ValueError as e:
        raise CliError(str(e)) from None

    waiting = waiting_runs(graph, saver, repo, number)
    if not waiting:
        raise CliError(f"no draft waiting for {ref}")
    if len(waiting) > 1:  # triage cancels older drafts, so this shouldn't happen
        raise CliError(f"several drafts waiting for {ref}: {', '.join(waiting)}")
    return waiting[0]


def _print_human(result: TriageState) -> None:
    c = result.classification
    if c is not None:
        print(f"type:       {c.type} ({c.confidence:.2f})")
        print(f"reasoning:  {c.reasoning}")
    print()
    print("---- draft ----")
    print(result.draft or "(no draft)")


# --- Commands ---------------------------------------------------------------


def _triage(args, graph, saver) -> int:
    state = _triage_input(args.source)

    # A newer triage replaces any draft of this issue still waiting for approval,
    # so at most one draft per issue is ever waiting.
    for old in cancel_waiting_runs(graph, saver, state.repo, state.issue_number):
        print(f"Cancelled previous draft: {old}", file=sys.stderr)

    # A fresh thread per run: reusing one would carry the old run's state over.
    thread = new_thread_id(state.repo, state.issue_number)
    config = _run_config(state.repo, state.issue_number, thread)
    result = TriageState.model_validate(graph.invoke(state, config=config))

    if args.json:
        print(result.model_dump_json(indent=2))  # stdout stays pure JSON
        print(f"Run: {thread}", file=sys.stderr)
    else:
        _print_human(result)
        key = issue_key(state.repo, state.issue_number)
        print(f"\nRun: {thread}")
        print(f"Waiting for approval: githubtriage approve|edit|reject {key}")

    # Degraded runs still produced a draft, so they exit 0 — but say so clearly.
    for error in result.errors:
        print(f"Warning: {error}", file=sys.stderr)
    # Tell the maintainer now, not after they approve: post enforces the allowlist.
    if (
        settings.post_mode == "github"
        and state.repo.lower() not in settings.post_allowed_repos
    ):
        print(
            f"Note: posting to {state.repo} is not allowed (POST_ALLOWED_REPOS); "
            "approving this draft will fail.",
            file=sys.stderr,
        )
    return EXIT_OK


def _show(args, graph, saver) -> int:
    thread = _resolve(graph, saver, args.ref)
    state = TriageState.model_validate(graph.get_state(thread_config(thread)).values)
    req = build_approval_request(state)

    print(f"Run:        {thread}")
    print(f"Issue:      {req.issue_title}  ({req.issue_url})")
    print(f"type:       {req.issue_type} ({req.confidence:.2f})")
    print(f"reasoning:  {req.reasoning}")
    if req.completeness is not None:
        print("completeness:")
        for field, review in req.completeness.items():
            note = f" — {review.note}" if review.note else ""
            print(f"  {field:<12} {review.status}{note}")
    for error in req.errors:
        print(f"error:      {error}")
    print("\n---- draft ----")
    print(req.draft)
    return EXIT_OK


def _decide(args, graph, saver, **decision_fields) -> int:
    """Resume the waiting run with a maintainer's decision."""
    thread = _resolve(graph, saver, args.ref)
    try:
        decision = Decision(
            approver=args.approver, decided_at=datetime.now(UTC), **decision_fields
        )
    except ValidationError as e:
        raise CliError(f"invalid decision:\n{e}") from None

    return _resume(graph, thread, Command(resume=decision.model_dump(mode="json")))


def _resume(graph, thread: str, command) -> int:
    """Run a paused or stuck run to the end and report the outcome."""
    repo, number = parse_thread_id(thread)
    try:
        result = TriageState.model_validate(
            graph.invoke(command, config=_run_config(repo, number, thread))
        )
    except (GitHubError, PostNotAllowedError) as e:
        # The decision is already saved; the run is stuck before/at `post`.
        raise CliError(
            f"posting failed: {e}\nThe decision is saved. Fix the problem, then run: "
            f"githubtriage retry {issue_key(repo, number)}"
        ) from None

    if result.stale_reason:
        # Approved, but the issue changed after triage: the draft was discarded.
        print(
            f"Not posted: {result.stale_reason}. The draft was discarded.\n"
            f"Re-triage to draft a reply from the current issue: "
            f"githubtriage triage {issue_key(repo, number)}",
            file=sys.stderr,
        )
        return EXIT_NOT_POSTED

    posted = result.posted_comment_id
    if posted and posted.startswith("github:"):
        comment_id = posted.removeprefix("github:")
        print(
            f"Posted: https://github.com/{repo}/issues/{number}#issuecomment-{comment_id}"
        )
    elif posted:
        print(f"Posted: {posted}")
    else:
        print(f"Rejected: {thread} — nothing posted.")
    return EXIT_OK


def _retry(args, graph, saver) -> int:
    """Resume an approved run that failed before finishing (e.g. posting)."""
    try:
        if "@" in args.ref:
            parse_thread_id(args.ref)
            stuck = (
                [args.ref]
                if graph.get_state(thread_config(args.ref)).next
                in (
                    ("stale_check",),
                    ("post",),
                )
                else []
            )
        else:
            stuck = stuck_runs(graph, saver, *_issue_from_ref(args.ref))
    except ValueError as e:
        raise CliError(str(e)) from None
    if len(stuck) != 1:
        raise CliError(f"no approved run waiting to be retried for {args.ref}")
    # invoke(None) resumes from the last checkpoint: the failed step runs again,
    # and post's marker check makes that safe.
    return _resume(graph, stuck[0], None)


def _approve(args, graph, saver) -> int:
    return _decide(args, graph, saver, action="approve")


def _edit(args, graph, saver) -> int:
    try:
        body = args.body_file.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise CliError(f"file not found: {args.body_file}") from None
    return _decide(args, graph, saver, action="edit", edited_body=body)


def _reject(args, graph, saver) -> int:
    return _decide(args, graph, saver, action="reject", reason=args.reason)


COMMANDS = {
    "triage": _triage,
    "show": _show,
    "approve": _approve,
    "edit": _edit,
    "reject": _reject,
    "retry": _retry,
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    # Must run before any LangChain/LangGraph code: the SDK caches env values.
    try:
        tracing = configure_tracing(settings)
    except TracingConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if tracing:
        logger.info("LangSmith tracing on (project: %s)", settings.langsmith_project)

    try:
        with sqlite_checkpointer(settings.checkpoint_path) as saver:
            graph = build_graph(checkpointer=saver)
            return COMMANDS[args.command](args, graph, saver)
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    finally:
        # Traces upload in a background thread; a CLI can exit before it finishes.
        if tracing:
            wait_for_all_tracers()
