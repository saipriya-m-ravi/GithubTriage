"""The one definition of a valid GitHub repository name ("owner/name").

A security check, not just formatting: repository names end up in file paths
(outbox), URL paths (GitHub API) and run ids. Rejecting anything else stops path
traversal ("../..") and URL manipulation ("owner/repo/../../other").
"""

import re

REPO_PATTERN = re.compile(r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")


def validate_repo(repo: str) -> str:
    """Return the repo unchanged if it is "owner/name"; otherwise ValueError."""
    if not REPO_PATTERN.fullmatch(repo):
        raise ValueError(f"invalid repository name: {repo!r} (expected owner/name)")
    return repo
