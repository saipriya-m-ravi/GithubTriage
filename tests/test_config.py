import pytest
from pydantic import ValidationError

from githubtriage.config import Settings

REQUIRED = {"groq_api_key": "k", "groq_model": "m"}


def make_settings(**overrides) -> Settings:
    # _env_file=None: only these values, never the real .env.
    return Settings(_env_file=None, **{**REQUIRED, **overrides})


def test_github_defaults_are_safe():
    s = make_settings()
    assert s.post_mode == "outbox"  # nothing goes public by default
    assert s.post_allowed_repos == []
    assert s.github_token is None
    assert s.github_api_url == "https://api.github.com"


def test_allowed_repos_parse_from_a_comma_separated_env_value(monkeypatch):
    # The real path: read from an environment variable, not passed in code.
    monkeypatch.setenv("POST_ALLOWED_REPOS", " Me/Sandbox , me/other-repo ,")

    s = make_settings()

    # Trimmed, empty parts dropped, lower-cased (GitHub names are case-insensitive).
    assert s.post_allowed_repos == ["me/sandbox", "me/other-repo"]


def test_invalid_allowed_repo_is_rejected():
    with pytest.raises(ValidationError, match="invalid repository"):
        make_settings(post_allowed_repos=["../../etc"])


def test_github_mode_needs_a_token():
    with pytest.raises(ValidationError, match="needs GITHUB_TOKEN"):
        make_settings(post_mode="github", post_allowed_repos=["me/sandbox"])


def test_github_mode_needs_an_allowlist():
    with pytest.raises(ValidationError, match="needs POST_ALLOWED_REPOS"):
        make_settings(post_mode="github", github_token="t")


def test_github_mode_fully_configured_is_accepted():
    s = make_settings(
        post_mode="github", github_token="t", post_allowed_repos=["me/sandbox"]
    )
    assert s.post_mode == "github"


def test_unknown_post_mode_is_rejected():
    with pytest.raises(ValidationError):
        make_settings(post_mode="everywhere")


def test_token_is_masked_when_printed():
    s = make_settings(github_token="ghp_secret123")
    assert "ghp_secret123" not in repr(s)
    assert "ghp_secret123" not in str(s)
