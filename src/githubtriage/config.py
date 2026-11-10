from pathlib import Path
from typing import Annotated, Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from githubtriage.repo_names import REPO_PATTERN


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required: the app can't classify without an LLM.
    groq_api_key: SecretStr
    groq_model: str
    # Only for reasoning models (e.g. gpt-oss: "low" / "medium" / "high").
    # None = don't send the parameter; the model uses its own default.
    groq_reasoning_effort: str | None = None

    # Optional: tracing sends data to a third party, so it is off unless enabled.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "githubtriage-dev"

    # Local working data: paused runs (checkpoints) and the outbox of "posted"
    # comments. Relative paths resolve from the directory you run the CLI in.
    data_dir: Path = Path(".githubtriage")

    # GitHub. The token is optional overall (triaging a local file doesn't need it)
    # but required by the commands that call GitHub — they check when they run.
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"  # GitHub Enterprise, or tests
    # Where approved replies go. "outbox" (default) never touches GitHub; posting
    # publicly has to be switched on deliberately.
    post_mode: Literal["outbox", "github"] = "outbox"
    # Repositories the bot may post to, even in "github" mode — anything else is
    # refused. Comma-separated in .env: POST_ALLOWED_REPOS=owner/repo,owner/other
    post_allowed_repos: Annotated[list[str], NoDecode] = []

    @field_validator("post_allowed_repos", mode="before")
    @classmethod
    def split_repo_list(cls, value):
        # NoDecode: read the env value as a plain string instead of JSON,
        # so the comma-separated form above works.
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("post_allowed_repos")
    @classmethod
    def repos_are_owner_slash_name(cls, repos: list[str]) -> list[str]:
        for repo in repos:
            if not REPO_PATTERN.fullmatch(repo):
                raise ValueError(f"invalid repository in POST_ALLOWED_REPOS: {repo!r}")
        # GitHub repository names are case-insensitive; compare in lower case.
        return [repo.lower() for repo in repos]

    @model_validator(mode="after")
    def github_mode_is_fully_configured(self) -> "Settings":
        # Fail at startup for settings that are wrong whatever command runs.
        if self.post_mode == "github":
            if self.github_token is None or not self.github_token.get_secret_value():
                raise ValueError("POST_MODE=github needs GITHUB_TOKEN")
            if not self.post_allowed_repos:
                raise ValueError(
                    "POST_MODE=github needs POST_ALLOWED_REPOS (no repo would be allowed)"
                )
        return self

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"


settings = Settings()
