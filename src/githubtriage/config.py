from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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


settings = Settings()
