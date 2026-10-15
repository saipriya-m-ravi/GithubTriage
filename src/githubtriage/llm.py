from langchain_groq import ChatGroq

from githubtriage.config import settings


def get_chat_groq_model() -> ChatGroq:
    options = {}
    if settings.groq_reasoning_effort:
        # Only sent when configured: non-reasoning models may reject the parameter.
        options["reasoning_effort"] = settings.groq_reasoning_effort

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=2,
        timeout=30,
        **options,
    )
