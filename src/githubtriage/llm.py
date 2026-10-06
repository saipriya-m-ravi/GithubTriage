from langchain_groq import ChatGroq

from githubtriage.config import settings


def get_chat_groq_model() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=2,
        timeout=30
    )
