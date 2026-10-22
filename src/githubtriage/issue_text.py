"""Turn an issue into safe, size-limited text for an LLM prompt.

Shared by every node that sends issue text to a model (classify, completeness, ...).
"""

import re

MAX_BODY_CHARS = 8000

# Matches <issue>, </issue>, <title>, </TITLE >, < body> ... in any letter case.
_TAG_PATTERN = re.compile(r"<\s*/?\s*(issue|title|body)\s*>", re.IGNORECASE)


def sanitise(text: str) -> str:
    """Remove our delimiter tags so user text can't close them early."""
    return _TAG_PATTERN.sub("", text)


def truncate_middle(text: str, max_chars: int = MAX_BODY_CHARS) -> str:
    """Keep the start and the end of long text, dropping the middle.

    Bug reports often end with a traceback whose last lines hold the actual error,
    so keeping only the start would throw away the most useful part. Limited by
    characters, not lines: a single log line can be tens of thousands of characters.
    """
    if len(text) <= max_chars:
        return text
    head = max_chars * 6 // 10  # description and repro steps usually come first
    tail = max_chars - head  # the end of a traceback holds the error
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted} characters omitted]...\n{text[-tail:]}"


def build_issue_block(title: str, body: str, max_chars: int = MAX_BODY_CHARS) -> str:
    """Sanitise, truncate and wrap the issue in delimiter tags."""
    title = sanitise(title)
    body = truncate_middle(sanitise(body), max_chars)
    return f"<issue>\n<title>{title}</title>\n<body>\n{body}\n</body>\n</issue>"
