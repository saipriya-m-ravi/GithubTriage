"""The index of past issues for the duplicate finder (slice 5 module 3).

GitHub issue JSON -> `issues` rows; title + body -> an embedding in
`issue_embeddings`. Idempotent and incremental: issues are upserted, and an
issue is (re-)embedded only when the text it would embed has changed.

Embedding model (decision D2): BAAI/bge-small-en-v1.5, run locally on CPU with
fastembed — free, no rate limit, private. 384 dimensions (matches the schema).
It reads ~512 tokens; longer text is silently ignored, so we cut it ourselves
and keep the START (title + description). The keyword index (`issues.search`)
still sees the whole body, including long tracebacks with exact error names.
"""

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pgvector.psycopg import register_vector
from pydantic import AwareDatetime, BaseModel

from githubtriage.config import settings
from githubtriage.github_issues import NotAnIssueError
from githubtriage.repo_names import validate_repo

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384
# ~512 tokens ≈ 2,000 characters of English prose; code and tracebacks use
# more tokens per character, so this errs on the side of what the model reads.
EMBED_MAX_CHARS = 2000


class IssueRecord(BaseModel):
    repo: str
    number: int
    title: str
    body: str
    state: str
    state_reason: str | None
    labels: list[str]
    html_url: str
    created_at: AwareDatetime
    updated_at: AwareDatetime
    closed_at: AwareDatetime | None


def record_from_github(raw: dict) -> IssueRecord:
    """GitHub issue JSON -> IssueRecord. Pull requests are refused."""
    if "pull_request" in raw:
        raise NotAnIssueError(f"#{raw.get('number')} is a pull request, not an issue")
    repo = validate_repo(raw["repository_url"].split("/repos/", 1)[-1])
    return IssueRecord(
        repo=repo,
        number=raw["number"],
        title=raw["title"],
        body=raw.get("body") or "",
        state=raw["state"],
        state_reason=raw.get("state_reason"),
        labels=[label["name"] for label in raw.get("labels") or []],
        html_url=raw["html_url"],
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
        closed_at=raw.get("closed_at"),
    )


def embed_text(record: IssueRecord) -> str:
    """What gets embedded: title, then the start of the body, cut to what the
    model actually reads (so the hash below matches what was embedded)."""
    text = f"{record.title}\n\n{record.body}".strip()
    return text[:EMBED_MAX_CHARS]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def model_cache_dir() -> Path:
    """Where the embedding model's files live: DATA_DIR/models (gitignored).

    Explicit rather than fastembed's default (the OS temp folder, which
    clean-up tools may empty — forcing a re-download that needs internet).
    """
    return settings.data_dir / "models"


class FastEmbedder:
    """The real embedder: runs locally on CPU (onnxruntime), no API.

    fastembed downloads a quantised ONNX copy of the model once (~64 MB, from
    the Hugging Face Hub: Qdrant/bge-small-en-v1.5-onnx-Q) into
    model_cache_dir(); afterwards nothing leaves the machine. Tests use a fake.

    Not yet pinned to a model REVISION: fastembed fetches the repository's
    current files. The model name stored with every vector guards against
    mixing models, not against the same name changing — pinning (and baking the
    model into the Docker image) is a slice 7 item.
    """

    def __init__(self, model: str = EMBED_MODEL, batch_size: int = 32):
        from fastembed import TextEmbedding  # heavy import: only when used

        self.model = model
        cache = model_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model, cache_dir=str(cache))
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            vector.tolist()
            for vector in self._model.embed(list(texts), batch_size=self._batch_size)
        ]


class IndexReport(BaseModel):
    issues_upserted: int = 0
    embedded: int = 0
    unchanged: int = 0  # text identical to what is already embedded


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def index_issues(
    conn, records: Iterable[IssueRecord], embedder: Embedder, *, batch_size: int = 64
) -> IndexReport:
    """Upsert issues, then embed those whose embed_text changed (or is new).

    One transaction per batch: a crash keeps finished batches, and a re-run
    skips everything whose hash already matches. That needs a connection with
    autocommit=True (db.connect(autocommit=True)), where each
    `conn.transaction()` block is a real commit. Inside an outer transaction
    (as in the tests) the blocks are savepoints and nothing is committed.
    """
    register_vector(conn)
    report = IndexReport()
    for batch in _batches(list(records), batch_size):
        with conn.transaction():
            for r in batch:
                conn.execute(UPSERT_ISSUE, r.model_dump())
            report.issues_upserted += len(batch)

            existing = dict(
                conn.execute(
                    "SELECT number, text_sha256 FROM issue_embeddings"
                    " WHERE repo = %s AND model = %s AND number = ANY(%s)",
                    (batch[0].repo, embedder.model, [r.number for r in batch]),
                ).fetchall()
            )
            todo = []
            for r in batch:
                text = embed_text(r)
                digest = text_hash(text)
                if existing.get(r.number) == digest:
                    report.unchanged += 1
                else:
                    todo.append((r, text, digest))
            if todo:
                vectors = embedder.embed([text for _, text, _ in todo])
                for (r, _, digest), vector in zip(todo, vectors, strict=True):
                    if len(vector) != EMBED_DIMS:
                        raise ValueError(
                            f"{embedder.model} gave {len(vector)} dims, schema has {EMBED_DIMS}"
                        )
                    conn.execute(
                        UPSERT_EMBEDDING,
                        (r.repo, r.number, embedder.model, vector, digest),
                    )
                report.embedded += len(todo)
    return report


# All values are passed as parameters — never formatted into the SQL.
UPSERT_ISSUE = """
INSERT INTO issues (repo, number, title, body, state, state_reason, labels,
                    html_url, created_at, updated_at, closed_at)
VALUES (%(repo)s, %(number)s, %(title)s, %(body)s, %(state)s, %(state_reason)s,
        %(labels)s, %(html_url)s, %(created_at)s, %(updated_at)s, %(closed_at)s)
ON CONFLICT (repo, number) DO UPDATE SET
    title = EXCLUDED.title, body = EXCLUDED.body, state = EXCLUDED.state,
    state_reason = EXCLUDED.state_reason, labels = EXCLUDED.labels,
    html_url = EXCLUDED.html_url, updated_at = EXCLUDED.updated_at,
    closed_at = EXCLUDED.closed_at
"""

UPSERT_EMBEDDING = """
INSERT INTO issue_embeddings (repo, number, model, embedding, text_sha256)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (repo, number, model) DO UPDATE SET
    embedding = EXCLUDED.embedding, text_sha256 = EXCLUDED.text_sha256,
    embedded_at = now()
"""


def fetch_issue_records(
    gh, repo: str, *, created_since: datetime
) -> Iterator[IssueRecord]:
    """All issues (open and closed, no PRs) created since `created_since`.

    GitHub lists newest-created first, so paging stops at the first older one.
    Pydantic's list is mostly pull requests (filtered out by the client), so a
    generous page limit is needed.
    """
    for raw in gh.list_issues(repo, state="all", max_pages=300):
        if datetime.fromisoformat(raw["created_at"]) < created_since:
            return
        yield record_from_github(raw)
