"""Database connection: configuration and error handling (no database needed).

The one real-database test is opt-in (GITHUBTRIAGE_DB_TESTS=1 + docker compose
up), so the normal suite never depends on Docker running on someone's machine.
"""

import os

import pytest
from pydantic import SecretStr

from githubtriage.config import settings
from githubtriage.db import DatabaseNotConfigured, DatabaseUnavailable, check, connect


def test_database_url_is_secret_and_masked(monkeypatch):
    monkeypatch.setattr(
        settings, "database_url", SecretStr("postgresql://u:hunter2@h/db")
    )
    assert "hunter2" not in repr(settings.database_url)
    assert "hunter2" not in str(settings.database_url)


def test_missing_database_url_says_which_setting(monkeypatch):
    monkeypatch.setattr(settings, "database_url", None)
    with pytest.raises(DatabaseNotConfigured, match="DATABASE_URL"), connect():
        pass


def test_unreachable_database_never_leaks_the_password(monkeypatch):
    # Port 1 on localhost: refused immediately — nothing is listening there.
    url = "postgresql://u:hunter2@127.0.0.1:1/db"
    monkeypatch.setattr(settings, "database_url", SecretStr(url))
    with pytest.raises(DatabaseUnavailable) as caught, connect(connect_timeout=1):
        pass
    assert "hunter2" not in str(caught.value)
    assert "docker compose up" in str(caught.value)


def test_migrations_are_numbered_and_ordered():
    from githubtriage.db import migrations

    versions = [v for v, _ in migrations()]
    assert versions == sorted(versions)
    assert versions[0].startswith("0001:")


needs_db = pytest.mark.skipif(
    os.environ.get("GITHUBTRIAGE_DB_TESTS") != "1",
    reason="needs a running database: docker compose up -d, GITHUBTRIAGE_DB_TESTS=1",
)


@needs_db
def test_real_database_has_pgvector():
    # Reads DATABASE_URL from .env (the autouse fixtures don't touch it).
    versions = check()
    assert versions["postgres"].startswith("17")
    assert versions["pgvector"]


@needs_db
def test_migrate_is_idempotent():
    from githubtriage.db import migrate

    migrate()  # whatever is pending
    assert migrate() == []  # a second run applies nothing


@needs_db
def test_schema_keyword_search_and_cascade():
    # Everything inside a transaction that is rolled back: no test rows remain.
    from datetime import UTC, datetime

    from githubtriage.db import migrate

    migrate()
    now = datetime.now(UTC)
    with connect() as conn:
        try:
            conn.execute(
                "INSERT INTO issues (repo, number, title, body, state, html_url,"
                " created_at, updated_at) VALUES"
                " ('t/t', 1, 'KeyError in model_validator', 'crashes on startup',"
                " 'open', 'https://github.com/t/t/issues/1', %s, %s)",
                (now, now),
            )
            # The generated tsvector is filled in automatically (stemmed: crash).
            hit = conn.execute(
                "SELECT number FROM issues WHERE search @@ plainto_tsquery('english', 'crash')"
                " AND repo = 't/t'"
            ).fetchall()
            assert hit == [(1,)]
            conn.execute(
                "INSERT INTO issue_embeddings (repo, number, model, embedding, text_sha256)"
                " VALUES ('t/t', 1, 'test-model', %s::vector, 'abc')",
                (str([0.0] * 383 + [1.0]),),
            )
            conn.execute("DELETE FROM issues WHERE repo = 't/t'")
            left = conn.execute(
                "SELECT count(*) FROM issue_embeddings WHERE repo = 't/t'"
            ).fetchone()
            assert left == (0,)  # embeddings go with their issue
        finally:
            conn.rollback()
