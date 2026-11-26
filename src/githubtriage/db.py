"""Postgres connection for the duplicate finder (slice 5).

One place that knows how to connect, so callers never handle the password: the
URL is a SecretStr in settings and is unwrapped only here. Errors never include
the URL (it contains the password) — they say which setting to check instead.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources

import psycopg

from githubtriage.config import settings

MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class DatabaseNotConfigured(RuntimeError):
    """DATABASE_URL isn't set (only the duplicate finder needs it)."""


class DatabaseUnavailable(RuntimeError):
    """The database couldn't be reached (is Docker running?)."""


@contextmanager
def connect(
    *, connect_timeout: int = 5, autocommit: bool = False
) -> Iterator[psycopg.Connection]:
    """A connection, closed afterwards.

    autocommit=False (psycopg's default): everything runs in one open
    transaction that YOU must commit — closing without commit() rolls it back.
    autocommit=True: each `with conn.transaction():` block is a real
    transaction (commit on success, rollback on error). Without autocommit,
    conn.transaction() is only a savepoint inside the outer transaction.
    """
    url = settings.database_url.get_secret_value() if settings.database_url else ""
    if not url:
        raise DatabaseNotConfigured(
            "DATABASE_URL is not set: see .env.example (database section)"
        )
    try:
        conn = psycopg.connect(
            url, connect_timeout=connect_timeout, autocommit=autocommit
        )
    except psycopg.OperationalError as e:
        # Never echo the URL or the driver message (may contain the URL/password).
        raise DatabaseUnavailable(
            "cannot reach the database at DATABASE_URL — is it running? "
            "(docker compose up -d)"
        ) from e
    try:
        yield conn
    finally:
        conn.close()


def migrations() -> list[tuple[str, str]]:
    """(version, sql) for every migration file, in order. Names must be
    NNNN_snake_case.sql and versions unique — checked, not assumed."""
    files = resources.files("githubtriage.migrations")
    found = []
    for entry in files.iterdir():
        if entry.name.endswith(".sql"):
            match = MIGRATION_NAME.match(entry.name)
            if not match:
                raise ValueError(f"bad migration file name: {entry.name}")
            found.append((match[1], entry.name, entry.read_text(encoding="utf-8")))
    found.sort()
    versions = [v for v, _, _ in found]
    if len(versions) != len(set(versions)):
        raise ValueError(f"duplicate migration versions: {versions}")
    return [(f"{v}:{name}", sql) for v, name, sql in found]


def migrate() -> list[str]:
    """Apply every migration not applied yet; return the ones applied now.

    Each migration runs in its own transaction together with its bookkeeping
    row, so a failing migration leaves nothing half-applied, and running
    migrate() twice is harmless (applied ones are skipped).
    """
    applied_now = []
    # autocommit: each conn.transaction() below is a REAL transaction. (The
    # first version used a normal connection: the blocks became savepoints in
    # one uncommitted transaction, and closing the connection rolled every
    # migration back — while reporting it as applied. A test caught it.)
    with connect(autocommit=True) as conn:
        with conn.transaction():
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version text PRIMARY KEY,"
                " applied_at timestamptz NOT NULL DEFAULT now())"
            )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, sql in migrations():
            if version in done:
                continue
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                )
            applied_now.append(version)
    return applied_now


def check() -> dict[str, str]:
    """Connect, make sure the vector extension exists, report versions."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("SELECT current_setting('server_version')")
        server = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        vector = cur.fetchone()[0]
        conn.commit()
    return {"postgres": server, "pgvector": vector}
