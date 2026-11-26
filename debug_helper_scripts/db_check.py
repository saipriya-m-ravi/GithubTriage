"""Check the duplicate-finder database: connect, enable pgvector, print versions.

docker compose up -d
uv run python debug_helper_scripts/db_check.py
"""

import sys

from githubtriage.db import DatabaseNotConfigured, DatabaseUnavailable, check, migrate

sys.stdout.reconfigure(encoding="utf-8")
try:
    versions = check()
    applied = migrate()
except (DatabaseNotConfigured, DatabaseUnavailable) as e:
    print(f"✗ {e}", file=sys.stderr)
    sys.exit(1)
print(f"✓ connected: Postgres {versions['postgres']}, pgvector {versions['pgvector']}")
print(f"✓ migrations applied now: {applied or 'none (up to date)'}")
