"""SQLAlchemy schema source-of-truth.

This project primarily uses `src.logging.db` for runtime logging, but some
tooling and deployment docs refer to `src.database.models`.

We keep this file as the canonical import location and re-export the actual
SQLAlchemy models defined in `src.logging.db`.
"""

from __future__ import annotations

# Re-export the runtime schema.
from src.logging.db import Base, DefectEventRow  # noqa: F401

# Multi-tenant hardening marker: available on DefectEventRow as a nullable String column.
INSPECTION_PROFILE_COLUMN = "inspection_profile"
