from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


# Engine cache: ensures we reuse connections across EventLogger instances and allows
# deterministic disposal on shutdown.
_ENGINE_CACHE: dict[str, Engine] = {}
_SESSION_CACHE: dict[str, Any] = {}


class DefectEventRow(Base):
    __tablename__ = "defect_events"

    event_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String, nullable=False)
    defect_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(Integer, nullable=False)

    batch_id = Column(String, nullable=True)
    operator_id = Column(String, nullable=True)
    machine_id = Column(String, nullable=True)
    shift_id = Column(String, nullable=True)
    line_id = Column(String, nullable=True)
    station_id = Column(String, nullable=True)
    temperature = Column(Float, nullable=True)
    vibration = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    lot_id = Column(String, nullable=True)

    full_image_path = Column(String, nullable=False)
    overlay_path = Column(String, nullable=True)
    crop_path = Column(String, nullable=True)
    audit_hash = Column(String, nullable=True)
    iso_clause = Column(String, nullable=True)
    vlm_narrative = Column(String, nullable=True)
    inspection_profile = Column(String, nullable=True)


def _migrate_defect_events(conn) -> None:
    """Best-effort in-place schema upgrades for existing SQLite deployments.

    Why this exists:
    - SQLite has no native ALTER TABLE ADD COLUMN IF NOT EXISTS.
    - SQLAlchemy's create_all() will not add new columns to an existing table.
    - Older deployments may have a partially populated defect_events schema,
      leading to runtime failures like: "OperationalError: no such column: line_id".

    This function is intentionally idempotent.
    """

    # If the table doesn't exist yet, PRAGMA returns an empty set and we do nothing.
    try:
        rows = conn.execute(text("PRAGMA table_info(defect_events)"))
        existing_cols = {str(r[1]) for r in rows.fetchall()}
    except Exception:
        existing_cols = set()

    if not existing_cols:
        return

    # NOTE: Keep this list in sync with DefectEventRow.
    desired: list[tuple[str, str]] = [
        ("batch_id", "VARCHAR"),
        ("operator_id", "VARCHAR"),
        ("machine_id", "VARCHAR"),
        ("shift_id", "VARCHAR"),
        ("line_id", "VARCHAR"),
        ("station_id", "VARCHAR"),
        ("temperature", "FLOAT"),
        ("vibration", "FLOAT"),
        ("speed", "FLOAT"),
        ("lot_id", "VARCHAR"),
        ("overlay_path", "VARCHAR"),
        ("crop_path", "VARCHAR"),
        ("audit_hash", "VARCHAR"),
        ("iso_clause", "VARCHAR"),
        ("vlm_narrative", "TEXT"),
        ("inspection_profile", "VARCHAR"),
    ]

    for col, col_type in desired:
        if col in existing_cols:
            continue
        try:
            conn.execute(text(f"ALTER TABLE defect_events ADD COLUMN {col} {col_type}"))
        except Exception:
            # Backward-compatible migration: column may already exist or DB may be read-only/locked.
            pass


def init_db(sqlite_path: str):
    """Return a SQLAlchemy sessionmaker for the given sqlite_path.

    This function is idempotent and caches the Engine so that:
    - multiple EventLogger instances reuse a single engine
    - CameraManager.stop() can dispose the engine deterministically
    """

    sqlite_path = str(sqlite_path)
    if sqlite_path in _SESSION_CACHE:
        return _SESSION_CACHE[sqlite_path]

    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_path}", echo=False, future=True)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        _migrate_defect_events(conn)
        conn.commit()
    Base.metadata.create_all(engine)

    _ENGINE_CACHE[sqlite_path] = engine
    SessionLocal = sessionmaker(bind=engine, future=True)
    _SESSION_CACHE[sqlite_path] = SessionLocal
    return SessionLocal


def get_engine(sqlite_path: str) -> Engine | None:
    """Return cached engine if it exists (does not create a new one)."""
    return _ENGINE_CACHE.get(str(sqlite_path))


def dispose_db(sqlite_path: str) -> None:
    """Dispose cached engine and clear caches.

    Best-effort, never raises.
    """
    try:
        key = str(sqlite_path)
        eng = _ENGINE_CACHE.pop(key, None)
        _SESSION_CACHE.pop(key, None)
        if eng is not None:
            eng.dispose()
    except Exception:
        pass
