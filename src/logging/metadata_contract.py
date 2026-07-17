from __future__ import annotations

"""Immutable metadata dictionary contract for Qualyxus events.

This module is the *single source of truth* for the Event Metadata object that
flows across:
- Ingest (camera_manager)
- Inference/postprocess
- Reasoning/VLM prompt injection
- Persistence (SQLAlchemy)
- Dispatch (MQTT/UNS)

Design goals:
- snake_case keys only
- stable ordering & deterministic audit hashing
- easy validation + safe JSON serialization
"""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "1.0.0"

# Canonical key set. If a key exists in the SQL schema, it must exist here.
REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "event_id",
    "timestamp",
    "source",
    "defect_type",
    "confidence",
    "severity",
    "batch_id",
    "operator_id",
    "machine_id",
    "shift_id",
    "line_id",
    "station_id",
    "temperature",
    "vibration",
    "speed",
    "lot_id",
    "full_image_path",
    "overlay_path",
    "crop_path",
    "audit_hash",
    "iso_clause",
    "vlm_narrative",
    "inspection_profile",
)


def _isoformat(ts: Any) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat()
    if hasattr(ts, "isoformat"):
        try:
            return str(ts.isoformat())
        except Exception:
            pass
    return str(ts)


def _json_safe(value: Any) -> Any:
    """Convert unknown objects to JSON-safe scalars (never raises)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    # Avoid accidental numpy / torch / cv2 objects leaking into the audit trail.
    return str(value)


def to_metadata_dict(event: Any, *, include_schema_version: bool = True) -> dict[str, Any]:
    """Build a canonical metadata dict from a DefectEvent-like object.

    Variable names are preserved exactly as the contract specifies.
    """

    # Dataclass or plain object.
    if hasattr(event, "__dataclass_fields__"):
        raw = asdict(event)
    else:
        raw = {k: getattr(event, k, None) for k in REQUIRED_KEYS if k != "schema_version"}

    out: dict[str, Any] = {}
    if include_schema_version:
        out["schema_version"] = SCHEMA_VERSION

    # Copy keys in canonical order.
    for k in REQUIRED_KEYS:
        if k == "schema_version":
            continue
        if k == "timestamp":
            out[k] = _isoformat(raw.get(k))
        else:
            out[k] = _json_safe(raw.get(k))

    # Ensure audit_hash placeholder exists even if not yet generated.
    out.setdefault("audit_hash", None)
    return out


def validate_metadata(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [k for k in REQUIRED_KEYS if k not in metadata]
    # Enforce snake_case (no uppercase letters) to prevent drift.
    bad_keys = [k for k in metadata.keys() if any(c.isupper() for c in str(k))]
    errors: list[str] = []
    if missing:
        errors.append(f"missing_keys={missing}")
    if bad_keys:
        errors.append(f"non_snake_case_keys={bad_keys}")
    return (len(errors) == 0), errors


def compute_audit_hash(metadata: dict[str, Any]) -> str:
    """Deterministic SHA-256 over canonical JSON (excluding audit_hash itself)."""
    canonical = {k: metadata.get(k) for k in REQUIRED_KEYS if k != "audit_hash"}
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
