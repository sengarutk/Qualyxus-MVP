from __future__ import annotations

from typing import Any

METADATA_KEY_ALIASES: dict[str, str] = {
    "lineId": "line_id",
    "stationId": "station_id",
    "auditHash": "audit_hash",
    "machineId": "machine_id",
    "shiftId": "shift_id",
    "batchId": "batch_id",
    "operatorId": "operator_id",
    "lotId": "lot_id",
    "fullImagePath": "full_image_path",
    "overlayPath": "overlay_path",
    "cropPath": "crop_path",
    "defectType": "defect_type",
}

def normalize_metadata_keys(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    out: dict[str, Any] = {}
    for key, value in dict(metadata).items():
        normalized = METADATA_KEY_ALIASES.get(str(key), str(key))
        out[normalized] = value
    return out


def severity_from_conf(conf: float) -> int:
    return 3 if conf >= 0.85 else 2 if conf >= 0.6 else 1
