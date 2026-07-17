from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SystemState(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass(frozen=True)
class SystemVitals:
    cpu_load_percent: float = 0.0
    ram_percent: float = 0.0
    cpu_temperature_c: float = 0.0
    gpu_temperature_c: float = 0.0
    vram_percent: float = 0.0
    timestamp: float = 0.0


@dataclass
class DefectEvent:
    event_id: str
    timestamp: datetime
    source: str
    defect_type: str
    confidence: float
    severity: int
    batch_id: str | None
    operator_id: str | None
    machine_id: str | None
    shift_id: str | None
    line_id: str | None
    station_id: str | None
    temperature: float | None
    vibration: float | None
    speed: float | None
    lot_id: str | None
    full_image_path: str
    overlay_path: str | None
    crop_path: str | None
    audit_hash: str | None = None
    iso_clause: str | None = None
    vlm_narrative: str | None = None
    inspection_profile: str | None = None
    frame_data: object | None = None  # Raw frame data for logger to save
