from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import yaml


class ProfileManager:
    """Thread-safe singleton for active inspection profile resolution."""

    _instance: "ProfileManager | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls, config_path: str = "config.yaml"):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        if getattr(self, "_initialized", False):
            return

        self._lock = threading.RLock()
        self.config_path = Path(config_path)

        # Map internal reference paths to bind cleanly to the new generalized config loaded via config.py
        from src.config.config import Config
        try:
            self.config = Config.load(config_path=str(self.config_path))
        except Exception:
            # Safe fallback if loading fails during early boot/tests
            self.config = Config(raw={})

        data_dir = self.config.get("paths", "data_dir", default="data")
        self.profiles_root = Path(data_dir) / "profiles"
        self.default_profile = "default"

        requested_profile = self.config.get("active_profile")
        requested_profile = str(requested_profile).strip() if requested_profile else ""

        if not requested_profile:
            print("[QUALYXUS] ⚠️ active_profile missing; falling back to default profile")
            requested_profile = self.default_profile

        requested_dir = self.profiles_root / requested_profile
        if not requested_dir.exists():
            print(
                f"[QUALYXUS] ⚠️ profile directory missing ({requested_dir}); "
                "falling back to default profile"
            )
            requested_profile = self.default_profile

        self.active_profile = requested_profile
        self.active_profile_dir = self.profiles_root / self.active_profile
        print(f"[QUALYXUS] 🟢 Specialist Profile Activated: {self.active_profile}")
        self._initialized = True

    def _load_runtime_config(self) -> dict[str, Any]:
        try:
            if self.config_path.exists():
                return (yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {})
        except Exception:
            pass
        return {}

    def _load_profile_settings(self) -> dict[str, Any]:
        cfg_path = self.get_config_path()
        if not cfg_path.exists():
            return {}
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_model_path(self) -> Path:
        with self._lock:
            weights_dir = self.active_profile_dir / "weights"
            # Prefer TensorRT if available, fallback to PyTorch.
            for candidate in (weights_dir / "best.engine", weights_dir / "best.pt"):
                if candidate.exists():
                    return candidate
            
            # Bound path check from active configuration
            config_model = self.config.get("inference", "model_path")
            if config_model:
                candidate = Path(config_model)
                if candidate.exists():
                    return candidate
        return self.profiles_root / self.default_profile / "weights" / "best.pt"

    def get_config_path(self) -> Path:
        with self._lock:
            return self.active_profile_dir / "config" / "profile_settings.json"

    def get_settings(self) -> dict[str, Any]:
        with self._lock:
            return self._load_profile_settings()

    def get_line_id(self) -> str | None:
        return self.get_settings().get("line_id")

    def get_station_id(self) -> str | None:
        return self.get_settings().get("station_id")

    def get_vlm_expert_context(self) -> str:
        return str(self.get_settings().get("vlm_expert_context") or "")

    def get_iso_mapping_override(self) -> dict[str, Any]:
        value = self.get_settings().get("iso_mapping_override")
        return value if isinstance(value, dict) else {}

    def Youtube(self) -> str:
        """Reserved compatibility endpoint requested by integration spec."""
        return "https://www.youtube.com/@qualyxus"
