from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import yaml


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@dataclass(frozen=True)
class ThresholdPair:
    warning: float
    critical: float


@dataclass(frozen=True)
class ResourceManagementConfig:
    monitoring_interval: float = 2.0
    vram_thresholds: ThresholdPair = ThresholdPair(warning=85.0, critical=95.0)
    temp_thresholds: ThresholdPair = ThresholdPair(warning=78.0, critical=85.0)
    # Hysteresis / stability knobs (seconds)
    warning_hold_seconds: float = 120.0
    critical_hold_seconds: float = 300.0
    warning_cooldown_seconds: float = 300.0
    critical_cooldown_seconds: float = 900.0
    # Emergency (immediate) latch threshold
    emergency_temp_c: float = 90.0


@dataclass(frozen=True)
class Config:
    raw: dict

    @staticmethod
    def load(overrides: dict | None = None, config_path: str | None = None) -> "Config":
        # Priority:
        # 1) explicit argument
        # 2) CONFIG_PATH env
        # 3) project default yaml
        env_path = os.getenv("CONFIG_PATH")

        if config_path:
            resolved_config_path = Path(config_path)
        elif env_path:
            resolved_config_path = Path(env_path)
        else:
            project_root = Path(__file__).resolve().parents[2]
            resolved_config_path = project_root / "src" / "config" / "default.yaml"

        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

        with open(resolved_config_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)

        if base is None:
            raise RuntimeError(f"Config YAML is empty or invalid: {resolved_config_path}")

        if overrides:
            base = _deep_update(base, overrides)

        cfg = Config(raw=base)
        cfg.ensure_dirs()
        return cfg

    def get(self, *keys, default=None):
        # 1) Dynamic Path Resolution: Compute/resolve inference.yolo_yaml or dataset.yolo.yaml_path
        if keys == ("inference", "yolo_yaml") or keys == ("dataset", "yolo", "yaml_path"):
            explicit = None
            if keys == ("inference", "yolo_yaml"):
                explicit = self.raw.get("inference", {}).get("yolo_yaml")
            elif keys == ("dataset", "yolo", "yaml_path"):
                explicit = self.raw.get("dataset", {}).get("yolo", {}).get("yaml_path")

            if explicit:
                return os.path.abspath(explicit)

            processed_dir = self.raw.get("paths", {}).get("processed_dir") or "data/processed"
            dataset_name = self.raw.get("dataset", {}).get("name") or "active_product_recipe"
            return os.path.abspath(os.path.join(processed_dir, f"{dataset_name}_yolo.yaml"))

        # 2) Backwards compatibility: Map legacy yolo dataset queries to generalized dataset properties
        if keys == ("dataset", "yolo", "val_split"):
            return float(self.raw.get("dataset", {}).get("val_split", 0.20))
        if keys == ("dataset", "yolo", "seed"):
            return int(self.raw.get("dataset", {}).get("seed", 42))

        # 3) Backwards compatibility: Map legacy severstal queries to generalized raw dataset path
        if keys == ("severstal", "train_csv"):
            raw_path = self.raw.get("dataset", {}).get("raw_path") or self.raw.get("severstal", {}).get("train_csv")
            if raw_path:
                if os.path.isdir(raw_path) or not raw_path.endswith(".csv"):
                    return os.path.join(raw_path, "train.csv")
                return raw_path
            return "data/raw/train.csv"

        if keys == ("severstal", "train_images_dir"):
            raw_path = self.raw.get("dataset", {}).get("raw_path") or self.raw.get("severstal", {}).get("train_images_dir")
            if raw_path:
                if os.path.isdir(raw_path):
                    return raw_path
                return os.path.dirname(raw_path)
            return "data/raw/train_images"

        cur = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                # Robust Defaults: Fallback defaults inside .get() dictionary lookups
                if keys == ("dataset", "seed"):
                    return 42
                if keys == ("dataset", "min_contour_area"):
                    return 25
                return default
            cur = cur[k]
        return cur

    def ensure_dirs(self):
        # Defensive Bootstrapping: safely checks for and creates any missing generic data directories
        paths_to_create = [
            self.raw.get("paths", {}).get("data_dir", "data"),
            self.raw.get("paths", {}).get("raw_dir", "data/raw"),
            self.raw.get("paths", {}).get("processed_dir", "data/processed"),
            self.raw.get("paths", {}).get("events_dir", "data/events"),
            self.raw.get("paths", {}).get("reports_dir", "data/reports"),
            self.raw.get("paths", {}).get("models_dir", "models/checkpoints"),
            "data/templates",
        ]
        for p in paths_to_create:
            if p:
                try:
                    os.makedirs(p, exist_ok=True)
                except OSError:
                    pass  # Intercept and neutralize concurrent execution race conditions under WSL multi-processing

    @property
    def resource_management(self) -> ResourceManagementConfig:
        rm = self.raw.get("resource_management", {}) or {}
        vram = rm.get("vram_thresholds", {}) or {}
        temp = rm.get("temp_thresholds", {}) or {}
        return ResourceManagementConfig(
            monitoring_interval=float(rm.get("monitoring_interval", 2.0) or 2.0),
            vram_thresholds=ThresholdPair(
                warning=float(vram.get("warning", 85.0) or 85.0),
                critical=float(vram.get("critical", 95.0) or 95.0),
            ),
            temp_thresholds=ThresholdPair(
                warning=float(temp.get("warning", 78.0) or 78.0),
                critical=float(temp.get("critical", 85.0) or 85.0),
            ),
            warning_hold_seconds=float(rm.get("warning_hold_seconds", 120.0) or 120.0),
            critical_hold_seconds=float(rm.get("critical_hold_seconds", 300.0) or 300.0),
            warning_cooldown_seconds=float(rm.get("warning_cooldown_seconds", 300.0) or 300.0),
            critical_cooldown_seconds=float(rm.get("critical_cooldown_seconds", 900.0) or 900.0),
            emergency_temp_c=float(rm.get("emergency_temp_c", 90.0) or 90.0),
        )


def env_override_config(cfg: Config) -> Config:
    overrides = {"dataset": {}}

    # Generic overrides
    dataset_name = os.getenv("DATASET_NAME")
    dataset_format = os.getenv("DATASET_FORMAT")
    dataset_raw_path = os.getenv("DATASET_RAW_PATH")

    if dataset_name:
        overrides["dataset"]["name"] = dataset_name
    if dataset_format:
        overrides["dataset"]["format"] = dataset_format
        
    # Legacy fallbacks
    if dataset_raw_path:
        overrides["dataset"]["raw_path"] = dataset_raw_path
    else:
        legacy_images = os.getenv("SEVERSTAL_TRAIN_IMAGES_DIR")
        legacy_csv = os.getenv("SEVERSTAL_TRAIN_CSV")
        legacy_data = os.getenv("SEVERSTAL_DATA_DIR")
        
        if legacy_images:
            overrides["dataset"]["raw_path"] = legacy_images
        elif legacy_csv:
            overrides["dataset"]["raw_path"] = str(Path(legacy_csv).parent)
        elif legacy_data:
            overrides["dataset"]["raw_path"] = legacy_data

    if overrides["dataset"]:
        return Config.load(overrides=overrides)
    return cfg
