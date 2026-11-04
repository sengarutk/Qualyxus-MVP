from __future__ import annotations
import copy
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
        # 3) project deployment config (<root>/config.yaml) - the file main.py,
        #    the dashboard, CameraManager and ProfileManager are all built around
        # 4) packaged defaults (src/config/default.yaml)
        env_path = os.getenv("CONFIG_PATH")
        project_root = Path(__file__).resolve().parents[2]

        if config_path:
            resolved_config_path = Path(config_path)
        elif env_path:
            resolved_config_path = Path(env_path)
        elif (project_root / "config.yaml").exists():
            resolved_config_path = project_root / "config.yaml"
        else:
            resolved_config_path = project_root / "src" / "config" / "default.yaml"

        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

        with open(resolved_config_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)

        if base is None:
            raise RuntimeError(f"Config YAML is empty or invalid: {resolved_config_path}")

        # `extends: <file>` layers this config over another one (relative to this file).
        extends = base.pop("extends", None)
        if extends:
            with open(resolved_config_path.parent / extends, "r", encoding="utf-8") as f:
                base = _deep_update(yaml.safe_load(f), base)

        if overrides:
            base = _deep_update(base, overrides)

        cfg = Config(raw=base)
        cfg.ensure_dirs()
        return cfg

    def _resolve_dataset_yaml(self) -> str:
        """Single source of truth for the compiled YOLO dataset YAML.

        Both the dataset preparation scripts (writers) and the training/evaluation
        entry points (readers) resolve the path through here so they always agree:
        1) explicit dataset.yolo.yaml_path
        2) explicit inference.yolo_yaml
        3) <paths.processed_dir>/<dataset.name>_yolo.yaml
        """
        explicit = (self.raw.get("dataset", {}) or {}).get("yolo", {}) or {}
        explicit = explicit.get("yaml_path") if isinstance(explicit, dict) else None
        if not explicit:
            explicit = (self.raw.get("inference", {}) or {}).get("yolo_yaml")
        if explicit:
            return os.path.abspath(explicit)

        processed_dir = self.raw.get("paths", {}).get("processed_dir") or "data/processed"
        dataset_name = self.raw.get("dataset", {}).get("name") or "active_product_recipe"
        return os.path.abspath(os.path.join(processed_dir, f"{dataset_name}_yolo.yaml"))

    def _severstal_base_dirs(self) -> list[str]:
        """Candidate directories that may hold the Severstal train.csv / train_images."""
        bases: list[str] = []
        raw_path = (self.raw.get("dataset", {}) or {}).get("raw_path")
        if raw_path:
            raw_path = str(raw_path)
            bases.append(os.path.dirname(raw_path) if raw_path.lower().endswith(".csv") else raw_path)
        bases.append(str(self.raw.get("paths", {}).get("raw_dir") or "data/raw"))
        # De-duplicate while preserving priority order.
        return list(dict.fromkeys(bases))

    def _resolve_severstal_csv(self) -> str:
        explicit = (self.raw.get("severstal", {}) or {}).get("train_csv")
        if explicit:
            return str(explicit)
        raw_path = str((self.raw.get("dataset", {}) or {}).get("raw_path") or "")
        candidates = [raw_path] if raw_path.lower().endswith(".csv") else []
        candidates += [os.path.join(base, "train.csv") for base in self._severstal_base_dirs()]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return candidates[0]

    def _resolve_severstal_images_dir(self) -> str:
        explicit = (self.raw.get("severstal", {}) or {}).get("train_images_dir")
        if explicit:
            return str(explicit)
        candidates: list[str] = []
        for base in self._severstal_base_dirs():
            candidates.append(os.path.join(base, "train_images"))
            candidates.append(base)
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        return candidates[0]

    def get(self, *keys, default=None):
        # 1) Dynamic Path Resolution: inference.yolo_yaml and dataset.yolo.yaml_path always agree.
        if keys == ("inference", "yolo_yaml") or keys == ("dataset", "yolo", "yaml_path"):
            return self._resolve_dataset_yaml()

        # 2) Backwards compatibility: Map legacy yolo dataset queries to generalized dataset properties
        if keys == ("dataset", "yolo", "val_split"):
            return float(self.raw.get("dataset", {}).get("val_split", 0.20))
        if keys == ("dataset", "yolo", "seed"):
            return int(self.raw.get("dataset", {}).get("seed", 42))

        # 3) Backwards compatibility: legacy Severstal layout (train.csv + train_images/)
        if keys == ("severstal", "train_csv"):
            return self._resolve_severstal_csv()
        if keys == ("severstal", "train_images_dir"):
            return self._resolve_severstal_images_dir()

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

        def _num(source: dict, key: str, fallback: float) -> float:
            # Only fall back when the key is absent/null: an explicit 0 (e.g. an
            # immediate critical hold) is a legitimate configuration value.
            value = source.get(key)
            if value is None:
                return float(fallback)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(fallback)

        return ResourceManagementConfig(
            monitoring_interval=_num(rm, "monitoring_interval", 2.0),
            vram_thresholds=ThresholdPair(
                warning=_num(vram, "warning", 85.0),
                critical=_num(vram, "critical", 95.0),
            ),
            temp_thresholds=ThresholdPair(
                warning=_num(temp, "warning", 78.0),
                critical=_num(temp, "critical", 85.0),
            ),
            warning_hold_seconds=_num(rm, "warning_hold_seconds", 120.0),
            critical_hold_seconds=_num(rm, "critical_hold_seconds", 300.0),
            warning_cooldown_seconds=_num(rm, "warning_cooldown_seconds", 300.0),
            critical_cooldown_seconds=_num(rm, "critical_cooldown_seconds", 900.0),
            emergency_temp_c=_num(rm, "emergency_temp_c", 90.0),
        )


def env_override_config(cfg: Config) -> Config:
    """Apply DATASET_* / legacy SEVERSTAL_* environment overrides on top of `cfg`.

    The overrides are merged into a copy of the already-loaded configuration so a
    config loaded from an explicit path is never silently swapped for the default one.
    """
    overrides: dict = {"dataset": {}, "severstal": {}}

    # Generic overrides
    dataset_name = os.getenv("DATASET_NAME")
    dataset_format = os.getenv("DATASET_FORMAT")
    dataset_raw_path = os.getenv("DATASET_RAW_PATH")

    if dataset_name:
        overrides["dataset"]["name"] = dataset_name
    if dataset_format:
        overrides["dataset"]["format"] = dataset_format
    if dataset_raw_path:
        overrides["dataset"]["raw_path"] = dataset_raw_path

    # Legacy Severstal layout: map each variable onto the key it actually describes.
    legacy_images = os.getenv("SEVERSTAL_TRAIN_IMAGES_DIR")
    legacy_csv = os.getenv("SEVERSTAL_TRAIN_CSV")
    legacy_data = os.getenv("SEVERSTAL_DATA_DIR")
    if legacy_images:
        overrides["severstal"]["train_images_dir"] = legacy_images
    if legacy_csv:
        overrides["severstal"]["train_csv"] = legacy_csv
    if legacy_data and not dataset_raw_path:
        overrides["dataset"]["raw_path"] = legacy_data

    overrides = {k: v for k, v in overrides.items() if v}
    if not overrides:
        return cfg
    return Config(raw=_deep_update(copy.deepcopy(cfg.raw), overrides))
