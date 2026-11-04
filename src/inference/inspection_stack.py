"""The per-recipe set of vision models that judges one part, and its decision rule.

A stack is the anomaly detector (catches anything unusual), the supervised segmenter
(names and outlines known defect types) and the optional defect-type classifier, for one
recipe. Stacks come from the model registry (active and shadow versions, hot-reloaded
when someone promotes or rolls back) or, without a registry, from explicit paths.
"""
from __future__ import annotations

import collections
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("QualyxusInspection")


@dataclass
class PartJudgement:
    failed: bool
    defect_type: str | None
    bbox: tuple[int, int, int, int] | None
    confidence: float | None
    anomaly: Any = None                 # AnomalyResult | None
    instances: list = field(default_factory=list)
    top: Any = None                     # strongest DefectInstance | None

    @property
    def normalized_score(self) -> float | None:
        return self.anomaly.normalized_score if self.anomaly is not None else None


def _load_anomaly(path: str | Path | None):
    if not path:
        return None
    if not os.path.exists(path):
        logger.warning("[Inspection] Anomaly model %s not found; train one with: "
                       "python -m src.inference.anomaly.benchmark --select --save-best <path>", path)
        return None
    try:
        from src.inference.anomaly.registry import load_detector

        det = load_detector(path)
        logger.info("[Inspection] Loaded %s anomaly model %s (threshold %.4f)", det.kind, path, det.threshold)
        return det
    except Exception as e:
        logger.error("[Inspection] Failed to load anomaly model %s: %s", path, e)
        return None


def _load_segmenter(path: str | Path | None):
    if not path:
        return None
    if not os.path.exists(os.path.join(path, "model_card.json")):
        logger.warning("[Inspection] Segmentation model %s not found (no model_card.json).", path)
        return None
    try:
        from src.inference.segmentation.base import load_segmenter

        seg = load_segmenter(path)
        logger.info("[Inspection] Loaded %s segmentation model %s (classes %s, %s)",
                    seg.card.get("arch"), path, seg.class_names, seg.card.get("license"))
        return seg
    except Exception as e:
        logger.error("[Inspection] Failed to load segmentation model %s: %s", path, e)
        return None


def _load_typer(path: str | Path | None):
    if not path:
        return None
    if not os.path.exists(path):
        logger.warning("[Inspection] Defect-type model %s not found.", path)
        return None
    try:
        from src.inference.defect_typing import DefectTypeClassifier

        return DefectTypeClassifier.load(path)
    except Exception as e:
        logger.error("[Inspection] Failed to load defect-type model %s: %s", path, e)
        return None


def _version_of(path: str | Path | None, registry_version: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    from src.inference.model_registry import file_digest

    try:
        digest = file_digest(path)
    except OSError:
        digest = None
    return {"version": registry_version or "unregistered", "digest": digest, "path": str(path)}


class InspectionStack:
    def __init__(self, recipe: str, *, anomaly_path=None, segmentation_path=None, typing_path=None,
                 versions: dict[str, str | None] | None = None):
        self.recipe = recipe
        versions = versions or {}
        self.anomaly = _load_anomaly(anomaly_path)
        self.segmenter = _load_segmenter(segmentation_path)
        self.typer = _load_typer(typing_path)
        self.versions = {
            role: v for role, v in (
                ("anomaly", _version_of(anomaly_path, versions.get("anomaly")) if self.anomaly else None),
                ("segmentation", _version_of(segmentation_path, versions.get("segmentation")) if self.segmenter else None),
                ("typing", _version_of(typing_path, versions.get("typing")) if self.typer else None),
            ) if v
        }
        self.loaded_ts = time.time()

    @property
    def empty(self) -> bool:
        return self.anomaly is None and self.segmenter is None

    def judge(self, frame: np.ndarray, *, frame_defect_types: list[str] | None = None) -> PartJudgement:
        """Any model failing the part fails it. Type: segmenter > frame detector > classifier."""
        anomaly = self.anomaly.predict(frame) if self.anomaly is not None else None
        instances = self.segmenter.predict(frame) if self.segmenter is not None else []
        anomalous = bool(anomaly and anomaly.is_anomalous)
        failed = anomalous or bool(instances) or bool(frame_defect_types)
        h, w = frame.shape[:2]
        top = max(instances, key=lambda i: i.confidence) if instances else None
        if top is not None:
            return PartJudgement(True, top.class_name, top.bbox, float(top.confidence), anomaly, instances, top)
        if frame_defect_types:
            return PartJudgement(True, frame_defect_types[0], (0, 0, w, h), None, anomaly, instances)
        if anomalous:
            bbox = anomaly.bbox or (0, 0, w, h)
            confidence = float(min(0.999, 1.0 - 1.0 / (1.0 + anomaly.normalized_score)))
            defect_type = "anomaly"
            if self.typer is not None:
                x1, y1, x2, y2 = bbox
                pad = 16
                crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
                if crop.size:
                    typed, _p = self.typer.predict(crop)
                    defect_type = typed if typed != "unknown" else "anomaly"
            return PartJudgement(True, defect_type, bbox, confidence, anomaly, instances)
        return PartJudgement(failed, None, None, None, anomaly, instances)


class StackProvider:
    """Resolves, caches and hot-reloads the active (and shadow) stack per recipe."""

    def __init__(self, inspection_cfg: dict[str, Any]):
        self.cfg = inspection_cfg or {}
        self.default_recipe = str(self.cfg.get("recipe") or "default")
        self.registry = None
        reg_dir = self.cfg.get("registry_dir")
        if reg_dir:
            from src.inference.model_registry import VisionModelRegistry

            self.registry = VisionModelRegistry(reg_dir)
        self.shadow_enabled = bool((self.cfg.get("shadow") or {}).get("enabled", True))
        self.max_cached = int(self.cfg.get("max_cached_recipes", 3))
        self.reload_check_s = float(self.cfg.get("reload_check_s", 10.0))
        self._stacks: collections.OrderedDict[str, tuple[InspectionStack, InspectionStack | None, tuple]] = \
            collections.OrderedDict()
        self._last_check: dict[str, float] = {}
        self._lock = threading.RLock()

    def _registry_versions(self, recipe: str) -> tuple:
        if self.registry is None:
            return ()
        out = []
        for role in ("anomaly", "segmentation", "typing"):
            m = self.registry.manifest(recipe, role)
            out.append((role, m.get("active"), m.get("shadow") if self.shadow_enabled else None))
        return tuple(out)

    def _build(self, recipe: str, key: tuple) -> tuple[InspectionStack, InspectionStack | None]:
        reg = {role: (active, shadow) for role, active, shadow in key}
        if self.registry is not None and any(a for a, _ in reg.values()):
            def path(role, which):
                version = reg.get(role, (None, None))[0 if which == "active" else 1]
                return self.registry.path_of(recipe, role, version), version

            (ap, av), (sp, sv), (tp, tv) = path("anomaly", "active"), path("segmentation", "active"), path("typing", "active")
            active = InspectionStack(recipe, anomaly_path=ap, segmentation_path=sp, typing_path=tp,
                                     versions={"anomaly": av, "segmentation": sv, "typing": tv})
            shadow = None
            if any(s for _, s in reg.values()):
                (ap2, av2), (sp2, sv2), (tp2, tv2) = path("anomaly", "shadow"), path("segmentation", "shadow"), path("typing", "shadow")
                # A role without a shadow version uses the active model, so only the challenger differs.
                shadow = InspectionStack(recipe, anomaly_path=ap2 or ap, segmentation_path=sp2 or sp,
                                         typing_path=tp2 or tp,
                                         versions={"anomaly": av2 or av, "segmentation": sv2 or sv, "typing": tv2 or tv})
            logger.info("[Inspection] Recipe '%s': active %s%s", recipe,
                        {r: v["version"] for r, v in active.versions.items()},
                        f", shadow {({r: v['version'] for r, v in shadow.versions.items()})}" if shadow else "")
            return active, shadow
        # No registry entry: explicit paths from the config (single, unversioned stack).
        recipes_cfg = (self.cfg.get("recipes") or {}).get(recipe) or {}
        src = recipes_cfg if recipes_cfg else self.cfg
        active = InspectionStack(recipe, anomaly_path=src.get("anomaly_model"),
                                 segmentation_path=src.get("segmentation_model"),
                                 typing_path=src.get("defect_type_model"))
        return active, None

    def get(self, recipe: str | None = None) -> tuple[InspectionStack, InspectionStack | None]:
        recipe = recipe or self.default_recipe
        with self._lock:
            now = time.time()
            cached = self._stacks.get(recipe)
            if cached is not None and now - self._last_check.get(recipe, 0.0) < self.reload_check_s:
                self._stacks.move_to_end(recipe)
                return cached[0], cached[1]
            self._last_check[recipe] = now
            key = self._registry_versions(recipe)
            if cached is not None and cached[2] == key:
                self._stacks.move_to_end(recipe)
                return cached[0], cached[1]
            if cached is not None:
                logger.warning("[Inspection] Model versions for recipe '%s' changed: reloading", recipe)
            active, shadow = self._build(recipe, key)
            self._stacks[recipe] = (active, shadow, key)
            self._stacks.move_to_end(recipe)
            while len(self._stacks) > self.max_cached:
                evicted, _ = self._stacks.popitem(last=False)
                logger.info("[Inspection] Unloaded models for recipe '%s' (cache of %d)", evicted, self.max_cached)
            return active, shadow
