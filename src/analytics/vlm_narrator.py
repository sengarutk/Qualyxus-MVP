from __future__ import annotations

import json
from typing import Any

import requests

from src.inference.postprocess import normalize_metadata_keys
from src.reasoning.vlm_engine import VLMEngine, VLMRequest
from src.utils.profile_manager import ProfileManager


class VLMNarrator:
    """Asynchronous visual root-cause narrator with graceful local fallback."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.model = str(self.config.get("model", "moondream2"))
        self.local_api = str(self.config.get("local_api", "http://localhost:11434"))
        self.timeout_seconds = float(self.config.get("timeout_seconds", 10.0))
        self.backend = str(self.config.get("backend", "ollama")).lower()
        self._transformers_pipeline = None
        self._engine = VLMEngine(base_url=self.local_api, timeout_seconds=5.0)
        self._profile_manager = ProfileManager()
        self._vlm_expert_context = self._profile_manager.get_vlm_expert_context().strip()

        if not self.enabled:
            return

        try:
            if self.backend == "transformers":
                self._init_transformers_backend()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"VLM backend initialization failed: {exc}") from exc

    def _init_transformers_backend(self) -> None:
        try:
            import torch
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(f"transformers backend unavailable: {exc}") from exc

        if torch.cuda.is_available():
            try:
                free_mem, total_mem = torch.cuda.mem_get_info()
                if free_mem < (2 * 1024 * 1024 * 1024):
                    raise RuntimeError("VRAM_EXHAUSTED: insufficient free VRAM for VLM backend")
            except RuntimeError:
                raise
            except Exception:
                pass

        # Kept lightweight and optional; actual local API path is preferred.
        self._transformers_pipeline = pipeline("image-to-text", model=self.model)

    def _system_prompt(self) -> str:
        base = (
            "You are a Senior Manufacturing Quality Engineer."
            " Provide a 1-sentence industrial root-cause summary."
        )
        if self._vlm_expert_context:
            base += f" Specialist context: {self._vlm_expert_context}."
        return base

    @staticmethod
    def _batch_system_prompt() -> str:
        return (
            "You are a Lead Industrial Process Engineer. Review multiple defect events and telemetry trends to identify "
            "the common systemic thread. Return exactly three concise lines describing the shared process failure, the "
            "supporting telemetry pattern, and the likely plant-level corrective direction."
        )

    @staticmethod
    def _telemetry_summary(telemetry_context: dict[str, Any] | None) -> str:
        """Create a bounded-size telemetry summary string.

        Malicious-audit hardening:
        - telemetry_context may contain huge blobs (e.g., 10k+ chars). We must not forward it verbatim.
        - keep only whitelisted keys + truncate oversized scalar values.
        """

        def _safe_scalar(value: Any, max_chars: int = 256) -> Any:
            if value is None:
                return None
            if isinstance(value, (int, float, bool)):
                return value
            text = str(value)
            if len(text) > max_chars:
                return text[: max_chars - 24] + f"...[TRUNCATED {len(text)} chars]"
            return text

        ctx = telemetry_context or {}
        # Whitelist only the minimal set used in prompts.
        compact = {
            "machine_id": _safe_scalar(ctx.get("machine_id")),
            "shift_id": _safe_scalar(ctx.get("shift_id")),
            "temperature": ctx.get("temperature"),
            "vibration": ctx.get("vibration"),
            "speed": ctx.get("speed"),
            "line_id": _safe_scalar(ctx.get("line_id")),
            "station_id": _safe_scalar(ctx.get("station_id")),
        }
        return json.dumps(compact, ensure_ascii=False)

    def generate_narrative(
        self,
        image_path: str | None,
        telemetry_context: dict[str, Any] | None,
        defect_type: str,
    ) -> str | None:
        if not self.enabled:
            return None
        if not image_path:
            return None

        if self.backend == "transformers" and self._transformers_pipeline is not None:
            try:
                result = self._transformers_pipeline(
                    image_path,
                    prompt=(
                        f"{self._system_prompt()} Defect type: {defect_type}. "
                        f"Telemetry: {self._telemetry_summary(telemetry_context)}"
                    ),
                )
                if isinstance(result, list) and result:
                    text = str(result[0].get("generated_text", "") or "").strip()
                    return text or None
            except Exception as exc:
                raise RuntimeError(f"Transformers VLM inference failed: {exc}") from exc

        # Ollama local API path (version-agnostic) with strict timeout and safe fallback.
        ctx = normalize_metadata_keys(telemetry_context or {})
        prompt = (
            f"{self._system_prompt()} "
            f"Analyze this defect: {defect_type} detected at {ctx.get('station_id')} on {ctx.get('line_id')}. "
            f"Reference ISO Clause {ctx.get('iso_clause')}. "
            f"Telemetry: {self._telemetry_summary(telemetry_context)}. "
            f"Image path: {image_path}."
        )
        narrative = self._engine.generate(VLMRequest(model=self.model, prompt=prompt))
        return narrative or None

    def generate_batch_insight(
        self,
        event_batch: list[dict[str, Any]],
        telemetry_snapshot: dict[str, Any] | None,
    ) -> str | None:
        if not self.enabled or not event_batch:
            return None

        representative = sorted(
            event_batch,
            key=lambda e: (float(e.get("severity", 0) or 0), float(e.get("confidence", 0) or 0)),
            reverse=True,
        )[:5]
        summary = [
            {
                "event_id": e.get("event_id"),
                "source": e.get("source"),
                "defect_type": e.get("defect_type"),
                "severity": e.get("severity"),
                "confidence": e.get("confidence"),
                "location": e.get("station_id") or e.get("line_id"),
            }
            for e in representative
        ]

        prompt = (
            f"{self._batch_system_prompt()}\n"
            f"Event batch: {json.dumps(summary, ensure_ascii=False)}\n"
            f"Telemetry trends: {json.dumps(telemetry_snapshot or {}, ensure_ascii=False)}\n"
        )
        narrative = self._engine.generate(VLMRequest(model=self.model, prompt=prompt))
        return narrative or None