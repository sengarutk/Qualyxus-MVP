from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


DEFAULT_PENDING_NARRATIVE = "Systemic status verified; detailed VLM narrative pending"


@dataclass(frozen=True)
class VLMRequest:
    model: str
    prompt: str


class VLMEngine:
    """Ollama-version agnostic client.

    - Prefer legacy /api/generate.
    - On 404, fallback to /api/chat.
    - Never block runtime threads: strict timeout and safe fallback narrative.
    """

    def __init__(self, *, base_url: str = "http://localhost:11434", timeout_seconds: float = 5.0):
        self.base_url = str(base_url).rstrip("/")
        # Strict upper bound per runtime hardening requirements.
        self.timeout_seconds = float(min(max(timeout_seconds, 0.1), 5.0))

    def generate(self, req: VLMRequest) -> str:
        """Return a cleaned 1-sentence narrative. Never raises on network errors."""
        try:
            text = self._try_generate(req)
            return self._clean_narrative(text)
        except Exception as exc:  # Defensive: never let VLM errors kill background worker.
            LOGGER.warning("VLMEngine failed: %s", exc)
            return DEFAULT_PENDING_NARRATIVE

    def _try_generate(self, req: VLMRequest) -> str:
        # 1) Legacy: /api/generate
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": req.model,
                    "prompt": req.prompt,
                    "stream": False,
                },
                timeout=self.timeout_seconds,
            )
            if response.status_code == 404:
                raise FileNotFoundError("/api/generate not found")
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "") or "")
        except requests.Timeout:
            return DEFAULT_PENDING_NARRATIVE
        except FileNotFoundError:
            pass
        except requests.RequestException as exc:
            # If server is offline or rejects request, return safe text rather than raising.
            LOGGER.warning("VLM /api/generate request failed: %s", exc)
            return DEFAULT_PENDING_NARRATIVE
        except Exception as exc:
            LOGGER.warning("VLM /api/generate parse failed: %s", exc)
            return DEFAULT_PENDING_NARRATIVE

        # 2) Modern: /api/chat
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": req.model,
                    "messages": [{"role": "user", "content": req.prompt}],
                    "stream": False,
                },
                timeout=self.timeout_seconds,
            )
            # Some Ollama builds do not expose /api/chat; treat as a silent fallback.
            if response.status_code == 404:
                return DEFAULT_PENDING_NARRATIVE
            response.raise_for_status()
            payload = response.json()
            msg = payload.get("message") or {}
            return str((msg or {}).get("content", "") or "")
        except requests.Timeout:
            return DEFAULT_PENDING_NARRATIVE
        except requests.RequestException as exc:
            # Avoid noisy logs for expected offline cases.
            try:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            except Exception:
                status = None
            if status == 404:
                return DEFAULT_PENDING_NARRATIVE
            LOGGER.warning("VLM /api/chat request failed: %s", exc)
            return DEFAULT_PENDING_NARRATIVE
        except Exception as exc:
            LOGGER.warning("VLM /api/chat parse failed: %s", exc)
            return DEFAULT_PENDING_NARRATIVE

    @staticmethod
    def _clean_narrative(text: str) -> str:
        # Normalize type + whitespace.
        raw = str(text or "").strip()
        if not raw:
            return DEFAULT_PENDING_NARRATIVE

        # Strip common wrappers.
        raw = raw.strip("`\"'")
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.replace("\r", "\n").strip()

        # If the model responded with JSON, try to extract the most likely fields.
        if raw.startswith("{") and raw.endswith("}"):
            try:
                payload = json.loads(raw)
                for key in ("response", "content", "message", "text"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        raw = val.strip()
                        break
                    if isinstance(val, dict) and isinstance(val.get("content"), str):
                        raw = str(val.get("content")).strip()
                        break
            except Exception:
                pass

        # Remove boilerplate/meta-talk.
        raw = re.sub(r"^as an ai language model[,:\s]+", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"^i (cannot|can't)\b.*", "", raw, flags=re.IGNORECASE)
        raw = raw.strip()

        # Enforce 1-sentence output: take first non-empty line, then first sentence.
        first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        if not first_line:
            return DEFAULT_PENDING_NARRATIVE

        m = re.match(r"^(.+?[.!?])(?:\s+|$)", first_line)
        if m:
            first_line = m.group(1)

        # Final guardrail.
        first_line = re.sub(r"\s+", " ", first_line).strip()
        return first_line or DEFAULT_PENDING_NARRATIVE
