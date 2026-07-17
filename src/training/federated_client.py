from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from src.config.config import Config


class FederatedClient:
    """Edge-side secure aggregation client for selective head sharing only."""

    HEAD_KEYWORDS = (
        "detect",
        "seg",
        "head",
        "proto",
        "cv2",
        "cv3",
        "dfl",
    )

    def __init__(self, config: Config | None = None, node_id: str = "edge_node"):
        self.config = config or Config.load()
        self.node_id = node_id
        self.runtime_dir = Path(self.config.get("paths", "data_dir", default="data")) / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.runtime_dir / "federated_status.json"

    def extract_trainable_weights(self, model) -> dict[str, torch.Tensor]:
        state_dict = model.state_dict() if hasattr(model, "state_dict") else {}
        selected: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in self.HEAD_KEYWORDS):
                selected[key] = value.detach().cpu().clone()
        if not selected:
            keys = list(state_dict.keys())
            for key in keys[- min(20, len(keys)) :]:
                selected[key] = state_dict[key].detach().cpu().clone()
        return selected

    @staticmethod
    def sanitize_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sanitized: dict[str, torch.Tensor] = {}
        for key, value in weights.items():
            clean_key = str(key).replace("\\", "_").replace("/", "_").replace(":", "_")
            sanitized[clean_key] = value.detach().cpu().contiguous().clone()
        return sanitized

    @staticmethod
    def _serialize_weights(weights: dict[str, torch.Tensor]) -> bytes:
        buffer = io.BytesIO()
        torch.save(weights, buffer, _use_new_zipfile_serialization=True)
        return buffer.getvalue()

    @staticmethod
    def _simulate_aes256_encrypt(payload: bytes, key_material: str) -> str:
        digest = hashlib.sha256(key_material.encode("utf-8")).digest()
        encrypted = bytes(payload[i] ^ digest[i % len(digest)] for i in range(len(payload)))
        return base64.b64encode(encrypted).decode("utf-8")

    def export_secure_payload(self, model, version: str = "v1") -> dict[str, Any]:
        extracted = self.extract_trainable_weights(model)
        sanitized = self.sanitize_weights(extracted)
        payload_bytes = self._serialize_weights(sanitized)
        sha256 = hashlib.sha256(payload_bytes).hexdigest()
        encrypted_payload = self._simulate_aes256_encrypt(payload_bytes, key_material=f"{self.node_id}:{version}:{sha256}")

        payload = {
            "node_id": self.node_id,
            "version": version,
            "privacy_level": str(self.config.get("federated_learning", "privacy_level", default="maximum")),
            "sha256": sha256,
            "encrypted_payload": encrypted_payload,
            "tensor_count": len(sanitized),
        }
        self._record_contribution(version)
        return payload

    def _record_contribution(self, version: str) -> None:
        status = {"collaborative_contribution_count": 0, "global_knowledge_version": version}
        if self.status_path.exists():
            try:
                status.update(json.loads(self.status_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        status["collaborative_contribution_count"] = int(status.get("collaborative_contribution_count", 0) or 0) + 1
        status["global_knowledge_version"] = version
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")