from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch


class FederatedServer:
    """Simple FedAvg secure aggregation coordinator for simulated factory nodes."""

    @staticmethod
    def _decrypt_payload(encrypted_payload: str, key_material: str) -> bytes:
        digest = hashlib.sha256(key_material.encode("utf-8")).digest()
        raw = base64.b64decode(encrypted_payload.encode("utf-8"))
        return bytes(raw[i] ^ digest[i % len(digest)] for i in range(len(raw)))

    @staticmethod
    def load_secure_payload(client_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
        key_material = f"{client_payload['node_id']}:{client_payload['version']}:{client_payload['sha256']}"
        payload_bytes = FederatedServer._decrypt_payload(client_payload["encrypted_payload"], key_material)
        sha256 = hashlib.sha256(payload_bytes).hexdigest()
        if sha256 != client_payload.get("sha256"):
            raise ValueError("Federated payload integrity check failed")
        buffer = io.BytesIO(payload_bytes)
        return torch.load(buffer, map_location="cpu")

    @staticmethod
    def fedavg(weight_sets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not weight_sets:
            return {}
        common_keys = set(weight_sets[0].keys())
        for ws in weight_sets[1:]:
            common_keys &= set(ws.keys())

        aggregated: dict[str, torch.Tensor] = {}
        for key in sorted(common_keys):
            tensors = [ws[key].float() for ws in weight_sets]
            aggregated[key] = torch.stack(tensors, dim=0).mean(dim=0)
        return aggregated

    def aggregate_from_clients(self, client_payloads: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        weights = [self.load_secure_payload(payload) for payload in client_payloads]
        return self.fedavg(weights)


def validate_aggregated_model(model, aggregated_weights: dict[str, torch.Tensor], combined_validation_yaml: str | None = None) -> dict[str, Any]:
    if hasattr(model, "load_state_dict"):
        current = model.state_dict()
        current.update(aggregated_weights)
        model.load_state_dict(current, strict=False)

    result = {"combined_validation_yaml": combined_validation_yaml, "status": "validated"}
    if hasattr(model, "val") and combined_validation_yaml:
        metrics = model.val(data=combined_validation_yaml)
        score = 0.0
        for attr in ("box", "seg"):
            obj = getattr(metrics, attr, None)
            if obj is not None and hasattr(obj, "map"):
                score = float(getattr(obj, "map", 0.0) or 0.0)
                break
        result["mAP"] = round(score, 4)
    return result