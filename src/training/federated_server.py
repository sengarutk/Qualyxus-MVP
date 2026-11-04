from __future__ import annotations

import hashlib
import io
from typing import Any

import torch
from cryptography.exceptions import InvalidTag

from src.training.federated_crypto import load_key, open_sealed


class FederatedServer:
    """Simple FedAvg secure aggregation coordinator for simulated factory nodes."""

    def __init__(self, key: bytes | None = None, key_path: str | None = None):
        self.key = key if key is not None else load_key(key_path)

    def load_secure_payload(self, client_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
        if client_payload.get("cipher") != "AES-256-GCM":
            raise ValueError("Unsupported or missing federated payload cipher")
        aad = f"{client_payload['node_id']}|{client_payload['version']}|{client_payload['sha256']}".encode("utf-8")
        try:
            payload_bytes = open_sealed(client_payload["nonce"], client_payload["encrypted_payload"], self.key, aad)
        except InvalidTag as exc:
            raise ValueError("Federated payload failed authentication (wrong key or tampered)") from exc
        if hashlib.sha256(payload_bytes).hexdigest() != client_payload.get("sha256"):
            raise ValueError("Federated payload integrity check failed")
        buffer = io.BytesIO(payload_bytes)
        # Payloads arrive from remote nodes: never unpickle arbitrary objects, tensors only.
        return torch.load(buffer, map_location="cpu", weights_only=True)

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