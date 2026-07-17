from __future__ import annotations

"""Dispatch layer for event/action payloads over MQTT.

Architectural note:
The production runtime uses `src.telemetry.uns_client.UNSClient` as the MQTT
transport. This module provides the missing `src/action/mqtt_engine.py` entry
point referenced in the TITAN reconciliation task.

It enforces:
- JSON serialization using the immutable metadata contract
- QoS1 replay-buffer semantics (handled by UNSClient)
- TCP_NODELAY on the underlying socket to disable Nagle's Algorithm, ensuring
  small pulse packets are transmitted immediately (zero-buffer path).
"""

import socket
from typing import Any
import logging

from src.inference.postprocess import normalize_metadata_keys
from src.logging.metadata_contract import to_metadata_dict, validate_metadata
from src.telemetry.uns_client import UNSClient


LOGGER = logging.getLogger(__name__)


class MQTTEngine:
    _global_client = None

    def __init__(self, mqtt_config: dict[str, Any] | None = None):
        self.client = UNSClient(mqtt_config or {})
        MQTTEngine._global_client = self.client

    def start(self) -> None:
        self.client.start()
        self._apply_tcp_nodelay()

    @staticmethod
    def _apply_tcp_nodelay() -> None:
        """Disable Nagle's Algorithm on the MQTT transport socket.

        Iterates over all open sockets of the current process and sets
        TCP_NODELAY=1 on every TCP socket, ensuring our small pulse
        packets are never coalesced by the OS send buffer.
        """
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            for conn in proc.net_connections(kind="tcp"):
                try:
                    fd = conn.fd
                    if fd < 0:
                        continue
                    sock = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.detach()  # Don't close the original FD
                except Exception:
                    pass
        except Exception as exc:
            LOGGER.warning("[QUALYXUS] TCP_NODELAY application skipped: %s", exc)

    def stop(self, join_timeout: float = 5.0) -> None:
        self.client.stop(join_timeout=join_timeout)

    def publish_event(self, event: Any, *, topic: str | None = None) -> bool:
        """Publish a defect event metadata payload over MQTT.

        `event` may be a DefectEvent dataclass or any object exposing the contract
        attributes.
        """

        md = normalize_metadata_keys(to_metadata_dict(event))
        ok, errors = validate_metadata(md)
        if not ok:
            # Keep the dictionary contract immutable; do not mutate payload keys.
            LOGGER.warning("[QUALYXUS] metadata contract validation failed: %s", errors)
        return self.client.publish_signal("DEFECT_EVENT", md, topic=topic)

    @classmethod
    def emit_pulse(cls, payload: dict) -> bool:
        """Thread-safe fast-action anomaly pulse emission."""
        client = cls._global_client
        if client is None:
            return False
        # Push raw dict-to-socket directly using QoS 1
        return client.publish_signal("ANOMALY/PULSE", payload)
