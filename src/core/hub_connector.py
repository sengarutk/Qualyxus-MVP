"""The plant uplink, wired into the runtime's life cycle."""
from __future__ import annotations

from typing import Any

from src.network.mqtt_bridge import MqttBridge


class HubConnector:
    def __init__(self, mqtt_cfg: dict[str, Any] | None = None):
        self.bridge = MqttBridge(mqtt_cfg)

    async def connect(self) -> None:
        await self.bridge.start()

    async def disconnect(self) -> None:
        await self.bridge.close()
