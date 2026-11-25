"""Optional MQTT uplink to the plant broker. Off unless ``mqtt.uplink_enabled`` is true.

Encrypted by default; a plain connection needs ``mqtt.allow_insecure: true``. Messages wait in
a small database while the broker is away, so a restart does not lose them. Nothing about
inspection depends on this.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import ssl
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("QualyxusMqttBridge")


class Spool:
    """Messages not yet taken by the broker, oldest first, bounded."""

    def __init__(self, path: str | Path, max_items: int = 50000):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.max_items = int(max_items)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS spool (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, topic TEXT, "
                           "payload BLOB, qos INTEGER)")
        self._conn.commit()
        self.dropped = 0

    def put(self, topic: str, payload: bytes, qos: int) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO spool (ts, topic, payload, qos) VALUES (?, ?, ?, ?)",
                               (time.time(), topic, payload, int(qos)))
            over = self._conn.execute("SELECT COUNT(*) FROM spool").fetchone()[0] - self.max_items
            if over > 0:
                self._conn.execute("DELETE FROM spool WHERE id IN (SELECT id FROM spool ORDER BY id LIMIT ?)", (over,))
                self.dropped += over
                logger.error("[MQTT] The waiting messages are full: %d oldest dropped so far.", self.dropped)
            self._conn.commit()

    def oldest(self, limit: int = 100) -> list[tuple[int, str, bytes, int]]:
        with self._lock:
            return self._conn.execute("SELECT id, topic, payload, qos FROM spool ORDER BY id LIMIT ?", (limit,)).fetchall()

    def remove(self, item_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM spool WHERE id = ?", (item_id,))
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM spool").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class MqttBridge:
    def __init__(self, cfg: dict[str, Any] | None = None, client_factory=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("uplink_enabled", False))
        self.host = str(cfg.get("host", "127.0.0.1"))
        self.port = int(cfg.get("port", 8883))
        self.client_id = str(cfg.get("client_id", "qualyxus_edge_01"))
        self.use_tls = bool(cfg.get("use_tls", True))
        self.certificates = {k: cfg.get(k) or os.getenv(f"MQTT_{k.upper()}") for k in ("ca_certs", "certfile", "keyfile")}
        self.username = cfg.get("username") or os.getenv("MQTT_USERNAME")
        self.password = os.getenv(str(cfg.get("password_env", "MQTT_PASSWORD")))
        self._factory = client_factory
        self.spool: Spool | None = None
        self.client: Any = None
        self._connected = asyncio.Event()
        self._task: asyncio.Task | None = None
        if not self.enabled:
            return
        if not self.use_tls and not cfg.get("allow_insecure"):
            raise ValueError("mqtt.use_tls is off: messages and the password would be readable on the network. Use TLS, or "
                             "set mqtt.allow_insecure: true on a closed machine network")
        self.spool = Spool(cfg.get("spool_path", "data/mqtt_spool.db"))

    def _ssl(self) -> ssl.SSLContext | bool:
        if not self.use_tls:
            return False
        context = ssl.create_default_context(cafile=self.certificates["ca_certs"])
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if self.certificates["certfile"]:
            context.load_cert_chain(self.certificates["certfile"], self.certificates["keyfile"])
        return context

    async def start(self) -> None:
        if self.enabled and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._run(), name="mqtt-uplink")

    def _new_client(self):
        if self._factory is not None:
            return self._factory(self.client_id)
        from gmqtt import Client

        return Client(self.client_id)

    async def _run(self) -> None:
        wait = 1.0
        while True:
            try:
                await self._connect()
                wait = 1.0
                while self._connected.is_set():
                    await self._drain()
                    await asyncio.sleep(0.2)
                logger.warning("[MQTT] Connection lost.")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("[MQTT] %s:%s: %s. Retrying in %.0f s.", self.host, self.port, e, wait)
            await self._disconnect()
            await asyncio.sleep(wait)
            wait = min(wait * 2.0, 60.0)

    async def _connect(self) -> None:
        self._connected.clear()
        self.client = self._new_client()
        self.client.on_connect = lambda *a: self._connected.set()
        self.client.on_disconnect = lambda *a, **k: self._connected.clear()
        if self.username:
            self.client.set_auth_credentials(self.username, self.password)
        await self.client.connect(self.host, self.port, ssl=self._ssl(), keepalive=10)
        await asyncio.wait_for(self._connected.wait(), timeout=10.0)
        logger.info("[MQTT] Connected to %s:%s%s.", self.host, self.port, "" if self.use_tls else " without encryption")

    async def _drain(self) -> None:
        assert self.spool is not None
        for item_id, topic, payload, qos in self.spool.oldest():
            if not self._connected.is_set():
                return
            self.client.publish(topic, payload, qos=qos)
            self.spool.remove(item_id)

    async def publish_telemetry(self, topic: str, payload: dict, qos: int = 1) -> None:
        if not self.enabled or self.spool is None:
            return
        await self.start()
        try:
            self.spool.put(topic, json.dumps(payload, default=str).encode(), qos)
        except (sqlite3.Error, TypeError, ValueError) as e:
            logger.error("[MQTT] Message not kept: %s", e)

    async def _disconnect(self) -> None:
        client, self.client = self.client, None
        self._connected.clear()
        if client is not None:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=2.0)
            except Exception as e:
                logger.debug("[MQTT] Disconnect: %s", e)

    async def close(self) -> None:
        if self._task is not None:
            if self._connected.is_set():
                await self._drain()
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._disconnect()
        if self.spool is not None:
            self.spool.close()
