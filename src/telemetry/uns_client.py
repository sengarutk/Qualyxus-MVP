from __future__ import annotations

import os
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - optional dependency at runtime
    mqtt = None


class UNSClient:
    """Non-blocking MQTT UNS publisher with queued outbound delivery."""

    def __init__(self, mqtt_config: dict[str, Any] | None = None):
        self.mqtt_config = mqtt_config or {}
        self.enabled = bool(self.mqtt_config.get("agentic_enabled", False))
        self.host = str(self.mqtt_config.get("host", "127.0.0.1"))
        self.port = int(self.mqtt_config.get("port", 1883))
        self.keepalive = int(self.mqtt_config.get("keepalive", 60))
        self.transport = str(self.mqtt_config.get("transport", "tcp"))
        self.username = self.mqtt_config.get("username")
        self.password = self.mqtt_config.get("password")
        self.client_id = str(self.mqtt_config.get("client_id", "qualyxus-agentic-gateway"))
        self.topic_root = str(self.mqtt_config.get("topic_root", "qualyxus/agentic"))
        self.qos = int(self.mqtt_config.get("qos", 1))
        self.retain = bool(self.mqtt_config.get("retain", False))
        self.reconnect_backoff_seconds = int(self.mqtt_config.get("reconnect_backoff_seconds", 30))
        self.max_queue_size = int(self.mqtt_config.get("max_queue_size", 1000))
        self.replay_buffer_path = Path(str(self.mqtt_config.get("replay_buffer_path", "data/runtime/uns_replay_buffer.jsonl")))
        self.replay_buffer_limit = int(self.mqtt_config.get("replay_buffer_limit", 5000))
        # QoS1 hardening: do not prune replay_buffer until PUBACK is confirmed.
        self.puback_timeout_seconds = float(self.mqtt_config.get("puback_timeout_seconds", 2.0) or 2.0)

        self._outbound_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, self.max_queue_size))
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._network_thread: threading.Thread | None = None
        self._publisher_thread: threading.Thread | None = None
        self._client = None
        self._replay_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return self.enabled and mqtt is not None

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("UNSClient disabled by config")
            return
        if mqtt is None:
            LOGGER.warning("UNSClient disabled because paho-mqtt is not installed")
            return
        if self._network_thread and self._network_thread.is_alive():
            return

        self._stop_event.clear()
        self._load_replay_buffer()
        self._publisher_thread = threading.Thread(target=self._publisher_loop, name="uns-publisher", daemon=True)
        self._network_thread = threading.Thread(target=self._network_loop, name="uns-network", daemon=True)
        self._publisher_thread.start()
        self._network_thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._connected_event.clear()
        client = self._client
        if client is not None:
            try:
                client.disconnect()
            except Exception as exc:
                LOGGER.warning("UNS disconnect warning: %s", exc)
            try:
                sock = client.socket()
                if sock is not None:
                    try:
                        import socket
                        sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    sock.close()
            except Exception as exc:
                LOGGER.warning("UNS socket close warning: %s", exc)

        for thread in (self._publisher_thread, self._network_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=join_timeout)

    def publish_signal(self, signal_type: str, payload: dict[str, Any], topic: str | None = None) -> bool:
        if not self.enabled:
            return False

        message_id = uuid.uuid4().hex

        # IMPORTANT: keep the payload JSON-serializable and include a stable message_id so
        # the replay buffer can be pruned deterministically only after PUBACK.
        enriched_payload = {
            "message_id": message_id,
            "queued_at": time.time(),
            "signal_type": signal_type,
            **(payload or {}),
        }

        message = {
            "topic": topic or f"{self.topic_root}/{signal_type.lower()}",
            "payload": enriched_payload,
            "qos": self.qos,
            "retain": self.retain,
            "message_id": message_id,
        }

        try:
            self._outbound_queue.put_nowait(message)
            if not self._connected_event.is_set():
                message["_from_replay"] = True
                self._append_replay_message(message)
            return True
        except queue.Full:
            try:
                _ = self._outbound_queue.get_nowait()
                self._outbound_queue.put_nowait(message)
                message["_from_replay"] = True
                self._append_replay_message(message)
                LOGGER.warning("UNS outbound queue full; dropped oldest message to preserve non-blocking behavior")
                return True
            except Exception:
                message["_from_replay"] = True
                self._append_replay_message(message)
                LOGGER.warning("UNS outbound queue full; dropping message")
                return False

    @staticmethod
    def _jsonl_sanitize(message: dict[str, Any]) -> dict[str, Any]:
        """Remove in-memory-only keys before persisting to jsonl."""
        out = dict(message)
        out.pop("_from_replay", None)
        return out

    def _atomic_write_text(self, path: Path, text_payload: str) -> None:
        """Atomic write helper (write temp then os.replace)."""
        tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text_payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))

    def _append_replay_message(self, message: dict[str, Any]) -> None:
        try:
            with self._replay_lock:
                self.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
                with self.replay_buffer_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(self._jsonl_sanitize(message), ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                # Hardening: avoid O(N) full-file reads on every append.
                # If limit is exceeded, trim by reading only when file is "large enough".
                try:
                    if self.replay_buffer_path.stat().st_size > 5_000_000:  # ~5MB heuristic
                        lines = self.replay_buffer_path.read_text(encoding="utf-8").splitlines()
                        if len(lines) > self.replay_buffer_limit:
                            trimmed = lines[-self.replay_buffer_limit :]
                            self._atomic_write_text(self.replay_buffer_path, "\n".join(trimmed) + "\n")
                except Exception:
                    pass
        except Exception as exc:
            LOGGER.warning("UNS replay buffer append failed: %s", exc)

    def _load_replay_buffer(self) -> None:
        if not self.replay_buffer_path.exists():
            return
        try:
            with self._replay_lock:
                lines = self.replay_buffer_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-self.replay_buffer_limit :]:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                    if isinstance(msg, dict):
                        msg["_from_replay"] = True
                    self._outbound_queue.put_nowait(msg)
                except queue.Full:
                    break
                except Exception:
                    continue
        except Exception as exc:
            LOGGER.warning("UNS replay buffer load failed: %s", exc)

    def _rewrite_replay_buffer(self, remaining: list[dict[str, Any]]) -> None:
        try:
            with self._replay_lock:
                self.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
                if not remaining:
                    if self.replay_buffer_path.exists():
                        self.replay_buffer_path.unlink()
                    return
                body_lines: list[str] = []
                for item in remaining[-self.replay_buffer_limit :]:
                    if not isinstance(item, dict):
                        continue
                    body_lines.append(json.dumps(self._jsonl_sanitize(item), ensure_ascii=False, default=str))
                self._atomic_write_text(self.replay_buffer_path, "\n".join(body_lines) + ("\n" if body_lines else ""))
        except Exception as exc:
            LOGGER.warning("UNS replay buffer rewrite failed: %s", exc)

    def _build_client(self):
        assert mqtt is not None
        callback_api = getattr(mqtt, "CallbackAPIVersion", None)
        if callback_api is not None:
            # paho-mqtt >= 2.0: opt into the current callback signatures explicitly.
            client = mqtt.Client(callback_api.VERSION2, client_id=self.client_id, transport=self.transport)
        else:  # pragma: no cover - paho-mqtt 1.x
            client = mqtt.Client(client_id=self.client_id, transport=self.transport)
        if self.username:
            client.username_pw_set(self.username, self.password)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.reconnect_delay_set(
            min_delay=self.reconnect_backoff_seconds,
            max_delay=self.reconnect_backoff_seconds,
        )
        return client

    def _network_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._client = self._build_client()
                self._client.connect(self.host, self.port, keepalive=self.keepalive)
                self._client.loop_forever(retry_first_connection=True)
            except Exception as exc:
                self._connected_event.clear()
                LOGGER.warning(
                    "UNS connection failed; retrying in %ss: %s",
                    self.reconnect_backoff_seconds,
                    exc,
                )
                if self._stop_event.wait(timeout=self.reconnect_backoff_seconds):
                    break

    def _publisher_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._outbound_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self._connected_event.wait(timeout=0.5):
                try:
                    self._outbound_queue.put_nowait(message)
                except queue.Full:
                    LOGGER.warning("UNS publish deferred but queue is full; dropping message")
                continue

            client = self._client
            if client is None:
                continue

            try:
                info = client.publish(
                    message["topic"],
                    json.dumps(message["payload"], default=str),
                    qos=int(message.get("qos", self.qos)),
                    retain=bool(message.get("retain", self.retain)),
                )
                if info.rc != 0:
                    # Publish did not even queue for delivery.
                    if not bool(message.get("_from_replay")):
                        message["_from_replay"] = True
                        self._append_replay_message(message)
                    LOGGER.warning("UNS publish returned non-zero rc=%s for topic=%s", info.rc, message["topic"])
                    continue

                # QoS hardening: only consider this delivered when PUBACK is confirmed.
                qos = int(message.get("qos", self.qos))
                if qos > 0:
                    try:
                        info.wait_for_publish(timeout=max(0.1, self.puback_timeout_seconds))
                    except Exception:
                        pass

                delivered = True
                try:
                    if qos > 0 and hasattr(info, "is_published"):
                        delivered = bool(info.is_published())
                except Exception:
                    delivered = False

                if delivered:
                    # Only prune if this message was actually persisted in the replay buffer.
                    if bool(message.get("_from_replay")):
                        self._ack_replay_message(str(message.get("message_id") or ""))
                else:
                    # Do NOT prune: PUBACK was not confirmed. Requeue for retry.
                    if not bool(message.get("_from_replay")):
                        message["_from_replay"] = True
                        self._append_replay_message(message)
                    try:
                        self._outbound_queue.put_nowait(message)
                    except queue.Full:
                        LOGGER.warning("UNS outbound queue full while requeueing un-acked publish")
            except Exception as exc:
                LOGGER.warning("UNS publish warning: %s", exc)
                if not bool(message.get("_from_replay")):
                    message["_from_replay"] = True
                    self._append_replay_message(message)
                try:
                    self._outbound_queue.put_nowait(message)
                except queue.Full:
                    LOGGER.warning("UNS outbound queue full while requeueing failed publish")
                time.sleep(0.5)

    def _ack_replay_message(self, message_id: str) -> None:
        """Remove a buffered entry only after PUBACK confirmation.

        This is intentionally tolerant:
        - if message_id is missing, do nothing
        - if buffer file is missing, do nothing
        """
        if not message_id:
            return
        if not self.replay_buffer_path.exists():
            return
        try:
            with self._replay_lock:
                lines = self.replay_buffer_path.read_text(encoding="utf-8").splitlines()

            remaining: list[dict[str, Any]] = []
            removed = False
            for line in lines:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not removed and isinstance(item, dict) and str(item.get("message_id") or "") == message_id:
                    removed = True
                    continue
                remaining.append(item)
            if removed:
                self._rewrite_replay_buffer(remaining)
        except Exception as exc:
            LOGGER.warning("UNS replay buffer ack prune failed: %s", exc)

    @staticmethod
    def _reason_failed(reason_code) -> bool:
        is_failure = getattr(reason_code, "is_failure", None)
        if is_failure is not None:
            return bool(is_failure)
        try:
            return int(reason_code) != 0
        except (TypeError, ValueError):
            return True

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):  # pragma: no cover - callback
        _ = (client, userdata, flags, properties)
        if not self._reason_failed(reason_code):
            self._connected_event.set()
            LOGGER.info("UNS connected to MQTT broker %s:%s", self.host, self.port)
        else:
            self._connected_event.clear()
            LOGGER.warning("UNS connection rejected with reason code %s", reason_code)

    def _on_disconnect(self, client, userdata, *args):  # pragma: no cover - callback
        # paho 2.x (VERSION2): (client, userdata, disconnect_flags, reason_code, properties)
        # paho 1.x:            (client, userdata, rc)
        reason_code = args[1] if len(args) >= 2 else (args[0] if args else None)
        _ = (client, userdata)
        self._connected_event.clear()
        if not self._stop_event.is_set():
            LOGGER.warning("UNS disconnected from MQTT broker with reason code %s", reason_code)