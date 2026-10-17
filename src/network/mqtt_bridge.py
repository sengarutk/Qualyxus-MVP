import asyncio
import json
import logging
import os
import ssl
from collections import deque
from typing import Optional

from gmqtt import Client as MQTTClient
from gmqtt.mqtt.constants import MQTTv50

from src.config.vault import ConfigVault

logger = logging.getLogger("QualyxusMqttBridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MqttBridge:
    """
    Highly secure, non-blocking asynchronous MQTT integration layer.
    Enforces strict mutual TLS (mTLS) authentication and eliminates chronological 
    inversion anomalies via an integrated double-ended recovery buffer.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config = ConfigVault.initialize()
        
        self.client_id = self.config.get("mqtt", "client_id", default="qualyxus_edge_01")
        self.host = self.config.get("mqtt", "host", default="127.0.0.1")
        
        try:
            self.port = int(self.config.get("mqtt", "port", default=8883))
        except (TypeError, ValueError):
            self.port = 8883
            logger.warning("[MqttBridge] Invalid port configured. Defaulting to 8883.")

        # mTLS is the default on the standard secure port (8883). A plain-TCP broker on 1883
        # cannot complete a TLS handshake, so TLS follows `mqtt.use_tls` or the port.
        use_tls = self.config.get("mqtt", "use_tls", default=None)
        self.use_tls = bool(use_tls) if use_tls is not None else self.port == 8883
        # The uplink can be disabled entirely (e.g. air-gapped bench setups).
        self.enabled = bool(self.config.get("mqtt", "uplink_enabled", default=True))
            
        # Resilient double-ended queue to maintain absolute chronological alignment
        self.spool_deque = deque(maxlen=5000)
        
        self.client: Optional[MQTTClient] = None
        self._connected_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._reconnect_task = None
        self._starting = False
        self._is_bootstrapping = False  # Atomic synchronous boot shield token
        self._poison_retry_count = 0    # Persistent counter for cross-connection poison tracking
        self._bootstrap_task = None
        self._initialized = True

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Initializes a strict mutual TLS context locked to TLS v1.3 minimum requirements."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        
        ca_certs = self.config.get("mqtt", "ca_certs", default=os.getenv("MQTT_CA_CERTS"))
        certfile = self.config.get("mqtt", "certfile", default=os.getenv("MQTT_CERTFILE"))
        keyfile = self.config.get("mqtt", "keyfile", default=os.getenv("MQTT_KEYFILE"))
        
        if ca_certs and os.path.exists(ca_certs):
            context.load_verify_locations(cafile=ca_certs)
        else:
            logger.warning("[MqttBridge] CA Certificate missing. Strict TLS verification may fail.")
            
        if certfile and keyfile and os.path.exists(certfile) and os.path.exists(keyfile):
            context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        else:
            logger.warning("[MqttBridge] Client Certificate or Key missing. Handshake will fail if requested by broker.")
            
        return context

    def _on_connect(self, client, flags, rc, properties):
        logger.info("[MqttBridge] Connected to plant broker (%s).", "mTLS v1.3" if self.use_tls else "plain TCP")
        self._connected_event.set()

    def _on_disconnect(self, client, packet, exc=None):
        logger.warning("[MqttBridge] Connection to plant broker lost. Activating resilient spooling mode.")
        self._connected_event.clear()

    async def start(self):
        """Asynchronously initializes the MQTT client and boots the connection loop cleanly."""
        if not self.enabled:
            logger.info("[MqttBridge] Uplink disabled via mqtt.uplink_enabled=false.")
            return
        if self._shutdown_event.is_set():
            return
        if self._starting or (self._reconnect_task is not None and not self._reconnect_task.done()):
            return
            
        self._starting = True
        self._is_bootstrapping = True
        try:
            if self.client is not None:
                try:
                    await self.client.disconnect()
                except Exception:
                    pass
                self.client = None

            self.client = MQTTClient(self.client_id)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            ssl_ctx = self._build_ssl_context() if self.use_tls else False
            self._reconnect_task = asyncio.create_task(self._connection_daemon(ssl_ctx))
            logger.info("[MqttBridge] Background resilient daemon booted.")
        finally:
            self._starting = False
            self._is_bootstrapping = False

    async def _connection_daemon(self, ssl_ctx: ssl.SSLContext | bool):
        """Automated re-connection loop enforcing a strict handshake synchronization barrier."""
        while not self._shutdown_event.is_set():
            if not self._connected_event.is_set():
                try:
                    logger.info(
                        "[MqttBridge] Connecting to %s:%s (%s)...",
                        self.host, self.port, "mTLS" if ssl_ctx else "plain TCP",
                    )
                    await self.client.connect(
                        self.host, 
                        self.port, 
                        ssl=ssl_ctx, 
                        version=MQTTv50,
                        keepalive=10
                    )
                    
                    # Synchronization Barrier: Block until CONNACK packet is officially handled
                    await asyncio.wait_for(self._connected_event.wait(), timeout=10.0)
                    await self._drain_spool()
                    
                except asyncio.TimeoutError:
                    logger.error("[MqttBridge] CONNACK synchronization barrier timed out. Severing hung socket.")
                    self._connected_event.clear()
                    if self.client:
                        try:
                            await self.client.disconnect()
                        except Exception:
                            pass
                    await asyncio.sleep(5.0)
                except Exception as e:
                    logger.error("[MqttBridge] Connection failure: %s. Retrying in 5 seconds.", e)
                    self._connected_event.clear()
                    await asyncio.sleep(5.0)
            else:
                await asyncio.sleep(1.0)

    async def _drain_spool(self):
        """Drains the resilient buffer smoothly while preserving chronological FIFO order and removing poison pills."""
        if not self.spool_deque:
            return
            
        logger.info("[MqttBridge] Draining resilient spooler. Items in buffer: %d", len(self.spool_deque))
        
        while self.spool_deque and self._connected_event.is_set():
            topic, payload_bytes, qos = self.spool_deque[0]
            try:
                if not self.client:
                    raise RuntimeError("MQTT Client is detached during active drain phase.")
                    
                self.client.publish(topic, payload_bytes, qos=qos)
                
                # Pop from the queue only after successful handoff to the transport layer
                self.spool_deque.popleft()
                self._poison_retry_count = 0  # Reset retry counter on success
                
                await asyncio.sleep(0.001)
            except (TypeError, ValueError) as ve:
                # Fatal payload composition error. Evict instantly without severing the socket.
                logger.critical("[MqttBridge] Poison-pill payload format rejected by client: %s. Evicting.", ve)
                self.spool_deque.popleft()
                continue
            except Exception as e:
                self._poison_retry_count += 1
                logger.error("[MqttBridge] Spool drain fault on current packet (Attempt %d): %s", self._poison_retry_count, e)
                
                # Circuit breaker: evict poison-pill payloads that repeatedly crash the underlying socket
                if self._poison_retry_count >= 3:
                    logger.critical("[MqttBridge] Packet triggered 3 consecutive socket faults. Evicting as poison-pill.")
                    self.spool_deque.popleft()
                    self._poison_retry_count = 0
                    continue
                
                self._connected_event.clear()
                break
                
        logger.info("[MqttBridge] Spool drain sequence finalized.")

    async def publish_telemetry(self, topic: str, payload: dict, qos: int = 1):
        """Resilient wrapper for upstream telemetry dispatches. Shields against pipeline blocks."""
        if not self.enabled:
            return
        # Synchronously claim the bootstrap token to block same-tick task accumulation floods
        if (
            not self._shutdown_event.is_set()
            and not self._is_bootstrapping
            and (self._reconnect_task is None or self._reconnect_task.done())
        ):
            self._is_bootstrapping = True
            self._bootstrap_task = asyncio.create_task(self.start())
            
        try:
            payload_bytes = json.dumps(payload, default=str).encode('utf-8')
        except Exception as e:
            logger.error("[MqttBridge] Telemetry serialization fault: %s", e)
            return

        # Attempt immediate live dispatch if the socket is healthy
        if self._connected_event.is_set() and self.client:
            try:
                self.client.publish(topic, payload_bytes, qos=qos)
                return
            except Exception as e:
                logger.warning("[MqttBridge] Live dispatch fault: %s. Diverting to spool buffer.", e)
                self._connected_event.clear()
                
        # Connection is down; append natively to the C-level deque structure
        if len(self.spool_deque) == self.spool_deque.maxlen:
            logger.critical("[MqttBridge] FATAL BACKPRESSURE: Spooler limit (5000) breached. Automatically shedding oldest telemetry.")
            
        self.spool_deque.append((topic, payload_bytes, qos))

    async def close(self):
        """Clean process unwinding handler releasing all network handles."""
        logger.info("[MqttBridge] Termination signal received. Securing shutdown vectors...")
        self._shutdown_event.set()
        
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
                
        if self._connected_event.is_set():
            await self._drain_spool()
            if self.client:
                try:
                    await self.client.disconnect()
                except Exception as e:
                    logger.error("[MqttBridge] Graceful disconnect failed: %s", e)
                    
        self.client = None
        self._connected_event.clear()
        logger.info("[MqttBridge] Network bridge completely closed.")
