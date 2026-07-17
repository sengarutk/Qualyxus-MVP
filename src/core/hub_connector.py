import logging
import asyncio
from src.network.mqtt_bridge import MqttBridge
from src.core.factory_manager import FactoryManager

logger = logging.getLogger("QualyxusHub")

class HubConnector:
    """
    Central plant integration gateway.
    Wires the hardened MqttBridge into the application lifecycle and 
    registers it with the FactoryManager to eradicate dead-code isolation.
    """
    def __init__(self):
        logger.info("[HubConnector] Bootstrapping central integration gateway.")
        self.bridge = MqttBridge()
        
        # Inject the active bridge into the global dependency map so all downstream 
        # inference pipelines can dynamically resolve and dispatch to it.
        FactoryManager.register_instance("MqttBridge", self.bridge)

    async def connect(self):
        """
        Asynchronous initialization hook triggered by the main orchestrator.
        Awaits the bootloader initialization task to ensure the daemon loop starts cleanly.
        """
        logger.info("[HubConnector] Dispatching secure uplink connection sequence.")
        try:
            await self.bridge.start()
        except Exception as e:
            logger.error("[HubConnector] Failed to initialize secure uplink sequence: %s", e)

    async def disconnect(self):
        """
        Asynchronous shutdown hook triggered by the main orchestrator.
        Strictly blocks the shutdown progression until the MqttBridge completes 
        its TLS teardown and flushes outstanding spooled frames safely to disk/wire.
        """
        logger.warning("[HubConnector] Severing secure uplinks and locking spoolers.")
        try:
            # Enforce true await execution to prevent the process from dropping out 
            # while data is mid-transit in the network buffers.
            await self.bridge.close()
        except Exception as e:
            logger.error("[HubConnector] Error encountered during network bridge teardown: %s", e)
