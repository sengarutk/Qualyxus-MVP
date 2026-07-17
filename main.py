import asyncio
import logging
import signal
import sys
import time
from pymodbus.client import ModbusTcpClient

from src.config.vault import ConfigVault
from src.core.factory_manager import FactoryManager
from src.core.hub_connector import HubConnector
from src.logging.event_logger import EventLogger
from src.api.pulse_server import PulseServer
from src.inference.infer_live import InferenceMultiplexer
from src.camera.camera_manager import CameraManager
from src.analytics.vlm_narrator import VLMNarrator
from src.fleet.historian import PredictiveHistorian

logger = logging.getLogger("QualyxusCore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CoreOrchestrator:
    """
    Master Application Lifecycle Orchestrator.
    Coordinates initialization, monitors background daemons, and enforces 
    a strict, linear, race-condition-free shutdown matrix across all OS targets.
    """
    def __init__(self):
        self.running = False
        self.t_pulse = None
        self.t_multi = None
        
        logger.info("[Main] Bootstrapping Master Lifecycle Orchestrator...")
        self.manager = FactoryManager()
        if not hasattr(self.manager, "register_service"):
            self.manager.register_service = self.manager.register_instance
        self.factory_manager = self.manager
        
        # Hardened Fix: Standardize on explicit single-instance registration syntax
        self.config = ConfigVault.initialize()
        self.manager.register_instance("ConfigVault", self.config)
        
        # Step A: Register the un-mocked VLMNarrator inside the service locator matrix
        self.narrator = VLMNarrator(config=self.config)
        self.manager.register_service("narrator", self.narrator)
        
        # Step B: Register the un-mocked PredictiveHistorian inside the service locator matrix
        self.historian = PredictiveHistorian()
        self.manager.register_instance("PredictiveHistorian", self.historian)
        
        self.event_logger = EventLogger()
        self.manager.register_instance("EventLogger", self.event_logger)
        
        self.hub_connector = HubConnector()
        self.manager.register_instance("HubConnector", self.hub_connector)
        
        # Core Edge Engine Wrappers
        self.multiplexer = InferenceMultiplexer(self.config)
        self.manager.register_instance("InferenceMultiplexer", self.multiplexer)
        self.pulse_server = PulseServer()
        
        # Instantiate Ingestion Plane Layer
        self.camera_manager = CameraManager(config_data=self.config.raw)
        self.manager.register_instance("CameraManager", self.camera_manager)
        
        # Centralized lifecycle event to coordinate thread-safe shutdowns
        self.shutdown_trigger_event = asyncio.Event()

    async def start(self):
        """Activates, monitors, and guarantees orderly reclamation of the edge engine processing space."""
        self.running = True
        logger.info("[Main] Qualyxus Edge Engine booting...")
        
        self.factory_manager.boot_singletons()
        
        # Sequential Boot
        logger.info("[Main] Awaiting synchronized bootloader sequences...")
        await self.event_logger.start()
        await self.hub_connector.connect()
        
        # Pass the event loop context down natively
        loop = asyncio.get_running_loop()
        FactoryManager().register_instance("MainEventLoop", loop)
        await self.camera_manager.start(loop)
        
        # Spawn background execution daemons
        logger.info("[Main] Spawning background execution daemons...")
        self.t_pulse = asyncio.create_task(self.pulse_server.run_async(), name="PulseServer_Daemon")
        self.t_multi = asyncio.create_task(self.multiplexer.run_batch_loop(), name="Inference_Daemon")
        
        # Create a monitoring task for our centralized shutdown event
        shutdown_monitor = asyncio.create_task(self.shutdown_trigger_event.wait(), name="Shutdown_Monitor")
        
        logger.info("[Main] Primary edge intelligence pipeline fully engaged.")
        
        try:
            done, pending = await asyncio.wait(
                [self.t_pulse, self.t_multi, shutdown_monitor], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Evaluate loop status safely without triggering CancelledError panics
            for task in done:
                if task.get_name() != "Shutdown_Monitor" and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        logger.critical("[Main] FATAL DAEMON CRASH DETECTED: %s", exc, exc_info=exc)
                        break
            
            # Cancel any remaining background monitors
            for task in pending:
                if not task.done():
                    task.cancel()
                        
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("[Main] Operator interrupt intercepted. Engaging execution barrier bypass.")
            self.shutdown_trigger_event.set()
        finally:
            # Deterministic closure path ensures the teardown matrix fires under all boot/run conditions
            await self.shutdown()

    def _execute_blocking_plc_clear(self, plc_ip, plc_port):
        """Executed inside an isolated thread worker pool to protect the async loop from GIL locks."""
        # Maintained Positive Mod: Explicit hard timeout 
        client = ModbusTcpClient(plc_ip, port=plc_port, timeout=1.5)
        try:
            if client.connect():
                client.write_register(100, 0x0000)
                logger.info("[Main] Successfully dispatched Advisory Clear to Safety PLC (Reg 100: 0x0000).")
            else:
                logger.error("[Main] Failed to connect to Safety PLC during shutdown.")
        except Exception as e:
            logger.warning("[Main] Modbus safe-state warning during cycle: %s", e)
        finally:
            # Deterministic connection severing protects file descriptor tables from leaking
            client.close()

    async def shutdown(self):
        if not self.running:
            return
        logger.warning("[Main] Initiating strict 4-stage sequential termination matrix...")
        self.running = False
        
        # Stage 1: Halt Ingestion
        logger.info("[Main] STAGE 1: Halting ingestion loops and freezing data generation...")
        await self.camera_manager.stop()
        
        # Drain active queue pipelines securely with an absolute maximum elapsed time window
        logger.info("[Main] Draining remaining frame packets from vision pipeline cache...")
        drain_timeout_barrier = time.perf_counter() + 2.0  # Force break after 2.0 seconds maximum
        
        while len(self.multiplexer.queue) > 0 and time.perf_counter() < drain_timeout_barrier:
            if not self.t_multi or self.t_multi.done():
                logger.warning("[Main] Inference daemon missing or completed early. Ending cache drain loop.")
                break
            await asyncio.sleep(0.01)
            
        tasks_to_cancel = []
        if self.t_multi and not self.t_multi.done():
            self.t_multi.cancel()
            tasks_to_cancel.append(self.t_multi)
        if self.t_pulse and not self.t_pulse.done():
            self.t_pulse.cancel()
            tasks_to_cancel.append(self.t_pulse)
            
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        try:
            await self.pulse_server.stop()
        except Exception as e:
            logger.warning("[Main] Pulse server explicit stop warning: %s", e)
            
        # Stage 2: Safe-State Hardware
        logger.info("[Main] STAGE 2: Safe-stating hardware layer via Modbus...")
        plc_ip = self.config.get("plc_ip", "10.0.0.50")
        plc_port = self.config.get("plc_port", 502)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._execute_blocking_plc_clear, plc_ip, plc_port)
        
        # Stage 3: Evacuate Persistence
        logger.info("[Main] STAGE 3: Evacuating persistent logger queues to SQLite/CSV...")
        
        historian = self.factory_manager.get_instance("PredictiveHistorian")
        if historian:
            # Execute blocking thread pool shutdown inside an executor to protect the async event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, historian.close)

        if getattr(self.event_logger, "worker_task", None) and not self.event_logger.worker_task.done():
            self.event_logger.worker_task.cancel()
            await asyncio.gather(self.event_logger.worker_task, return_exceptions=True)
            
        # Stage 4: Sever Network Sockets
        logger.info("[Main] STAGE 4: Severing mutual TLS network sockets cleanly...")
        await self.hub_connector.disconnect()
        
        self.factory_manager.shutdown_singletons()
        logger.info("[Main] Core shutdown sequence complete. Process exiting cleanly.")

async def main():
    orchestrator = CoreOrchestrator()
    loop = asyncio.get_running_loop()
    
    def _shutdown_handler():
        logger.warning("[Main] Intercepted OS termination signal. Priming shutdown sequence.")
        orchestrator.shutdown_trigger_event.set()

    # Secure, loop-aware async signal mapping
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_handler)
        except NotImplementedError:
            pass

    try:
        await orchestrator.start()
    finally:
        logger.info("[Main] Reclaiming OS resources. Process terminating cleanly.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[Main] Execution manually interrupted by operator. Halting immediately.")
        sys.exit(0)
