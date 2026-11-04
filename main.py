import asyncio
import logging
import signal
import sys
from pymodbus.client import ModbusTcpClient

from src.config.vault import ConfigVault
from src.core.factory_manager import FactoryManager
from src.core.hub_connector import HubConnector
from src.logging.event_logger import EventLogger
from src.api.pulse_server import PulseServer
from src.camera.camera_manager import CameraManager
from src.analytics.vlm_narrator import VLMNarrator
from src.fleet.historian import PredictiveHistorian
from src.process.service import ProcessIntelligenceService
from src.process.sources import build_cycle_source

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
        
        # Persist to the same SQLite/CSV targets the dashboard and analytics read.
        self.event_logger = EventLogger(
            sqlite_path=self.config.get("logging", "sqlite_path", default="data/events/events.db"),
            csv_path=self.config.get("logging", "csv_path", default="data/events/events.csv"),
        )
        self.manager.register_instance("EventLogger", self.event_logger)
        
        self.hub_connector = HubConnector()
        self.manager.register_instance("HubConnector", self.hub_connector)
        
        api_cfg = self.config.get("api", default={}) or {}
        self.pulse_server = PulseServer(
            host=str(api_cfg.get("host", "0.0.0.0")),
            port=int(api_cfg.get("port", 8000)),
        )
        
        # Instantiate Ingestion Plane Layer. CameraManager owns the single batched
        # InferenceMultiplexer that its capture workers submit frames to.
        self.camera_manager = CameraManager(config_data=self.config.raw)
        self.manager.register_instance("CameraManager", self.camera_manager)
        self.multiplexer = self.camera_manager.multiplexer

        # Process-quality intelligence: machine cycles <-> inspected parts.
        self.process_service = None
        process_cfg = self.config.get("process", default={}) or {}
        if process_cfg.get("enabled", False):
            self.process_service = ProcessIntelligenceService(
                process_cfg, cameras_cfg=self.config.get("cameras", default=[]) or [],
            )
            self.process_service.attach_source(build_cycle_source(process_cfg))
            self.process_service.add_alert_callback(self._on_process_alert)
            self.manager.register_instance("ProcessIntelligence", self.process_service)
            self.camera_manager.attach_process_service(self.process_service)
        
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
        self._loop = loop
        if self.process_service is not None:
            self.process_service.start()
        await self.camera_manager.start(loop)
        
        # Spawn background execution daemons
        logger.info("[Main] Spawning background execution daemons...")
        self.t_pulse = asyncio.create_task(self.pulse_server.run_async(), name="PulseServer_Daemon")
        
        # Create a monitoring task for our centralized shutdown event
        shutdown_monitor = asyncio.create_task(self.shutdown_trigger_event.wait(), name="Shutdown_Monitor")
        
        logger.info("[Main] Primary edge intelligence pipeline fully engaged.")
        
        try:
            done, pending = await asyncio.wait(
                [self.t_pulse, shutdown_monitor], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Evaluate loop status safely without triggering CancelledError panics
            for task in done:
                if task.get_name() != "Shutdown_Monitor" and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        logger.critical("[Main] FATAL DAEMON CRASH DETECTED: %s", exc, exc_info=exc)
                        break
            
            # Only the shutdown monitor is cancelled here; the daemons are stopped
            # gracefully (in order) by shutdown().
            if not shutdown_monitor.done():
                shutdown_monitor.cancel()
                        
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("[Main] Operator interrupt intercepted. Engaging execution barrier bypass.")
            self.shutdown_trigger_event.set()
        finally:
            # Deterministic closure path ensures the teardown matrix fires under all boot/run conditions
            await self.shutdown()

    def _on_process_alert(self, assessment, cycle, prescriptions):
        """High/watch-risk cycle: publish to the plant MQTT uplink and live SSE clients."""
        payload = {
            "type": "process_risk",
            "cycle_id": assessment.cycle_id,
            "machine_id": cycle.machine_id,
            "risk": assessment.risk,
            "decision": assessment.decision,
            "likely_defect": assessment.likely_defect,
            "model_version": assessment.model_version,
            "top_factors": assessment.top_factors,
            "prescriptions": [p.rationale for p in prescriptions],
            "ts": assessment.ts,
        }
        if assessment.decision == "high":
            logger.warning("[Process] HIGH defect risk %.2f on %s (%s)", assessment.risk, assessment.cycle_id,
                           ", ".join(f["feature"] for f in assessment.top_factors[:3]) or "no dominant factor")
        self.event_logger.broadcast_to_subscribers(payload)
        loop = getattr(self, "_loop", None)
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.hub_connector.bridge.publish_telemetry(f"qualyxus/process/risk/{cycle.machine_id}", payload),
                loop,
            )

    def _execute_blocking_plc_clear(self, plc_ip, plc_port, unit_id=1):
        """Executed inside an isolated thread worker pool to protect the async loop from GIL locks."""
        # Maintained Positive Mod: Explicit hard timeout, no reconnect storm during shutdown.
        client = ModbusTcpClient(plc_ip, port=plc_port, timeout=1.5, retries=0)
        try:
            if client.connect():
                client.write_register(100, 0x0000, device_id=unit_id)
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
        
        # Stage 1: Halt Ingestion. CameraManager joins the capture workers first (no new frames),
        # then stops the shared inference multiplexer, then drains its disk/reflex writers.
        # It blocks on thread joins, so it runs off the event loop.
        logger.info("[Main] STAGE 1: Halting ingestion loops and freezing data generation...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.camera_manager.stop_all_streams)
        if self.process_service is not None:
            await loop.run_in_executor(None, self.process_service.stop)
        
        # Ask the API server to exit gracefully; cancel it only if it does not comply.
        try:
            await self.pulse_server.stop()
        except Exception as e:
            logger.warning("[Main] Pulse server explicit stop warning: %s", e)
        if self.t_pulse and not self.t_pulse.done():
            try:
                await asyncio.wait_for(asyncio.shield(self.t_pulse), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.t_pulse.cancel()
                await asyncio.gather(self.t_pulse, return_exceptions=True)
            except Exception as e:
                logger.warning("[Main] Pulse server task ended with error: %s", e)
            
        # Stage 2: Safe-State Hardware (only when a physical PLC endpoint is configured)
        plc_ip = self.config.get("plc_ip", default=None) or self.config.get("telemetry", "endpoint", default=None)
        if plc_ip:
            logger.info("[Main] STAGE 2: Safe-stating hardware layer via Modbus (%s)...", plc_ip)
            plc_port = int(self.config.get("plc_port", default=502) or 502)
            unit_id = int(self.config.get("plc_unit_id", default=1) or 1)
            await loop.run_in_executor(None, self._execute_blocking_plc_clear, plc_ip, plc_port, unit_id)
        else:
            logger.info("[Main] STAGE 2: No PLC endpoint configured (plc_ip / telemetry.endpoint). Skipping Modbus safe-state.")
        
        # Stage 3: Evacuate Persistence
        logger.info("[Main] STAGE 3: Evacuating persistent logger queues to SQLite/CSV...")
        
        historian = self.factory_manager.get_instance("PredictiveHistorian")
        if historian:
            # Execute blocking thread pool shutdown inside an executor to protect the async event loop
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
