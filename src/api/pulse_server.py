import asyncio
import json
import logging
import copy
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.core.factory_manager import FactoryManager
from src.logging.event_logger import EventLogger  # FIXED: Imported to resolve lazy-initialization requirements

logger = logging.getLogger("PulseServer")

app = FastAPI(title="Qualyxus Core v1 API Engine")

# Activate explicit CORS bindings to enable seamless cross-origin web dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TelemetryBroadcaster:
    """
    Singleton Broadcaster Pattern for non-blocking in-memory pub-sub.
    Protects client boundary isolation via pre-serialized immutable strings 
    and bounded memory drop ring buffers, eliminating slow deep copies.
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
        self._subscribers = set()
        self._lock = asyncio.Lock()
        self._initialized = True

    async def register(self) -> asyncio.Queue:
        # Bounded allocation to prevent infinite memory expansion under slow consumers
        queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        logger.info("[Broadcaster] New client registered. Total active: %d", len(self._subscribers))
        return queue

    async def unregister(self, queue: asyncio.Queue):
        async with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
        logger.info("[Broadcaster] Client disconnected. Total active: %d", len(self._subscribers))

    async def broadcast(self, payload: dict):
        """Dispatches isolated snapshots to all active client queues asynchronously."""
        try:
            # Pre-serialize to an immutable string outside the lock.
            # This completely eliminates cross-client state bleeding and avoids 
            # the massive performance penalty of `copy.deepcopy()`.
            immutable_json_payload = json.dumps(payload)
        except Exception as e:
            logger.error("[Broadcaster] Failed to serialize telemetry payload: %s", e)
            return

        async with self._lock:
            if not self._subscribers:
                return
                
            for queue in self._subscribers:
                try:
                    queue.put_nowait(immutable_json_payload)
                except asyncio.QueueFull:
                    try:
                        # Drop oldest frame from the buffer ring if client backpressure builds
                        queue.get_nowait()
                        queue.put_nowait(immutable_json_payload)
                    except asyncio.QueueEmpty:
                        pass
                except Exception as e:
                    logger.error("[Broadcaster] Error dispatching to subscriber: %s", e)

broadcaster = TelemetryBroadcaster()

@app.get("/api/v1/stream/telemetry")
async def stream_telemetry(request: Request):
    """
    Exposes an optimized text/event-stream connection broadcasting real-time 
    industrial anomaly diagnostics straight out of volatile memory boundaries.
    """
    factory = FactoryManager()
    event_logger = factory.get_instance("EventLogger")
    
    # --- FIXED: SELF-HEALING HOOK TO ELIMINATE INSTANT EXIT LOGGING ERROR ---
    if not event_logger:
        logger.warning("[PulseServer] EventLogger absent from FactoryManager cache. Invoking self-healing instantiation...")
        event_logger = EventLogger()
        if not hasattr(event_logger, "subscribers") or event_logger.subscribers is None:
            event_logger.subscribers = []
        factory.register_instance("EventLogger", event_logger)
        
    if not hasattr(event_logger, "subscribers") or event_logger.subscribers is None:
        event_logger.subscribers = []

    # Allocate an isolated asyncio Queue for this web client session thread
    client_queue = asyncio.Queue(maxsize=256)
    event_logger.subscribers.append(client_queue)
    logger.info(f"[SSE] Frontend channel connected. Active subscribers count: {len(event_logger.subscribers)}")

    async def event_generator():
        try:
            while True:
                # Proactive Disconnect Detection
                if await request.is_disconnected():
                    break
                
                try:
                    # Non-blocking await with a low-overhead heartbeat timeout fallthrough
                    payload = await asyncio.wait_for(client_queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # Enforce low-overhead keepalive heartbeats to keep the TCP link open
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # CRITICAL SHIELD: Unconditionally evict the queue to prevent memory drift leaks
            if client_queue in event_logger.subscribers:
                event_logger.subscribers.remove(client_queue)
            logger.info(f"[SSE] Frontend channel disconnected. Active subscribers count: {len(event_logger.subscribers)}")

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Pragma": "no-cache",
        "Expires": "0"
    }

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

@app.get("/api/v1/telemetry/stream")
async def telemetry_stream():
    """
    Asynchronous Server-Sent Events (SSE) network engine.
    Injects transmission headers to bypass corporate proxy buffering.
    """
    async def event_generator():
        queue = await broadcaster.register()
        try:
            while True:
                # Payload is already serialized into an immutable JSON string by the broadcaster
                json_string = await queue.get()
                yield f"data: {json_string}\n\n"
        except asyncio.CancelledError:
            logger.info("[PulseServer] Connection reset by client window close request.")
            raise
        finally:
            await broadcaster.unregister(queue)

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Pragma": "no-cache",
        "Expires": "0"
    }

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

@app.get("/api/v1/health")
async def health_check():
    """
    Dedicated HTTP Health Probe Endpoint.
    Resolves Flaw where hitting the SSE stream endpoint causes connection leaks.
    """
    return {"status": "ok", "timestamp": time.time()}

class PulseServer:
    def __init__(self):
        self.app = app
        self.server = None
        
    async def run_async(self):
        logger.info("[PulseServer] Booting ASGI SSE Telemetry Engine on port 8000")
        config = uvicorn.Config(
            self.app, 
            host="0.0.0.0", 
            port=8000, 
            log_level="error"
        )
        self.server = uvicorn.Server(config)
        
        try:
            await self.server.serve()
        except asyncio.CancelledError:
            # Hardening Guard: Intercept the master loop's raw task cancellation
            logger.warning("[PulseServer] Task cancellation intercepted. Diverting to clean port severance pipeline...")
            await self.stop()
            raise  # Re-raise the CancelledError to preserve async task contract compliance

    async def stop(self):
        """
        Awaitable Port Severance Hook.
        Forcefully unbinds HTTP listeners from the OS network stack and guarantees socket release.
        """
        if self.server is None or not self.server.started:
            return
            
        logger.info("[PulseServer] Triggering active listener drop sequence...")
        self.server.should_exit = True
        
        # Validation Poll Loop with a strict maximum bound to guarantee closure
        start_time = time.perf_counter()
        timeout = 2.0  # 2.0 seconds maximum socket severance budget
        
        try:
            while self.server.started:
                if time.perf_counter() - start_time > timeout:
                    logger.warning("[PulseServer] Socket teardown timeout exceeded. Forcing socket closure.")
                    for srv in getattr(self.server, "servers", []):
                        srv.close()
                    break
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Flaw Fix: Catch the inevitable CancelledError thrown by `asyncio.sleep()` when 
            # this method is called from within a cancellation handler.
            logger.warning("[PulseServer] Sleep trap triggered during stop loop. Forcing synchronous socket closure.")
            for srv in getattr(self.server, "servers", []):
                srv.close()
            
        logger.info("[PulseServer] Network engine shutdown complete.")

# --- ADDED EXECUTION ENTRY-POINT BLOCK TO SOLVE SILENT EXIT ISSUE ---
if __name__ == "__main__":
    server = PulseServer()
    asyncio.run(server.run_async())