import asyncio
import hmac
import json
import logging
import os
import time
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.core.factory_manager import FactoryManager
from src.system.tls import check_exposure

logger = logging.getLogger("PulseServer")

app = FastAPI(title="Qualyxus Core v1 API Engine")

# Browsers may only call this API from origins the plant lists; none by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("QUALYXUS_API_CORS_ORIGINS", "").split(",") if o.strip()],
    allow_methods=["GET"],
    allow_headers=["Authorization"],
)


def require_token(request: Request) -> None:
    """With QUALYXUS_API_TOKEN set, the telemetry streams need that token as a bearer token."""
    token = os.getenv("QUALYXUS_API_TOKEN")
    sent = request.headers.get("authorization", "").encode()
    if token and not hmac.compare_digest(sent, f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong API token")


@app.get("/api/v1/stream/telemetry", dependencies=[Depends(require_token)])
async def stream_telemetry(request: Request):
    """
    Exposes an optimized text/event-stream connection broadcasting real-time 
    industrial anomaly diagnostics straight out of volatile memory boundaries.
    """
    factory = FactoryManager()
    event_logger = factory.get_instance("EventLogger")
    
    if event_logger is None:
        raise HTTPException(status_code=503, detail="The runtime is starting")
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
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                except asyncio.TimeoutError:
                    # Enforce low-overhead keepalive heartbeats to keep the TCP link open
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # Unconditionally evict the queue to prevent memory drift leaks
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

@app.get("/api/v1/health")
async def health_check():
    """200 only while the station can really inspect (the same check Docker and systemd use)."""
    manager = FactoryManager().get_instance("CameraManager")
    problems = manager.health_problems() if manager is not None else ["the runtime is starting"]
    return JSONResponse({"status": "ok" if not problems else "unhealthy", "problems": problems, "timestamp": time.time()},
                        status_code=200 if not problems else 503)


class PulseServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, *, tls_cert: str | None = None,
                 tls_key: str | None = None, behind_proxy: bool = False):
        problem = check_exposure(host, tls=bool(tls_cert and tls_key), protected=bool(os.getenv("QUALYXUS_API_TOKEN")),
                                 behind_proxy=behind_proxy)
        if problem:
            raise ValueError(f"api.host {host}: the API {problem}")
        self.tls = {"ssl_certfile": tls_cert, "ssl_keyfile": tls_key} if tls_cert and tls_key else {}
        self.app = app
        self.host = host
        self.port = int(port)
        self.server = None
        self._finished: asyncio.Event | None = None
        
    async def run_async(self):
        logger.info("[PulseServer] Booting ASGI SSE Telemetry Engine on port %d", self.port)
        config = uvicorn.Config(
            self.app, 
            host=self.host, 
            port=self.port, 
            log_level="error",
            # Open SSE streams never end on their own; bound the graceful drain on shutdown.
            timeout_graceful_shutdown=1,
            **self.tls,
        )
        self.server = uvicorn.Server(config)
        self._finished = asyncio.Event()
        
        try:
            await self.server.serve()
        except SystemExit:
            # uvicorn exits the process when it cannot bind. The API is optional; inspection is not.
            logger.error("[PulseServer] Cannot listen on %s:%d. Telemetry API is off; inspection continues.",
                         self.host, self.port)
        except asyncio.CancelledError:
            # Intercept the master loop's raw task cancellation
            logger.warning("[PulseServer] Task cancellation intercepted. Diverting to clean port severance pipeline...")
            self._force_close()
            raise  # Re-raise the CancelledError to preserve async task contract compliance
        finally:
            self._finished.set()

    def _force_close(self):
        for srv in getattr(self.server, "servers", []) or []:
            srv.close()

    async def stop(self, timeout: float = 3.0):
        """
        Awaitable Port Severance Hook.
        Requests a graceful uvicorn exit and waits for serve() to return, forcing the
        listening sockets closed if the drain exceeds `timeout`.
        """
        if self.server is None or self._finished is None or self._finished.is_set():
            return
            
        logger.info("[PulseServer] Triggering active listener drop sequence...")
        self.server.should_exit = True
        try:
            await asyncio.wait_for(self._finished.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[PulseServer] Socket teardown timeout exceeded. Forcing socket closure.")
            self.server.force_exit = True
            self._force_close()
        except asyncio.CancelledError:
            logger.warning("[PulseServer] Stop interrupted. Forcing synchronous socket closure.")
            self._force_close()
            raise
            
        logger.info("[PulseServer] Network engine shutdown complete.")
