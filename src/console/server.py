"""Read-only operations console: one page, no internet, nothing it can change on the line.

    CONFIG_PATH=config.360.yaml python -m src.console        # http://127.0.0.1:8600
"""
from __future__ import annotations

import argparse
import hmac
import os
import re
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from src.console.data import ConsoleData

STATIC = Path(__file__).parent / "static"
_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")
_IMAGE = re.compile(r"^[A-Za-z0-9_-]+(\.heat)?\.jpg$")
LOCAL = {"127.0.0.1", "localhost", "::1"}


def _authorise(request: Request) -> None:
    token = os.getenv("QUALYXUS_API_TOKEN")
    if not token:
        return
    sent = request.headers.get("authorization", "").removeprefix("Bearer ") or request.cookies.get("qx_token", "")
    if not hmac.compare_digest(sent.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong token")


def create_app(cfg: dict[str, Any], data: ConsoleData | None = None) -> FastAPI:
    data = data or ConsoleData(cfg)
    app = FastAPI(title="Qualyxus Console", docs_url=None, redoc_url=None, openapi_url=None)
    api = [Depends(_authorise)]

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    def found(value: Any) -> Any:
        if value is None:
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(value)

    @app.get("/")
    def index(token: str | None = None):
        response = FileResponse(STATIC / "index.html")
        if token:
            response.set_cookie("qx_token", token, httponly=True, samesite="strict")
        return response

    @app.get("/static/{name}")
    def static(name: str):
        if name not in {"console.css", "console.js"}:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name)

    @app.get("/img/{folder}/{name}", dependencies=api)
    def image(folder: str, name: str):
        if not _SAFE.match(folder) or not _IMAGE.match(name):
            raise HTTPException(status_code=404)
        path = data.views_dir / folder / name
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/state", dependencies=api)
    def state():
        return found(data.state())

    @app.get("/api/parts", dependencies=api)
    def parts(limit: int = 120):
        return found(data.part_list(max(1, min(limit, 1000))))

    @app.get("/api/part/{cycle_id}", dependencies=api)
    def part(cycle_id: str, cavity: int = 0):
        return found(data.part_detail(cycle_id, cavity))

    @app.get("/api/spc", dependencies=api)
    def spc(signal: str = "hold_pressure.active_median"):
        return found(data.spc(signal))

    @app.get("/api/risk", dependencies=api)
    def risk():
        return found(data.risk())

    @app.get("/api/quality", dependencies=api)
    def quality():
        return found(data.quality())

    @app.get("/api/audit", dependencies=api)
    def audit():
        return found(data.audit())

    @app.get("/api/prescriptions", dependencies=api)
    def prescriptions():
        return found(data.prescriptions())

    return app


def main(argv=None) -> None:
    import uvicorn

    from src.config.config import Config

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args(argv)
    if args.host not in LOCAL and not os.getenv("QUALYXUS_API_TOKEN"):
        parser.error("set QUALYXUS_API_TOKEN before opening the console to the network")
    print(f"Qualyxus console: http://{args.host}:{args.port}")
    uvicorn.run(create_app(Config.load().raw), host=args.host, port=args.port, log_level="warning")
