"""The operations console: one page, no internet.

Reading is open on this machine until accounts exist; after that everyone signs in. Changing
anything (approving, closing, promoting, reviewing) always needs an account, the role for it
and a reason, and is written to the audit trail before it happens. Nothing here can write to
the machine or change a verdict.

    CONFIG_PATH=config.360.yaml python -m src.console        # http://127.0.0.1:8600
    python -m src.console --host 0.0.0.0 --tls-cert secrets/tls/cert.pem --tls-key secrets/tls/key.pem
"""
from __future__ import annotations

import argparse
import hmac
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from src.console.actions import ActionError, ConsoleActions
from src.console.data import ConsoleData
from src.operations.auth import SESSION_S, UserStore
from src.operations.rbac import Permission, permissions_of
from src.system.tls import check_exposure

STATIC = Path(__file__).parent / "static"
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMAGE = re.compile(r"^[A-Za-z0-9_-]+(\.heat)?\.jpg$")
COOKIE = "qx_session"


class Sessions:
    """Who is signed in. The cookie is a random number; nothing about the person is in it."""

    def __init__(self, users: UserStore):
        self.users = users
        self._open: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def start(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.time()
            self._open = {t: v for t, v in self._open.items() if v[1] > now}
            self._open[token] = (username, now + SESSION_S)
        return token

    def person(self, token: str | None) -> tuple[str, str] | None:
        """(name, role today): a removed or demoted person is looked up again on every request."""
        with self._lock:
            username, expires = self._open.get(token or "", ("", 0.0))
        role = self.users.role_of(username) if username and expires > time.time() else None
        return (username, role) if role else None

    def end(self, token: str | None) -> None:
        with self._lock:
            self._open.pop(token or "", None)


def create_app(cfg: dict[str, Any], data: ConsoleData | None = None, *, secure_cookie: bool = False) -> FastAPI:
    data = data or ConsoleData(cfg)
    users = UserStore((cfg.get("console") or {}).get("users_file", "secrets/users.json"))
    sessions = Sessions(users)
    actions = ConsoleActions(cfg, data)
    app = FastAPI(title="Qualyxus Console", docs_url=None, redoc_url=None, openapi_url=None)

    def reader(request: Request) -> tuple[str, str] | None:
        person = sessions.person(request.cookies.get(COOKIE))
        token = os.getenv("QUALYXUS_API_TOKEN")
        sent = request.headers.get("authorization", "").removeprefix("Bearer ")
        if person or (token and sent and hmac.compare_digest(sent.encode(), token.encode())):
            return person
        if users.has_users() or token:
            raise HTTPException(status_code=401, detail="Sign in")
        return None

    def may(permission: Permission):
        def check(request: Request) -> tuple[str, str]:
            person = sessions.person(request.cookies.get(COOKIE))
            if person is None:
                raise HTTPException(status_code=401, detail="Sign in to change anything")
            # A page on another site cannot set this header, and the cookie is SameSite=Strict as well.
            if request.headers.get("x-qualyxus") != "1":
                raise HTTPException(status_code=403, detail="Not sent by the console")
            if permission not in permissions_of(person[1]):
                raise HTTPException(status_code=403, detail=f"Your role may not {permission.value.replace('_', ' ')}")
            return person
        return Depends(check)

    api = [Depends(reader)]

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; "
                                                       "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        if secure_cookie:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    def found(value: Any) -> Any:
        if value is None:
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(value)

    def done(job, *args, **kwargs) -> Any:
        try:
            return found(job(*args, **kwargs))
        except ActionError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str):
        if name not in {"console.css", "console.js"}:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name)

    # ----------------------------------------------------------------- who
    @app.get("/api/me")
    def me(request: Request):
        person = sessions.person(request.cookies.get(COOKIE))
        return {"accounts": users.has_users(), "name": person[0] if person else None,
                "role": person[1] if person else None,
                "may": sorted(p.value for p in permissions_of(person[1])) if person else []}

    @app.post("/api/login")
    def login(request: Request, body: dict = Body(...)):
        address = request.client.host if request.client else "unknown"
        name, password = str(body.get("name", ""))[:100], str(body.get("password", ""))[:1000]
        wait = users.locked_for(name, address=address)
        if wait > 0:
            raise HTTPException(status_code=429, detail=f"Too many attempts. Try again in {int(wait) + 1} s.")
        if users.authenticate(name, password, address=address) is None:
            raise HTTPException(status_code=401, detail="Wrong name or password")
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, sessions.start(name), httponly=True, samesite="strict", secure=secure_cookie,
                            max_age=int(SESSION_S))
        return response

    @app.post("/api/logout")
    def logout(request: Request):
        sessions.end(request.cookies.get(COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE)
        return response

    # ----------------------------------------------------------------- read
    @app.get("/img/{folder}/{name}", dependencies=api)
    def image(folder: str, name: str):
        path = data.views_dir / folder / name
        if not _SAFE.match(folder) or folder.startswith(".") or not _IMAGE.match(name) or not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/review/{camera}/{item}", dependencies=api)
    def review_image(camera: str, item: str, overlay: bool = False):
        path = actions.review_image(camera, item, overlay)
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(path)

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
    def spc(signal: str = "hold_pressure.active_median", machine: str | None = None):
        return found(data.spc(signal, machine=machine))

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

    @app.get("/api/work", dependencies=api)
    def work():
        return found(actions.work())

    @app.get("/metrics", dependencies=api)
    def metrics():
        return PlainTextResponse(data.metrics(), media_type="text/plain; version=0.0.4")

    # --------------------------------------------------------------- change
    @app.post("/api/actions/{action_id}/{step}")
    def action(action_id: str, step: str, request: Request, body: dict = Body(default={})):
        permission = {"approve": Permission.APPROVE_ACTIONS, "reject": Permission.APPROVE_ACTIONS,
                      "start": Permission.CLOSE_ACTIONS, "close": Permission.CLOSE_ACTIONS}.get(step)
        if permission is None:
            raise HTTPException(status_code=404)
        person = may(permission).dependency(request)
        return done(actions.action, person, action_id, step, str(body.get("reason", "")))

    @app.post("/api/requests/{request_id}/{step}")
    def retraining(request_id: str, step: str, body: dict = Body(default={}), person=may(Permission.MANAGE_MODELS)):
        if step not in ("approve", "reject"):
            raise HTTPException(status_code=404)
        return done(actions.retraining, person, request_id, step == "approve", str(body.get("reason", "")))

    @app.post("/api/models/{recipe}/{step}")
    def model(recipe: str, step: str, body: dict = Body(default={}), person=may(Permission.MANAGE_MODELS)):
        if step not in ("promote", "rollback"):
            raise HTTPException(status_code=404)
        return done(actions.model, person, recipe, step, str(body.get("reason", "")))

    @app.post("/api/review/{camera}/{item}")
    def review(camera: str, item: str, body: dict = Body(default={}), person=may(Permission.REVIEW_PARTS)):
        return done(actions.review, person, camera, item, str(body.get("verdict", "")),
                    str(body.get("defect_type", "")) or None)

    @app.post("/api/baseline/{machine}")
    def baseline(machine: str, body: dict = Body(default={}), person=may(Permission.CLOSE_ACTIONS)):
        return done(actions.baseline, person, machine, bool(body.get("process_change")), str(body.get("reason", "")))

    return app


def main(argv=None) -> None:
    import uvicorn

    from src.config.config import Config

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--behind-proxy", action="store_true", help="Another server in front provides HTTPS")
    args = parser.parse_args(argv)
    cfg = Config.load().raw
    users = UserStore((cfg.get("console") or {}).get("users_file", "secrets/users.json"))
    tls = bool(args.tls_cert and args.tls_key)
    problem = check_exposure(args.host, tls=tls, protected=users.has_users(), behind_proxy=args.behind_proxy)
    if problem:
        parser.error(f"the console {problem}")
    uvicorn.run(create_app(cfg, secure_cookie=tls or args.behind_proxy), host=args.host, port=args.port,
                log_level="warning", ssl_certfile=args.tls_cert, ssl_keyfile=args.tls_key)


if __name__ == "__main__":
    main()
