"""Tell someone: webhook, e-mail or an andon output, when the station stops being healthy.

``alerts:`` in the config. A problem is sent once, again after ``repeat_s``, and once more when
it is over. Sending never holds up inspection, and a failed delivery is tried again.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import subprocess
import threading
import time
import urllib.request
from email.message import EmailMessage
from typing import Any, Callable

logger = logging.getLogger("QualyxusAlerts")


class AlertSender:
    def __init__(self, cfg: dict[str, Any], station: str, problems: Callable[[], list[str]], *, every_s: float = 5.0):
        self.cfg, self.station, self.problems, self.every_s = cfg, station, problems, float(every_s)
        self.hold_s = float(cfg.get("hold_s", 15.0))         # this long before it is worth waking someone
        self.repeat_s = float(cfg.get("repeat_s", 3600.0))
        self._since: dict[str, float] = {}
        self._sent: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("webhook") or self.cfg.get("email") or self.cfg.get("command"))

    def check(self, now: float) -> None:
        current = set(self.problems())
        for problem in current:
            self._since.setdefault(problem, now)
            due = now - self._since[problem] >= self.hold_s and now - self._sent.get(problem, -1e18) >= self.repeat_s
            if due and self.send("PROBLEM", problem):
                self._sent[problem] = now
        for problem in [p for p in self._since if p not in current]:
            if problem in self._sent and not self.send("OVER", problem):
                continue
            self._since.pop(problem)
            self._sent.pop(problem, None)

    def send(self, kind: str, text: str) -> bool:
        ok = True
        for name, job in (("webhook", self._webhook), ("email", self._email), ("command", self._command)):
            if not self.cfg.get(name):
                continue
            try:
                job(kind, text)
            except Exception as e:
                logger.error("[Alerts] %s not delivered: %s", name, e)
                ok = False
        logger.warning("[Alerts] %s: %s", kind, text)
        return ok

    def _webhook(self, kind: str, text: str) -> None:
        url = str(self.cfg["webhook"])
        if not url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("the webhook must use https")
        body = json.dumps({"station": self.station, "kind": kind, "text": text, "ts": time.time()}).encode()
        headers = {"Content-Type": "application/json"}
        token = os.getenv(str(self.cfg.get("webhook_token_env", "QUALYXUS_WEBHOOK_TOKEN")))
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=5.0) as reply:     # noqa: S310 - scheme checked above
            if reply.status >= 300:
                raise OSError(f"HTTP {reply.status}")

    def _email(self, kind: str, text: str) -> None:
        mail = self.cfg["email"]
        message = EmailMessage()
        message["Subject"] = f"[{self.station}] {kind}: {text}"[:180]
        message["From"], message["To"] = mail["from"], ", ".join(mail["to"])
        message.set_content(f"{self.station}\n{kind}: {text}\n{time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        with smtplib.SMTP(mail["host"], int(mail.get("port", 587)), timeout=10) as smtp:
            if mail.get("starttls", True):
                smtp.starttls(context=ssl.create_default_context())
            if mail.get("username"):
                smtp.login(mail["username"], os.environ[str(mail.get("password_env", "QUALYXUS_SMTP_PASSWORD"))])
            smtp.send_message(message)

    def _command(self, kind: str, text: str) -> None:
        """An andon light or an SMS modem: run with the kind and the text as its last two arguments, no shell."""
        subprocess.run([*map(str, self.cfg["command"]), kind, text], timeout=10, check=True, capture_output=True)

    def start(self) -> None:
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="alerts", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=12.0)

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                self.check(time.monotonic())
            except Exception as e:
                logger.error("[Alerts] %s", e, exc_info=True)
