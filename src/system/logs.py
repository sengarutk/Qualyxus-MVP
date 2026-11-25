"""One place that decides how the logs look: text for people, JSON lines for a log collector."""
from __future__ import annotations

import json
import logging
import os
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
               "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure() -> None:
    """``QUALYXUS_LOG_FORMAT=json`` for JSON lines, ``QUALYXUS_LOG_LEVEL`` for the level."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if os.getenv("QUALYXUS_LOG_FORMAT") == "json"
                         else logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("QUALYXUS_LOG_LEVEL", "INFO").upper())
