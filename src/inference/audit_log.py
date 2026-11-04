"""Tamper-evident audit trail of every part verdict.

Customer quality audits (IATF 16949, ISO 9001, medical ISO 13485) ask: *which model, with
which threshold, judged this part, from which image?* Each verdict is appended as one JSON
line to a daily file, with the image's SHA-256, the exact model versions/digests and the
threshold. Every record also carries the hash of the previous record, forming a hash
chain: editing or deleting any past record breaks verification from that point on.

    python -m src.inference.audit_log verify data/audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

GENESIS = "0" * 64


def image_sha256(image: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def _record_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class InspectionAuditLog:
    def __init__(self, directory: str | Path = "data/audit"):
        self.dir = Path(directory)
        self._lock = threading.Lock()
        self._last_hash: str | None = None

    def _tail_hash(self) -> str:
        files = sorted(self.dir.glob("*.jsonl"))
        for f in reversed(files):
            lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return json.loads(lines[-1])["hash"]
        return GENESIS

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            if self._last_hash is None:
                self._last_hash = self._tail_hash()
            ts = float(record.get("ts") or time.time())
            entry = {**record, "ts": ts, "prev_hash": self._last_hash}
            entry["hash"] = _record_hash(entry)
            path = self.dir / (time.strftime("%Y-%m-%d", time.gmtime(ts)) + ".jsonl")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str, sort_keys=True) + "\n")
            self._last_hash = entry["hash"]
            return entry


def verify_chain(directory: str | Path) -> dict[str, Any]:
    """Check every record's hash and its link to the previous record."""
    prev = GENESIS
    n = 0
    for f in sorted(Path(directory).glob("*.jsonl")):
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("prev_hash") != prev:
                return {"ok": False, "records": n, "error": f"{f.name}:{lineno} broken link (record removed or reordered)"}
            if _record_hash(rec) != rec.get("hash"):
                return {"ok": False, "records": n, "error": f"{f.name}:{lineno} content does not match its hash (edited)"}
            prev = rec["hash"]
            n += 1
    return {"ok": True, "records": n, "head": prev}


def main(argv=None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("directory", nargs="?", default="data/audit")
    args = parser.parse_args(argv)
    result = verify_chain(args.directory)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
