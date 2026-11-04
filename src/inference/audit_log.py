"""Tamper-evident audit trail of every part verdict.

Customer quality audits (IATF 16949, ISO 9001, medical ISO 13485) ask: *which model, with
which threshold, judged this part, from which image?* Each verdict is appended as one JSON
line to a daily file, with the image's SHA-256, the exact model versions/digests and the
threshold.

Three things make tampering visible:

* every record carries the previous record's hash, so editing or removing one breaks the chain
* hashes are keyed (HMAC-SHA256), so nobody can rebuild the chain after an edit without the key
* an anchor file outside the log directory records how many records exist and the latest
  hash, so deleting the newest records, which a chain alone cannot show, is caught too
* if the log and anchor disagree when logging starts up, an ``audit_gap`` record goes into the
  chain, so restarting cannot quietly bury the evidence

Records are forced to disk before the anchor moves. A power cut can tear only the last line,
which is set aside in a ``.torn`` file at the next start, not mistaken for tampering.

    python -m src.inference.audit_log verify data/audit --key-file secrets/audit.key

Someone holding the key can rewrite everything, so keep it on a separate volume or account
from the log, and copy the anchor off the box regularly (a printed copy works).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("AuditLog")

GENESIS = "0" * 64
AUDIT_GAP = "audit_gap"


def image_sha256(image: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def _record_hash(record: dict[str, Any], key: bytes) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hmac.new(key, json.dumps(body, sort_keys=True, default=str).encode(), hashlib.sha256).hexdigest()


def _anchor_mac(records: int, head: str, key: bytes) -> str:
    return hmac.new(key, f"{records}:{head}".encode(), hashlib.sha256).hexdigest()


def _anchor_problem(records: int, head: str, anchor_file: Path, key: bytes) -> str | None:
    """Why this log does not match its anchor, or None when it does."""
    if not anchor_file.exists():
        return f"anchor file missing: {anchor_file}" if records else None
    try:
        anchor = json.loads(anchor_file.read_text(encoding="utf-8"))
        expected, mark = int(anchor["records"]), str(anchor["head"])
    except (ValueError, KeyError, TypeError):
        return "anchor file is unreadable"
    if not hmac.compare_digest(_anchor_mac(expected, mark, key), str(anchor.get("mac"))):
        return "anchor file was altered"
    if records < expected:
        return f"{expected - records} newest records are missing"
    if records == expected and head != mark:
        return "latest record does not match the anchor"
    if records > expected + 1:  # one extra record is a crash between the two writes
        return "log has records the anchor never saw"
    return None


def _set_aside_torn_line(path: Path) -> bool:
    """A line with no newline was cut off mid-write; move it out so the chain can continue."""
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return False
    keep = raw.rfind(b"\n") + 1
    with open(path.with_suffix(".torn"), "ab") as fh:
        fh.write(raw[keep:] + b"\n")
    with open(path, "r+b") as fh:
        fh.truncate(keep)
    return True


def anchor_path(directory: str | Path) -> Path:
    path = Path(directory)
    return path.with_name(path.name + ".anchor.json")


def _key_path(directory: str | Path, key_file: str | Path | None) -> Path:
    if key_file or os.getenv("QUALYXUS_AUDIT_KEY_FILE"):
        return Path(key_file or os.environ["QUALYXUS_AUDIT_KEY_FILE"])
    path = Path(directory)
    return path.with_name(path.name + ".key")


class InspectionAuditLog:
    def __init__(self, directory: str | Path = "data/audit", key_file: str | Path | None = None):
        self.dir = Path(directory)
        self.key_path = _key_path(directory, key_file)
        self.anchor = anchor_path(directory)
        self._lock = threading.Lock()
        self._last_hash: str | None = None
        self._records = 0
        self._last_file = ""
        self._key: bytes | None = None

    def _load_key(self) -> bytes:
        if self._key is None:
            if not self.key_path.exists():
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(secrets.token_hex(32))
            self._key = bytes.fromhex(self.key_path.read_text(encoding="utf-8").strip())
        return self._key

    def _restore_tail(self, key: bytes) -> None:
        self._last_hash, self._records = GENESIS, 0
        files = sorted(self.dir.glob("*.jsonl"))
        if files and _set_aside_torn_line(files[-1]):
            logger.warning("[Audit] Last line of %s was cut off mid-write; kept in its .torn file.", files[-1].name)
        for f in files:
            lines = [ln for ln in f.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            if lines:
                self._records += len(lines)
                self._last_hash = json.loads(lines[-1])["hash"]
                self._last_file = f.stem
        problem = _anchor_problem(self._records, self._last_hash, self.anchor, key)
        if problem:
            logger.critical("[Audit] Log does not match its anchor: %s", problem)
            self._write({"ts": time.time(), "record": AUDIT_GAP, "problem": problem}, key)

    def _write_anchor(self, key: bytes) -> None:
        tmp = self.anchor.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"records": self._records, "head": self._last_hash,
                       "mac": _anchor_mac(self._records, self._last_hash, key)}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.anchor)

    def _write(self, record: dict[str, Any], key: bytes) -> dict[str, Any]:
        ts = float(record.get("ts") or time.time())
        # Hash what will be on disk: JSON turns some values (numeric dict keys, say) into others.
        entry = json.loads(json.dumps({**record, "ts": ts, "prev_hash": self._last_hash}, default=str))
        entry["hash"] = _record_hash(entry, key)
        # Files follow write order, not record time: verification reads them in name order,
        # so a part inspected before midnight but logged after it must not sort backwards.
        name = max(time.strftime("%Y-%m-%d", time.gmtime()), self._last_file)
        self._last_file = name
        path = self.dir / (name + ".jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._last_hash = entry["hash"]
        self._records += 1
        self._write_anchor(key)
        return entry

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            key = self._load_key()
            if self._last_hash is None:
                self._restore_tail(key)
            return self._write(record, key)


def verify_chain(directory: str | Path, key_file: str | Path | None = None) -> dict[str, Any]:
    """Check every record's hash and link, then the record count against the anchor."""
    key_path = _key_path(directory, key_file)
    anchor_file = anchor_path(directory)
    files = sorted(Path(directory).glob("*.jsonl"))
    if not key_path.exists():
        if files or anchor_file.exists():
            return {"ok": False, "records": 0, "error": f"audit key not found: {key_path}"}
        return {"ok": True, "records": 0, "head": GENESIS}
    try:
        key = bytes.fromhex(key_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return {"ok": False, "records": 0, "error": f"audit key is unreadable: {key_path}"}
    prev = GENESIS
    n = 0
    gaps: list[str] = []
    torn = False
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                if f == files[-1] and lineno == len(lines) and not text.endswith("\n"):
                    torn = True  # cut off mid-write: the next start sets it aside
                    continue
                return {"ok": False, "records": n, "error": f"{f.name}:{lineno} is not a valid record"}
            if rec.get("prev_hash") != prev:
                return {"ok": False, "records": n, "error": f"{f.name}:{lineno} broken link (record removed or edited before it)"}
            if not hmac.compare_digest(_record_hash(rec, key), str(rec.get("hash"))):
                return {"ok": False, "records": n, "error": f"{f.name}:{lineno} content does not match its hash"}
            prev = rec["hash"]
            n += 1
            if rec.get("record") == AUDIT_GAP:
                gaps.append(f"{f.name}:{lineno} {rec.get('problem')}")
    problem = _anchor_problem(n, prev, anchor_file, key)
    if problem:
        return {"ok": False, "records": n, "error": problem}
    if gaps:
        return {"ok": False, "records": n, "error": f"logging restarted after a gap: {'; '.join(gaps)}"}
    return {"ok": True, "records": n, "head": prev, **({"torn_tail": True} if torn else {})}


def main(argv=None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("directory", nargs="?", default="data/audit")
    p.add_argument("--key-file", default=None)
    args = parser.parse_args(argv)
    result = verify_chain(args.directory, args.key_file)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
