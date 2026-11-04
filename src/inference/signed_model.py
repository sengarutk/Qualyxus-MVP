"""Joblib files can run code when loaded, so each one must carry a keyed signature.

A model is signed when it is saved on this box. To bring in a model from another box, check
where it came from, then sign it here:

    python -m src.inference.signed_model sign models/process/risk_x.joblib

The key is ``secrets/model.key`` (or ``QUALYXUS_MODEL_KEY_FILE``). Keep it off shared storage.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Any

import joblib


def _key() -> bytes:
    path = Path(os.getenv("QUALYXUS_MODEL_KEY_FILE", "secrets/model.key"))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(secrets.token_hex(32))
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


def signature_path(path: str | Path) -> Path:
    return Path(f"{path}.sig")


def _signature(path: str | Path) -> str:
    mac = hmac.new(_key(), digestmod=hashlib.sha256)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            mac.update(chunk)
    return mac.hexdigest()


def sign(path: str | Path) -> None:
    signature_path(path).write_text(_signature(path), encoding="utf-8")


def save(value: Any, path: str | Path) -> None:
    joblib.dump(value, path)
    sign(path)


def load(path: str | Path) -> Any:
    """Check the signature first: nothing in the file runs unless it matches."""
    sig = signature_path(path)
    if not sig.exists():
        raise ValueError(f"{path} has no signature; sign it with: python -m src.inference.signed_model sign {path}")
    if not hmac.compare_digest(sig.read_text(encoding="utf-8").strip(), _signature(path)):
        raise ValueError(f"{path} does not match its signature: file changed or signed with another key")
    return joblib.load(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sign").add_argument("path")
    args = parser.parse_args(argv)
    sign(args.path)
    print(f"Signed {args.path}")


if __name__ == "__main__":
    main()
