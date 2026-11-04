"""Dashboard accounts: who is signed in decides the role, not a dropdown.

Passwords are stored as salted scrypt hashes in a local file, so logins work on an
air-gapped box. Create accounts from a shell on that box:

    python -m src.operations.auth add alice --role QA
    python -m src.operations.auth list
    python -m src.operations.auth remove alice
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path

from src.operations.rbac import parse_role

_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}
MIN_PASSWORD = 10
MAX_FAILURES = 5
LOCKOUT_S = 300.0
SESSION_S = 8 * 3600.0


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT).hex()


class UserStore:
    def __init__(self, path: str | Path = "secrets/users.json"):
        self.path = Path(path)
        self._failures: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _write(self, users: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)
        os.replace(tmp, self.path)

    def has_users(self) -> bool:
        return bool(self._read())

    def users(self) -> dict[str, str]:
        return {name: entry["role"] for name, entry in self._read().items()}

    def add(self, username: str, password: str, role: str) -> None:
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"Password must be at least {MIN_PASSWORD} characters")
        salt = secrets.token_bytes(16)
        users = self._read()
        users[username] = {"role": parse_role(role).value, "salt": salt.hex(), "hash": _hash(password, salt)}
        self._write(users)

    def remove(self, username: str) -> bool:
        users = self._read()
        removed = users.pop(username, None) is not None
        if removed:
            self._write(users)
        return removed

    def locked_for(self, username: str, now: float | None = None) -> float:
        """Seconds until this account can try again after repeated failures."""
        now = time.time() if now is None else now
        with self._lock:
            count, last = self._failures.get(username, (0, 0.0))
        return max(0.0, last + LOCKOUT_S - now) if count >= MAX_FAILURES else 0.0

    def authenticate(self, username: str, password: str, now: float | None = None) -> str | None:
        """The account's role, or None. Unknown users cost the same time as wrong passwords."""
        now = time.time() if now is None else now
        if self.locked_for(username, now) > 0:
            return None
        entry = self._read().get(username)
        salt = bytes.fromhex(entry["salt"]) if entry else b"\0" * 16
        candidate = _hash(password, salt)
        if entry is not None and hmac.compare_digest(candidate, entry["hash"]):
            with self._lock:
                self._failures.pop(username, None)
            return entry["role"]
        with self._lock:
            count, last = self._failures.get(username, (0, 0.0))
            count = 0 if now - last > LOCKOUT_S else count
            self._failures[username] = (count + 1, now)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", default="secrets/users.json")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("username")
    add.add_argument("--role", required=True, help="operator | maintenance | QA | manager")
    sub.add_parser("list")
    remove = sub.add_parser("remove")
    remove.add_argument("username")
    args = parser.parse_args(argv)

    store = UserStore(args.file)
    if args.cmd == "add":
        password = getpass.getpass("Password: ") if sys.stdin.isatty() else sys.stdin.readline().rstrip("\n")
        if sys.stdin.isatty() and password != getpass.getpass("Repeat password: "):
            print("Passwords do not match.", file=sys.stderr)
            return 1
        store.add(args.username, password, args.role)
        print(f"Saved {args.username} ({args.role}).")
    elif args.cmd == "list":
        for name, role in sorted(store.users().items()):
            print(f"{name}\t{role}")
    elif not store.remove(args.username):
        print(f"No such user: {args.username}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
