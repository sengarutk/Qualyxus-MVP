"""Accounts for the console and the command line: who is signed in decides the role, not a dropdown.

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

from src.operations.rbac import enforce, parse_role

_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}
MIN_PASSWORD = 10
MAX_FAILURES = 5
LOCKOUT_S = 300.0
MAX_ADDRESS_FAILURES = 30
SESSION_S = 8 * 3600.0


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT).hex()


class UserStore:
    def __init__(self, path: str | Path = "secrets/users.json"):
        self.path = Path(path)
        # Failures are counted per account *and* where they come from: someone else's wrong
        # guesses must not lock a person out. Kept on disk, so a restart does not reset them.
        self._failures_path = self.path.with_name(self.path.stem + ".failures.json")
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

    def role_of(self, username: str) -> str | None:
        """The role now. A removed or demoted person loses it at once, whatever they signed in with."""
        entry = self._read().get(username)
        return entry["role"] if entry else None

    def _failures(self, now: float) -> dict[str, list[float]]:
        try:
            raw = json.loads(self._failures_path.read_text(encoding="utf-8"))
            return {k: [int(v[0]), float(v[1])] for k, v in raw.items() if now - float(v[1]) < LOCKOUT_S}
        except (OSError, ValueError, TypeError, IndexError):
            return {}

    def _save_failures(self, failures: dict[str, list[float]]) -> None:
        self._failures_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._failures_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(failures), encoding="utf-8")
        os.replace(tmp, self._failures_path)

    def locked_for(self, username: str, now: float | None = None, address: str = "local") -> float:
        """Seconds until this account can be tried again from this address."""
        now = time.time() if now is None else now
        failures = self._failures(now)
        wait = 0.0
        for key, limit in ((f"{username}@{address}", MAX_FAILURES), (f"@{address}", MAX_ADDRESS_FAILURES)):
            count, last = failures.get(key, (0, 0.0))
            if count >= limit:
                wait = max(wait, last + LOCKOUT_S - now)
        return wait

    def authenticate(self, username: str, password: str, now: float | None = None, address: str = "local") -> str | None:
        """The account's role, or None. Unknown users cost the same time as wrong passwords."""
        now = time.time() if now is None else now
        with self._lock:
            if self.locked_for(username, now, address) > 0:
                return None
            entry = self._read().get(username)
            salt = bytes.fromhex(entry["salt"]) if entry else b"\0" * 16
            candidate = _hash(password, salt)
            good = entry is not None and hmac.compare_digest(candidate, entry["hash"])
            failures = self._failures(now)
            if good:
                if failures.pop(f"{username}@{address}", None) is not None:
                    self._save_failures(failures)
                return entry["role"]
            for key in (f"{username}@{address}", f"@{address}"):
                failures[key] = [failures.get(key, (0, 0.0))[0] + 1, now]
            while len(failures) > 10000:
                failures.pop(next(iter(failures)))
            self._save_failures(failures)
            return None


def signed_in(username: str, permission, users_file: str | Path = "secrets/users.json") -> str:
    """For commands that change something: prove who it is, and that they may.

    The password comes from QUALYXUS_PASSWORD or is asked for. A box with no accounts yet
    accepts any name, and says so: that is for setting it up, not for production.
    """
    store = UserStore(users_file)
    if not store.has_users():
        print("WARNING: no accounts exist, so this name is not checked. Create accounts: "
              "python -m src.operations.auth add NAME --role QA", file=sys.stderr)
        return username
    password = os.environ.get("QUALYXUS_PASSWORD")
    if password is None:
        password = getpass.getpass(f"Password for {username}: ")
    role = store.authenticate(username, password)
    if role is None:
        raise PermissionError("wrong name or password (or too many attempts; wait five minutes)")
    enforce(role, permission)
    return username


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
