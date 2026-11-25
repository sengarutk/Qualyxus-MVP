"""Back up and restore a station: databases, models, configuration, audit trail, accounts.

    python -m src.system.backup create --to /mnt/backup
    python -m src.system.backup verify /mnt/backup/qualyxus-20261125-120000
    python -m src.system.backup restore /mnt/backup/qualyxus-20261125-120000 --into /opt/qualyxus

Safe while the runtime runs. Keys and accounts are included only with a passphrase
(QUALYXUS_BACKUP_PASSPHRASE) and are encrypted with it; without them a restored audit trail
cannot be verified, so keep that passphrase. Restoring never overwrites: it needs an empty place.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
import time
from pathlib import Path
from typing import Any

FORMAT = "qualyxus-backup-1"
SECRETS = "secrets.enc"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(passphrase: str, salt: bytes) -> bytes:
    return base64.urlsafe_b64encode(hashlib.scrypt(passphrase.encode(), salt=salt, n=2 ** 15, r=8, p=1, dklen=32,
                                                   maxmem=2 ** 26))


def sources(cfg: dict[str, Any]) -> dict[str, list[Path]]:
    process, inspection = cfg.get("process") or {}, cfg.get("inspection") or {}
    audit = inspection.get("audit") or {}
    audit_dir = Path(audit.get("dir", "data/audit"))
    process_db = Path(process.get("db_path", "data/events/process.db"))
    events_db = Path((cfg.get("logging") or {}).get("sqlite_path", "data/events/events.db"))
    models = {Path(process.get("models_dir", "models/process"))}
    for section in [inspection, *(inspection.get("recipes") or {}).values()]:
        for role in ("anomaly_model", "segmentation_model", "defect_type_model"):
            if section.get(role):
                models.add(Path(section[role]))
                models.add(Path(section[role]).with_suffix(".card.json"))
                models.add(Path(f"{section[role]}.sig"))
    if inspection.get("registry_dir"):
        models.add(Path(inspection["registry_dir"]))
    users = Path((cfg.get("console") or {}).get("users_file", "secrets/users.json"))
    key = audit.get("key_file") or os.getenv("QUALYXUS_AUDIT_KEY_FILE") or audit_dir.with_name(audit_dir.name + ".key")
    return {"databases": [process_db, events_db, process_db.with_name("actions.db")],
            "files": [*sorted(models), Path(os.getenv("CONFIG_PATH") or "config.yaml"),
                      Path((inspection.get("review") or {}).get("dir", "data/review"))],
            "audit": [audit_dir, audit_dir.with_name(audit_dir.name + ".anchor.json"),
                      audit_dir.with_name(audit_dir.name + ".base.json")],
            "secrets": [users, Path(key), Path(os.getenv("QUALYXUS_MODEL_KEY_FILE", "secrets/model.key"))]}


def _inside(path: Path) -> str:
    """Where a file goes in the backup: its own path, or under ``absolute/`` when it is not relative."""
    return str(Path("absolute") / path.resolve().relative_to("/")) if path.is_absolute() else os.path.normpath(path)


def _copy(path: Path, root: Path) -> None:
    target = root / _inside(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".lock", "*.tmp"))
    else:
        shutil.copy2(path, target)


def create(cfg: dict[str, Any], destination: str | Path, *, passphrase: str | None = None) -> Path:
    from src.inference.audit_log import _only_writer, verify_chain

    out = Path(destination) / f"qualyxus-{time.strftime('%Y%m%d-%H%M%S')}"
    work = out.with_name(out.name + ".part")
    files = work / "files"
    files.mkdir(parents=True)
    found = sources(cfg)
    for db in found["databases"]:
        if db.exists():
            target = files / _inside(db)
            target.parent.mkdir(parents=True, exist_ok=True)
            # The online copy is consistent even while the runtime writes.
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as src, sqlite3.connect(target) as dst:
                src.backup(dst)
            dst.close()
            src.close()
    for path in found["files"]:
        if path.exists():
            _copy(path, files)
    audit_dir = found["audit"][0]
    audit_ok = None
    if audit_dir.is_dir():
        with _only_writer(audit_dir):       # log and anchor must be from the same moment
            for path in found["audit"]:
                if path.exists():
                    _copy(path, files)
        audit_ok = verify_chain(files / _inside(audit_dir), found["secrets"][1])["ok"]
    if passphrase:
        _encrypt([p for p in found["secrets"] if p.exists()], work / SECRETS, passphrase)
    listing = {str(p.relative_to(work)): _digest(p) for p in sorted(work.rglob("*")) if p.is_file()}
    (work / "manifest.json").write_text(json.dumps({"format": FORMAT, "created": time.time(), "files": listing,
                                                    "audit_intact": audit_ok, "secrets": bool(passphrase)}, indent=1))
    os.replace(work, out)
    return out


def _encrypt(paths: list[Path], target: Path, passphrase: str) -> None:
    from cryptography.fernet import Fernet

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=_inside(path))
    salt = os.urandom(16)
    target.write_bytes(salt + Fernet(_key(passphrase, salt)).encrypt(buffer.getvalue()))


def verify(backup: str | Path) -> list[str]:
    """What is wrong with a backup; empty when every file is there and unchanged."""
    backup = Path(backup)
    try:
        manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"manifest unreadable: {e}"]
    if manifest.get("format") != FORMAT:
        return ["not a backup of this kind"]
    wrong = []
    for name, digest in manifest["files"].items():
        path = backup / name
        if not path.is_file():
            wrong.append(f"missing: {name}")
        elif _digest(path) != digest:
            wrong.append(f"changed: {name}")
    for db in (backup / "files").rglob("*.db"):
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                wrong.append(f"damaged database: {db.name}")
        conn.close()
    if manifest.get("audit_intact") is False:
        wrong.append("the audit trail was already broken when this backup was made")
    return wrong


def restore(backup: str | Path, into: str | Path, *, passphrase: str | None = None) -> Path:
    backup, into = Path(backup), Path(into)
    wrong = [w for w in verify(backup) if not w.startswith("the audit trail was already")]
    if wrong:
        raise ValueError("the backup is not usable: " + "; ".join(wrong[:5]))
    if into.exists() and any(into.iterdir()):
        raise ValueError(f"{into} is not empty: restoring never overwrites")
    into.mkdir(parents=True, exist_ok=True)
    shutil.copytree(backup / "files", into, dirs_exist_ok=True)
    if (backup / SECRETS).exists():
        if not passphrase:
            print("The keys and accounts were left out: give QUALYXUS_BACKUP_PASSPHRASE to restore them.")
            return into
        from cryptography.fernet import Fernet, InvalidToken

        raw = (backup / SECRETS).read_bytes()
        try:
            data = Fernet(_key(passphrase, raw[:16])).decrypt(raw[16:])
        except InvalidToken as e:
            raise ValueError("wrong passphrase") from e
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for member in tar.getmembers():
                target = (into / member.name).resolve()
                if not member.isfile() or not target.is_relative_to(into.resolve()):
                    raise ValueError(f"unsafe entry in the backup: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                content = tar.extractfile(member)
                assert content is not None
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(content.read())
    return into


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create").add_argument("--to", required=True)
    sub.add_parser("verify").add_argument("backup")
    p = sub.add_parser("restore")
    p.add_argument("backup")
    p.add_argument("--into", required=True)
    args = parser.parse_args(argv)
    passphrase = os.getenv("QUALYXUS_BACKUP_PASSPHRASE")
    if args.cmd == "create":
        from src.config.config import Config

        out = create(Config.load().raw, args.to, passphrase=passphrase)
        print(out)
        if not passphrase:
            print("Keys and accounts are NOT in it: set QUALYXUS_BACKUP_PASSPHRASE to include them, encrypted.")
        return 1 if verify(out) else 0
    if args.cmd == "verify":
        wrong = verify(args.backup)
        print("\n".join(wrong) or "Backup is complete and unchanged.")
        return 1 if wrong else 0
    print("Restored into", restore(args.backup, args.into, passphrase=passphrase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
