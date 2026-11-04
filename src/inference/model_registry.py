"""Versioned vision-model registry: per-recipe models, rollback, shadow mode, retraining.

Layout (``models/registry`` by default)::

    <root>/<recipe>/<role>/manifest.json       active / shadow version + full history
    <root>/<recipe>/<role>/v0003/...           immutable copy of one model version
    <root>/<recipe>/<role>/shadow_log.jsonl    active-vs-shadow verdicts on live parts
    <root>/_requests/<id>.json                 retraining requests (human approval gate)

Roles: ``anomaly`` (a .pt / .onnx file + sidecars), ``segmentation`` (a model directory
with model_card.json), ``typing`` (a .joblib file).

Every change of the active model is a recorded action (who, when, why) and can be rolled
back. A new model is normally registered as the *shadow*: it scores every live part next
to the active model without affecting verdicts, and ``shadow-report`` compares the two
(agreement, extra rejects, and accuracy on human-reviewed parts) before ``promote``.

CLI::

    python -m src.inference.model_registry register --recipe housing_a --role anomaly --path models/anomaly/x.pt --shadow
    python -m src.inference.model_registry list
    python -m src.inference.model_registry shadow-report --recipe housing_a --role anomaly
    python -m src.inference.model_registry promote --recipe housing_a --role anomaly --by alice --reason "500 parts, 0 extra rejects"
    python -m src.inference.model_registry rollback --recipe housing_a --role anomaly --by alice --reason "false rejects"
    python -m src.inference.model_registry requests
    python -m src.inference.model_registry approve <request-id> --by alice
    python -m src.inference.model_registry retrain <request-id> --candidate patchcore_dinov2_s --confidence 0.9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("QualyxusModelRegistry")

ROLES = ("anomaly", "segmentation", "typing")
_SIDECAR_SUFFIXES = (".card.json", ".json")


def file_digest(path: str | Path, length: int = 12) -> str:
    """Short content hash of a model file or directory (identifies exactly what is deployed)."""
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        h.update(f.name.encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:length]


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


class VisionModelRegistry:
    def __init__(self, root: str | Path = "models/registry"):
        self.root = Path(root)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ manifest

    def _dir(self, recipe: str, role: str) -> Path:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if not recipe or "/" in recipe or recipe.startswith((".", "_")):
            raise ValueError(f"invalid recipe name {recipe!r}")
        return self.root / recipe / role

    def manifest(self, recipe: str, role: str) -> dict[str, Any]:
        path = self._dir(recipe, role) / "manifest.json"
        if not path.exists():
            return {"recipe": recipe, "role": role, "active": None, "shadow": None, "versions": {}, "history": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, m: dict[str, Any]) -> None:
        _atomic_write_json(self._dir(m["recipe"], m["role"]) / "manifest.json", m)

    def _log(self, m: dict[str, Any], action: str, version: str | None, by: str, reason: str) -> None:
        m["history"].append({"ts": time.time(), "action": action, "version": version, "by": by, "reason": reason})

    # ------------------------------------------------------------ versions

    def register(self, recipe: str, role: str, path: str | Path, *, metrics: dict[str, Any] | None = None,
                 notes: str = "", activate: bool = False, shadow: bool = False, by: str = "system",
                 source: str = "manual") -> str:
        """Copy a trained model into the registry as a new immutable version."""
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(src)
        with self._lock:
            m = self.manifest(recipe, role)
            version = f"v{len(m['versions']) + 1:04d}"
            target = self._dir(recipe, role) / version
            target.mkdir(parents=True, exist_ok=False)
            if src.is_dir():
                shutil.copytree(src, target / src.name)
                entry = src.name
            else:
                shutil.copy2(src, target / src.name)
                entry = src.name
                for suffix in _SIDECAR_SUFFIXES:   # .card.json / ONNX .json sidecars travel with the model
                    side = src.with_suffix(suffix)
                    if side.exists() and side != src:
                        shutil.copy2(side, target / side.name)
            card = {}
            for name in ("model_card.json",):
                p = target / entry / name
                if p.exists():
                    card = json.loads(p.read_text(encoding="utf-8"))
            card_path = (target / entry).with_suffix(".card.json")
            if not card and card_path.exists():
                card = json.loads(card_path.read_text(encoding="utf-8"))
            m["versions"][version] = {
                "entry": entry, "created_ts": time.time(), "digest": file_digest(target / entry),
                "metrics": metrics or {}, "notes": notes, "source": source, "registered_by": by,
                "card": {k: card.get(k) for k in ("kind", "arch", "backbone", "backbone_license", "license",
                                                    "threshold", "validation") if k in card},
            }
            self._log(m, "register", version, by, notes or source)
            if activate or m["active"] is None and not shadow:
                m["active"] = version
                self._log(m, "activate", version, by, "first version" if not activate else (notes or "activated"))
            elif shadow:
                m["shadow"] = version
                self._log(m, "shadow", version, by, notes or "registered as shadow")
            self._save(m)
            logger.info("[Registry] %s/%s %s registered (active=%s shadow=%s)", recipe, role, version,
                        m["active"], m["shadow"])
            return version

    def activate(self, recipe: str, role: str, version: str, *, by: str, reason: str) -> None:
        with self._lock:
            m = self.manifest(recipe, role)
            if version not in m["versions"]:
                raise KeyError(f"{recipe}/{role} has no version {version}")
            m["active"] = version
            if m.get("shadow") == version:
                m["shadow"] = None
            self._log(m, "activate", version, by, reason)
            self._save(m)

    def promote(self, recipe: str, role: str, *, by: str, reason: str) -> str:
        m = self.manifest(recipe, role)
        if not m.get("shadow"):
            raise ValueError(f"{recipe}/{role} has no shadow model to promote")
        version = m["shadow"]
        self.activate(recipe, role, version, by=by, reason=f"promoted from shadow: {reason}")
        return version

    def rollback(self, recipe: str, role: str, *, by: str, reason: str) -> str:
        """Re-activate the version that was active before the current one."""
        with self._lock:
            m = self.manifest(recipe, role)
            actives = [h["version"] for h in m["history"] if h["action"] in ("activate", "rollback")]
            current = m["active"]
            previous = next((v for v in reversed(actives) if v != current), None)
            if previous is None:
                raise ValueError(f"{recipe}/{role}: no earlier active version to roll back to")
            m["active"] = previous
            self._log(m, "rollback", previous, by, reason)
            self._save(m)
            return previous

    def set_shadow(self, recipe: str, role: str, version: str | None, *, by: str, reason: str) -> None:
        with self._lock:
            m = self.manifest(recipe, role)
            if version is not None and version not in m["versions"]:
                raise KeyError(f"{recipe}/{role} has no version {version}")
            m["shadow"] = version
            self._log(m, "shadow" if version else "clear_shadow", version, by, reason)
            self._save(m)

    def path_of(self, recipe: str, role: str, version: str | None) -> Path | None:
        if not version:
            return None
        m = self.manifest(recipe, role)
        info = m["versions"].get(version)
        return None if info is None else self._dir(recipe, role) / version / info["entry"]

    def resolve(self, recipe: str, role: str) -> dict[str, Any]:
        m = self.manifest(recipe, role)
        return {
            "active_version": m.get("active"), "active": self.path_of(recipe, role, m.get("active")),
            "shadow_version": m.get("shadow"), "shadow": self.path_of(recipe, role, m.get("shadow")),
        }

    def recipes(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("_"))

    def overview(self) -> list[dict[str, Any]]:
        rows = []
        for recipe in self.recipes():
            for role in ROLES:
                m = self.manifest(recipe, role)
                if m["versions"]:
                    rows.append({"recipe": recipe, "role": role, "active": m["active"], "shadow": m["shadow"],
                                 "versions": len(m["versions"]),
                                 "last_change": m["history"][-1] if m["history"] else None})
        return rows

    # ------------------------------------------------------------ shadow log

    def shadow_log_path(self, recipe: str, role: str) -> Path:
        return self._dir(recipe, role) / "shadow_log.jsonl"

    def log_shadow(self, recipe: str, role: str, record: dict[str, Any]) -> None:
        path = self.shadow_log_path(recipe, role)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def shadow_report(self, recipe: str, role: str = "anomaly", *, review_dir: str | Path | None = "data/review",
                      version: str | None = None) -> dict[str, Any]:
        """Active vs shadow on live parts, plus accuracy on human-reviewed parts."""
        m = self.manifest(recipe, role)
        version = version or m.get("shadow")
        path = self.shadow_log_path(recipe, role)
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if version is None or r.get("shadow_version") == version:
                    rows.append(r)
        n = len(rows)
        report: dict[str, Any] = {"recipe": recipe, "role": role, "shadow_version": version,
                                  "active_version": m.get("active"), "parts": n}
        if not n:
            return report
        a_fail = sum(r["active_verdict"] == "fail" for r in rows)
        s_fail = sum(r["shadow_verdict"] == "fail" for r in rows)
        report.update({
            "agreement": round(sum(r["active_verdict"] == r["shadow_verdict"] for r in rows) / n, 4),
            "active_reject_rate": round(a_fail / n, 4),
            "shadow_reject_rate": round(s_fail / n, 4),
            "shadow_only_rejects": sum(r["active_verdict"] == "pass" and r["shadow_verdict"] == "fail" for r in rows),
            "active_only_rejects": sum(r["active_verdict"] == "fail" and r["shadow_verdict"] == "pass" for r in rows),
        })
        labels = _review_labels(review_dir) if review_dir else {}
        judged = [(r, labels[r["image_sha"]]) for r in rows if r.get("image_sha") in labels]
        if judged:
            def acc(key):
                return round(sum(r[key] == truth for r, truth in judged) / len(judged), 4)

            def escapes(key):
                return sum(truth == "fail" and r[key] == "pass" for r, truth in judged)

            report["reviewed"] = {
                "parts": len(judged), "active_accuracy": acc("active_verdict"), "shadow_accuracy": acc("shadow_verdict"),
                "active_escapes": escapes("active_verdict"), "shadow_escapes": escapes("shadow_verdict"),
            }
        return report

    # ------------------------------------------------------- retrain requests

    def _requests_dir(self) -> Path:
        return self.root / "_requests"

    def request_retrain(self, recipe: str, *, camera_id: str | None, reason: str, role: str = "anomaly",
                        since_ts: float | None = None, dedupe_s: float = 6 * 3600) -> dict[str, Any] | None:
        """File a retraining request (needs human approval). Duplicate open requests are suppressed."""
        with self._lock:
            now = time.time()
            for req in self.requests():
                if (req["recipe"] == recipe and req["role"] == role and req["status"] in ("pending", "approved")
                        and now - req["created_ts"] < dedupe_s):
                    return None
            req = {"id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}", "recipe": recipe, "role": role,
                   "camera_id": camera_id, "reason": reason, "created_ts": now, "since_ts": since_ts or now,
                   "status": "pending", "decided_by": None, "decided_ts": None, "result": None}
            _atomic_write_json(self._requests_dir() / f"{req['id']}.json", req)
            logger.warning("[Registry] Retraining requested for %s/%s: %s (approve with: python -m "
                           "src.inference.model_registry approve %s --by <name>)", recipe, role, reason, req["id"])
            return req

    def requests(self, status: str | None = None) -> list[dict[str, Any]]:
        d = self._requests_dir()
        if not d.exists():
            return []
        out = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]
        return [r for r in out if status is None or r["status"] == status]

    def decide(self, request_id: str, *, approve: bool, by: str, note: str = "") -> dict[str, Any]:
        path = self._requests_dir() / f"{request_id}.json"
        req = json.loads(path.read_text(encoding="utf-8"))
        if req["status"] != "pending":
            raise ValueError(f"request {request_id} is already {req['status']}")
        req.update({"status": "approved" if approve else "rejected", "decided_by": by,
                    "decided_ts": time.time(), "note": note})
        _atomic_write_json(path, req)
        return req

    def complete_request(self, request_id: str, result: dict[str, Any]) -> None:
        path = self._requests_dir() / f"{request_id}.json"
        req = json.loads(path.read_text(encoding="utf-8"))
        req.update({"status": "done", "result": result})
        _atomic_write_json(path, req)


def _review_labels(review_dir: str | Path) -> dict[str, str]:
    """image_sha -> human verdict for reviewed queue items."""
    out = {}
    d = Path(review_dir)
    if not d.exists():
        return out
    for p in d.rglob("*.json"):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        verdict = (meta.get("review") or {}).get("verdict")
        if verdict in ("pass", "fail") and meta.get("image_sha"):
            out[meta["image_sha"]] = verdict
    return out


# ----------------------------------------------------------------------- CLI

def _retrain(reg: VisionModelRegistry, request_id: str, args) -> dict[str, Any]:
    """Train a candidate on the recipe's good-part pool and register it as the shadow model."""
    from src.inference.anomaly.registry import create_candidate
    from src.inference.good_pool import GoodPartPool

    req = next((r for r in reg.requests() if r["id"] == request_id), None)
    if req is None:
        raise KeyError(request_id)
    if req["status"] != "approved":
        raise ValueError(f"request {request_id} is {req['status']}; it must be approved first")
    pool = GoodPartPool(args.pool_dir, req["recipe"])
    images = pool.load(since_ts=req["since_ts"] if args.since_request else None, limit=args.limit)
    if len(images) < args.min_images:
        raise ValueError(f"only {len(images)} pooled good images for {req['recipe']} (need {args.min_images}); "
                         "let the line run longer after the change, or lower --min-images")
    tiles = tuple(int(v) for v in args.tiles.lower().split("x")) if args.tiles else None
    detector, spec = create_candidate(args.candidate, register=args.register, tiles=tiles)
    extra = {"iterations": args.iterations} if spec["method"] in ("dinomaly", "ensemble") else {}
    calibration = detector.fit(images, confidence=args.confidence, **extra)
    out = Path(args.work_dir) / f"{req['recipe']}_{request_id}.pt"
    detector.save(out)
    version = reg.register(req["recipe"], "anomaly", out, metrics={"calibration": calibration}, shadow=True,
                           by=req["decided_by"] or "system", source=f"retrain:{request_id}",
                           notes=f"retrained on {len(images)} pooled good parts ({args.candidate})")
    result = {"version": version, "images": len(images), "candidate": args.candidate}
    reg.complete_request(request_id, result)
    return result


def main(argv=None) -> Any:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="models/registry")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("register")
    p.add_argument("--recipe", required=True)
    p.add_argument("--role", choices=ROLES, required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--by", default=os.environ.get("USER", "operator"))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--activate", action="store_true")
    g.add_argument("--shadow", action="store_true")

    sub.add_parser("list")
    for name in ("promote", "rollback"):
        p = sub.add_parser(name)
        p.add_argument("--recipe", required=True)
        p.add_argument("--role", choices=ROLES, default="anomaly")
        p.add_argument("--by", required=True)
        p.add_argument("--reason", required=True)
    p = sub.add_parser("activate")
    p.add_argument("--recipe", required=True)
    p.add_argument("--role", choices=ROLES, default="anomaly")
    p.add_argument("--version", required=True)
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("shadow-report")
    p.add_argument("--recipe", required=True)
    p.add_argument("--role", choices=ROLES, default="anomaly")
    p.add_argument("--review-dir", default="data/review")
    sub.add_parser("requests")
    for name in ("approve", "reject"):
        p = sub.add_parser(name)
        p.add_argument("request_id")
        p.add_argument("--by", required=True)
        p.add_argument("--note", default="")
    p = sub.add_parser("retrain")
    p.add_argument("request_id")
    p.add_argument("--candidate", default="patchcore_dinov2_s")
    p.add_argument("--pool-dir", default="data/good_pool")
    p.add_argument("--since-request", action="store_true", help="Only use parts pooled after the request's change point")
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--min-images", type=int, default=150)
    p.add_argument("--confidence", type=float, default=0.9)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--register", action="store_true")
    p.add_argument("--tiles", default=None)
    p.add_argument("--work-dir", default="models/anomaly/retrained")
    args = parser.parse_args(argv)

    reg = VisionModelRegistry(args.root)
    if args.cmd == "register":
        out = reg.register(args.recipe, args.role, args.path, notes=args.notes, activate=args.activate,
                           shadow=args.shadow, by=args.by)
    elif args.cmd == "list":
        out = reg.overview()
    elif args.cmd == "promote":
        out = reg.promote(args.recipe, args.role, by=args.by, reason=args.reason)
    elif args.cmd == "rollback":
        out = reg.rollback(args.recipe, args.role, by=args.by, reason=args.reason)
    elif args.cmd == "activate":
        reg.activate(args.recipe, args.role, args.version, by=args.by, reason=args.reason)
        out = args.version
    elif args.cmd == "shadow-report":
        out = reg.shadow_report(args.recipe, args.role, review_dir=args.review_dir)
    elif args.cmd == "requests":
        out = reg.requests()
    elif args.cmd in ("approve", "reject"):
        out = reg.decide(args.request_id, approve=args.cmd == "approve", by=args.by, note=args.note)
    else:
        out = _retrain(reg, args.request_id, args)
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
