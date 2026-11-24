"""The judged image of every view of the newest parts, so a verdict can be seen and not just read."""
from __future__ import annotations

import os
import re
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np


def part_folder(part_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", part_key)


class ViewStore:
    def __init__(self, root: str | Path = "data/live/parts", keep: int = 200):
        self.root = Path(root)
        self.keep = max(1, int(keep))
        self._lock = threading.Lock()
        self._known: set[str] = set()

    def save(self, part_key: str, camera_id: str, frame: np.ndarray, heat: np.ndarray | None = None) -> None:
        folder = self.root / part_folder(part_key)
        folder.mkdir(parents=True, exist_ok=True)
        name = part_folder(camera_id)
        self._write(folder / f"{name}.jpg", frame)
        if heat is not None:
            self._write(folder / f"{name}.heat.jpg", heat)
        with self._lock:
            if folder.name not in self._known:
                self._known.add(folder.name)
                self._prune()

    @staticmethod
    def _write(path: Path, image: np.ndarray) -> None:
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data.tobytes())
            os.replace(tmp, path)

    def _prune(self) -> None:
        folders = sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        for old in folders[:-self.keep]:
            shutil.rmtree(old, ignore_errors=True)
            self._known.discard(old.name)
