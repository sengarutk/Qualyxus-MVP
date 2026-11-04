"""Bundle pretrained backbones next to the models so a plant box never needs the internet.

A saved anomaly model holds its own head or memory bank, but the frozen DINO or ResNet
backbone comes from Hugging Face or torchvision. Run this once on a connected machine and
ship ``models/`` with the deployment:

    python -m src.inference.anomaly.bundle --model models/anomaly/housing.pt --verify

``--verify`` reloads the model in a fresh process with an empty download cache and all
network access to the model hubs disabled, which is what a clean air-gapped box sees.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from src.inference.anomaly.backbones import CNN_BACKBONES, VIT_CHECKPOINTS, bundle_dir

logger = logging.getLogger("AnomalyBundle")


def bundle_backbone(name: str, out_dir: Path) -> Path:
    """Save one backbone's weights into ``out_dir``; returns where they went."""
    if name in VIT_CHECKPOINTS:
        from transformers import AutoModel

        target = out_dir / name
        AutoModel.from_pretrained(VIT_CHECKPOINTS[name][0]).save_pretrained(target)
        return target
    if name in CNN_BACKBONES:
        from src.inference.anomaly.backbones import CNNBackbone

        target = out_dir / f"{name}.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(CNNBackbone(name, device=torch.device("cpu")).model.state_dict(), target)
        return target
    raise ValueError(f"Unknown backbone '{name}'")


def backbones_of(model_path: str | Path) -> list[str]:
    """Backbone names a saved anomaly model needs (an ensemble needs each member's)."""
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    names: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            value = node.get("backbone")
            if isinstance(value, str):
                names.extend(value.split("+"))
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(payload.get("state") or payload)
    return list(dict.fromkeys(n for n in names if n in VIT_CHECKPOINTS or n in CNN_BACKBONES))


def verify_offline(model_path: str | Path, *, backbones: Path) -> dict:
    """Load the model and score one frame as a clean, disconnected machine would."""
    code = (
        "import json, sys, numpy as np\n"
        "from src.inference.anomaly.registry import load_detector\n"
        "det = load_detector(sys.argv[1])\n"
        "res = det.predict(np.zeros((480, 640, 3), np.uint8))\n"
        "print(json.dumps({'ok': True, 'kind': det.kind, 'score': float(res.normalized_score)}))\n"
    )
    with tempfile.TemporaryDirectory() as empty:
        env = {**os.environ, "HF_HOME": empty, "TORCH_HOME": empty, "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "QUALYXUS_BACKBONE_DIR": str(Path(backbones).resolve())}
        run = subprocess.run([sys.executable, "-c", code, str(model_path)], env=env,
                             capture_output=True, text=True, timeout=300)
    if run.returncode != 0:
        return {"ok": False, "error": run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "failed"}
    return json.loads(run.stdout.strip().splitlines()[-1])


def main(argv=None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Saved anomaly model whose backbones to bundle")
    parser.add_argument("--backbone", action="append", default=[], help="Backbone name (repeatable)")
    parser.add_argument("--out", default=None, help="Default: models/backbones")
    parser.add_argument("--verify", action="store_true", help="Reload --model as a clean offline box would")
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else bundle_dir()
    names = list(args.backbone) + (backbones_of(args.model) if args.model else [])
    if not names:
        parser.error("Nothing to bundle: pass --model or --backbone")
    for name in dict.fromkeys(names):
        logger.info("Bundled %s -> %s", name, bundle_backbone(name, out_dir))
    result: dict = {"bundled": list(dict.fromkeys(names)), "dir": str(out_dir)}
    if args.verify:
        if not args.model:
            parser.error("--verify needs --model")
        result["offline_check"] = verify_offline(args.model, backbones=out_dir)
        if not result["offline_check"].get("ok"):
            raise SystemExit(f"Offline check failed: {result['offline_check'].get('error')}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
