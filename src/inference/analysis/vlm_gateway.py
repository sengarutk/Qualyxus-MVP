#!/usr/bin/env python3
import base64
import logging
import cv2
import numpy as np
import httpx
from src.config.config import Config

logger = logging.getLogger("VLMGateway")

# Stored in front of the text, so a model's guess is never read as a measured cause.
SUGGESTION = "AI suggestion, not verified: "


def analyze_defect_crop_sync(crop_matrix: np.ndarray, expert_context: str, config: Config) -> str:
    """Ask the local vision-language model about a defect crop; "" when there is no answer.

    An error is logged, never returned: the result is stored with the defect record.
    """
    if crop_matrix is None or crop_matrix.size == 0:
        return ""
    vlm_cfg = config.raw.get("vlm_reasoning", {}) or {}
    if not bool(vlm_cfg.get("enabled", False)):
        return ""
    base = str(vlm_cfg.get("local_api", "http://localhost:11434")).rstrip("/")
    ollama = str(vlm_cfg.get("backend", "ollama")).lower() == "ollama"
    model = vlm_cfg.get("model", "moondream2")
    try:
        ok, buffer = cv2.imencode(".jpg", crop_matrix)
        if not ok:
            raise ValueError("the crop could not be compressed")
        image = base64.b64encode(buffer).decode("utf-8")
        prompt = (f"System Persona: {expert_context}\nTask: Inspect this high-resolution manufacturing defect crop and "
                  "output a single sentence naming the most likely mechanical root cause or component failure mode.")
        if ollama:
            url, payload = f"{base}/api/generate", {"model": model, "prompt": prompt, "stream": False, "images": [image]}
        else:
            url, payload = f"{base}/v1/predictions", {"model": model, "prompt": prompt,
                                                      "image": f"data:image/jpeg;base64,{image}"}
        with httpx.Client(timeout=float(vlm_cfg.get("timeout_seconds", 10.0))) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            text = str(response.json().get("response" if ollama else "output") or "").strip()
    except Exception as e:
        logger.warning("[VLM] No analysis for this defect: %s", e)
        return ""
    return SUGGESTION + text if text else ""
