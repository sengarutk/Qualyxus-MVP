#!/usr/bin/env python3
import base64
import logging
import cv2
import numpy as np
import httpx
from src.config.config import Config

logger = logging.getLogger("VLMGateway")

def analyze_defect_crop_sync(crop_matrix: np.ndarray, expert_context: str, config: Config) -> str:
    """
    Encodes an anomalous image crop matrix to base64 and queries the local air-gapped VLM 
    backend for real-time forensic root-cause analysis.
    """
    # Defensive Shape/Dimension Verification Guard
    if crop_matrix is None or crop_matrix.size == 0 or crop_matrix.shape[0] == 0 or crop_matrix.shape[1] == 0:
        return "Root cause analysis suspended: Bounding box crop matrix possesses null or empty dimensions."

    try:
        vlm_cfg = config.raw.get("vlm_reasoning", {}) or {}
        if not bool(vlm_cfg.get("enabled", True)):
            return ""
        local_api = vlm_cfg.get("local_api", "http://localhost:11434")
        backend = str(vlm_cfg.get("backend", "ollama")).lower()
        model = vlm_cfg.get("model", "moondream2")
        timeout_sec = float(vlm_cfg.get("timeout_seconds", 10.0))

        # Compress matrix natively into memory block prior to encoding
        ret, buffer = cv2.imencode('.jpg', crop_matrix)
        if not ret:
            return "Root cause analysis aborted: Image compression allocation failure."
        b64_str = base64.b64encode(buffer).decode('utf-8')

        prompt = f"System Persona: {expert_context}\nTask: Inspect this high-resolution manufacturing defect crop and output a single, definitive sentence diagnosing the specific mechanical root cause or component failure mode."

        if backend == "ollama":
            url = f"{local_api.rstrip('/')}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "images": [b64_str]
            }
        else:
            url = f"{local_api.rstrip('/')}/v1/predictions"
            payload = {
                "model": model,
                "prompt": prompt,
                "image": f"data:image/jpeg;base64,{b64_str}"
            }

        with httpx.Client(timeout=timeout_sec) as client:
            response = client.post(url, json=payload)
            if response.status_code == 200:
                res_json = response.json()
                if backend == "ollama":
                    return str(res_json.get("response", "Diagnostics returned inconclusive metadata.")).strip()
                return str(res_json.get("output", "Diagnostics returned inconclusive metadata.")).strip()
            else:
                return f"Root cause classification error: Local platform server returned status code {response.status_code}."

    except httpx.TimeoutException:
        logger.warning("Local air-gapped VLM endpoint timed out during inference execution.")
        return "Root cause classification timeout; diagnostics pending offline manual line audit."
    except Exception as e:
        logger.error(f"VLM Gateway handler crashed during internal execution loop: {e}")
        return f"Root cause classification suspended due to internal pipeline runtime error: {str(e)}"
