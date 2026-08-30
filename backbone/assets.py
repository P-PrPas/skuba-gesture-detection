"""MediaPipe Tasks model files (.task). Downloaded on first use to
backbone/models/ and read as bytes (the Tasks API resolves a plain path oddly
on Windows). These ship with the deployment package (Phase 7) — the classifier
is only valid for the exact backbone that produced its features.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# float16 "latest" — pin by re-hosting if reproducibility ever bites
_URLS = {
    "pose_landmarker_lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand_landmarker": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def asset_bytes(name: str) -> bytes:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.task"
    if not path.exists():
        if name not in _URLS:
            raise KeyError(f"unknown asset {name!r}")
        urllib.request.urlretrieve(_URLS[name], path)
    return path.read_bytes()
