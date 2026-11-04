"""Raise-hand MLP head + (de)serialisation shared by train / evaluate / detector."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from features.keypoints import FEATURE_NAMES, RAW_DIM


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(256, 128), p: float = 0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]; d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def columns(mode: str) -> np.ndarray:
    """'raw' = normalized keypoints + conf only; 'all' = raw + derived distances/angles."""
    return np.arange(RAW_DIM if mode == "raw" else len(FEATURE_NAMES))


def save(path: Path, model: MLP, mean, std, mode: str, thr: float, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "mode": mode, "thr": thr,
                "d_in": len(mean)}, path)
    path.with_suffix(".json").write_text(json.dumps({"mode": mode, "thr": thr, **extra}, indent=2, default=str))


class Classifier:
    """Loaded model: .predict(F [N,FEATURE_DIM]) -> prob [N]."""

    def __init__(self, path: Path, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.cols = columns(ck["mode"]); self.thr = float(ck["thr"]); self.mode = ck["mode"]
        self.mean = torch.as_tensor(ck["mean"], device=device); self.std = torch.as_tensor(ck["std"], device=device)
        self.m = MLP(ck["d_in"]).to(device).eval(); self.m.load_state_dict(ck["state"]); self.device = device

    @torch.no_grad()
    def predict(self, F: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(F[:, self.cols], dtype=torch.float32, device=self.device)
        return torch.sigmoid(self.m((x - self.mean) / self.std)).cpu().numpy()
