"""MLP + a squeeze-and-excitation gate on the input features (Phase 4 Track C
follow-up — the user asked to try a simple SE layer since the plain MLP is the
locked pick).

Feature-wise SE (Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018,
adapted for a flat tabular vector): a small bottleneck branch reads the whole
standardised input and emits one multiplier in (0,1) per feature, so the trunk
sees a vector with the currently-irrelevant slices damped — e.g. the 84 hand
coordinates pulled toward zero on a `sit` / `laying` frame, the body-angle
block pulled down on a static hand gesture. It is `input -> Linear(d, d/r) ->
ReLU -> Linear(d/r, d) -> sigmoid -> elementwise *`, then the unchanged
`TorchMLP` trunk. The last SE layer starts at ~identity (gate ~= 0.88) so the
trunk trains first and the gate only learns to suppress.

Subclasses `TorchMLP` — same fit / predict_proba / classes_ / temperature
surface, so `classifier.evaluate` scores it exactly like every other candidate.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from classifier.mlp import TorchMLP


class _SEMLP(nn.Module):
    """Module-level (picklable) SE wrapper: (floor + (1-floor)*gate(x)) * x -> trunk.

    `floor` > 0 caps how far any feature can be suppressed — added because a
    hard gate (floor 0) zeroes the evidence the `idle` class needs to rule out
    gestures and collapses its recall (Phase 4 5-seed check)."""

    def __init__(self, in_dim: int, se_hidden: int, trunk: nn.Module, floor: float = 0.0):
        super().__init__()
        self.floor = float(floor)
        self.se_fc1 = nn.Linear(in_dim, se_hidden)
        self.se_fc2 = nn.Linear(se_hidden, in_dim)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.constant_(self.se_fc2.bias, 2.0)   # sigmoid(2) ~= 0.88 -> near identity
        self.act = nn.ReLU()
        self.trunk = trunk

    def gate(self, x):
        g = self.se_fc2(self.act(self.se_fc1(x))).sigmoid()
        return self.floor + (1.0 - self.floor) * g

    def forward(self, x):
        return self.trunk(x * self.gate(x))


class TorchSEMLP(TorchMLP):
    def __init__(self, *args, se_ratio: int = 6, se_floor: float = 0.0, **kw):
        super().__init__(*args, **kw)
        self.se_ratio = se_ratio
        self.se_floor = se_floor

    def _build(self, in_dim):
        trunk = super()._build(in_dim)          # identical Linear/BN/ReLU/Dropout stack
        return _SEMLP(in_dim, max(8, in_dim // self.se_ratio), trunk, self.se_floor)

    def feature_gate(self, X):
        """Mean SE multiplier per input feature over X — for inspecting which
        feature slices the gate learned to keep vs suppress."""
        import torch

        Xs = (np.asarray(X, np.float32) - self._mu) / self._sd
        self._net.eval()
        with torch.no_grad():
            return self._net.gate(torch.tensor(Xs)).mean(0).numpy()


def demo():
    rng = np.random.default_rng(0)
    n, d = 3000, 24
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    # only features 0,1,2 carry the label; 3..23 are pure noise
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int) + (X[:, 0] > 1).astype(int)
    m = TorchSEMLP(n_classes=3, hidden=(64, 32), max_epochs=120, patience=15,
                   se_ratio=4, seed=0).fit(X[:2400], y[:2400], X[2400:], y[2400:])
    p = m.predict_proba(X[2400:])
    assert p.shape == (600, 3) and np.allclose(p.sum(1), 1, atol=1e-4)
    acc = (p.argmax(1) == y[2400:]).mean()
    base = np.bincount(y[2400:]).max() / 600
    assert acc > base + 0.1, (acc, base)
    g = m.feature_gate(X[2400:])
    # the gate should keep the 3 real features open more than the 21 noise ones
    assert g[:3].mean() > g[3:].mean() + 0.02, (g[:3].mean(), g[3:].mean())
    assert 0.5 <= m.temperature <= 5.0
    print(f"semlp.demo OK  acc {acc:.2f}  gate real {g[:3].mean():.2f} "
          f"noise {g[3:].mean():.2f}  T {m.temperature:.2f}")


if __name__ == "__main__":
    demo()
