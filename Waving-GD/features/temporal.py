"""Per-track temporal smoothing shared by evaluation and the realtime detector."""
from __future__ import annotations
from collections import defaultdict, deque


class TrackSmoother:
    """Mean of the last `window` probabilities per track id. Window is in frames of the *processing* rate,
    so at ~5 FPS use 3 (~0.6 s); at 30 FPS use ~9."""

    def __init__(self, window: int = 3):
        self.window = max(1, window)
        self.buf = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, tid: int, prob: float) -> float:
        b = self.buf[tid]; b.append(prob)
        return sum(b) / len(b)

    def drop(self, alive: set) -> None:
        for t in [t for t in self.buf if t not in alive]:
            del self.buf[t]
