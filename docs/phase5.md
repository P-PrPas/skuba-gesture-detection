# Phase 5 — temporal smoothing & real-time integration

Per-frame classifier (Phase 4) → live camera loop + a sliding-window vote + an
OpenCV GUI. No backbone or classifier change.

## What was built

| file | role |
|---|---|
| `pipeline/runtime.py` | `frame_to_vec()` (shared with `extract_features.py` so live and training features can't drift), `GestureClassifier` (locked bundle + the 0.20 floor), `TemporalSmoother` (majority vote + idle fallback). `demo()` self-check, no camera. |
| `pipeline/demo.py` | the real-time GUI — webcam / `--video` / `--benchmark` / `--record`. |
| `scripts/phase5_flicker.py` | picks N on the s01/s02 clips. |

## Run it

```bash
python -m pipeline.demo                              # webcam
python -m pipeline.demo --video data/clips/ok/ok_01.mp4
python -m pipeline.demo --benchmark --video <clip>   # latency / FPS, headless
python -m pipeline.demo --video <clip> --record out.mp4   # annotated mp4, headless
python scripts/phase5_flicker.py                     # N sweep
```

GUI keys: `q` quit · `space` pause · `[` `]` window N∓1 · `h` cycle hand-every
(1/2/3) · `r` reset smoother · `s` screenshot.

## Results

**Smoothing (`scripts/phase5_flicker.py`, 13 s01/s02 clips, 1040 frames):**

| N | switches / 100f | clip accuracy | lag (frames) |
|---|---|---|---|
| 1 (raw) | 6.95 | 0.873 | 1.4 |
| 5 | 2.79 | 0.889 | 2.2 |
| **7** | **2.36** | **0.896** | **2.3** |
| 11 | 2.26 | 0.902 | 2.6 |
| 15 | 1.73 | 0.903 | 2.6 |

**N = 7 locked** — 66 % fewer label switches than raw, accuracy *up* 2 points,
~0.1 s lag. Meets the exit criterion (smoothing measurably reduces flicker
without hurting accuracy). Floor stays 0.20 global (Phase 4 Track E).

**Latency (`--benchmark`, this machine — RTX 3050 laptop, CPU pipeline, proxy
for the Acer):**

| hand-every | frame p50 | pose | hands | classify | FPS (p50) |
|---|---|---|---|---|---|
| 1 | 72 ms | 20 | 51 | 1.6 | **~14** |
| 2 | 52 ms | 20 | 19* | 1.6 | ~19 |
| 3 | 22 ms | 20 | 0* | 1.6 | ~30 avg |

*amortised — hands run every K-th frame, last landmarks reused between.
Hands are ~70 % of a full frame (matches Phase 1). The 0.3 MB MLP is 1.6 ms —
the classifier is free; the backbone is the whole budget.

## Known items / carry-over

- **Acer benchmark still pending** (Phase 1 open item too). Expect a bit slower
  than 14 FPS on the deploy laptop; `--hand-every 2` is the lever if needed.
  Also run the MediaPipe GPU (GL) delegate there — Linux-only, can't measure on
  Windows.
- **`data/features/*.npz` (s01/s02) drift** ~0.006 median / 0.04 p95 (normalised
  units) from the current MediaPipe 1.0.1 output — a handful of marginal
  right-hand detections flip. Not from Phase 5 (the un-refactored extractor
  shows the same delta); the committed set predates a mediapipe patch or a
  confidence-threshold tweak. Harmless for the demo; **re-extract before Phase 6**
  and re-confirm the Phase 4 test numbers on a clean set.
- Robot perception-stack integration is Phase 5/7 — this is the standalone
  demo; packaging (weights + `frame_to_vec` + model card as one unit) is Phase 7.
- The s01/s02 clips hold each gesture for most of the clip, so they under-test
  transition flicker. Phase 6 field recordings with natural gesture on/offsets
  are the real smoothing test.
