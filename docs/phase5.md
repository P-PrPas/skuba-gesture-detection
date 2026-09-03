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
python -m pipeline.demo --video data/main.MOV --record demo.mp4 \
       --labels data/segments.csv --floor 0.45       # + ground-truth overlay + score
python scripts/phase5_flicker.py                     # N sweep
```

GUI keys: `q` quit · `space` pause · `[` `]` window N∓1 · `h` cycle hand-every
(1/2/3) · `r` reset smoother · `s` screenshot.

`--floor T` — below-window-confidence `T` the smoother emits **`unknown`**
(orange in the GUI), distinct from a confident **`idle`** (gray). The lock is
0.20; pass a higher `--floor` to make uncertainty visible in a demo.
`--labels segments.csv` overlays the ground-truth class for a `--video` and
prints a frame-accuracy tally at the end (cut classes count as misses).

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

**End-to-end run — `data/main.MOV`** (`results/phase5/demo_main.mp4`, the full
76 s s01 take, 15 gestures back-to-back, GT overlay + `--floor 0.45`):

| GT segment | frames | accuracy | main misroute |
|---|---|---|---|
| idle | 60 | **1.00** | |
| raise_right_hand | 96 | 0.72 | 24 → idle (arm-lowering tail) |
| raise_left_hand | 45 | **1.00** | |
| ok | 45 | 0.71 | 7 → sit, 5 → idle |
| two_finger | 87 | **1.00** | |
| mini_heart | 45 | 0.73 | 11 → unknown (low conf) |
| t_pose | 36 | 0.78 | 5 → sit |
| glico_pose | 66 | 0.95 | (aug-only class) |
| squat | 108 | **1.00** | (aug-only class) |
| sit | 314 | 0.96 | 10 → unknown |
| **overall (in-vocab)** | **902** | **91.6 %** | |

Per-frame, N=7 smoothed, floor 0.45. Matches the Phase 4 offline numbers
(raise_*_hand ~0.76, mini_heart ~0.73, idle/sit/two_finger strong) — main.MOV
is the s01 take, already in the Phase 4 test set, so this confirms the live
pipeline reproduces the offline accuracy rather than adding new evidence.

The 4 cut classes in the take (`i_love_you`, `rock`, `heart` ×2) correctly land
on **`unknown`** — the classifier's max-prob for an out-of-vocabulary gesture
stays under the floor, so the smoother refuses to guess. `squat` / `glico_pose`
are aug-only-leak classes (Phase 4), so their high scores here are not a
generalisation claim.

Gotcha found + fixed: raw `.MOV` files carry a rotation flag that
`cv2.VideoCapture` ignores — the demo now sets `CAP_PROP_ORIENTATION_AUTO` (and
has a `--rotate` fallback). The pre-cut `data/clips/*.mp4` are already upright
(ffmpeg baked it in), so `extract_features.py` was never affected.

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
