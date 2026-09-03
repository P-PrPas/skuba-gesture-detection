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

## Multi-person

`config.NUM_POSES` (repo-root `config.py`) is the knob — `1` = the original
single-person path, `> 1` = multi-person. `backbone/pose.py` reads it as the
default; `estimate_all()` returns every detected person (largest bbox first,
capped at `NUM_POSES`), `estimate()` still returns the primary one.
`extract_features.py` + the Phase 1 benchmark pin `num_poses=1` regardless.

`pipeline/multiperson.py` — `MultiPerson.process()` fans the frame out per
person and runs a light IoU tracker (`config.TRACK_IOU` / `TRACK_MAX_MISSED`) so
each person keeps their own `TemporalSmoother` window across frames. The demo
switches to per-person labels (anchored above each person, colour-keyed by
track id) automatically when `NUM_POSES > 1`, or with `--multi`.

**MediaPipe multi-pose is a config change, not a backbone swap** — but the
per-person landmarking + hand crops mean latency grows ~linearly with the
person count. `config.MAX_CLASSIFY` (N > 0) caps it to the N largest people.

### Test — `num_poses=2` on `data/main3.MOV` (`results/phase5/demo_main3_multi.mp4`)

Two people present most of the 13.5 s clip (one on the floor, one standing),
handheld camera, cluttered room. `MultiPerson` + N=7 + floor 0.45:

| metric | value |
|---|---|
| frames with **2 people** detected | **118 / 404 (29 %)** |
| frames with 1 | 260 (64 %) · with 0 | 26 (6 %) |
| latency p50 (2 people, hands every frame) | **257 ms → 3.9 FPS** (p90 364, p99 438) |
| distinct track ids over the clip | **10** (for 2 real people — heavy churn) |

**MediaPipe multi-pose is a real weak spot here.** It catches both people only
~a third of the time, prefers the upright person over the larger person lying
on the floor (BlazePose is trained on upright bodies — same reason `laying` was
a Phase 1 concern), drops out under motion blur, and the tracker keeps losing
people. Speed is also marginal — 3.9 FPS median with 2 people; `--hand-every 2`
+ `MAX_CLASSIFY=2` claws back to ~6–8 but the detection rate does not improve.

A fixed robot camera with people standing in front of it facing it would be an
easier case than this handheld take — but the failure modes (non-upright poses,
occlusion, blur, id churn) are what the robot will hit. If reliable
multi-person is a hard requirement, the options are: (a) a dedicated
person-detector front-end (small YOLO-detect ~30 ms CPU) feeding per-crop
MediaPipe pose — fixes the missed-person problem, keeps the 33-landmark
features, still linear landmarking cost; or (b) YOLO-pose (native multi-person,
one pass) — but that needs the GPU and a full feature+classifier retrain
(COCO-17). Decision for the team; the config toggle is in place to keep testing.

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
