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
occlusion, blur, id churn) are what the robot will hit.

### YOLO-detect front-end — `config.PERSON_DETECTOR = "yolo"`

Built as the alternative: YOLO11n **detection** (`backbone/detect.py`, weights
auto-download, ~5 MB) + ByteTrack IDs finds the person boxes, then MediaPipe
Pose runs on each crop (`PoseEstimator.estimate_crop()`, IMAGE mode) for the
33 landmarks — so the feature vector and the trained classifier are unchanged.
`DETECT_EVERY` amortises the YOLO cost (run every K frames, hold boxes between).

Same `main3.MOV`, `--detector yolo`, `DETECT_EVERY=2`, imgsz 512, after the
crop fixes below (`results/phase5/demo_main3_yolo.mp4`):

| | MediaPipe multi-pose | **YOLO-detect front-end** |
|---|---|---|
| frames with 2 people | 29 % | **52 %** |
| latency p50 | 257 ms (3.9 FPS) | 232 ms (4.3 FPS) |
| track ids over the clip (2 real people) | 10 | **6** |

YOLO's person detector is trained on COCO's huge pose/occlusion variety, so it
catches the person on the floor that BlazePose's upright-biased detector
misses. The min-area filter + pose sanity gate drop the false "person" (the
Pepper mannequin in shot) — so `2 people` went 29 → 52 % *clean*, and the
track-id churn from 10 → 6. The 22 % crop margin is the cost: bigger crops,
p50 down to 4.3 FPS (was 6.3 at tight-crop, but that had the mannequin +
random-class hits). `DETECT_EVERY 3` / imgsz 448 buy the speed back.

**Crop-quality bugs found + fixed** (`scratchpad` diag, then `_process_yolo`):
- *Tight crop breaks MediaPipe Pose.* At 10 % box padding, 14 % of detections
  got no pose at all, and a lying person's crop came back as an *upright*
  skeleton (`torso_lean` ≈ 0 instead of ≈ ±2 rad) → classified `sit`. Padding
  22 % (following the box shape — a square crop is worse, it pulls in
  background) drops no-pose to 10 % and recovers `laying` (main3 frame 320:
  `[sit 0.91, laying 0.80]` — both people, both right).
- *Spurious far/small boxes → random class* (a 171 px box → `thumb 0.66`).
  `DETECT_MIN_AREA_FRAC` 0.015 → 0.025, plus a pose sanity gate (drop a crop
  whose skeleton has a collapsed torso or < 12 visible joints).

**Remaining misclassification is domain gap, not a bug.** main3's people are
doing real activities in odd postures — bent over a desk, sitting on the floor
with a suitcase — not performing the 12 vocab gestures. Full-frame and
YOLO-crop give the *identical* verdict for these (`sit 0.77`, `torso_lean
−0.2`), so it isn't the crop: the classifier has no "standing / working / doing
stuff" class and a forward-leaning person lands nearest `sit`. Phase 6 field
data (people in the robot's environment doing non-gesture things, labelled
`idle`) is the fix; a higher `--floor` routes the borderline calls to
`unknown` in the meantime.

**Recommendation:** if multi-person is required, `PERSON_DETECTOR = "yolo"` is
the path — it keeps the whole Phase 2–4 investment (33-landmark features,
locked classifier) and only swaps the "who is in frame" step. A full switch to
**YOLO-pose** (native multi-person, one pass) is *not* worth it: it needs the
GPU (contended on the Acer) and COCO-17 keypoints → a full feature + classifier
retrain. Confirm on real robot-camera footage in Phase 6.

## Test clips — `scripts/make_test_clips.py`

The recorded material has no single clip covering every action and none with
two people doing labelled actions. Derived (git-ignored under `data/`,
regenerate with the script):

- **`data/main_actions.mp4`** — `main.MOV` + `main3.MOV` concatenated (89 s):
  idle / raise_L / raise_R / sit / squat / laying + the hand gestures, one
  subject at a time. GT in `data/segments_main_actions.csv`. Single-person
  pipeline, N=7, floor 0.45, per segment:

  | idle | raise_L | two_finger | squat | sit | laying | glico | raise_R | t_pose | ok |
  |---|---|---|---|---|---|---|---|---|---|
  | 100 % | 100 % | 100 % | 100 % | 97 % | 83 % | 94 % | 73 % | 75 % | 71 % |

  (raise_R loses its arm-lowering tail to `idle`; `ok`/`t_pose` drift to sit/idle.
  This is the Phase 4 picture — the pipeline is fine when the person is *doing*
  the gesture.)

- **`data/2p_raise_lr.mp4` / `2p_sit_laying.mp4` / `2p_idle_squat.mp4`** —
  synthetic 2-person (two single-person clips side by side, 540×960 each). The
  multi-person pipeline (`--detector yolo`, `DETECT_EVERY=1`) gets both people
  right on the clean pairs:
  - `2p_raise_lr` → left `raise_left_hand`, right `raise_right_hand` ✓
  - `2p_idle_squat` → left `idle`, right `squat` ✓
  - `2p_sit_laying` → left `sit` ✓, right `laying` intermittent (the lying
    person, plus track churn on the sitter)

**Bug found + fixed:** `DETECT_EVERY = 2` swapped ByteTrack ids — and therefore
the per-person labels — between two similar people standing close together
(`2p_*` clips reproduced it). Default is now `1`.

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
