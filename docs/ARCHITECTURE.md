# Architecture — technical reference

Companion to CLAUDE.md. This is where the concrete schemas and formulas live so the classification-layer experiments (Phase 4) have a stable contract to work against.

## Tech stack (Phase 1 — backbone locked)

- **Body pose:** MediaPipe `pose_landmarker` (Tasks API, **lite** model, VIDEO mode), 33-landmark BlazePose topology. `backbone/pose.py`. ~25–38 ms/frame CPU, 0 VRAM. Call `PoseEstimator.new_sequence()` before each independent clip so VIDEO-mode tracking does not leak across clips.
- **Hand landmarks:** MediaPipe `hand_landmarker` (Tasks API, IMAGE mode), 21 landmarks/hand, on wrist-anchored crops. `backbone/hands.py`. ~40 ms/crop.
- **`mediapipe==1.0.1`.** `mp.solutions` (the legacy API) was removed in 1.0 — code uses `mediapipe.tasks.python.vision`. `.task` files auto-download to `backbone/models/` via `backbone/assets.py`.
- Not fully signed off: no clean `laying` clip yet; combined FPS unconfirmed on the Acer (see `docs/phase1_report.md` open items).
- **Evidence** — full report `results/phase1/backbone_report.docx` (numbers + montages), summary `docs/phase1_report.md`, reproduce with `scripts/phase1_eval.py` + `scripts/phase1_report.py`. Benchmarked MediaPipe Solutions + Tasks (lite/full/heavy) vs YOLO11n/s-pose vs RTMPose-t/m, CPU **and** GPU (RTX 3050 as proxy for the Acer laptop).
  - Latency squat, ms/frame (CPU / GPU): MP Tasks-lite **25 / Linux-only**. MP Solutions 38 / no GPU path. YOLO11n 119 / 24. YOLO11s 198 / 21. RTMPose-t 167 / 37. RTMPose-m 356 / 83.
  - VRAM on CUDA: MediaPipe 0 (CPU) — its GPU delegate is OpenGL-ES (not CUDA), Linux-only, tens of MB. YOLO11n 70 MB / YOLO11s 130 / RTMPose-t 359 / RTMPose-m 611, plus a torch or onnxruntime-gpu runtime.
  - Keypoint stability: MediaPipe (both APIs) cleanest on `squat`/`sit`, single-person so it never grabs the bystander; YOLO threw flyaway head keypoints and picked the bystander; RTMPose's detector also picked the bystander (RTMPose-m best skeleton otherwise).
  - The GPU verdict: no pose backbone gets a worthwhile speed win from the GPU — the fastest GPU option (YOLO11n, 24 ms) only ties MediaPipe Tasks-lite on CPU and reintroduces the keypoint bugs. Keep pose on CPU, GPU stays free for the CUDA modules.
  - Combined MediaPipe pose + 2 hand crops ≈ **116 ms/frame ≈ 8.6 FPS** (CPU, Solutions API); ~10 FPS with Tasks-lite. Hands are ~2/3 of it.
  - Hands on wrist crops: rock/ILY/two-finger 100% detection; `ok` 78% (misses only during entry/exit motion blur).
- **2D only** (RGB camera, no depth).
- **Open items** (Phase 1 not fully signed off): (1) no clean `laying` clip yet for the low-camera-angle case; (2) confirm combined FPS + real-time budget on the Acer, and run the MediaPipe GPU delegate there for a real number. (`backbone/pose.py` is now on the Tasks API and features are re-extracted — done.)
- Any change to backbone version/config invalidates every extracted feature file and the classifier — re-extract and retrain (see "Versioning note").

## Feature vector schema

Implemented in `features/schema.py` (`FEATURE_DIM = 152`). The classifier's input is a single fixed-length vector per frame:

| slice | content |
|---|---|
| `[0:66]` | 33 body landmarks, normalized (x, y), MediaPipe Pose order |
| `[66:108]` | 21 left-hand landmarks, normalized (x, y) |
| `[108]` | left-hand presence flag (1 detected / 0 not) |
| `[109:151]` | 21 right-hand landmarks, normalized (x, y) |
| `[151]` | right-hand presence flag |

Built by concatenating:

1. **Body features** — normalized (x, y) for each body keypoint, in MediaPipe's fixed landmark order.
2. **Left hand features** — normalized (x, y) for each of the 21 hand landmarks, plus a **presence flag** (1 if detected this frame, 0 if not).
3. **Right hand features** — same as left, mirrored.

If a hand's presence flag is 0, that hand's coordinate slots are zero-filled. The classifier is trained on data that includes both presence states, so it learns to treat a flagged-absent hand as "no information" rather than as a literal position at the origin.

## Normalization

Applied separately to body and to each hand, before fusion:

- **Center**: subtract a fixed reference point — hip-shoulder midpoint for body, wrist landmark for that hand's own points.
- **Scale**: divide by a fixed reference length — shoulder-to-hip distance for body, palm width (wrist-to-middle-finger-MCP distance) for hand — so the feature is invariant to distance from camera and to body/hand size.
- Do not scale body and hand independently in a way that discards relative size information if a future class needs it (e.g. "hand near face") — document any such class's requirements before finalizing the scale reference, since it constrains what the classifier can express.

## Hand cropping strategy

Do not run the hand landmark model on the full frame. Instead:

1. From body pose output, take the wrist keypoint for each side.
2. Crop a square region around that wrist, sized relative to a body-scale reference (e.g. a multiple of shoulder width) so the crop scales correctly with the person's distance from the camera.
3. Run the hand landmark model on each crop independently.
4. Map landmark coordinates back to a hand-local normalized frame (see Normalization above) — they do not need to be re-projected into the original frame's coordinates, since the classifier only ever sees normalized features.

## Augmentation recipe

Implemented in `features/augment.py` (`AugParams`). Applied to the **train split only**; originals are kept. Current starting values (tune in Phase 4, then update here and the dataset card):

| step | parameter | value |
|---|---|---|
| Mirror | `mirror_p` | 0.5 |
| Rotation jitter | `rot_deg` (max \|tilt\|) | 12° |
| Keypoint dropout | `kp_dropout_frac` | 0.05 |
| Whole-hand drop | `hand_drop_p` | 0.10 |
| Coordinate noise | `coord_noise_std` (normalized units) | 0.02 |
| Copies per original | `n_per_sample` | 4 |

- **Mirror**: flip x-coordinates of body and both hands; swap left/right body landmark pairs; swap the left/right hand feature slices **and** their presence flags; relabel any class with a left/right variant (`raise_right_hand` <-> `raise_left_hand`). Classes without a left/right variant keep their label. `features/augment.py::demo()` asserts double-mirror is identity and the relabel happens.
- **Rotation jitter**: rotate all keypoints by a random angle in ±`rot_deg` about the origin (body is already centered at origin post-normalization; each hand about its own wrist).
- **Keypoint dropout**: zero `kp_dropout_frac` of the (33 + 21 + 21) keypoints at random.
- **Whole-hand drop**: with prob `hand_drop_p`, zero a *present* hand's slice and clear its presence flag — real occlusion, teaches the presence-flag behavior.
- **Coordinate noise**: add N(0, `coord_noise_std`) to every coordinate.

## Classifier interface

Whatever model is chosen (LightGBM, RF, MLP), it must implement a consistent interface: `predict(feature_vector) -> (class_label, confidence)`. This lets Phase 4 swap model families without touching the rest of the pipeline.

## Temporal smoothing

Operates on the stream of per-frame `(class_label, confidence)` outputs, not on raw keypoints:

- Maintain a sliding window of the last N frames' predictions.
- Emit the majority class in the window, using its mean confidence.
- If the majority class's mean confidence is below its class-specific threshold, emit `idle`/`unknown` instead.

N and per-class thresholds are tuning parameters set during Phase 4/5 using validation data — document final values here once locked.

## Versioning note

The classifier's weights are only valid for the exact backbone version/config and the exact normalization code they were trained against. Any change to the backbone (even a version bump) or to the normalization formulas requires re-extracting features and retraining the classifier — treat backbone+normalization+classifier as one versioned unit when packaging for deployment.
