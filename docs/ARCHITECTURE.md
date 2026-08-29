# Architecture — technical reference

Companion to CLAUDE.md. This is where the concrete schemas and formulas live so the classification-layer experiments (Phase 4) have a stable contract to work against.

## Feature vector schema

The classifier's input is a single fixed-length vector per frame, built by concatenating:

1. **Body features** — normalized (x, y) [or (x, y, z) if using a 3D-capable backbone] for each body keypoint, in a fixed, documented order.
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

## Augmentation recipe (starting parameters — tune in Phase 4)

- **Mirror**: flip x-coordinates of body and both hands; swap left/right hand feature slices; relabel any class with a left/right variant (e.g. `raise_right_hand` -> `raise_left_hand`). Classes without a left/right variant (e.g. `sit`) keep their label.
- **Rotation jitter**: rotate keypoints by a small random angle around the body center, to simulate camera tilt and imperfect posture.
- **Keypoint dropout**: randomly zero out a small fraction of keypoints (with their own presence-style handling if applicable) to simulate partial occlusion.
- **Coordinate noise**: add small Gaussian noise to coordinates, to simulate backbone jitter.

Document the exact ranges/probabilities used once tuned — they affect reproducibility of every downstream result.

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
