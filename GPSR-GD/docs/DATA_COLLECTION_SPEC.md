# Data collection spec

## Classes to record

All classes from CLAUDE.md's target list, **including `idle`**. `idle` recordings should include normal standing, walking, and casual arm movement that does not match any target gesture — this class is as important to collect well as the "real" gestures, since it defines the rejection boundary for all of them.

## Protocol per class

For each class, record clips varying:

- **Subjects**: as many different people as feasible — body/hand proportions vary and the classifier must not learn one person's specific geometry.
- **Camera angle**: at minimum, the range of angles the robot's camera will actually see in deployment (including any low-angle views relevant to `laying`/`squat`).
- **Distance**: typical human-robot interaction distances, not just close-up.
- **Execution variation**: ask subjects to perform the gesture slightly differently each take (tilted, off-center, non-perfect form) — do not only record "textbook perfect" examples, since that's exactly the case the system must go beyond.
- **Occlusion**: include some clips with partial occlusion (e.g. one hand out of frame, object partially blocking the body) so the presence-flag/dropout handling has real examples to learn from, not only synthetic augmentation.
- **Lighting/background**: vary across the conditions expected in deployment.

## Labeling

- Label at the clip level (one label per recorded clip), not per frame — this matches how the data will be recorded and avoids frame-level labeling overhead.
- Store subject ID and session ID alongside every extracted frame's features, so Phase 2's split can be done correctly by subject/session rather than by frame.

## Storage

- Raw clips: keep for re-extraction if the backbone or normalization changes (see ARCHITECTURE.md versioning note).
- Extracted features: one record per frame with `subject_id`, `session_id`, `class_label`, `feature_vector`, `presence_flags`.
- Keep raw and extracted data versioned together — a feature dataset is only valid for the backbone version that produced it.

## Split strategy

Split by `subject_id` (or `session_id` if a subject has multiple unrelated sessions), not by frame or by clip within the same session. Frames from the same recording session are highly correlated; leaking them across train/val/test will silently inflate reported accuracy and hide real generalization failures.
