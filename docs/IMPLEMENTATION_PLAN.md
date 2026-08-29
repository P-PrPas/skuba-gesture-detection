# Implementation plan — gesture/posture recognition

Phased roadmap. Each phase has a goal, concrete tasks, a deliverable, and an exit criterion — don't move to the next phase until the exit criterion is met.

## Phase 0 — Environment & scaffolding

**Goal:** repo builds and runs the two candidate backbones end to end on a webcam/sample video.

- Set up repo structure (see CLAUDE.md)
- Install and smoke-test candidate pose backbones (e.g. YOLO-pose, MediaPipe Pose) and MediaPipe Hands
- Confirm target deployment hardware (Jetson model / onboard PC / RGB vs RGB-D camera) — this affects Phase 1 choices, get it confirmed before proceeding

**Deliverable:** a script that reads a video/webcam stream and overlays body + hand keypoints.

**Exit criterion:** keypoints visibly track a person in a live/test feed without crashing.

## Phase 1 — Backbone sanity check (not a full benchmark)

**Goal:** pick one body-pose backbone and confirm MediaPipe Hands is adequate, based on the project's actual hard cases — not a generic accuracy comparison.

- Collect a small set of hard-case clips: person laying down (from robot's camera height), person doing rock/fist-like hand shapes with fingers overlapping, person at typical robot-interaction distance (2-4m)
- Run each candidate body-pose backbone on the laying/squat clips — check visually whether keypoints stay stable and anatomically plausible
- Run MediaPipe Hands on the fist/rock clips at typical distance and on wrist-anchored crops (see ARCHITECTURE.md) — check landmark stability
- Benchmark inference latency of the chosen combination on target hardware

**Deliverable:** ARCHITECTURE.md's "Tech stack" section filled in with the chosen backbone(s) and the evidence for the choice.

**Exit criterion:** chosen backbone(s) handle all identified hard cases acceptably, and combined latency fits the robot's real-time budget. Once this is signed off, the backbone is not revisited except for a proven, reproduced failure.

## Phase 2 — Data collection pipeline

**Goal:** a repeatable way to go from "record a clip" to "labeled, normalized feature vectors" with minimal manual effort.

- Define recording protocol (see DATA_COLLECTION_SPEC.md): multiple subjects, angles, distances, lighting conditions per class, including `idle`
- Build the extraction script: clip -> per-frame keypoints (body + both hand crops) -> normalized feature vector -> labeled record
- Build the augmentation step: mirror+relabel, rotation jitter, keypoint dropout, coordinate noise (see ARCHITECTURE.md for parameters)
- Decide storage format (e.g. parquet/JSON per clip, indexed by subject and session for correct splitting)

**Deliverable:** running the pipeline on N recorded clips produces a versioned, augmented training dataset.

**Exit criterion:** dataset covers all initial classes with multiple subjects each, split by subject (not frame) into train/val/test.

## Phase 3 — Baseline classifier

**Goal:** first working classifier, correctness over performance.

- Train LightGBM/Random Forest baseline on normalized+fused features
- Evaluate per-class precision/recall and full confusion matrix — pay special attention to confusable pairs (e.g. sit vs squat, rock vs fist/idle, ok vs thumb)
- Calibrate confidence threshold per class using the validation set (not test set)

**Deliverable:** trained baseline model + evaluation report (confusion matrix, per-class metrics, chosen thresholds).

**Exit criterion:** baseline meets a minimum bar the team agrees on before comparing alternatives — don't tune this baseline extensively yet.

## Phase 4 — Classifier iteration

**Goal:** the actual experimentation phase — this is where most of the project's effort goes.

- Compare feature representations: raw normalized coordinates vs. derived features (joint angles, inter-point distances)
- Compare model families: LightGBM/RF vs. MLP vs. (if needed later) a small temporal model
- Tune augmentation strength against validation performance — too little augmentation underfits generalization, too much washes out real class boundaries
- Re-run full confusion matrix after every change; a fix for one class must not silently break another as the class count grows

**Deliverable:** final chosen classifier configuration with documented rationale for why it beat the alternatives.

**Exit criterion:** acceptable accuracy on held-out subjects (not seen during training) across all classes including `idle`, with no single class catastrophically confused with another.

## Phase 5 — Temporal smoothing & real-time integration

**Goal:** stable, real-time behavior on live camera input, not just single-frame offline accuracy.

- Implement majority-vote (or similar) smoothing over a short frame window
- Wire the full pipeline (backbone -> features -> classifier -> smoothing -> thresholding) into a real-time loop
- Benchmark end-to-end latency and frame rate on target hardware

**Deliverable:** a real-time demo that classifies gestures live from camera input with visible confidence/idle output.

**Exit criterion:** stable frame rate on target hardware, and smoothing measurably reduces flicker/false triggers compared to raw per-frame output.

## Phase 6 — Field testing & iteration

**Goal:** find and close the gap between lab conditions and the robot's real operating environment.

- Test in the actual environment/lighting/camera setup the robot will use
- Log failure cases (misclassifications, missed detections) with enough context to reproduce
- Retrain only the classification layer on new data derived from failure cases — do not touch the backbone unless a failure is clearly backbone-attributable (see CLAUDE.md constraint)

**Deliverable:** a log of failure cases and the resulting dataset/model updates.

**Exit criterion:** acceptable performance in the real deployment environment across repeated test runs.

## Phase 7 — Packaging for deployment

**Goal:** a versioned, deployable artifact — the final ask of this project.

- Export final classifier weights + the exact normalization/feature-fusion code it depends on (these must ship together — a classifier without its matching feature pipeline is useless)
- Document the exact backbone version/config used, since features depend on it
- Record class list, confidence thresholds, and known limitations in a short model card

**Deliverable:** a single versioned package (weights + feature pipeline code + model card) ready to load into the robot's perception stack.

**Exit criterion:** the packaged model runs correctly in the robot's actual runtime environment, not just the development environment.
