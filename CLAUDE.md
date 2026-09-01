# CLAUDE.md

This file gives any AI coding agent (Claude Code or similar) the context needed to work on this project correctly. Read this before making architectural changes.

## Project

Gesture and posture recognition system for a service robot. The robot's camera observes a person and must classify which predefined gesture/posture they are performing, from a growing vocabulary of classes, robustly across imperfect execution, camera angle, lighting, and partial occlusion.

## Target classes (expected to grow over time)

Body posture: `raise_right_hand`, `raise_left_hand`, `sit`, `squat`, `laying`
Hand gesture: `ok`, `two_finger` (peace), `thumb`
Pose: `t_pose`, `glico_pose` (Glico / running-man: arms up + one knee up), `mini_heart` (hands-together small heart)
Special: `idle` (no recognized gesture — must always be present in the label set)

**12 classes.** New classes will be added over the project's lifetime. The architecture must support this without retraining the backbone (see below). The authoritative list lives in `features/schema.py` (`CLASSES`) — update it there and keep this section in sync.

**Cut 2026-09-01** (Phase 3): `i_love_you` and `rock` — MediaPipe Hands cannot resolve them apart from `two_finger` on our conversational-distance footage (all three read as "loose fist"; reproduced backbone failure, see `docs/phase3_baseline.md`). `heart` (overhead two-arm) — no public dataset, only 55 noisy COCO candidates. Revisit if a better hand model is adopted (Phase 4) or with Phase 6 field recordings. The 606 ILY feature rows are kept at `data/features_ext/roboflow_ily__*.npz`.

`thumb` has no s01/s02 recording — trained + tested on held-out HaGRID subjects only.

## Core architecture decision — do not deviate without strong reason

```
camera -> person detect -> body pose estimator (pretrained)
                              |--> crop @ left wrist  -> hand landmark model (pretrained)
                              `--> crop @ right wrist -> hand landmark model (pretrained)
       -> normalize keypoints (body + both hands)
       -> concat into single feature vector (+ presence flags for missing hands)
       -> single classifier (LightGBM/RF primary; MLP alternative) -> class + confidence
       -> temporal smoothing (majority vote over N frames)
       -> final output (or `idle`/`unknown` if below confidence threshold)
```

**Why this shape, specifically:**

- The pose/hand estimators are pretrained on large diverse human datasets and already generalize to lighting, clothing, background, and most camera angles. Do not fine-tune them unless a specific, reproduced failure case proves they need it (e.g. `laying` posture from the robot's low camera angle). Fine-tuning the backbone on our small dataset first is very unlikely to beat the pretrained one.
- All experimentation effort goes into the classification layer, not the backbone. This is the layer we iterate on constantly: feature engineering, model choice, augmentation, thresholds.
- Both branches (body, hand) run every frame, always. There is no gating logic that decides "check hands now" vs "check body now" — that would just reintroduce hand-written rules/thresholds in a different place. The classifier sees a fused feature vector and decides everything itself, including whether nothing relevant is happening (`idle`).
- Hand crops are anchored on the wrist keypoint from the body pose output, not run on the full frame. This keeps hand detection usable at typical robot-to-person distances, where hands are a small fraction of the frame.
- If a hand is not detected (out of frame, occluded, angle), do not drop the frame or crash — fill that hand's feature slice with zeros and set its presence flag to 0. The classifier learns to ignore zeroed, flagged-absent slices.

## Hard constraints (do not violate these)

1. **No hand-written thresholds** anywhere in the classification path (no `if angle > X: return "sit"`). If you find yourself writing one, that logic belongs in the classifier's training data, not in code.
2. **Mirror augmentation must swap left/right labels.** Any script that mirrors a training sample and does NOT relabel `raise_right_hand` <-> `raise_left_hand` (and swap left/right hand feature slices) is a bug, not a feature.
3. **`idle` is a real class**, not a threshold hack applied after the fact. Training data must include real examples of people standing/moving normally without performing any target gesture.
4. **Never retrain the backbone as a first response to a failure.** First: check if it's a classifier-layer problem (bad features, insufficient data for that class, bad threshold). Only fine-tune the backbone after a reproduced, backbone-attributable failure.
5. **Splits are by subject/recording session, not by frame.** Frames from the same person/clip in both train and test leak information and silently inflate accuracy. *(Current reality: only 2 subjects exist and no more recording is possible — see `docs/ARCHITECTURE.md` "Evaluating with a small subject pool". `build_dataset.py` does a cross-person hold-out fold for the classes it can, plus limb-length augmentation; the train-only accuracy is never reported as a generalization number, and Phase 6 field testing is the real gate.)*

## Tech stack (backbone LOCKED at Phase 1)

- Body pose backbone: **MediaPipe Pose — Tasks API** (`pose_landmarker` lite model, 33 landmarks, VIDEO running mode). CPU, 0 VRAM. Wrapper: `backbone/pose.py`. Benchmark evidence: `results/phase1/backbone_report.docx`, `docs/phase1_report.md`.
- Hand landmark backbone: **MediaPipe Hands — Tasks API** (`hand_landmarker`, 21 landmarks/hand, IMAGE mode, on wrist-anchored crops). Wrapper: `backbone/hands.py`.
- `mediapipe==1.0.1` (pinned). The legacy `mp.solutions` API is gone in 1.0 — everything uses `mediapipe.tasks.python.vision`. `.task` model files auto-download to `backbone/models/` (git-ignored) via `backbone/assets.py`.
- Classifier: LightGBM or Random Forest (primary), small MLP (secondary/comparison) — *not yet chosen, Phase 3/4*
- Feature vector: 152-dim, defined in `features/schema.py`
- Language/runtime: Python 3.11, `.venv` at repo root (mediapipe 1.0.1 is `py3-none`, so 3.9–3.13 all work)
- Target deployment hardware: Acer notebook, Ubuntu, RGB camera only. Optimize for minimal VRAM + latency (VRAM is shared with other robot modules).

## How to add a new gesture class later

1. Record clips of the new gesture across multiple people, angles, distances.
2. Run the fixed backbone pipeline to extract + normalize keypoints (no backbone changes).
3. Augment (mirror+relabel if left/right applies, rotate, keypoint dropout, coordinate noise).
4. Add to the training set and retrain only the classification layer.
5. Re-run the full evaluation suite (see IMPLEMENTATION_PLAN.md Phase 4) — check for new confusion pairs against existing classes, not just the new class's own accuracy.

## Repo structure (proposed)

```
/data/           raw clips, extracted keypoints, labels (see DATA_COLLECTION_SPEC.md)
/backbone/       thin wrappers around pose/hand models -- no training code here
/features/       normalization, fusion, augmentation
/classifier/     training scripts, saved model weights, evaluation
/pipeline/       real-time inference pipeline (temporal smoothing, thresholding, robot integration)
/docs/           this file, IMPLEMENTATION_PLAN.md, ARCHITECTURE.md, DATA_COLLECTION_SPEC.md
```
