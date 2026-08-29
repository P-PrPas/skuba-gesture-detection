# Phase 1 — backbone sanity check

Reproduce: `python scripts/phase1_eval.py` (annotated montages land in `scratch/phase1/`).
Hard-case clips from `data/clips/`; subject is at ~2–4 m in the footage.
Latency measured on the **Windows dev machine, CPU** — used as a proxy for the
Ubuntu deploy laptop (per the owner's call). Numbers on the Acer will differ;
treat the *ranking* as the durable result, the absolute ms as indicative.

## Body pose — posture hard cases (`laying_01`, `squat_01`, `sit_02`)

| backbone | keypoint quality on the hard poses | latency (CPU) | person selection |
|---|---|---|---|
| **MediaPipe Pose** (complexity 1) | stable; skeleton plausible on all visible joints; occluded legs (robot in foreground) inferred sensibly, no wild jumps | **32–38 ms/frame** | single-person model — locked onto the subject in every clip |
| YOLO11n-pose | body OK when locked, but frequent **flyaway head/face keypoints** placed at frame corners with high confidence; latches onto the **background bystander** in several frames | 49–62 ms/frame | naive top-1 box — picks the bystander |
| RTMPose-t (rtmlib, ONNX, `lightweight`) | skeleton ≈ MediaPipe when locked; its YOLOX-tiny detector **also picked the bystander** (2/8 squat frames) | 153–213 ms/frame (≈5–6× MediaPipe) | two-stage — same bystander problem |
| RTMPose-m (rtmlib, `balanced`) | not assessed | **1200–4600 ms/frame** — unusable on CPU | — |

`laying` caveat: `laying_01` has **no clean lying-flat segment** (mostly the
squat→floor transition, heavily occluded). The "laying from the robot's low
camera angle" hard case is **not yet properly tested** — needs a real recording,
then re-run this.

## Hands — MediaPipe Hands on wrist-anchored crops

| clip | detection rate | notes |
|---|---|---|
| `rock_01` | 10/10 | folded fingers slightly jittery; index+pinky and overall handshape captured |
| `i_love_you_01` | 10/10 | very stable |
| `two_finger_01` | 8/8 | stable |
| `ok_01` | 7/10 | 3 misses are all hand entering/leaving frame (motion blur), **none on the held gesture** |

≈ 35–38 ms per hand crop. Brief misses are exactly what the presence-flag +
temporal-smoothing design absorbs.

## Combined latency

MediaPipe **pose + 2 hand crops per frame**, full `squat_01` (108 frames):
**≈ 104 ms/frame ≈ 9.7 FPS** on the dev-machine CPU. Hands dominate (~76 ms for
the two crops). No GPU delegate, no optimization.

## Decision

**Body pose: MediaPipe Pose. Hands: MediaPipe Hands.** Reasons, in order:

1. Deployment target is a laptop whose VRAM is shared across robot modules
   (CLAUDE.md). MediaPipe runs on CPU, 0 VRAM. YOLO-pose needs torch; RTMPose
   needs onnxruntime + a separate detector.
2. Best keypoint stability on the posture hard cases — no flyaway keypoints,
   no bystander confusion (built-in single-person model).
3. Fastest of the three on CPU.
4. Hand landmarks on the overlapping-finger shapes (rock, ILY) are stable.

## Open items before Phase 1 is fully signed off

- [ ] Record a clean `laying` clip from the robot's camera height; re-run the
      posture check for that pose specifically.
- [ ] Confirm latency on the actual Acer/Ubuntu laptop, and agree a real-time
      budget. ~10 FPS on CPU may be fine with temporal smoothing; if not, options
      are the MediaPipe Tasks GPU delegate, lower-res hand crops, or
      `static_image_mode=False` for the hand model.
