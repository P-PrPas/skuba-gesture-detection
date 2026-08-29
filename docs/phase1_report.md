# Phase 1 — backbone benchmark

Full report with embedded montages: **`results/phase1/backbone_report.docx`**.
Reproduce:

```bash
pip install -r requirements-phase1-gpu.txt --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple --trusted-host download.pytorch.org
python scripts/phase1_eval.py --device cpu --annotate
python scripts/phase1_eval.py --device gpu
python scripts/phase1_report.py            # -> results/phase1/backbone_report.docx
```

Host: Windows, RTX 3050 Laptop (4 GB), used as a proxy for the Acer/Ubuntu
deploy laptop. Hard-case clips: `laying_01`, `squat_01`, `sit_02` (posture);
`rock_01`, `ok_01`, `i_love_you_01`, `two_finger_01` (hands). Subject at ~2–4 m.

## Verdict: MediaPipe Pose + MediaPipe Hands

| Backbone | VRAM | CPU ms/f (squat) | GPU ms/f | Detect | Keypoint quality |
|---|---|---|---|---|---|
| **MediaPipe Pose** | **0 (CPU)** | **38** (26 fps) | 37 (no GPU delegate) | 98% | Stable; no flyaway keypoints; single-person → never grabs the bystander |
| YOLO11n-pose | 70 MB | 119 (8 fps) | **24** (42 fps) | 100% | Head keypoints fly to frame corners; skeleton jumps to the background bystander |
| YOLO11s-pose | 130 MB | 198 (5 fps) | 21 (47 fps) | 100% | Torso cleaner than 11n, same flyaway + bystander issues |
| RTMPose-t (rtmlib) | 359 MB | 167 (6 fps) | 37 (27 fps) | 100% | Usually clean; YOLOX-tiny detector picks the bystander sometimes |
| RTMPose-m (rtmlib) | 611 MB | 356 (3 fps) | 83 (12 fps) | 100% | **Best skeleton** — clean through the deep crouch; but slow + bystander-prone |

Combined MediaPipe (pose + 2 hand crops/frame): **116 ms ≈ 8.6 FPS** CPU
(unchanged on the GPU pass — MediaPipe does not use the GPU). Hands are ~2/3 of it.

Hands (MediaPipe Hands on wrist crops): `rock` 102/102, `i_love_you` 141/141,
`two_finger` 87/87, `ok` 35/45 (misses are hand entry/exit motion blur, not the
held gesture). ~40–57 ms/crop.

### Why MediaPipe, not the faster GPU option

YOLO11n on CUDA is the fastest at 24 ms / 42 FPS, but it costs 70 MB of the
shared 4 GB GPU, a torch+CUDA runtime (~5 GB on disk), the flyaway-keypoint and
bystander bugs, and a person-selection rewrite. The robot's GPU budget is the
binding constraint (CLAUDE.md), so the 0-VRAM CPU option wins. RTMPose-m's nicer
skeleton is not worth 600 MB + a second-stage detector.

## Open items before Phase 1 is fully signed off

- [ ] Record a clean `laying` clip from the robot's camera height and re-run —
      `laying_01` is mostly squat→floor transition and heavy occlusion.
- [ ] Confirm the combined ~8–9 FPS on the actual Acer/Ubuntu laptop and agree a
      real-time budget. If short: MediaPipe Tasks GPU delegate, smaller hand
      crops, or `static_image_mode=False` for the hand model.
