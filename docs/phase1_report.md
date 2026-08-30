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

Host: Windows, RTX 3050 Laptop (4 GB), proxy for the Acer/Ubuntu deploy laptop.
Hard-case clips: `laying_01`, `squat_01`, `sit_02` (posture); `rock_01`,
`ok_01`, `i_love_you_01`, `two_finger_01` (hands). Subject at ~2–4 m.

## Verdict: MediaPipe Pose via the Tasks API (lite model) + MediaPipe Hands

| Backbone | VRAM | CPU ms/f (squat) | GPU ms/f | Keypoint quality |
|---|---|---|---|---|
| **MediaPipe Tasks — lite** | **0** | **25** (40 fps) | Linux-only (see below) | clean; single-person; no flyaways |
| MediaPipe Tasks — full | 0 | 37 (27 fps) | Linux-only | slightly steadier when occluded |
| MediaPipe Tasks — heavy | 0 | 109 (9 fps) | Linux-only | best MP skeleton, too slow |
| MediaPipe Pose (Solutions API) | 0 | 38 (26 fps) | no GPU path | = Tasks-lite quality, 1.5× slower |
| YOLO11n-pose | 70 MB | 119 (8 fps) | **24** (42 fps) | flyaway head keypoints; picks the bystander |
| YOLO11s-pose | 130 MB | 198 (5 fps) | 21 (47 fps) | same issues, milder |
| RTMPose-t (rtmlib) | 359 MB | 167 (6 fps) | 37 (27 fps) | detector grabs the bystander sometimes |
| RTMPose-m (rtmlib) | 611 MB | 356 (3 fps) | 83 (12 fps) | best skeleton, slow, bystander-prone |

### The GPU question (you asked to see MediaPipe on GPU)

- **MediaPipe Tasks GPU delegate is Linux-only.** On Windows it raises
  `NotImplementedError: GPU Delegate is not yet supported for Windows` (recorded
  in `metrics_gpu.json`). It **can** run on the Acer (Ubuntu).
- It uses **OpenGL ES compute, not CUDA** — so it will not compete with the CUDA
  modules for compute scheduling. VRAM footprint is small (tens of MB, GL
  buffers), non-zero, and only shows in `nvidia-smi`, not `torch.cuda` stats.
- Expected speed on desktop-class HW: roughly on-par to ~2× the CPU path for
  lite/full (the per-frame image upload can eat the gain); helps most for heavy.
- **It doesn't matter much either way:** MediaPipe Tasks *lite on CPU* is already
  25 ms — the same as YOLO11n on CUDA (24 ms) — with 0 VRAM and none of YOLO's
  bugs. No pose backbone here gets a worthwhile win from the GPU.
- To close it with a real number: run
  `python scripts/phase1_eval.py --device gpu --only mediapipe_tasks --merge` on
  the Acer; `MPTasksPose` already requests the GPU delegate and logs whether it
  loaded.

Combined pipeline (MediaPipe pose + 2 hand crops/frame): **116 ms ≈ 8.6 FPS**
CPU with the Solutions API; ~10 FPS with Tasks-lite pose. Hands are the
bottleneck (~2/3) — Phase 5 speed work goes there.

Hands (MediaPipe Hands on wrist crops): `rock` 102/102, `i_love_you` 141/141,
`two_finger` 87/87, `ok` 35/45 (misses are entry/exit motion blur). ~40–57 ms/crop.

## Open items before Phase 1 is fully signed off

- [ ] Record a clean `laying` clip from the robot's camera height and re-run.
- [ ] On the Acer/Ubuntu: confirm combined FPS + real-time budget; run the
      MediaPipe GPU delegate for a real number.
- [ ] Port `backbone/pose.py` from the Solutions API to the Tasks API
      (pose_landmarker lite), then re-extract features — features are only valid
      for the exact backbone (ARCHITECTURE.md "Versioning note"). Landmark order
      is the same 33-point BlazePose topology; verify before retraining.
