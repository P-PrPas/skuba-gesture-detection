# skuba-gesture-detection

Monorepo for SKUBA's **Gesture Detection (GD)** projects. Each folder is an
independent project for a different competition task — its own code, docs,
requirements and README. Nothing is shared between them; `cd` into a folder
and work there.

| Folder | Gesture Detection for… | Status | What it recognizes |
|---|---|---|---|
| [`GPSR-GD/`](GPSR-GD/) | **GPSR** (General Purpose Service Robot) | Archived — old project, kept as reference | 12 body-posture / hand-gesture / pose classes (`sit`, `squat`, `laying`, `raise_*_hand`, `ok`, `two_finger`, `thumb`, `t_pose`, `glico_pose`, `mini_heart`, `idle`) from MediaPipe Pose + Hands keypoints and a fused-feature classifier |
| [`Waving-GD/`](Waving-GD/) | **Waving detection** | Active — new work starts here | A single gesture: is the person waving or not |

## Where to work

- New development → `Waving-GD/`.
- `GPSR-GD/` is frozen. Its pipeline (backbone wrappers, feature schema,
  classifier, multi-person demo) and its docs are a useful reference or source
  of reusable pieces, but copy what you need into `Waving-GD/` rather than
  importing across folders.
- Each folder's imports are rooted at that folder (`from backbone...`,
  `import config`), so run Python commands from inside the folder, e.g.
  `cd GPSR-GD && python -m pipeline.demo`.

## Layout

```
skuba-gesture-detection/
├── README.md        this file
├── GPSR-GD/         Gesture Detection for GPSR (archived)
└── Waving-GD/       Gesture Detection for waving (active)
```
