# Waving-GD — raise-hand detector

Binary, per-person: **raise hand (1)** = either arm raised above the shoulder, still or waving; **0** otherwise
(including scratching head, adjusting glasses, drinking, phone ...). Target: ROS2 node on an RTX 3050, >= 5 FPS,
output per person: `class 0/1, conf, fps, track id`.

```
frame -> YOLO11-pose + ByteTrack  -> person box, track id, 17 keypoints (x, y, conf)
      -> features/keypoints.py    -> person-centred, scale-normalised keypoints + wrist/elbow/face distances & angles (73-d)
      -> classifier (MLP)         -> P(raise hand)
      -> features/temporal.py     -> per-track smoothing -> class / conf
```

Only change vs GPSR-GD: MediaPipe -> YOLO-pose (which also supplies the person box and the track id).

## Layout
| Path | What |
|---|---|
| `config.py` | paths, pose model (`yolo11s-pose`), thresholds |
| `features/keypoints.py` | the feature contract (normalise, mirror, featurise) - shared by training and runtime |
| `pipeline/extract_features.py` | YOLO-pose over K700/HMDB51/SCB/test clips -> `data/kpts/` (resumable) |
| `pipeline/build_dataset.py` | frame labels + group-wise train/val split -> `data/dataset/` |
| `classifier/train.py` | MLP training, threshold selection, per-source false-positive report |
| `classifier/evaluate.py` | held-out clips (`dataset/`) vs rule baseline, simulated at 5 FPS |
| `pipeline/detector.py` | `RaiseHandDetector` class (for the ROS2 node) + CLI demo |
| `scripts/fetch_datasets.py` | downloads the open datasets into `data/raw/` |
| `data/test_segments.csv` | **draft** frame labels for the two test clips (eyeballed, please verify) |
| `docs/datasets.md` | where the data came from, licences, label cleaning, known gaps |

## Run
```bash
cd Waving-GD
bash scripts/run_pipeline.sh          # GPU=1 by default; everything below, in order

# or step by step
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python -m pipeline.extract_features test
.venv/bin/python -m pipeline.extract_features videos     # --shard 0/2 and 1/2 to use two GPUs
.venv/bin/python -m pipeline.extract_features scb
.venv/bin/python -m pipeline.build_dataset
.venv/bin/python -m classifier.train --name mlp_all
.venv/bin/python -m classifier.train --name mlp_raw --features raw
.venv/bin/python -m classifier.evaluate --name mlp_all --name mlp_raw
.venv/bin/python -m pipeline.detector --model mlp_all --source dataset/IMG_8243.MOV --save outputs/demo.mp4
```

## Weights
* Trained classifier heads: `outputs/runs/<version>/model.pt` (+ `model.json` with threshold/features, `train.log`, `eval_test.txt`) are
  tracked with **Git LFS** (`git lfs install` before cloning/pulling). Recommended: `v3_all_rec95`
  (`v1_*`/`v2_*` are identical models kept for the record). Videos, probes and data are not in git.
* YOLO-pose weights are **not** in git: put `yolo11s-pose.pt` in `weights/` (`yolo pose` / ultralytics downloads it).
* Use: `RaiseHandDetector("v3_all_rec95")` from `pipeline/detector.py`; needs both files.

## Status
Trained on K700 + HMDB51 + SCB; evaluated on the two held-out clips (87 labelled frames, one person - a pipeline check, not
evidence of generalisation). `v3_all_rec95`: val recall 95 %, hard-negative FPR 4.6 %. Weakest on hand-near-face actions
(adjusting glasses, brushing hair, drinking). Next: record robot-camera clips incl. hard negatives, measure latency on the RTX 3050.
