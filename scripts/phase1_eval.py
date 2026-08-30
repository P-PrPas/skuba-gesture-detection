"""Phase 1 backbone benchmark (IMPLEMENTATION_PLAN.md Phase 1).

Runs the candidate body-pose backbones (+ MediaPipe Hands) on the project's
hard-case clips, on CPU and on GPU, and records detailed numbers +
annotated output for review. Feeds scripts/phase1_report.py (the .docx).

    python scripts/phase1_eval.py --device cpu
    python scripts/phase1_eval.py --device gpu
    python scripts/phase1_eval.py --device gpu --annotate      # also write annotated mp4s

Writes:
    results/phase1/metrics_<device>.json
    results/phase1/montages/<model>__<clip>.jpg
    results/phase1/annotated/<model>__<clip>.mp4         (with --annotate)

Latency is measured on THIS machine. The deploy laptop will differ - the
*ranking* and the VRAM figures are the durable results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# onnxruntime-gpu needs the CUDA 12 / cuDNN 9 DLLs on the path; the torch cu124
# wheel bundles them. Register that dir before importing anything CUDA.
_torch_lib = ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.is_dir():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

GPU_BASELINE_MB = None  # set in main() before any model loads
CLIPS = ROOT / "data" / "clips"
OUT = ROOT / "results" / "phase1"

POSTURE = {
    "laying": CLIPS / "laying" / "laying_01.mp4",
    "squat": CLIPS / "squat" / "squat_01.mp4",
    "sit": CLIPS / "sit" / "sit_02.mp4",
}
HAND_CLIPS = {
    "rock": CLIPS / "rock" / "rock_01.mp4",
    "ok": CLIPS / "ok" / "ok_01.mp4",
    "i_love_you": CLIPS / "i_love_you" / "i_love_you_01.mp4",
    "two_finger": CLIPS / "two_finger" / "two_finger_01.mp4",
}
COMBINED_CLIP = POSTURE["squat"]

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


# ---------- helpers ----------
def gpu_mem_used_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def read_all(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def draw(frame, xy, edges, conf=None, thr=0.3):
    f = frame.copy()
    for i, j in edges:
        if conf is not None and (conf[i] < thr or conf[j] < thr):
            continue
        cv2.line(f, tuple(xy[i].astype(int)), tuple(xy[j].astype(int)), (0, 255, 0), 3)
    for k, (x, y) in enumerate(xy):
        if conf is not None and conf[k] < thr:
            continue
        cv2.circle(f, (int(x), int(y)), 4, (0, 165, 255), -1)
    return f


def montage(tiles, path: Path, cols=8):
    if not tiles:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tw = 260
    th = int(tw * tiles[0].shape[0] / tiles[0].shape[1])
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, t in enumerate(tiles):
        im = Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)).resize((tw, th))
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    sheet.save(path, quality=85)


def timed(fn, frames, warm=5):
    for f in frames[:warm]:
        fn(f)
    per = []
    for f in frames:
        t0 = time.perf_counter()
        fn(f)
        per.append((time.perf_counter() - t0) * 1000)
    a = np.array(per)
    return {"ms_mean": float(a.mean()), "ms_p50": float(np.percentile(a, 50)),
            "ms_p90": float(np.percentile(a, 90)), "fps": float(1000 / a.mean()), "n": len(a)}


# ---------- pose backends ----------
class MPPose:
    key = "mediapipe_pose"
    fmt = "mp33"

    def __init__(self, device):
        from backbone.pose import PoseEstimator
        self.est = PoseEstimator("mediapipe")
        self.edges = self.est.edges
        self.device_actual = "cpu"  # legacy Solutions API — CPU only, all platforms

    def infer(self, frame):
        kp = self.est.estimate(frame)
        return (None, None) if kp is None else (kp.xy, kp.visibility)


_MP_TASK_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_MP_POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (0, 2), (0, 5),
]


class MPTasksPose:
    """MediaPipe Tasks pose_landmarker. GPU delegate works on Linux (OpenGL ES,
    NOT CUDA) but not Windows. On the deploy laptop (Ubuntu) --device gpu will
    exercise it; on Windows it records the NotImplementedError and stays on CPU."""

    fmt = "mp33"

    def __init__(self, device, variant="lite"):
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.key = f"mediapipe_tasks_{variant}"
        self.edges = _MP_POSE_EDGES
        self._mp = mp
        model = ROOT / "results" / "phase1" / f"pose_landmarker_{variant}.task"
        model.parent.mkdir(parents=True, exist_ok=True)
        if not model.exists():
            urllib.request.urlretrieve(_MP_TASK_URLS[variant], model)
        buf = model.read_bytes()  # model_asset_path is resolved oddly on Windows; use buffer

        want_gpu = device == "gpu"
        self.delegate_note = ""
        deleg = BaseOptions.Delegate.GPU if want_gpu else BaseOptions.Delegate.CPU
        try:
            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=buf, delegate=deleg),
                running_mode=vision.RunningMode.VIDEO, num_poses=1)
            self.m = vision.PoseLandmarker.create_from_options(opts)
            self.device_actual = "gpu(gl)" if want_gpu else "cpu"
        except Exception as e:  # noqa: BLE001
            self.delegate_note = f"GPU delegate unavailable: {type(e).__name__}: {e}"
            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=buf, delegate=BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO, num_poses=1)
            self.m = vision.PoseLandmarker.create_from_options(opts)
            self.device_actual = "cpu (gpu n/a)"
        self._t = 0

    def infer(self, frame):
        h, w = frame.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._t += 33
        res = self.m.detect_for_video(img, self._t)
        if not res.pose_landmarks:
            return None, None
        lm = res.pose_landmarks[0]
        xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        cf = np.array([getattr(p, "visibility", 1.0) for p in lm], dtype=np.float32)
        return xy, cf


class YoloPose:
    def __init__(self, device, weights="yolo11n-pose.pt"):
        from ultralytics import YOLO
        import torch
        self.key = f"{Path(weights).stem}"
        self.edges = COCO_EDGES
        self.fmt = "coco17"
        self.dev = "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        self.device_actual = "cuda" if self.dev.startswith("cuda") else "cpu"
        self.m = YOLO(weights)
        self.m.to(self.dev)

    def infer(self, frame):
        r = self.m(frame, verbose=False, conf=0.3, device=self.dev)[0]
        if r.keypoints is None or r.keypoints.xy.shape[0] == 0:
            return None, None
        xy = r.keypoints.xy[0].cpu().numpy()
        cf = r.keypoints.conf
        cf = cf[0].cpu().numpy() if cf is not None else np.ones(len(xy))
        return (None, None) if xy.shape[0] == 0 else (xy, cf)


class RtmPose:
    def __init__(self, device, mode="lightweight"):
        from rtmlib import Body
        self.key = f"rtmpose_{mode}"
        self.edges = COCO_EDGES
        self.fmt = "coco17"
        want = "cuda" if device == "gpu" else "cpu"
        try:
            import onnxruntime as ort
            has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            has_cuda = False
        self.device_actual = "cuda" if (want == "cuda" and has_cuda) else "cpu"
        self.m = Body(mode=mode, backend="onnxruntime", device=self.device_actual)

    def infer(self, frame):
        kpts, scores = self.m(frame)
        if kpts is None or len(kpts) == 0:
            return None, None
        return np.asarray(kpts[0]), np.asarray(scores[0])


POSE_SPECS = [
    (MPPose, {}),
    (MPTasksPose, {"variant": "lite"}),
    (MPTasksPose, {"variant": "full"}),
    (MPTasksPose, {"variant": "heavy"}),
    (YoloPose, {"weights": "yolo11n-pose.pt"}),
    (YoloPose, {"weights": "yolo11s-pose.pt"}),
    (RtmPose, {"mode": "lightweight"}),
    (RtmPose, {"mode": "balanced"}),
]


def probe_vram(be) -> dict:
    """VRAM the loaded model occupies. torch models: exact via torch stats.
    onnxruntime models: nvidia-smi delta vs the baseline captured at import."""
    if be.device_actual != "cuda":
        return {"method": "n/a (CPU)", "model_mb": 0.0}
    try:
        import torch
        if torch.cuda.is_available() and hasattr(be, "m") and "ultralytics" in type(be.m).__module__:
            return {"method": "torch.max_memory_allocated",
                    "model_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
    except Exception:
        pass
    cur = gpu_mem_used_mb()
    return {"method": "nvidia-smi delta", "model_mb": None if cur is None else round(cur - GPU_BASELINE_MB, 1)}


# ---------- eval passes ----------
TIMING_FRAMES = 40  # cap per clip so RTMPose-m CPU doesn't take 15 min


def eval_posture(be, device, annotate):
    res = {}
    for cls, path in POSTURE.items():
        if not path.exists():
            continue
        allf = read_all(path)
        sub = max(1, len(allf) // TIMING_FRAMES)
        frames = allf[::sub][:TIMING_FRAMES]
        stats = timed(be.infer, frames)
        results = [be.infer(f) for f in frames]
        hit = sum(1 for xy, _ in results if xy is not None)
        mstep = max(1, len(frames) // 8)
        tiles = [draw(fr, xy, be.edges, cf) if xy is not None else fr
                 for fr, (xy, cf) in list(zip(frames, results))[::mstep][:8]]
        montage(tiles, OUT / "montages" / f"{be.key}__{cls}__{device}.jpg")
        if annotate and be.key in ("mediapipe_pose", "yolo11n-pose", "rtmpose_lightweight"):
            af = read_all(path)
            ar = [be.infer(f) for f in af]
            _write_video(af, ar, be.edges, OUT / "annotated" / f"{be.key}__{cls}.mp4")
        res[cls] = {**stats, "detect_rate": f"{hit}/{len(frames)}",
                    "detect_pct": round(100 * hit / len(frames), 1)}
    return res


def _write_video(frames, results, edges, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for fr, (xy, cf) in zip(frames, results):
        vw.write(draw(fr, xy, edges, cf) if xy is not None else fr)
    vw.release()


def eval_hands(device, annotate):
    from backbone.hands import HandLandmarker, wrist_crop_box
    from backbone.pose import PoseEstimator
    from smoke_test import HAND_EDGES
    pose = PoseEstimator("mediapipe")
    hands = HandLandmarker()
    res = {}
    for cls, path in HAND_CLIPS.items():
        if not path.exists():
            continue
        frames = read_all(path)
        hit, tot, times, tiles = 0, 0, [], []
        for fr in frames:
            kp = pose.estimate(fr)
            if kp is None:
                continue
            w = kp.left_wrist if kp.left_wrist[1] < kp.right_wrist[1] else kp.right_wrist
            x0, y0, x1, y1 = wrist_crop_box(w, kp.shoulder_width or 90, fr.shape)
            crop = fr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            tot += 1
            t0 = time.perf_counter()
            lm = hands.detect(crop)
            times.append((time.perf_counter() - t0) * 1000)
            if lm is not None:
                hit += 1
        res[cls] = {"detect_rate": f"{hit}/{tot}", "detect_pct": round(100 * hit / max(1, tot), 1),
                    "ms_mean": round(float(np.mean(times)), 1) if times else None}
    return res


def eval_combined(device):
    from backbone.hands import HandLandmarker, wrist_crop_box
    from backbone.pose import PoseEstimator
    pose = PoseEstimator("mediapipe")
    hands = HandLandmarker()
    frames = read_all(COMBINED_CLIP)

    def step(f):
        kp = pose.estimate(f)
        if kp is None:
            return
        for wr in (kp.left_wrist, kp.right_wrist):
            x0, y0, x1, y1 = wrist_crop_box(wr, kp.shoulder_width or 90, f.shape)
            c = f[y0:y1, x0:x1]
            if c.size:
                hands.detect(c)

    return {**timed(step, frames), "note": "MediaPipe pose + 2 hand crops per frame (CPU)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--only", default="", help="comma-substrings; run only matching backend keys")
    ap.add_argument("--merge", action="store_true", help="merge into existing metrics_<device>.json instead of overwriting")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    global GPU_BASELINE_MB
    GPU_BASELINE_MB = gpu_mem_used_mb()
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu_name = None

    report = {
        "device_requested": args.device,
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "gpu": gpu_name, "gpu_mem_total_mb": _gpu_total(),
                 "gpu_mem_baseline_mb": GPU_BASELINE_MB},
        "pose": {}, "hands": {}, "combined_mediapipe": {},
    }

    print(f"\n### device={args.device} ###  (GPU baseline {GPU_BASELINE_MB} MB)")
    for cls, kw in POSE_SPECS:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        t_init = time.perf_counter()
        try:
            be = cls(args.device, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {cls.__name__} {kw}: {e}")
            continue
        init_s = round(time.perf_counter() - t_init, 2)
        if only and not any(s in be.key for s in only):
            del be
            continue
        note = getattr(be, "delegate_note", "")
        print(f"  {be.key} (actual: {be.device_actual}, init {init_s}s){'  ' + note if note else ''}")
        clips = eval_posture(be, args.device, args.annotate)
        report["pose"][be.key] = {"device_actual": be.device_actual, "format": be.fmt,
                                  "init_s": init_s, "vram": probe_vram(be),
                                  "delegate_note": note, "clips": clips}
        del be

    if not only:
        report["hands"] = {"model": "mediapipe_hands", "device_actual": "cpu",
                           "clips": eval_hands(args.device, args.annotate)}
        report["combined_mediapipe"] = eval_combined(args.device)

    path = OUT / f"metrics_{args.device}.json"
    if args.merge and path.exists():
        prev = json.loads(path.read_text())
        prev.setdefault("pose", {}).update(report["pose"])
        for k in ("hands", "combined_mediapipe", "host", "device_requested"):
            if report.get(k):
                prev[k] = report[k]
        report = prev
    path.write_text(json.dumps(report, indent=2))
    print(f"\n-> {path}")
    _print_summary(report)


def _gpu_total():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _print_summary(r):
    print("\n  pose backbone         actual  squat ms/f   fps    detect%   VRAM MB")
    for k, v in r["pose"].items():
        s = v["clips"].get("squat", {})
        print(f"  {k:20s} {v['device_actual']:6s} {s.get('ms_mean', 0):8.1f}  {s.get('fps', 0):6.1f}   "
              f"{s.get('detect_pct', 0):5.1f}   {(v.get('vram') or {}).get('model_mb')}")
    c = r.get("combined_mediapipe")
    if c:
        print(f"\n  combined MediaPipe: {c['ms_mean']:.1f} ms/frame -> {c['fps']:.1f} FPS")


if __name__ == "__main__":
    main()
