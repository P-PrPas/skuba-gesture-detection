"""Build the Phase 1 backbone test report (.docx) from the eval JSON.

    python scripts/phase1_eval.py --device cpu --annotate
    python scripts/phase1_eval.py --device gpu --annotate
    python scripts/phase1_report.py            # -> results/phase1/backbone_report.docx

Reads results/phase1/metrics_{cpu,gpu}.json and the montages, writes one .docx
with the verdict up top, full latency/VRAM/accuracy tables, and the annotated
montages embedded so the reader can judge keypoint quality directly.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase1"

POSE_LABEL = {
    "mediapipe_pose": "MediaPipe Pose",
    "yolo11n-pose": "YOLO11n-pose",
    "yolo11s-pose": "YOLO11s-pose",
    "rtmpose_lightweight": "RTMPose-t (rtmlib)",
    "rtmpose_balanced": "RTMPose-m (rtmlib)",
}

# qualitative keypoint-quality notes from the montage review (section 5)
QUALITY = {
    "mediapipe_pose": "Stable on squat/sit. No flyaway keypoints; single-person model so it never jumps to the bystander. Occluded legs are inferred, occasionally implausible on the worst floor frames.",
    "yolo11n-pose": "Head/face keypoints repeatedly shoot to the frame corners at high confidence; skeleton jumps to the background bystander in several frames.",
    "yolo11s-pose": "Torso cleaner than 11n but the corner-flyaway head keypoints and bystander pickup remain.",
    "rtmpose_lightweight": "Skeleton usually clean; the bundled YOLOX-tiny detector picks the bystander in a minority of frames.",
    "rtmpose_balanced": "Best skeleton of the lot — clean through the whole deep crouch, no flyaways. Costs: 356 ms/frame on CPU, and the YOLOX-m detector can still grab a bystander.",
}
RECOMMEND = {"mediapipe_pose": "YES - default"}


def scorecard_rows(cpu, gpu):
    rows = []
    for key, v in cpu["pose"].items():
        g = (gpu or {}).get("pose", {}).get(key, {})
        gv = (g.get("vram") or {}).get("model_mb")
        cpu_ms = v["clips"].get("squat", {}).get("ms_mean", 0)
        gd = g.get("clips", {}).get("squat", {})
        gpu_ms = gd.get("ms_mean")
        gpu_ran = g.get("device_actual", "")
        det = [c.get("detect_pct", 0) for c in v["clips"].values()]
        rows.append([
            POSE_LABEL.get(key, key),
            "0 (CPU)" if not gv else f"{gv:.0f}",
            f"{cpu_ms:.0f}",
            "-" if gpu_ms is None else (f"{gpu_ms:.0f}" + ("" if gpu_ran == "cuda" else " (CPU)")),
            f"{sum(det) / len(det):.0f}%" if det else "-",
            QUALITY.get(key, ""),
            RECOMMEND.get(key, "no"),
        ])
    return rows


def _load(dev):
    p = OUT / f"metrics_{dev}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cell(table, r, c, text, bold=False):
    cell = table.cell(r, c)
    cell.text = ""
    run = cell.paragraphs[0].add_run(str(text))
    run.bold = bold
    run.font.size = Pt(9)


def _table(doc, headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    for c, h in enumerate(headers):
        _cell(t, 0, c, h, bold=True)
    for r, row in enumerate(rows, 1):
        for c, val in enumerate(row):
            _cell(t, r, c, val)
    return t


def latency_rows(data):
    rows = []
    for key, v in data["pose"].items():
        sq = v["clips"].get("squat", {})
        la = v["clips"].get("laying", {})
        si = v["clips"].get("sit", {})
        vram = (v.get("vram") or {}).get("model_mb")
        rows.append([
            POSE_LABEL.get(key, key), v["device_actual"],
            f"{sq.get('ms_mean', 0):.1f}", f"{sq.get('ms_p90', 0):.1f}",
            f"{sq.get('fps', 0):.1f}",
            f"{la.get('ms_mean', 0):.1f}", f"{si.get('ms_mean', 0):.1f}",
            "0" if vram == 0 else ("-" if vram is None else f"{vram:.0f}"),
        ])
    return rows


def detect_rows(data):
    clips = ["laying", "squat", "sit"]
    rows = []
    for key, v in data["pose"].items():
        rows.append([POSE_LABEL.get(key, key)] +
                    [v["clips"].get(c, {}).get("detect_rate", "-") for c in clips])
    return rows


def main():
    cpu, gpu = _load("cpu"), _load("gpu")
    if cpu is None:
        raise SystemExit("run scripts/phase1_eval.py --device cpu first")

    doc = Document()
    doc.add_heading("SKUBA gesture detection — Phase 1 backbone test", 0)
    doc.add_paragraph(f"Generated {date.today().isoformat()}").runs[0].italic = True

    h = cpu["host"]
    verdict = doc.add_paragraph()
    r = verdict.add_run("Verdict: MediaPipe Pose + MediaPipe Hands.")
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0x1a, 0x7f, 0x37)
    doc.add_paragraph(
        "MediaPipe wins on the constraint that actually matters for this project — "
        "GPU memory. It runs entirely on CPU (0 MB VRAM), while the shared GPU on the "
        "deploy laptop has to be left free for the other robot modules. It also gave "
        "the most stable keypoints on the hard poses and had no second-person "
        "confusion. YOLO and RTMPose are faster on GPU but every one of those "
        "milliseconds is bought with VRAM the robot cannot spare, and both mis-locked "
        "onto a bystander in the test clips."
    )

    doc.add_heading("Scorecard", 1)
    _table(doc,
           ["Backbone", "VRAM MB", "CPU ms/f", "GPU ms/f", "Detect (avg)", "Keypoint quality", "Use it?"],
           scorecard_rows(cpu, gpu))
    doc.add_paragraph(
        "VRAM: MediaPipe 0 (CPU). CPU/GPU ms are per frame on squat_01, this "
        "machine. 'Detect' = frames with a person found, averaged over the three "
        "posture clips."
    ).runs[0].font.size = Pt(9)

    doc.add_heading("1. Test setup", 1)
    doc.add_paragraph(
        f"Host: {h['platform']}, Python {h['python']}. "
        f"GPU: {h.get('gpu') or 'n/a'} ({h.get('gpu_mem_total_mb') or '?'} MB). "
        "Latency is measured on this machine and is only a proxy for the Acer/Ubuntu "
        "deploy laptop; treat the ranking and the VRAM figures as the durable result."
    )
    doc.add_paragraph(
        "Hard-case clips (from data/clips/, subject at ~2–4 m): posture — laying_01, "
        "squat_01, sit_02; hands — rock_01, ok_01, i_love_you_01, two_finger_01. "
        "For each backbone we record per-frame latency (mean / p90 / FPS), the "
        "detection rate, peak VRAM, and an 8-frame annotated montage for visual "
        "keypoint-quality review."
    )
    doc.add_paragraph(
        "Note: the MediaPipe Python wheel is CPU-only on both Windows and Linux — "
        "there is no GPU delegate through pip — so its 'GPU' row is identical to CPU."
    )

    doc.add_heading("2. Body-pose latency & VRAM — CPU", 1)
    _table(doc,
           ["Backbone", "Ran on", "squat ms/f", "p90 ms", "FPS", "laying ms/f", "sit ms/f", "VRAM MB"],
           latency_rows(cpu))

    if gpu:
        doc.add_heading("3. Body-pose latency & VRAM — GPU requested", 1)
        _table(doc,
               ["Backbone", "Ran on", "squat ms/f", "p90 ms", "FPS", "laying ms/f", "sit ms/f", "VRAM MB"],
               latency_rows(gpu))
    else:
        doc.add_heading("3. GPU pass", 1)
        doc.add_paragraph("Not available / not run on this machine.")

    doc.add_heading("4. Detection rate (person found / total frames)", 1)
    doc.add_paragraph("CPU pass:")
    _table(doc, ["Backbone", "laying", "squat", "sit"], detect_rows(cpu))

    doc.add_heading("5. Keypoint quality — visual review", 1)
    doc.add_paragraph(
        "8 frames per clip, evenly spaced. Green = skeleton, orange = keypoints. "
        "Look for: keypoints flying to the frame edge, the skeleton jumping to a "
        "second person, limbs snapping to implausible angles. Keypoints are the "
        "same on CPU and GPU (identical models) — only latency changes. Full "
        "annotated clips: results/phase1/annotated/."
    )
    for key in cpu["pose"]:
        for clip in ("squat", "laying", "sit"):
            img = OUT / "montages" / f"{key}__{clip}__cpu.jpg"
            if img.exists():
                doc.add_paragraph(f"{POSE_LABEL.get(key, key)} — {clip}", style="Heading 3")
                doc.add_picture(str(img), width=Inches(6.5))

    doc.add_heading("5b. Why not just run the fast one on the GPU?", 1)
    tot = cpu["host"].get("gpu_mem_total_mb")
    doc.add_paragraph(
        f"The dev GPU has {tot or '~4096'} MB; the deploy laptop is similar and its "
        "VRAM is shared with the other perception / navigation modules. Every model "
        "that touches the GPU takes a fixed slice away from them for the whole "
        "session. The GPU-latency win (below) does not change that MediaPipe's "
        "0-MB footprint is the right default; revisit only if a measured accuracy "
        "gap forces it."
    )

    doc.add_heading("6. Hands — MediaPipe Hands on wrist-anchored crops", 1)
    hrows = [[c, v["detect_rate"], f"{v['ms_mean']} ms" if v["ms_mean"] else "-"]
             for c, v in cpu["hands"]["clips"].items()]
    _table(doc, ["Gesture clip", "detected / crops", "ms/crop"], hrows)
    doc.add_paragraph(
        "Misses are concentrated at the moments the hand enters/leaves frame "
        "(motion blur), not on the held gesture — which is exactly what the "
        "presence-flag + temporal-smoothing design absorbs."
    )

    doc.add_heading("7. Combined pipeline (the real workload)", 1)
    c = cpu["combined_mediapipe"]
    cg = (gpu or {}).get("combined_mediapipe", {})
    doc.add_paragraph(
        f"MediaPipe Pose + 2 wrist-crop hand detections per frame: "
        f"{c['ms_mean']:.0f} ms/frame ≈ {c['fps']:.1f} FPS"
        + (f" (CPU pass) / {cg['ms_mean']:.0f} ms ≈ {cg['fps']:.1f} FPS (GPU pass — "
           "unchanged, MediaPipe does not use the GPU)." if cg else " on this CPU.")
        + " The two hand crops are the bottleneck (~2/3 of the time)."
    )

    doc.add_heading("8. Recommendation", 1)
    doc.add_paragraph("Lock MediaPipe Pose + MediaPipe Hands.", style="List Bullet")
    doc.add_paragraph(
        "Rationale in priority order: (1) 0 VRAM — the deploy GPU is shared; "
        "(2) most stable keypoints on squat/sit, no bystander lock-on; "
        "(3) on CPU it is 3–9× faster than every alternative; "
        "(4) hand landmarks stable on the overlapping-finger shapes (rock, ILY).",
        style="List Bullet")
    doc.add_paragraph(
        "The GPU alternative, stated plainly: YOLO11n-pose on CUDA is the fastest "
        "option at ~24 ms (42 FPS) but costs ~70 MB VRAM, a torch+CUDA runtime "
        "(~5 GB on disk), the flyaway-keypoint / bystander bugs, and a re-write of "
        "the person-selection logic. RTMPose-m has the nicest skeleton but needs "
        "~600 MB VRAM and a separate detector. None of that is worth it while the "
        "robot's GPU budget is the binding constraint.", style="List Bullet")
    doc.add_heading("Open items before Phase 1 is fully signed off", 2)
    doc.add_paragraph("Record a clean 'laying' clip from the robot's camera height and "
                      "re-run — the current laying_01 is mostly transition and occlusion.",
                      style="List Bullet")
    doc.add_paragraph(f"Confirm the combined ~{c['fps']:.0f} FPS on the actual Acer/Ubuntu "
                      "laptop and agree a real-time budget. If short: MediaPipe Tasks GPU "
                      "delegate, smaller hand crops, or static_image_mode=False for hands.",
                      style="List Bullet")

    out = OUT / "backbone_report.docx"
    doc.save(str(out))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
