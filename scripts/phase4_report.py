"""Build results/phase4/classifier_report.docx — the Track C model-family
comparison: N candidate classifiers on the finalised 12-class dataset, all
scored by the same harness.

    python -m classifier.train --model all
    python -m classifier.evaluate --all
    python scripts/phase4_report.py

Reads results/phase4/model_comparison.json + the eval jsons in
classifier/models/ + the figures in results/phase4/fig/ and results/phase3/fig/.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pareto_scatter import pareto_scatter  # noqa: E402

LOCK = "mlp"  # Phase 4 user decision — see docs/phase4_baseline.md
DS = ROOT / "data" / "dataset"
MODELS = ROOT / "classifier" / "models"
OUT = ROOT / "results" / "phase4"
FIG = OUT / "fig"
P3FIG = ROOT / "results" / "phase3" / "fig"

GREEN = RGBColor(0x1A, 0x7F, 0x37)
AMBER = RGBColor(0x88, 0x66, 0x00)

# nicer display order (best-known first)
ORDER = ["lgbm", "catboost", "hgb", "rf", "et", "svm", "mlp", "semlp", "logreg"]
HEADLINE_ROWS = [
    ("macro_f1", "macro-F1 (12)"),
    ("macro_f1_real_eval", "macro-F1 (real-eval)"),
    ("micro_f1_accuracy", "accuracy"),
    ("balanced_accuracy", "balanced acc"),
    ("expected_calibration_error", "ECE (lower better)"),
    ("mean_conf_correct", "conf | correct"),
    ("mean_conf_wrong", "conf | wrong"),
    ("model_size_mb", "size (MB)"),
    ("fit_seconds", "fit (s)"),
]


def _cell(t, r, c, text, bold=False, color=None):
    cell = t.cell(r, c)
    cell.text = ""
    run = cell.paragraphs[0].add_run("" if text is None else str(text))
    run.bold = bold
    run.font.size = Pt(9)
    if color:
        run.font.color.rgb = color


def _table(doc, headers, rows, row_colors=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    for c, h in enumerate(headers):
        _cell(t, 0, c, h, bold=True)
    for r, row in enumerate(rows, 1):
        for c, v in enumerate(row):
            _cell(t, r, c, v, color=(row_colors[r - 1] if row_colors else None))
    return t


def _fig(doc, path: Path, width=6.4):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
    else:
        doc.add_paragraph(f"[missing figure {path.name}]")


def _classifier_scatter(hl, models):
    # the two torch models are stochastic — plot their 5-seed MEAN real-eval, not
    # the lucky/unlucky single seed, so the frontier is honest. tree/linear
    # models are deterministic single fits.
    sm = _bde("semlp") or {}
    seed_mean = {"mlp": (sm.get("plain_mlp_5seed") or {}).get("real_eval", [None])[0],
                 "semlp": ((sm.get("variants") or {}).get("se_floor_0.5", {})
                           .get("summary", {}).get("real_eval", [None])[0])}
    pts = []
    for m in models:
        h = hl[m]
        gap = ((h.get("mean_conf_correct") or 0) - (h.get("mean_conf_wrong") or 0))
        y = seed_mean.get(m) or h.get("macro_f1_real_eval") or 0.0
        off = {"mlp": (-4, 9), "semlp": (10, -14)}.get(m, (0, 0))
        pts.append({"name": m.upper(),
                    "x": max(h.get("model_size_mb") or 0.0, 0.05),
                    "y": y, "size": max(gap, 0.02), "star": m == LOCK,
                    "label_offset": off})
    out = FIG / "compare_scatter.png"
    pareto_scatter(pts, xlabel="model size on disk (MB, log)",
                   ylabel="real-eval macro-F1",
                   title="Phase 4 classifiers — accuracy vs size (bubble = confidence separation)",
                   out=out, xlog=True, size_label="conf(correct) − conf(wrong)",
                   note="MLP/SEMLP = 5-seed mean real-eval; others = single deterministic fit. "
                        "Pareto-optimal = green dashed line; gold star = locked pick.")
    return out


def _bde(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _semlp_section(doc):
    d = _bde("semlp")
    doc.add_heading("4b. SE-MLP — a squeeze-and-excitation variant", 1)
    if not d:
        doc.add_paragraph("[results/phase4/semlp.json missing]")
        return
    doc.add_paragraph(
        "Follow-up asked after the MLP lock: does a feature-wise "
        "squeeze-and-excitation gate before the MLP trunk help? The gate "
        "(classifier/semlp.py) reads the whole 189-d input and emits one "
        "multiplier per feature, so the trunk sees the currently-irrelevant "
        "slices damped. Two variants, 5 seeds each, scored on test:")
    pm = d["plain_mlp_5seed"]
    rows = [["plain MLP (lock)", f"{pm['real_eval'][0]:.3f} ± {pm['real_eval'][1]:.3f}",
             f"{pm['macro_f1'][0]:.3f} ± {pm['macro_f1'][1]:.3f}",
             f"{pm['acc'][0]:.3f} ± {pm['acc'][1]:.3f}",
             f"{pm['idle'][0]:.3f} ± {pm['idle'][1]:.3f}"]]
    colors = [GREEN]
    for key, label in (("se_floor_0.0", "SE gate (hard, floor 0)"),
                       ("se_floor_0.5", "SE gate (floor 0.5)")):
        s = d["variants"][key]["summary"]
        rows.append([label, f"{s['real_eval'][0]:.3f} ± {s['real_eval'][1]:.3f}",
                     f"{s['macro_f1'][0]:.3f} ± {s['macro_f1'][1]:.3f}",
                     f"{s['acc'][0]:.3f} ± {s['acc'][1]:.3f}",
                     f"{s['idle'][0]:.3f} ± {s['idle'][1]:.3f}"])
        colors.append(None)
    _table(doc, ["model", "real-eval", "macro-F1", "acc", "idle F1"], rows, colors)
    dec = d["decision"]
    doc.add_paragraph(dec[:1].upper() + dec[1:])
    doc.add_paragraph(
        "Worth keeping: the gate is interpretable — it suppresses the off-hand "
        "for one-hand gestures (raise_right → left-hand block ≈ 0.16, right-hand "
        "≈ 0.48; raise_left the mirror), leans on the joint-angle block for "
        "postures, and keeps that block the most open overall. This is a third "
        "independent confirmation (with Track B's body_raw_hands_derived) that "
        "the idle class is fragile to anything that reduces feature information.")


def _track_b(doc):
    d = _bde("track_b_seeds")
    doc.add_heading("5. Track B — feature set (MLP, 5 seeds)", 1)
    if not d:
        doc.add_paragraph("[results/phase4/track_b_seeds.json missing]")
        return
    fm = d["feature_modes"]
    rows = []
    for key in ("both", "body_raw_hands_derived"):
        s = fm[key]["summary"]
        def ms(k):
            v = s[k]
            return f"{v[0]:.3f} ± {v[1]:.3f}"
        rows.append([f"{key} ({fm[key]['dim']}-d)", ms("macro_f1"), ms("real_eval"),
                     ms("acc"), ms("mini_heart"), ms("idle")])
    _table(doc, ["features", "macro-F1", "real-eval", "acc", "mini_heart F1", "idle F1"],
           rows, [GREEN, None])
    o = d.get("one_seed_also_ran", {})
    doc.add_paragraph(
        f"Also run at 1 seed: raw (152-d) real-eval {o.get('raw', {}).get('real_eval')}, "
        f"derived-only ({o.get('derived', {}).get('dim')}-d) "
        f"{o.get('derived', {}).get('real_eval')} — both clearly worse. "
        + d.get("decision", ""))
    doc.add_paragraph(
        "body_raw_hands_derived (drops the raw hand coordinates, keeps the "
        "scale-invariant derived hand features) is a real, stable win for "
        "mini_heart — the raw hand coords carry MediaPipe's hand-crop "
        "domain-gap noise. But it misses the ≥0.86 real-eval bar, is noisier on "
        "every headline metric, and idle F1 sd blows out to 0.31 (collapses to "
        "~0.21 on 2 of 5 seeds). idle is a mandated class, so the lock stays at "
        "'both'. The mini_heart signal is filed for a Phase 6 feature pass.")


def _track_d(doc):
    d = _bde("track_d")
    doc.add_heading("6. Track D — augmentation strength (MLP 'both', seed 0)", 1)
    if not d:
        doc.add_paragraph("[results/phase4/track_d.json missing]")
        return
    rows, colors = [], []
    for r in d:
        t = r["test"]
        rows.append([r["tag"], f"{t['real_eval']:.3f}", f"{t['mini_heart']:.2f}",
                     f"{t['raise_right_hand']:.2f}", f"{t['raise_left_hand']:.2f}",
                     f"{t['idle']:.2f}", f"{t['acc']:.3f}"])
        colors.append(GREEN if r["tag"].startswith("D0") else None)
    _table(doc, ["config", "real-eval", "mini_heart", "raise_R", "raise_L", "idle", "acc"],
           rows, colors)
    doc.add_paragraph(
        "Each config rebuilds the dataset (AUG_OVERRIDE / ARMS_UP_KEEP_ORIG env "
        "hooks in pipeline/build_dataset.py), retrains, and is scored on test. "
        "D1 keeps chest-level mini_heart originals; D1b also widens the "
        "arm-elevation range; D2 raises n_per_sample 4→7; D3 softens the "
        "occlusion aug (hand_drop .10→.05, kp_dropout .05→.03). No variant beats "
        "the defaults — extra augmentation washes out the mini_heart boundary "
        "(0.73 → 0.16 at n_per_sample=7). The val split pins mini_heart at ~0.99 "
        "for every config and cannot tune the scarce classes. AugParams "
        "defaults retained.")


def _track_e(doc):
    d = _bde("track_e")
    doc.add_heading("7. Track E — confidence thresholds for the idle fallback", 1)
    if not d:
        doc.add_paragraph("[results/phase4/track_e.json missing]")
        return
    tr = d["transfer"]
    rows = []
    for sp in ("val", "test"):
        for row, lbl in (("raw", "no gate"), ("val_per_class", "per-class val T"),
                         ("global_floor", "global 0.20 floor")):
            v = tr[sp][row]
            rows.append([f"{sp} — {lbl}", f"{v['macro_f1']:.3f}", f"{v['acc']:.3f}",
                         f"{v['idle_p']:.2f}", f"{v['idle_r']:.2f}", v["to_idle"]])
    _table(doc, ["split / rule", "macro-F1", "acc", "idle P", "idle R", "→idle"], rows)
    doc.add_paragraph(
        "Rule: if the max probability for the argmax class is below its "
        "threshold, emit idle instead. Per-class thresholds fit on val (keep "
        "85 % of val true-positives) collapse the honest test split — val is the "
        "same webcam domain as train so its confidences run ~0.2 higher; applied "
        "to test, mini_heart is gated to F1 0 and idle precision drops to 0.23. "
        "On test the idle errors are real gestures misread as idle, so "
        "reject-to-idle can only make idle precision worse. Only a garbage floor "
        "(global 0.20, ~1 frame) is active; real thresholds need Phase 6 field "
        "confidences. classifier/thresholds.py stays as the re-tuning harness.")


def main():
    cmp_path = OUT / "model_comparison.json"
    if not cmp_path.exists():
        raise SystemExit("run `python -m classifier.evaluate --all` first")
    cmp = json.loads(cmp_path.read_text())
    card = json.loads((DS / "dataset_card.json").read_text())
    classes = card["classes"]

    models = [m for m in ORDER if m in cmp["headline"]] + \
             [m for m in cmp["headline"] if m not in ORDER]
    hl = cmp["headline"]
    pcf1 = cmp["per_class_f1"]
    xper = cmp.get("cross_person", {})

    best_real = min(models, key=lambda m: (-(hl[m].get("macro_f1_real_eval") or 0),
                                           hl[m].get("model_size_mb") or 1e9))
    winner = LOCK if LOCK in models else best_real  # locked pick, not raw top score

    doc = Document()
    doc.add_heading("SKUBA gesture detection — Phase 4: classifier iteration", 0)
    doc.add_paragraph(f"Generated {date.today().isoformat()}").runs[0].italic = True

    p = doc.add_paragraph()
    r = p.add_run(f"{len(models)} candidate classifiers trained on the same "
                  f"12-class pipeline (train {card['train']['rows']:,} / val "
                  f"{card.get('val', {}).get('rows', 0):,} / test "
                  f"{card['test']['rows']:,}), scored by classifier/evaluate.py. "
                  f"LOCKED pick: {winner.upper()} (features 'both', seed 0). "
                  f"{best_real.upper()} is nominally top on raw real-eval "
                  f"({hl[best_real].get('macro_f1_real_eval')}) but "
                  f"{winner.upper()} ties it within seed noise and wins on "
                  f"confidence separation + size (section 5).")
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = GREEN
    doc.add_paragraph(
        "Sections 1–4: the model-family comparison (Track C). Sections 5–7: the "
        "feature-set (B), augmentation-strength (D) and confidence-threshold (E) "
        "sweeps done on the locked model. Section 8: verdict.")

    doc.add_heading("1. Headline metrics — every candidate", 1)
    rows = []
    for key, label in HEADLINE_ROWS:
        rows.append([label] + [_fmt(hl[m].get(key), key) for m in models])
    _table(doc, ["metric"] + [m.upper() for m in models], rows)
    doc.add_paragraph(
        "'real-eval' = macro-F1 over the classes with a genuine cross-domain / "
        "cross-subject test (excludes the aug-only-leak classes squat, "
        "glico_pose). 'conf | correct' vs 'conf | wrong' is the mean max-"
        "probability when the prediction is right vs wrong — the gap is what the "
        "Phase 5 idle/unknown threshold needs; a model that is accurate but "
        "piles all its confidence at ~1.0 needs post-hoc calibration first."
    )
    _fig(doc, FIG / "compare_overall.png")

    scat = _classifier_scatter(hl, models)
    doc.add_heading("1b. Accuracy vs model size — the Pareto view", 1)
    doc.add_paragraph(
        "Paper-style cost/quality scatter: x = model size on disk (log scale, "
        "smaller is better), y = real-eval macro-F1 (higher is better), bubble "
        "area = confidence separation (mean conf when correct minus when wrong — "
        "the bigger the bubble the more usable the Phase 5 idle threshold). "
        "Points on the green dashed line are Pareto-optimal (nothing beats them "
        "on both axes); the gold star is the locked pick. The frontier is "
        "LogReg → MLP/SE-MLP (statistically tied, overlapping) → LightGBM: "
        "LightGBM edges the MLPs on F1 but at 100x the size and a third of the "
        "confidence spread. Every other candidate — including CatBoost and "
        "HistGradientBoosting, run to close the GBDT-family gap — is dominated: "
        "3–11 real-eval points behind LightGBM at equal or larger size. The GBDT "
        "family does not cluster (section 8); the SE gate does not help "
        "(section 4b).")
    _fig(doc, scat, width=6.0)

    doc.add_heading("2. Per-class F1", 1)
    prows, pcolors = [], []
    for c in classes:
        vals = pcf1.get(c, {})
        if not any(vals.get(m) is not None for m in models):
            continue
        best = max((vals.get(m) or 0) for m in models)
        prows.append([c] + [f"{vals.get(m):.2f}" if vals.get(m) is not None else "-"
                            for m in models])
        pcolors.append(AMBER if best < 0.75 else None)
    _table(doc, ["class"] + [m.upper() for m in models], prows, pcolors)
    _fig(doc, FIG / "compare_f1.png")

    doc.add_heading("3. Cross-person (sit — the only multi-subject class)", 1)
    xrows = []
    for m in models:
        w = xper.get(m, {}).get("sit", {})
        if w:
            xrows.append([m.upper(), w.get("s01", "-"), w.get("s02", "-")])
    if xrows:
        _table(doc, ["model", "sit @ s01", "sit @ s02 (never trained)"], xrows)
        doc.add_paragraph("s02 recall is the single most trustworthy number — a "
                          "person who never appears in training.")

    doc.add_heading("4. The leading model in detail — " + winner.upper(), 1)
    ev_path = MODELS / f"{winner}.eval.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())
        drows, dcolors = [], []
        for c, rr in ev["per_class"].items():
            drows.append([c, rr["eval_type"], rr["n"], rr["precision"],
                          rr["recall"], rr["f1"], rr["mean_conf"]])
            dcolors.append(AMBER if rr["f1"] < 0.75 else None)
        _table(doc, ["class", "eval type", "n", "prec", "rec", "F1", "conf"],
               drows, dcolors)
    for name in (f"{winner}_confusion.png", f"{winner}_calibration.png"):
        f = FIG / name
        if not f.exists():
            f = P3FIG / name
        _fig(doc, f)

    _semlp_section(doc)
    _track_b(doc)
    _track_d(doc)
    _track_e(doc)

    doc.add_heading("8. Verdict", 1)
    doc.add_paragraph(
        f"Locked classifier: {winner.upper()}, features 'both' (189-d), seed 0 "
        "(classifier/models/mlp.joblib). Accuracy is tied with LightGBM inside "
        "5-seed noise (real-eval 0.857 ± 0.016 vs 0.866); MLP wins on the "
        "deployment axes — 0.3 MB vs 31 MB, and a confidence gap (0.85 correct / "
        "0.49 wrong) wide enough to drive the Phase 5 idle threshold without a "
        "separate calibration step.")
    doc.add_paragraph(
        "All 9 candidates were trained and scored (section 1): the original "
        "four, plus CatBoost / HistGradientBoosting / RBF-SVM / ExtraTrees "
        "(an earlier session had left them unrun), plus the SE-MLP variant "
        "(section 4b). None changes the pick. The GBDT family does not cluster "
        "around LightGBM as had been assumed (CatBoost −8, HGB −5.5 real-eval "
        "points at the screening configs); HGB is the best-calibrated model of "
        "the lot (ECE 0.022) but trails MLP on accuracy and confidence spread; "
        "and the SE gate ties plain MLP at best while making the idle class less "
        "stable. Plain MLP wins on Occam.")
    doc.add_paragraph(
        "Tracks B/D/E each tested one lever and each confirmed the current "
        "config: 'both' features beat the leaner alternative on stability, the "
        "AugParams defaults beat every variant tried, and val-fit confidence "
        "thresholds do not transfer to the cross-domain test split. The "
        "remaining weak classes (mini_heart 0.73, raise_*_hand 0.76, idle "
        "precision 0.62 on test) are data/feature-gap limited, not "
        "hyperparameter-limited — Phase 6 field recordings are the next real "
        "lever. Full rationale: docs/phase4_baseline.md.")

    out = OUT / "classifier_report.docx"
    try:
        doc.save(str(out))
    except PermissionError:
        out = OUT / "classifier_report.NEW.docx"
        doc.save(str(out))
        print("!! report open in Word — wrote", out.name)
    print(f"-> {out}  (winner: {winner})")


def _fmt(v, key):
    if v is None:
        return "-"
    if key == "model_size_mb":
        return f"{v:.0f}" if v >= 1 else "<1"
    if key == "fit_seconds":
        return f"{v:.0f}"
    return f"{v:.3f}"


if __name__ == "__main__":
    main()
