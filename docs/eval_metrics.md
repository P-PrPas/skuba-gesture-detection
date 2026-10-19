# Evaluation metrics & harness — Phase 3 / 4 / 5

*How every classifier candidate in this project is scored. One harness
(`classifier/evaluate.py`), one metric set, one JSON schema, so RF, LightGBM, the
MLP, and anything added later are compared like-for-like. This document defines
every score the harness emits and every score quoted in the phase reports.*

Companion to `docs/model_architecture_phase3_4.md`. Source of truth:
`classifier/evaluate.py`, `classifier/thresholds.py`, `scripts/phase5_flicker.py`,
`data/dataset/dataset_card.json`.

---

## 1. What the harness scores

A **model** is a bundle dict `{clf, classes, clip, features}` written by
`classifier/train.py`. The only requirement on `clf` is `.predict_proba(X)` and
`.classes_`. Every candidate implements that surface (§6 of the architecture
doc), so the harness is model-agnostic.

```
python -m classifier.evaluate --model mlp        # one model, on test
python -m classifier.evaluate --all              # every trained model + comparison
python -m classifier.evaluate --model mlp --split val
```

### Scoring procedure (`_evaluate`)

| step | what happens |
|---|---|
| 1 | load `data/dataset/<split>.npz` — `X` (N × 152), `y` (int labels), `subject_id` |
| 2 | clip `X` to ±`clip` (±10), apply `to_features(X, features)` → the model's input dim (189 for `both`) |
| 3 | **filter to modelled classes**: keep only rows whose `y ∈ clf.classes_`. Reported as `n_test_rows` (all) vs `n_scored_rows` (kept). A model that never learned a class is not penalised for it here — that gap is a dataset-card fact, not a metric. |
| 4 | `proba = clf.predict_proba(X)` ; `pred = argmax(proba)` ; `conf = max(proba)` ; `correct = (pred == y)` |
| 5 | compute every metric below from `(y, pred, conf, correct, subject_id)` |

Default split is **`test`** — every original s01/s02 frame + 15 % held-out
HaGRID subjects for `thumb`, never augmented. `val` scoring exists for Track E /
calibration diagnostics only.

---

## 2. Eval-type taxonomy — the trust level of each per-class number

Each class carries an `eval_type` from `dataset_card.json` (`per_class`). **This
is the single most important thing to read before any per-class score.** It says
whether that class's test frames are a real generalisation test or a leak.

| `eval_type` | meaning | is it a real number? |
|---|---|---|
| `cross_domain` | trained on external data (HaGRID / COCO), tested on our s01/s02 frames — different subjects **and** different camera/domain | **yes** — genuine generalisation |
| `held_out_external` | `thumb` only — trained on HaGRID, tested on held-out HaGRID *subjects*, same webcam domain | **yes** — cross-subject, same domain |
| `aug_only_leak` | `squat`, `glico_pose`, `laying` — train = augmented copies of the same s01/s02 frames that are in test | **no** — train/test overlap. `sit`@s02 is the exception: its aug is s01-only so the s02 frames are a clean cross-person check |
| `pending_data` / `n/a` / `-` | class present in test only, or no clean split | not scored as generalisation |

`REAL_EVAL = ("cross_domain", "held_out_external")`. The **headline number of
every experiment is `macro_f1_real_eval`** — the macro-F1 restricted to those
classes. `aug_only_leak` F1s are printed but never used to make a decision.

The `_perclass.png` figure colour-codes every class label by eval-type
(green = cross-domain, blue = held-out, amber = leak) so a leaked 1.00 can't be
mistaken for a real one.

---

## 3. Core classification metrics

All defined per class `k` from the confusion counts:
`TP = #(y=k ∧ pred=k)`, `FP = #(y≠k ∧ pred=k)`, `FN = #(y=k ∧ pred≠k)`.

### 3.1 Precision, recall, F1 (per class) — `_prf`

```
precision(k) = TP / (TP + FP)          "when the model says k, how often is it right"
recall(k)    = TP / (TP + FN)          "of the real k frames, how many did it catch"
F1(k)        = 2·P·R / (P + R)         harmonic mean
support(k)   = TP + FN                 (# true k frames in the split, field `n`)
```

Convention: `0/0 → 0.0` (a class with no predictions gets precision 0, not NaN).

Per-class interpretation notes used in the reports:
- **`idle` precision** is the one to watch — low precision (≈0.46 in Phase 3,
  0.62 in Phase 4 test) means real gestures are falling through to `idle`.
- **`laying` recall** — misses go to `sit` / `raise_left_hand` (a lying arm reads
  as raised in un-rotated normalized coords).

### 3.2 Macro-F1 (all modelled classes) — `macro_f1`

```
macro_f1 = mean_k F1(k)      over every modelled class present in the split
```

Unweighted mean — every class counts equally regardless of support, so the rare
classes (`t_pose`, `mini_heart`) are not drowned out by `idle`. This is the "12-class"
number in the tables.

### 3.3 Macro-F1, real-eval — `macro_f1_real_eval`  ← **headline**

```
macro_f1_real_eval = mean_k F1(k)      over classes with eval_type ∈ REAL_EVAL
```

The honest generalisation score. In Phase 4 this is the "real-eval" column and
the number the lock decision was made on (MLP 0.847, 5-seed 0.857 ± 0.016;
LightGBM 0.866).

### 3.4 Macro-F1, cross-domain only — `macro_f1_cross_domain`

Same, restricted to `cross_domain` (drops `thumb`'s same-domain held-out number).
A stricter cut; reported for completeness.

### 3.5 Accuracy / micro-F1 — `micro_f1_accuracy`

```
accuracy = mean(correct) = (# correct frames) / (# scored frames)
```

In single-label multiclass this equals micro-averaged F1, hence the field name.
Support-weighted, so it tracks the common classes (`idle`, `sit`). Always quoted
**alongside** macro-F1, never instead of it — a model can hit 0.92 accuracy while
a rare class sits at F1 0.4.

### 3.6 Balanced accuracy — `balanced_accuracy`

```
balanced_accuracy = mean_k recall(k)
```

Mean per-class recall — accuracy re-weighted so each class contributes equally.
The gap between this and raw accuracy shows how lopsided the errors are across
classes.

### 3.7 The 0.6 per-class floor

Not a computed metric — a **Phase 3 exit bar**. Every class must reach F1 ≥ 0.60
on its own eval-type. Drawn as a dashed red line on `_perclass.png` and
`compare_f1.png`. Phase 3 exited with all 12 classes ≥ 0.6.

---

## 4. Confidence & calibration metrics

`conf` = the model's max-softmax probability for its predicted class. These
metrics decide whether that number is usable as a **reject threshold** (the
Phase 5 "if conf < T → idle/unknown" rule).

### 4.1 Confidence separation — `mean_conf`, `mean_conf_correct`, `mean_conf_wrong`

```
mean_conf_correct = mean(conf | correct)
mean_conf_wrong   = mean(conf | wrong)
gap               = mean_conf_correct − mean_conf_wrong
```

Reported globally and per class (`mean_conf`, `mean_conf_correct`,
`mean_conf_wrong`). **A wide gap is what makes confidence thresholdable.** This
is the axis the MLP won the lock on:

| model | conf correct | conf wrong | gap | thresholdable? |
|---|---|---|---|---|
| MLP (lock) | 0.86 | **0.48** | **0.37** | yes — one cut near 0.5 |
| LightGBM | 0.95 | 0.80 | 0.15 | no — needs Platt/isotonic first |

### 4.2 Expected Calibration Error — `expected_calibration_error` (ECE)

`_calibration`: bin the `N` predictions into 10 equal-width confidence bins
`[0.0,0.1), … [0.9,1.0]`. Per bin `b`: `acc(b)` = fraction correct,
`conf(b)` = mean confidence.

```
ECE = Σ_b  (n_b / N) · | acc(b) − conf(b) |
```

The average distance between "how confident" and "how often right". **0 = perfectly
calibrated**; lower is better. A model claiming 0.9 confidence on frames it gets
right 0.7 of the time has poor calibration and a high ECE.

Caveat (Track E finding): temperature scaling can leave the MLP slightly
*under*-confident, which raises ECE (0.103) but is the **safe** direction for a
reject rule — better to occasionally doubt a correct frame than to trust a wrong
one.

### 4.3 Reliability diagram + confidence histogram — `calibration_bins`

`calibration_bins` is the per-bin `{lo, hi, n, acc, conf}` list. Rendered as
`_calibration.png`:
- **left** — reliability curve: `acc(b)` vs `conf(b)`, with the `y = x` diagonal
  ("perfect"). Points below the diagonal = over-confident.
- **right** — confidence histogram: `n` per bin. Shows whether predictions pile
  into the top bin (LightGBM) or spread across 0.2–1.0 (MLP). A spread histogram
  is a prerequisite for a single useful threshold.

---

## 5. Confusion analysis

### 5.1 Confusion matrix — `confusion_counts`, `confusion_row_normalised`

Square matrix over the classes present in the split, ordered by `CLASSES`.
Row = true class, column = predicted.

```
confusion_row_normalised[i,j] = count[i,j] / Σ_j count[i,j]
```

Row-normalised so each row sums to 1 — cell `[i,j]` reads "fraction of true `i`
frames predicted as `j`". The diagonal is per-class recall. Rendered as
`_confusion.png` (heatmap, raw counts annotated).

### 5.2 Top confusions — `top_confusions`

The 12 largest off-diagonal cells as `{true, pred, count}`, sorted descending.
The quick "what is this model actually getting wrong" list — e.g. Phase 3
`two_finger → rock` (before `rock` was cut), `laying → sit`.

---

## 6. Cross-person recall — `cross_person`

For any class present for **more than one `subject_id`** in the split:

```
cross_person[class][subject] = recall of that class, computed on that subject's frames only
```

This isolates the **cross-subject** component of generalisation (the dataset only
has s01 + s02, so it is thin, but `sit` has s01 train-aug + s02 test). The
canonical number: `sit`@s02 recall = **0.97** (Phase 3) / **1.00** (Phase 4) —
the payoff of de-leaking `sit` from COCO. `laying` cannot be scored this way (s02
only). See `docs/ARCHITECTURE.md` "Evaluating with a small subject pool".

---

## 7. Cost metrics

| field | source | why it's a decision axis |
|---|---|---|
| `model_size_mb` | `<model>.joblib` file size | shared-VRAM Acer deployment — RF's 389 MB is a liability, MLP's 0.3 MB is free |
| `fit_seconds` | `<model>.meta.json` (`train.py` timing) | experiment iteration cost; not a deployment metric |
| `n_features` | `meta.train_dim` | 189 for `both`, 105 for `body_raw_hands_derived`, etc. |
| `train_rows` | `meta.train_rows` | 52 738 for the locked build |

Phase 4's Pareto view is **`macro_f1_real_eval` vs `model_size_mb`** — the MLP
sits at the knee (tied accuracy, 100× smaller than LightGBM).

---

## 8. Stability — multi-seed mean ± sd

The honest test is ~1 254 rows — small enough that a 1–2 point macro-F1 move is
noise. For any close decision the model is retrained under **5 seeds** (0–4) and
reported as **mean ± sd**, not a single run:

```
MLP:  macro-F1 0.881 ± 0.014   real-eval 0.857 ± 0.016   acc 0.919 ± 0.008
```

Rule of thumb applied in the reports: two models are "tied on accuracy" if their
means are within one sd of each other (MLP 0.857 ± 0.016 vs LightGBM's
deterministic 0.866 → tied). The committed artifact is **seed 0** (a mid-range
seed — no cherry-picking).

---

## 9. Downstream harnesses (same metric vocabulary)

### 9.1 Track E — reject-threshold analysis (`classifier/thresholds.py`)

Tests the Phase 5 rule *"argmax = c; if `maxproba < T[c]` emit `idle` instead"*.
Metrics, computed on **both `val` and `test`** to check transfer:

| metric | definition |
|---|---|
| `macro_f1` | mean per-class F1 (over `np.unique(y)`) after the reject rule is applied |
| `idle_p` / `idle_r` | `idle` precision / recall after the rule — the rule can only *add* `idle` predictions, so it trades gesture recall for `idle` recall |
| `to_idle` | # frames the rule flipped to `idle` (the cost) |
| `global_floor` (0.20) | the only active threshold — a garbage floor, ≈1 test frame affected |
| `reference_val_pXX` | per-class thresholds that keep XX % of val true-positives — **computed, stored, but inactive** |

**Finding:** per-class val thresholds do **not** transfer — `val` confidences run
~0.2 higher than cross-domain `test`, so a val-fit threshold rejects a large
slice of correct `test` predictions (`mini_heart` → F1 0). Real per-class
thresholds wait for Phase 6 field confidences.

### 9.2 Phase 5 — temporal smoothing sweep (`scripts/phase5_flicker.py`)

Runs each clip through the locked classifier → `TemporalSmoother(n, floor)`,
sweeps `N ∈ {1,3,5,7,9,11,15}`. Metrics, averaged over clips:

| metric | definition | target |
|---|---|---|
| `switches_per_100f` | label changes per 100 frames — **flicker** | ↓ (steadier) |
| `accuracy` | fraction of frames whose smoothed label == the clip's single true label | ≥ (raw − 0.02) |
| `lag_frames` | frames after frame 0 until the smoothed label first equals truth | ≤ 15 (~0.5 s) |

Locked result: **N = 7** — flicker 7.0 → 2.4 per 100 f (−66 %) while clip
accuracy *rises* 0.873 → 0.896, lag ~2 frames.

---

## 10. Output files

Per model (`--model <m>`, on `test`):

| file | content |
|---|---|
| `classifier/models/<m>.eval.json` | every field in §3–§7 |
| `results/phase3/fig/<m>_confusion.png` | row-normalised confusion heatmap |
| `results/phase3/fig/<m>_perclass.png` | P / R / F1 bars, eval-type-coloured labels, 0.6 floor |
| `results/phase3/fig/<m>_calibration.png` | reliability diagram + confidence histogram |

With `--all`:

| file | content |
|---|---|
| `results/phase{3,4}/model_comparison.json` | headline metrics + per-class F1 + cross-person, all models |
| `.../fig/compare_f1.png` | per-class F1, all models side by side |
| `.../fig/compare_overall.png` | macro-F1 / real-eval / accuracy / balanced-acc / ECE, all models |

(`rf` + `lgbm` only → `results/phase3/`; any wider set → `results/phase4/`.)

---

## 11. Reading a result — worked example (locked MLP, `test`)

```
macro_f1                 0.872     all 12 classes, unweighted
macro_f1_real_eval       0.847     ← headline: the 8 cross-domain + held-out classes
micro_f1_accuracy        0.915     support-weighted; tracks idle/sit
balanced_accuracy        ~0.88     mean recall
expected_calibration_err 0.103     slightly under-confident (safe direction)
mean_conf_correct/wrong  0.86 / 0.48   gap 0.37 → confidence is thresholdable
cross_person  sit        {s02: 1.00}   clean cross-person check
model_size_mb            0.3       vs LightGBM 31, RF 389
```

Decision trail: real-eval tied with LightGBM (within seed sd) → look at the
confidence gap → MLP's 0.37 vs LightGBM's 0.15 → MLP is thresholdable without a
calibration step → **lock MLP**. Every number above comes from one
`python -m classifier.evaluate --model mlp` run.
