# Model Architecture — Phase 3 & 4 classification layer

*Reference write-up of every model configuration exercised in Phase 3 (baseline)
and Phase 4 (candidate comparison + lock). The backbone (MediaPipe Pose + Hands)
is frozen and out of scope — see `docs/ARCHITECTURE.md` "Tech stack". This
document describes only the **classification layer**: the fixed feature contract
it consumes, the per-model layer stacks, the calibration head, and the training
recipe.*

Source of truth for shapes: `features/schema.py`, `features/derived.py`,
`classifier/{train,mlp,semlp,thresholds}.py`, `pipeline/build_dataset.py`.

---

## 1. System overview

```
                     ┌─────────────────────────  FROZEN BACKBONE  ──────────────────────────┐
 RGB frame ─▶ person detect ─▶ MediaPipe Pose (33 lm) ─┬─▶ wrist-anchored crop ─▶ Hands (21 lm)  L
                                                       └─▶ wrist-anchored crop ─▶ Hands (21 lm)  R
                     └──────────────────────────────────────────────────────────────────────┘
                                                    │
        ┌──────────────────────────  CLASSIFICATION LAYER (this doc)  ─────────────────────────┐
        │  L2  normalize (body / L-hand / R-hand, independently)                               │
        │  L3  fuse ─▶ x ∈ ℝ¹⁵²  (66 body + 42 Lh + 1 flag + 42 Rh + 1 flag)                   │
        │  L4  derived features  d ∈ ℝ³⁷   (joint angles, ratios, inter-wrist dist)            │
        │  L5  feature assembly  →  z ∈ ℝ¹⁸⁹  ("both" mode);  clip to ±10                      │
        │  L6  per-model input transform (StandardScaler / z-score / none)                     │
        │  L7  classifier head  fθ : ℝ¹⁸⁹ → ℝ¹²  (logits)                                       │
        │  L8  probability calibration  (temperature T, or Platt, or none)                     │
        │  L9  argmax → (class, confidence)                                                    │
        └────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                      Phase 5:  L10 confidence floor (0.20)  →  L11 temporal majority vote (N=7)
```

**Task.** Single-frame multi-class classification, 12 classes:
`idle, raise_right_hand, raise_left_hand, sit, squat, laying, ok, two_finger,
thumb, mini_heart, t_pose, glico_pose` (`features/schema.py::CLASSES`, index order
is the label order).

**Design constraint (CLAUDE.md).** One classifier slot, no hand-written
thresholds in the classification path, `idle` is a first-class label. Every
candidate below implements the same bundle surface `{clf, classes, clip,
features}` with `clf.predict_proba` / `clf.classes_`, so `classifier/evaluate.py`
scores them identically.

---

## 2. Input representation (L1–L3) — shared by all models

### L1. Backbone keypoints (frozen)

| stream | model | output |
|---|---|---|
| body | MediaPipe `pose_landmarker` lite, VIDEO mode | 33 landmarks (x, y), BlazePose topology |
| left hand | MediaPipe `hand_landmarker`, IMAGE mode, on a wrist-anchored square crop | 21 landmarks (x, y) or **absent** |
| right hand | same | 21 landmarks (x, y) or **absent** |

Hand crops are anchored on the pose wrist landmark and sized relative to a
body-scale reference, so hands stay resolvable at robot-to-person distance. A
missing hand is not an error — it is propagated as a zero slice + presence flag 0.

### L2. Normalization (`features/normalize.py`)

Applied **independently** to body and to each hand, before fusion. Removes camera
distance and subject/hand size.

| stream | center | scale |
|---|---|---|
| body | midpoint of (shoulder-mid, hip-mid) | ‖shoulder-mid − hip-mid‖ |
| hand | wrist landmark (idx 0) | ‖middle-MCP (idx 9) − wrist‖  (palm length) |

`x_norm = (x − center) / max(scale, 1e-6)`

### L3. Fusion (`features/fuse.py`) → **x ∈ ℝ¹⁵²**

| slice | dim | content |
|---|---|---|
| `[0:66]` | 66 | 33 body landmarks, normalized (x, y) |
| `[66:108]` | 42 | 21 left-hand landmarks, normalized (x, y) |
| `[108]` | 1 | left-hand presence flag ∈ {0, 1} |
| `[109:151]` | 42 | 21 right-hand landmarks, normalized (x, y) |
| `[151]` | 1 | right-hand presence flag ∈ {0, 1} |

A flagged-absent hand has its 42 coordinates zero-filled. Training data contains
both presence states, so the classifier learns "flag 0 ⇒ ignore this slice"
rather than reading the origin as a literal position.

---

## 3. Derived feature layer (L4) — `features/derived.py`

Hand-crafted, scale-free geometric features. Motivation: MediaPipe Hands produces
a **proportionally distorted** skeleton on a tight robot-camera crop vs a large
webcam hand; raw coordinates carry that domain gap, joint **angles** and
**length ratios** do not. This layer is what let `two_finger` survive the Phase 3
cut of `rock` / `i_love_you`.

**d ∈ ℝ³⁷ = body(13) + left-hand(12) + right-hand(12)**

| group | features | count |
|---|---|---|
| body angles | L/R elbow, L/R shoulder, L/R hip, L/R knee (angle at vertex, radians) | 8 |
| body pose | torso lean vs vertical; L & R wrist height rel. shoulders; shoulder tilt; **inter-wrist distance** (in body units) | 5 |
| per hand — curl | angle at MCP for thumb / index / middle / ring / pinky | 5 |
| per hand — spread | thumb abduction; index–pinky MCP spread; index–pinky fingertip spread; thumb–pinky fingertip angle | 4 |
| per hand — ratios | (thumb-tip↔index-tip), (index-tip↔middle-tip), (thumb-tip↔pinky-tip), each ÷ palm length | 3 |

Absent hand ⇒ that hand's 12-vector is zeroed (mirrors the presence-flag
convention). The inter-wrist distance is the only explicit "hands together"
signal — each hand is normalized on its own wrist, so it is what separates
`mini_heart` (hands meet) from `t_pose` / `raise_*` (hands apart).

*(Code note: `derived.py` has a stale inline comment reading "12 + 24 = 36"; the
computed `DERIVED_DIM` is **37** — `N_BODY_DERIVED = 8 + 5 = 13`.)*

---

## 4. Feature-assembly modes (L5) — `features/derived.py::to_features`

| mode | composition | dim |
|---|---|---|
| `raw` | x (152) | 152 |
| `derived` | 2 presence flags + d (37) | 39 |
| **`both`** *(LOCK)* | x (152) ∥ d (37) | **189** |
| `body_raw_hands_derived` | body coords (66) + 2 flags + d (37) | 105 |

All models below are trained and reported on **`both`** unless noted. Track B
(§9.2) is the ablation that fixed this.

**Clipping (L5, `classifier/train.py::CLIP`).** Every feature is clipped to
**±10** before it reaches any model — shoulder-hip normalization blows up when the
hips leave the frame, and one un-clipped row can dominate a tree split or an SVM
kernel.

---

## 5. Per-model input transform (L6)

| model family | L6 transform |
|---|---|
| RF, ExtraTrees, LightGBM, HistGB, CatBoost | none (trees are scale-invariant) |
| LogisticRegression, RBF-SVM | `sklearn.StandardScaler` (fit on train) inside a `Pipeline` |
| MLP, SE-MLP | z-score `(z − μ)/(σ + 1e-6)`, μ/σ stored on the estimator from the train split |

---

## 6. Classifier heads (L7)

All heads map **ℝ¹⁸⁹ → 12 class scores**. Hyperparameters are verbatim from
`classifier/train.py` (`build_*`). Every head receives class weights (§8).

### 6.1 RandomForest — Phase 3 working default

```
L7  RandomForestClassifier
      n_estimators        = 300         (300 CART trees, bootstrap samples)
      criterion           = gini
      max_depth           = None        (grown until pure / min_samples_leaf)
      min_samples_leaf    = 2
      max_features        = sqrt(189) ≈ 13   per split
      class_weight        = balanced × WEIGHT_MULT
L8  probability = mean of per-tree leaf class-frequency vectors   (no calibration)
```

Phase 3: macro-F1 **0.84**, 10-class real-eval 0.81, ECE 0.148, **419 MB**.
Phase 4 (finalised dataset): regressed — macro-F1 0.823, `laying` collapsed to
0.41, **389 MB**. Under-confident and spread (conf 0.77 correct / 0.44 wrong).

### 6.2 ExtraTrees

```
L7  ExtraTreesClassifier
      n_estimators        = 300
      min_samples_leaf    = 2
      splits              = random thresholds (not optimised), no bootstrap
      class_weight        = balanced × WEIGHT_MULT
L8  mean of per-tree class-frequency vectors   (no calibration)
```

Phase 4: macro-F1 0.797, real-eval 0.757, ECE 0.172, **998 MB**. Same size
liability as RF, no accuracy or calibration gain. Dropped.

### 6.3 LightGBM — Phase 3 alternate, Phase 4 accuracy leader

```
L7  LGBMClassifier   (objective = multiclass, num_class = 12)
      n_estimators        = 400         (400 boosting rounds × 12 one-vs-rest trees)
      learning_rate       = 0.05
      num_leaves          = 63          (leaf-wise growth)
      min_child_samples   = 20
      subsample           = 0.8  (freq 1)   ·   colsample_bytree = 0.8
      class_weight        = balanced × WEIGHT_MULT
L8  softmax over summed leaf values           (no calibration)
```

Phase 4: **highest on every headline metric** — macro-F1 0.888, real-eval 0.866,
accuracy 0.923, ECE 0.036, 31 MB. **Rejected for the lock** on one axis: its
confidence histogram piles ~80 % of predictions into the top bin (mean 0.95
correct vs 0.80 wrong — a 0.15 gap), which is too tight for the Phase 5
"reject below T → idle/unknown" rule without a separate Platt/isotonic step.

### 6.4 HistGradientBoosting (sklearn, zero new dependency)

```
L7  HistGradientBoostingClassifier
      max_iter            = 400
      learning_rate       = 0.05
      max_leaf_nodes      = 63
      l2_regularization   = 1.0
      early_stopping      = True   (internal validation fraction)
      class_weight        = balanced × WEIGHT_MULT
L8  softmax over summed leaf values           (no calibration)
```

Phase 4: macro-F1 0.843, real-eval 0.811, **best ECE of any candidate (0.022)**,
23 MB. Well-calibrated but 5.5 real-eval points behind LightGBM and its
confidence spread doesn't beat the MLP. Not competitive on the deciding axes.

### 6.5 CatBoost

```
L7  CatBoostClassifier   (loss = MultiClass)
      iterations          = 400
      learning_rate       = 0.06
      depth               = 6           (symmetric / oblivious trees, 64 leaves each)
      ordered boosting    = on          (permutation-based, anti prediction-shift)
      class_weights       = balanced × WEIGHT_MULT
L8  softmax over summed leaf values           (no calibration)
```

Screening config (depth-6/400). Phase 4: macro-F1 0.821, real-eval 0.786,
ECE 0.091, **2.7 MB** (oblivious trees serialise tiny). Smallest tree model but
8 real-eval points behind LightGBM at the screening config — not accurate enough
to justify tuning.

### 6.6 RBF-SVM (subsampled)

```
L6  StandardScaler
L7  SVC(kernel = rbf)
      C                   = 4.0
      gamma               = "scale"  =  1 / (189 · Var(z))
      training set        = stratified subsample to 12 000 rows  (SVC is ~O(n²))
      class_weight        = balanced × WEIGHT_MULT
L8  Platt scaling — SVC(probability=True): internal 5-fold CV sigmoid fit
L9  decision = argmax of the 1-vs-1 vote, probability from the Platt layer
```

Phase 4: macro-F1 0.811, real-eval 0.775, ECE 0.056, 11 MB. A genuinely
different inductive bias (geometric margin), but the 12 k subsample throws away
augmented data the trees exploit and the rare classes suffer.

### 6.7 LogisticRegression — the linear floor

```
L6  StandardScaler
L7  LogisticRegression(multinomial, C = 1.0, max_iter = 2000)
      class_weight        = balanced × WEIGHT_MULT
L8  native softmax                            (no extra calibration)
```

Phase 4: macro-F1 0.667, real-eval 0.610. **Purpose: prove the problem is not
linearly separable** — a 25-point gap to LightGBM confirms a non-linear head is
required. Not a contender.

### 6.8 MLP — **LOCKED** (`classifier/mlp.py::TorchMLP`)

PyTorch, CPU, `hidden = (256, 128)`, `dropout = 0.3`. Input is the z-scored
189-vector.

```
 L6   z-score            z' = (z − μ) / (σ + 1e-6)                     ℝ¹⁸⁹
 ───  block 1
 L7.1 Linear             189 → 256
 L7.2 BatchNorm1d        256
 L7.3 ReLU
 L7.4 Dropout            p = 0.3
 ───  block 2
 L7.5 Linear             256 → 128
 L7.6 BatchNorm1d        128
 L7.7 ReLU
 L7.8 Dropout            p = 0.3
 ───  head
 L7.9 Linear             128 → 12                                     logits ∈ ℝ¹²
 L8   Temperature        p = softmax(logits / T),  T scalar (§7)
 L9   argmax → (class, max p)
```

≈ 8.5 × 10⁴ trainable parameters, **0.3 MB on disk** (100× smaller than LightGBM).

**Loss.** `CrossEntropyLoss(weight = class_weight, label_smoothing = 0.05)`.
**Optimizer.** AdamW, `lr = 1e-3`, `weight_decay = 1e-4`, `batch_size = 512`,
`max_epochs = 200`, **early stopping** on val loss, `patience = 15`, best
state restored. If no val split is passed, 10 % is carved internally.

**Why it is the lock (`docs/phase4_baseline.md`).** Accuracy is statistically
tied with LightGBM — macro-F1 0.872 (5-seed mean 0.881 ± 0.014), real-eval 0.847
(5-seed 0.857 ± 0.016), inside one SD of LightGBM's 0.866. The deciding axis:
**confidence separates right from wrong out of the box** — mean 0.85 correct vs
**0.48 wrong** (0.37 gap), histogram spread 0.2–1.0, so a single ~0.5 threshold
flags most errors and keeps the correct ones, with no extra calibration step.
Higher ECE (0.103) is temperature scaling leaving it slightly *under*-confident —
the safe direction. Committed model is **seed 0** (a mid-range seed, no
cherry-pick).

### 6.9 SE-MLP — Track C follow-up, **not adopted** (`classifier/semlp.py`)

Feature-wise squeeze-and-excitation gate before the MLP trunk (Hu et al., CVPR
2018, adapted to a flat tabular vector). Subclass of `TorchMLP` — identical
`fit` / `predict_proba` / temperature surface.

```
 L6    z-score                                                       z' ∈ ℝ¹⁸⁹
 ───   SE gate branch
 L6a   Linear              189 → 31        (bottleneck, ratio r = 6, min 8)
 L6b   ReLU
 L6c   Linear              31 → 189        (weight init 0, bias init 2.0
                                            ⇒ sigmoid(2) ≈ 0.88, starts ≈ identity)
 L6d   Sigmoid             → g ∈ (0,1)¹⁸⁹
 L6e   floor              g' = floor + (1 − floor)·g,   floor = 0.5
 L6f   gate              z'' = z' ⊙ g'         (features damped ≤ 50 %, never zeroed)
 ───   trunk = unchanged MLP L7.1 … L7.9
 L8    Temperature scaling
```

**Result:** a genuine tie with the plain MLP on accuracy (real-eval 0.859 ±
0.018 at floor 0.5) but worse `idle` robustness — `idle` F1 sd 0.23 vs the plain
MLP's 0.013, and a hard gate (floor 0) collapses `idle` recall on 4/5 seeds. Not
worth a new hyperparameter and a second module for a tie. **Kept as evidence:**
the learned per-class gates confirm the schema's feature-block design —
`raise_right_hand` holds its right-hand block open (0.72) vs the left (0.56),
`raise_left_hand` the mirror, `ok`/`two_finger` favour the right hand, postures
lean on the joint-angle block. Committed variant: floor 0.5, seed 0.

### 6.10 Not built

k-NN (undeployable — the "model" is the 55 k-row training set); XGBoost (too
close to LightGBM to earn the boosting slot); FT-Transformer / TabNet / SAINT
(need more data + a bigger honest test than exist); spatial GCN (highest upside
but real overfitting risk on the ~1.25 k-row test — deferred). Stacking / voting
ensembles are forbidden by the one-slot architecture rule.

---

## 7. Calibration head (L8) — temperature scaling

Applied to the MLP / SE-MLP only (tree models emit their leaf-frequency softmax
directly). Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.

```
after early stopping, on the held-out val split:
  logits_val  = net(z_val)
  T*          = argmin_T  NLL( softmax(logits_val / T), y_val )
                1-D grid search, T ∈ linspace(0.5, 5.0, 46)
inference:
  p(x) = softmax( net(z(x)) / T* )
```

One scalar, **does not change the argmax** (accuracy untouched), fixes raw-softmax
over-confidence. `T*` is stored in the bundle and in `<model>.meta.json`.

Post-hoc alternatives evaluated for LightGBM but not wired in: Platt
(`CalibratedClassifierCV(method="sigmoid")`) and isotonic — deferred to a
Track E re-run if the lock ever moves back to LightGBM.

---

## 8. Class weighting (all heads)

`classifier/train.py::_weights`:

```
w[c] = sklearn "balanced"(c)  ×  WEIGHT_MULT[c]
WEIGHT_MULT = { squat: 0.35, glico_pose: 0.35, idle: 0.6, else: 1.0 }
```

`squat` and `glico_pose` train on **augmented s01 copies only** (no external
data) — the down-weight forces the head to demand strong evidence before
predicting a class it has only ever seen as self-augmentation. `idle` is
down-weighted because it otherwise over-triggers (recall ~0.98, precision ~0.46
in Phase 3).

---

## 9. Training data & recipe

### 9.1 Splits (`pipeline/build_dataset.py`)

| split | rows | composition | never |
|---|---|---|---|
| `train` | 52 738 | external datasets, cleaned → augmented (+ aug-only s01 copies for `squat`/`glico_pose`) | contains no raw s01/s02 frame |
| `val` | 2 357 | 15 % of each external class's **clean, pre-augmentation** rows | augmented; in train |
| `test` | 1 254 | **every** original s01/s02 frame + 15 % held-out HaGRID subjects for `thumb` | augmented |

Only `test` classes tagged `cross_domain` or `held_out_external` are real
generalisation numbers ("real-eval" = macro-F1 over those). `squat`, `laying`
(no s01), `glico_pose` are `aug_only_leak` — train/test overlap; `sit`@s02 is a
clean cross-person check because its augmentation is s01-only. Phase 6 field
recording is the real gate. Per-class sources: `docs/phase2b_external_data.md`.

**External sources:** HaGRID v1/v2 (`ok`, `peace`, `like`, `no_gesture`,
`hand_heart`), COCO 2017 (`idle` negatives, `t_pose`, `raise_right_hand`, and the
`sit`/`laying` posture mine via `_classify_coco` + re-verification in
`_clean_external`). Extraction is streaming — keypoints only, raw images deleted
per shard (dev box has ~5 GB free).

### 9.2 Augmentation (L-pre, `features/augment.py::AugParams`) — train only

| step | param | value |
|---|---|---|
| mirror (+ L/R relabel & slice swap) | `mirror_p` | 0.5 |
| limb-length jitter (arm / leg / torso chains, pelvis-rooted) | `limb_jitter` | ±0.15 |
| rotation jitter (camera tilt) | `rot_deg` | ±12° |
| keypoint dropout | `kp_dropout_frac` | 0.05 |
| whole-hand drop (+ clear presence flag) | `hand_drop_p` | 0.10 |
| coordinate noise (normalized units) | `coord_noise_std` | 0.02 |
| copies per original | `n_per_sample` | 4 (scarce classes up to ~40 to reach ≈2 200 rows) |
| arm elevation (opt-in: `mini_heart`) | `elevate_dy` | U(−1.1, −0.3) |

**Mirror is a hard constraint (CLAUDE.md #2):** flip x, swap L/R body landmark
pairs, swap the L/R hand slices **and** presence flags, relabel
`raise_right_hand ↔ raise_left_hand`. `augment.py::demo()` asserts double-mirror
is identity and the relabel fires.

**Arm-elevation** (`raise_arms`) rigidly shifts both forearms + hand points up by
a shared `dy`, turning HaGRID's chest-level finger-heart into s01's overhead
`mini_heart` while keeping the hands together. Hand slices are wrist-local so the
handshape is untouched. Took `mini_heart` from 0.00 → 0.73–0.93 F1.

### 9.3 Config knobs

| knob | value | where |
|---|---|---|
| feature mode | `both` (189-d) | `train.py --features` |
| feature clip | ±10 | `train.py::CLIP` |
| MLP seed (lock) | 0 | `build_mlp` |
| build seed | 20260831 | `build_dataset.py::SEED` |
| augmentation seed offset | `class_index(cls)` | reproducible per-class |

---

## 10. Results

### 10.1 Phase 3 — baseline (12 classes, `both`)

| metric | RandomForest *(default)* | LightGBM |
|---|---|---|
| macro-F1 (12) | **0.84** | 0.83 |
| macro-F1 (10 real-eval) | **0.81** | 0.80 |
| accuracy | **0.88** | 0.87 |
| ECE | 0.148 | **0.061** |
| model size | 419 MB | **31 MB** |

Phase 3 also cut 3 classes (15 → 12): `i_love_you`, `rock` (MediaPipe Hands
can't resolve them vs `two_finger` on conversational-distance footage — same
feature vector), `heart` (no dataset). Cutting `rock` fixed `two_finger`
(0.72 → 0.94). `sit` was de-leaked from COCO: cross-person recall 0.53 → **0.97**.

### 10.2 Phase 4 — candidate comparison (finalised dataset, scored on `test`)

| model | macro-F1 | real-eval | acc | ECE | conf ok / wrong | size | fit |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.888** | **0.866** | **0.923** | 0.036 | 0.95 / 0.80 | 31 MB | 104 s |
| **MLP + temp-scale** *(LOCK)* | 0.872 | 0.847 | 0.915 | 0.103 | 0.86 / **0.48** | **0.3 MB** | 102 s |
| HistGradientBoosting | 0.843 | 0.811 | 0.899 | **0.022** | 0.90 / 0.63 | 23 MB | 160 s |
| RandomForest | 0.823 | 0.788 | 0.868 | 0.146 | 0.77 / 0.44 | 389 MB | 43 s |
| CatBoost (depth-6/400) | 0.821 | 0.786 | 0.881 | 0.091 | 0.93 / 0.53 | **2.7 MB** | 307 s |
| RBF-SVM (12k subsample) | 0.811 | 0.775 | 0.848 | 0.056 | 0.86 / 0.55 | 11 MB | 80 s |
| ExtraTrees | 0.797 | 0.757 | 0.867 | 0.172 | 0.79 / 0.53 | 998 MB | 29 s |
| LogisticRegression | 0.667 | 0.610 | 0.785 | 0.083 | 0.88 / 0.52 | ~0 | 31 s |

### 10.3 Phase 4 ablations (all on the locked MLP)

- **Track B — feature set.** `both` (189-d) vs `body_raw_hands_derived` (105-d):
  the reduced set fixes `mini_heart` (0.78 → 0.89) but destabilises the
  CLAUDE.md-mandated `idle` (F1 sd 0.31, collapses to ~0.21 on 2/5 seeds) and
  misses the real-eval bar. **Lock stays `both`.**
- **Track C — SE-MLP.** §6.9 — tie on accuracy, worse `idle` robustness. Not
  adopted.
- **Track D — augmentation strength.** 4 variants rebuilt; **none beats the
  defaults**. More augmentation washes out class boundaries; `mini_heart` is the
  most fragile. `AugParams` defaults retained.
- **Track E — confidence thresholds.** Per-class max-prob thresholds fit on
  `val` **do not transfer** to `test` (val confidences run ~0.2 high — same
  webcam domain as train). Only a garbage floor `global_floor = 0.20`
  (≈1 frame affected) is active. Real per-class thresholds wait for Phase 6.

---

## 11. Locked configuration (Phase 4 exit)

```
feature mode      both            x(152) ∥ derived(37)  =  189-d,  clip ±10
input transform   z-score         (z − μ)/(σ + 1e-6),  μ/σ from train
head              TorchMLP        189 → [Linear 256 → BN → ReLU → Drop 0.3]
                                      → [Linear 128 → BN → ReLU → Drop 0.3]
                                      → Linear 12
loss              CrossEntropy    class-weighted, label_smoothing 0.05
optimizer         AdamW           lr 1e-3, wd 1e-4, batch 512, early stop patience 15
calibration       temperature T*  argmin val-NLL, grid [0.5, 5.0]×46, argmax-preserving
seed              0
artifact          classifier/models/mlp.joblib   ({clf, classes, clip="10.0", features="both"})
                  + mlp.meta.json  + mlp.thresholds.json (global_floor 0.20, per-class inactive)
downstream        Phase 5: confidence floor 0.20 → temporal majority vote N = 7
```

Remaining weak classes — `mini_heart` 0.73, `raise_*_hand` 0.76, `idle`
precision 0.62 on `test` — are data / feature-gap limited, not
hyperparameter-limited. Phase 6 field recordings are the next lever.
