# Phase 4 — classification-layer candidates

Brainstorm + literature notes for the model-family comparison (Track C). The
input is a single-frame 152-d (or 189-d with derived features) fused
keypoint vector; 12 classes; ~55k augmented training rows; a **small
(~1,250-row) never-augmented cross-domain test** that is easy to overfit;
CPU / low-VRAM deployment. Temporal models are out of scope (Phase 5).

> A background research pass was started for primary-source verification but hit
> a session rate limit. This is the in-house synthesis; paper claims are cited by
> title/venue and flagged where a docs check is still owed.

## Verdict

The baseline (RandomForest) is already good on accuracy; its weaknesses are
**size (419 MB)** and **under-confidence**, and LightGBM's weakness is
**un-thresholdable confidence**. So Phase 4 should hunt for a model that keeps
~0.83 macro-F1 while being small and well-calibrated — not for a big accuracy
jump (the honest test is too small to trust a 1-2 point gain). Treat
**calibration as a separable post-hoc layer** so "which model" and "is it
calibrated" are answered independently.

## Ranked shortlist (worth the experiment time)

1. **CatBoost** — same GBDT family as LightGBM but ordered boosting gives better
   out-of-box probabilities; compact model; drop-in. Directly targets the
   LightGBM calibration problem. *Lowest effort, highest expected value.*
2. **sklearn HistGradientBoostingClassifier** — no new dependency, fast, a free
   third GBDT data point.
3. **Small MLP (PyTorch: 2-3 layers, dropout + batchnorm + label smoothing) +
   temperature scaling** — the plan's "MLP", done properly. Tiny, fast,
   calibrates cleanly with one parameter.
4. **RBF-SVM on a ~15-20k subsample** — a genuinely different inductive bias
   (geometric margin, not tree partitions); very small model. Slower to fit with
   probability calibration; worth one run.
5. *(stretch)* **Spatial Graph Convolutional Network** — the skeleton is a
   graph; this is the domain-standard architecture. Highest upside, but real
   effort and a real overfitting risk on the 1,250-row test.

## Comparison table

| model | library | ~model size | ~inference | calibration (out-of-box) | effort | main risk |
|---|---|---|---|---|---|---|
| RandomForest *(baseline)* | scikit-learn | 419 MB | fast | under-confident, **spread** | — | size |
| LightGBM *(baseline)* | lightgbm | 31 MB | fast | piled at ~1.0, not thresholdable | — | calibration |
| **CatBoost** | catboost | ~5-30 MB | fast | best of the GBDTs (ordered boosting) | low (drop-in) | marginal accuracy delta only |
| **HistGradientBoosting** | scikit-learn (native) | ~5-20 MB | fast | similar to LightGBM | very low | little new signal vs LGBM |
| ExtraTrees | scikit-learn | like RF (big) | fast | often better than RF | very low | still large |
| **MLP + temp-scaling** | PyTorch | <1 MB | very fast | poor raw, **excellent after temp-scaling** | medium | needs a val split; instability |
| **RBF-SVM (subsampled)** | scikit-learn | few MB | medium | Platt (built into `SVC(probability=True)`) | medium | O(n²) fit; subsample hurts rare classes |
| LinearSVC / LogReg | scikit-learn | tiny | very fast | LogReg native; LinearSVC needs calibration | very low | a floor, not a contender |
| k-NN | scikit-learn | = training set (huge) | slow | frequency-based, coarse | very low | undeployable — diagnostic only |
| XGBoost | xgboost | ~20-40 MB | fast | between LGBM and CatBoost | low | too close to LightGBM |
| **Spatial GCN** | PyTorch (hand-rolled or PyG) | ~1-5 MB | fast on CPU | poor raw, temp-scalable | high | overfits the small test; adjacency design |
| FT-Transformer / TabNet / SAINT | pytorch-tabular etc. | 5-50 MB | medium | model-dependent | high | needs more data + tuning than we have |

---

## 1. Other gradient boosting

### CatBoost
- *CatBoost: unbiased boosting with categorical features*, Prokhorenkova et al.,
  NeurIPS 2018. The core idea is **ordered boosting**: each training example's
  residual is computed with a model trained on examples that precede it in a
  random permutation, removing the "prediction shift" (a form of target leakage)
  that ordinary GBDT suffers. The practical, widely-reported consequence is
  **less over-confident probability output** than XGBoost/LightGBM. *(Verify the
  exact calibration wording against catboost.ai/docs before relying on it in the
  report — the mechanism is from the paper; the "better-calibrated" claim is a
  community-consensus practical result.)*
- `CatBoostClassifier(loss_function="MultiClass")`, `.predict_proba`,
  `.classes_` — plugs into the `{clf, classes, clip, features}` bundle unchanged.
- Model size: the trees are symmetric (oblivious) — every split at a depth uses
  the same feature/threshold — so the model serialises much smaller than an RF
  and comparably to LightGBM.
- Effort: **low**. New dependency `catboost` (single wheel, no CUDA needed).
- Risk: on pure accuracy it will land within noise of LightGBM; the value is the
  calibration + size, so judge it on ECE and conf-correct-vs-wrong, not F1.

### sklearn HistGradientBoostingClassifier
- Native since scikit-learn 0.21; histogram-based, inspired by LightGBM
  (sklearn docs, "Histogram-Based Gradient Boosting"). Supports
  `class_weight`, `early_stopping`, missing values.
- **No new dependency**, fast to fit, `.predict_proba` available.
- Effort: **very low** — a `build_hgb()` in `train.py`.
- Risk: likely behaves like LightGBM (same family, same calibration profile), so
  mostly a sanity data-point rather than a contender.

### XGBoost
- *XGBoost: A Scalable Tree Boosting System*, Chen & Guestrin, KDD 2016.
  Level-wise growth (vs LightGBM's leaf-wise), different regularisation.
- Calibration profile sits between LightGBM and CatBoost in practice.
- Effort low, but it is close enough to LightGBM that CatBoost is the better use
  of a boosting slot. **Include only if CatBoost + HGB both disappoint.**

---

## 2. Kernel / margin methods

### RBF-SVM
- `sklearn.svm.SVC(kernel="rbf", class_weight=..., probability=True)`.
- Fits an implicit non-linear boundary via the kernel; historically a very
  strong baseline for pose / keypoint classification because normalized
  landmark vectors live on a low-dimensional manifold where a geometric margin
  is meaningful. Different inductive bias from every tree model — worth having
  one such point in the comparison.
- **Cost**: training is between O(n²) and O(n³) in the number of samples; at 55k
  rows it is impractical. Subsample the training set to ~15-20k (stratified,
  keeping rare classes fully). `probability=True` fits Platt scaling with an
  internal 5-fold CV on top of the SVM — roughly 6× the base fit cost.
- **Model size**: only the support vectors are stored (support_vectors_ ×
  feature_dim + duals). Usually a few MB — a big win over RF.
- Effort: **medium** (feature scaling — SVM needs standardised inputs, our
  features are already ~unit-scaled but clipping to ±10 leaves outliers;
  a `StandardScaler` in the pipeline; the subsample logic).
- Risk: subsampling throws away most of the augmented data the trees exploit;
  rare classes (t_pose n≈2.3k, mini_heart n≈5.6k after aug) may suffer.

### LinearSVC / Logistic Regression
- `LogisticRegression(class_weight="balanced", multi_class="multinomial")` —
  native `predict_proba`, tiny, instant. `LinearSVC` is faster to fit but has no
  `predict_proba` (wrap in `CalibratedClassifierCV`).
- Purpose: a **floor**. If a linear model gets within a few points of the trees,
  the feature space is nearly linearly separable and the fancy models are
  overkill. Run once, report, move on.

---

## 3. Instance-based — k-NN

- `KNeighborsClassifier` on the normalized keypoint vector = skeleton **template
  matching**. Pose-recognition tutorials and several MediaPipe sample apps
  (e.g. Google's "MediaPipe Pose Classification" Colab, which uses a k-NN over
  normalized-and-flattened landmarks) do exactly this and get usable accuracy.
- **Undeployable here**: the "model" is the entire training set (~55k × 189
  floats ≈ 40 MB in RAM, plus a slow per-frame nearest-neighbour search) — worse
  than the RF on both axes.
- **Use it once as a diagnostic**: k-NN accuracy is a lower bound on how
  separable the classes are in this feature space. If k-NN already hits ~0.80,
  the problem is "easy" and model choice should be driven by size/calibration,
  not accuracy.

---

## 4. Tree variants — ExtraTrees

- `ExtraTreesClassifier`: like RandomForest but split thresholds are drawn at
  random (not optimised) and, by default, no bootstrap — more variance
  reduction, often marginally better accuracy, and frequently better-calibrated
  than RF because the extra randomisation softens the vote.
- **Same size problem as RF** — it still stores N full trees. Only interesting
  if paired with aggressive depth/leaf limits.
- Effort: **very low** (`build_et()` mirroring `build_rf()`).

---

## 5. Neural, single-frame

### Properly-regularised MLP + temperature scaling
- Architecture: `input → [Linear → BatchNorm → ReLU → Dropout] × 2-3 → Linear →
  softmax`, ~128-256 hidden units. A few hundred k parameters, **<1 MB on disk**,
  microsecond CPU inference.
- Regularisation that matters for the small honest test: dropout (0.2-0.4),
  weight decay, **label smoothing** (0.05-0.1) — label smoothing both regularises
  and pre-empts over-confidence. Early stopping on a **validation split** (we
  must create one — Phase 4 prerequisite).
- Calibration: raw softmax from a trained net is typically over-confident
  (*On Calibration of Modern Neural Networks*, Guo et al., ICML 2017). Their
  fix — **temperature scaling**: divide logits by a single scalar T learned on
  the validation set by minimising NLL. One parameter, does not change the
  argmax (so accuracy is untouched), and is very effective. This is the cleanest
  calibration story of any candidate.
- `sklearn.neural_network.MLPClassifier` is the zero-dependency option but has
  no dropout/batchnorm/label-smoothing and no logit access for temp-scaling —
  so a ~40-line PyTorch module is the right call.
- Effort: **medium**. Risk: training instability on imbalanced data; needs the
  val split and a couple of seeds.

### Spatial Graph Convolutional Network (spatial-only ST-GCN)
- *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
  Recognition*, Yan, Xiong, Lin, AAAI 2018. The spatial graph convolution: for
  each joint, aggregate neighbour features weighted by a learned weight matrix
  **per graph partition** (the paper's best partitioning splits neighbours into
  root / centripetal / centrifugal by distance to the skeleton's centre of
  gravity). Full ST-GCN then stacks temporal 1-D convs on top — **we drop those**
  and keep: spatial-GCN blocks → global average pool over joints → FC → softmax.
- **Adjacency for our skeleton** (~75 nodes): MediaPipe Pose 33-landmark edges
  (`mp.solutions.pose.POSE_CONNECTIONS` equivalent) + 21-edge hand skeleton ×2 +
  **bridge edges** connecting body left-wrist(15)↔left-hand-wrist(0) and
  body right-wrist(16)↔right-hand-wrist(0) so the graph is connected. Node
  feature = (x, y) normalized coords; a hand's nodes carry that hand's
  presence flag as a third channel (0 when absent → the GCN learns to
  down-weight it, mirroring how the trees treat the zeroed slice).
- Implementation: the adjacency is **fixed**, so a hand-rolled layer
  (`D^-1/2 (A+I) D^-1/2 X W`) is ~15 lines and avoids the PyTorch-Geometric
  dependency. Model can be small — a few 100k params, **1-5 MB**.
- Why it might win: it is the only candidate that uses the *structure* of the
  input instead of treating it as a flat vector; it should need far less data to
  learn "left arm up" as one concept.
- **Risk**: our honest test is ~1,250 rows. A GCN has enough capacity to
  memorise the ~55k augmented train rows and still miss the cross-domain test.
  Heavy regularisation + the val split + temperature scaling are mandatory, and
  it may still lose to CatBoost on the number that matters. Highest effort of the
  shortlist (~half a day).

### 1D-CNN over the ordered landmark vector
- Treat the 152 values as a length-152 signal, `Conv1d` with small kernels to
  learn local groupings (adjacent landmarks in the vector are often the same
  body part). Lighter than a GCN, no graph to design.
- Likely lands near the MLP — the vector ordering only loosely reflects the
  skeleton topology, so the conv's locality prior is weak. **Low priority.**

### Tabular deep learning — FT-Transformer, TabNet, SAINT
- FT-Transformer (*Revisiting Deep Learning Models for Tabular Data*, Gorishniy
  et al., NeurIPS 2021) is competitive with GBDT **on larger tabular datasets**;
  the same paper shows GBDT still wins on many. TabNet (*TabNet: Attentive
  Interpretable Tabular Learning*, Arik & Pfister, AAAI 2021) uses sequential
  attention for feature selection but in independent benchmarks often
  underperforms tuned GBDT and is hyperparameter-sensitive.
- For this project the blocker is **evaluation size**: a 1,250-row honest test
  cannot distinguish these from the GBDTs, and their tuning burden is high.
- **Not worth it now** — revisit only if the class count grows a lot and a
  much larger real test set exists.

---

## 6. Post-hoc calibration (a separable layer)

Applies on top of whichever model wins the accuracy comparison:

| method | tool | fits on | notes |
|---|---|---|---|
| Platt scaling (sigmoid) | `CalibratedClassifierCV(method="sigmoid")` | held-out val (CV or prefit) | 1-2 params/class; robust on small val sets; built into `SVC(probability=True)` |
| Isotonic regression | `CalibratedClassifierCV(method="isotonic")` | held-out val | non-parametric, more flexible, **overfits small val sets** — risky here |
| Temperature scaling | ~10 lines, learn scalar T on val by NLL | val logits | neural-net standard (Guo et al. 2017); does not change argmax; needs logit access |

Plan: pick the model on **raw** accuracy + separability, then apply the
appropriate calibrator and re-score ECE / conf-correct-vs-wrong. A model that is
accurate but uncalibrated (LightGBM) is not disqualified if a cheap calibrator
fixes it.

---

## Prerequisite: a validation split

There is no validation set today — `train.npz` / `test.npz` only. Phase 4 needs
one for early stopping (MLP/GCN), temperature scaling, and per-class threshold
tuning, **without touching `test.npz`**. Options: (a) hold out a stratified
slice of the external training rows (cross-domain-ish, but same sources as
train), or (b) hold out one HaGRID subject-group and a slice of COCO. Simplest:
carve ~15% of the *cleaned external* rows before augmentation, per class,
tagged `val`, in `build_dataset.py`.

## Probably not worth it for this project

- **FT-Transformer / TabNet / SAINT** — need more data and a bigger honest test
  than we have; high tuning cost.
- **k-NN as a deployed model** — model = training set, slow inference. (Do run
  it once as a separability probe.)
- **XGBoost** — too similar to LightGBM to earn the boosting slot over CatBoost.
- **PointNet / Set Transformer / DeepSets** — permutation-invariance is not
  needed (our landmark order is fixed and meaningful); overkill.
- **Stacking / voting ensembles** — forbidden by the one-classifier-slot
  architecture decision (CLAUDE.md).
- **Full ST-GCN / any temporal model** — Phase 5's job.

## Suggested Track C sequence

1. `build_dataset.py`: add a stratified `val` split from the pre-augmentation
   external rows.
2. `train.py`: `--model {rf, lgbm, catboost, hgb, et, svm, mlp, logreg}` (+ `gcn`
   as a stretch), each emitting the same bundle.
3. Train + `evaluate.py --all` → the harness comparison table across all models.
4. Run k-NN once as a separability sanity check (not a candidate).
5. Take the top 2-3 by (accuracy AND size), apply Platt / temperature scaling,
   re-score calibration.
6. Tune augmentation strength (Track D) against the val split for the leading
   model.
7. **Lock one** with a written rationale in `docs/phase4_baseline.md`; regen the
   Phase 4 report from the harness outputs.
