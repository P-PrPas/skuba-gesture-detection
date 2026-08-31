# Phase 3 — baseline classifier

`classifier/train.py` (LightGBM + RandomForest), `classifier/evaluate.py`.
Train: `data/dataset/train.npz` (62,627 rows, hybrid — see
`docs/phase2b_external_data.md`). Test: `data/dataset/test.npz` (1,581 rows —
1,367 original s01/s02 frames + 214 held-out HaGRID `thumb`, never augmented).

Numbers mean different things per class (`dataset_card.json` `eval_type`):
**cross_domain** = external train, our test → a real generalisation number;
**held_out_external** = cross-subject within HaGRID; **aug_only_leak** = train is
augmented s01 frames → not a generalisation number.

## Results (RandomForest; LightGBM ≈ same)

| class | eval | F1 | note |
|---|---|---|---|
| raise_left_hand | cross_domain | 0.90 | |
| raise_right_hand | cross_domain | 0.81 | ~20% → idle |
| t_pose | cross_domain | 0.84 | |
| two_finger | cross_domain | 0.81 | ~25% → i_love_you |
| ok | cross_domain | 0.77 | recall 0.64 |
| thumb | held_out_external | 0.97 | same domain (HaGRID) |
| **rock** | cross_domain | **0.00** | all 102 → `i_love_you` |
| **mini_heart** | cross_domain | **0.00** | 40/45 → `heart` |
| idle | cross_domain | 0.65 | precision 0.48 — over-triggers |
| sit | aug_only | 0.98 | **s02 (clean): recall 0.77**, s01 (leak): 1.00 |
| squat / laying / glico_pose | aug_only | 1.00 | leak |
| heart | aug_only | 0.81 | precision 0.68 |
| i_love_you | aug_only | 0.69 | precision 0.53 — absorbs rock/two_finger/ok |

macro-F1: all 0.75, real-eval-only (9 classes) **0.64**.

## Why rock and mini_heart are 0.00

1. **s01-attractor effect.** `i_love_you` and `heart` are trained only on
   augmented s01 frames, so the model over-fits s01's specific hand/body and any
   *other* s01 test frame that is geometrically near gets pulled in. s01's `rock`
   and `mini_heart` are near s01's `i_love_you` / `heart`.
2. **Hand-landmark domain gap.** On s01's `rock` the normalised fingertip spread
   (thumb–pinky ≈ 1.3) is ~2× HaGRID's `rock` (≈ 0.5) — palm-width normalisation
   does **not** close it. MediaPipe Hands on a tight wrist-crop of a distant
   robot-camera hand produces a differently-proportioned skeleton than on a
   large frontal webcam hand. So HaGRID-trained `rock` doesn't recognise
   s01-`rock`, and the nearest thing it knows is s01-trained `i_love_you`.
3. **mini_heart body position.** HaGRID `hand_heart` is chest-level; s01
   `mini_heart` is hands-together overhead — matches s01 `heart` (also overhead).

## Phase 3 verdict: NOT passed

Two classes at 0.00 is below any bar. Phase 4 items, in priority order:

1. **Close the hand-landmark domain gap.** Options: add hand-scale + hand-rotation
   jitter to `features/augment.py`; canonicalise hand orientation in
   `normalize_hand`; re-check `wrist_crop_box` sizing against s01 vs HaGRID.
2. **`i_love_you` / `heart` need real data or isolation.** As aug-only s01
   classes they poison their neighbours. Either find external data (ILY: none
   found; `heart`: HaGRID `hand_heart` is the wrong pose) or down-weight them /
   raise their decision threshold so they stop absorbing `rock` / `two_finger`.
3. **`mini_heart`**: drop HaGRID (wrong body position) and treat as aug-only,
   or add overhead-position augmentation.
4. **`idle` over-triggers** — the `_ily_negative`→idle rows + COCO idle may be
   too aggressive; tune the idle threshold on a validation slice.
5. Compare derived features (joint angles, inter-point distances) vs raw
   coordinates — IMPLEMENTATION_PLAN Phase 4. Angles are scale-free and would
   sidestep the hand-scale gap.
