# External training datasets

Research task: find publicly available datasets to **train** the gesture/posture
classifier, so the two-subject in-house recording (`s01`, `s02`) can be kept
entirely as a held-out **test** set (the leakage concern in
`ARCHITECTURE.md` → "Evaluating with a small subject pool").

Scope reminder: our feature vector is MediaPipe Pose (33) + MediaPipe Hands
(21/hand), normalized per `ARCHITECTURE.md`. An external dataset is only usable if
it is **(a)** RGB images/video with a visible person we can run our own MediaPipe
extraction on, or **(b)** keypoints already convertible to MediaPipe-33 / -21
topology. Datasets shipping only COCO-17 / OpenPose-18 / Kinect-25 skeletons with
no imagery are flagged as low value.

Last verified: 2026-08-30. Every license/size/class claim below is cited to a
primary source; where a primary source could not be reached, that is stated
explicitly.

---

## 1. One-paragraph verdict

External, **license-clean**, directly-usable training data exists for the hand
gestures **`ok`, `rock`, `two_finger`, `thumb`** and for **`idle`** negatives —
all from **HaGRID**, whose images are webcam/phone upper-body shots we can push
straight through our MediaPipe pipeline. Add **`mini_heart`** to that list *if*
HaGRID**v2** is used and its license question (below) is resolved. For the body
postures **`sit`, `squat`, `laying`** there is abundant RGB video (NTU RGB+D,
the fall-detection corpora, Kaggle gym-workout videos), but almost all of it is
research-/non-commercial-licensed or has **unverifiable** licensing, so it is
fine for development and cross-subject validation but is a **deployment-license
risk**. For **`raise_right_hand` / `raise_left_hand`** and **`t_pose`** there is
no purpose-built dataset, but they can be mined from general pose datasets
(**COCO** is the only clearly commercial-friendly one) or, for the raise-hand
pair, produced from one side plus mirror augmentation. For **`i_love_you`** (ASL
ILY handshape), **`heart`** (big two-arm overhead heart) and **`glico_pose`**
(running-man) there is **no credible external dataset at all** — these remain
dependent on `s01` with the single-subject leakage caveat, and should be the
priority for any further in-house recording.

---

## 2. Class → best external source

| Our class | Best external dataset | Format | License | Coverage |
|---|---|---|---|---|
| `idle` | HaGRID `no_gesture` (123,589 imgs) + COCO person crops | images | HaGRID custom (BY-SA-style) / COCO CC BY 4.0 | **good** |
| `raise_right_hand` | mine COCO / MPII for arm-above-shoulder pose | images | COCO CC BY 4.0 / MPII BSD (non-commercial) | partial |
| `raise_left_hand` | mirror-augment from `raise_right_hand` (schema `MIRROR_LABEL_SWAP`) | — | — | partial |
| `sit` | NTU RGB+D `sitting down` (A008); fall-dataset ADL clips | RGB video | NTU research-only; URFD CC BY-NC-SA | partial (NC) |
| `squat` | Kaggle "Gym Workout/Exercises Video" squat class; Fit3D | video | unverified (Kaggle) / Fit3D research-only | partial (license) |
| `laying` | UR Fall, Le2i, UP-Fall, Multiple-Cameras-Fall; NTU `falling` (A043) | RGB(+D) video | mostly CC BY-NC-SA / unverified | partial (NC) |
| `ok` | HaGRID `ok` | images | HaGRID custom | **good** |
| `i_love_you` | none clean (WLASL gloss = NC video); use HaGRID `call` as hard negative | video | WLASL research-only | **none** |
| `rock` | HaGRID `rock` | images | HaGRID custom | **good** |
| `two_finger` | HaGRID `peace` (+ `peace_inverted`, `two_up`) | images | HaGRID custom | **good** |
| `thumb` | HaGRID `like` | images | HaGRID custom | **good** |
| `heart` (overhead 2-arm) | none | — | — | **none** |
| `mini_heart` | HaGRID**v2** `hand_heart` + `hand_heart2` | images | HaGRIDv2 — license disputed (see §3.1) | good* |
| `t_pose` | Yoga-82 (Warrior II / star pose, partial); filter COCO for arms-horizontal | images | Yoga-82 NC + link-rot / COCO CC BY 4.0 | partial |
| `glico_pose` | none | — | — | **none** |

\* contingent on resolving the HaGRIDv2 license question.

---

## 3. Per-dataset detail

### 3.1 HaGRID / HaGRIDv2 — HAnd Gesture Recognition Image Dataset

- **Primary sources:**
  - Repo: https://github.com/hukenovs/hagrid
  - v1 paper (WACV 2024): https://arxiv.org/abs/2206.08219 , HTML
    https://arxiv.org/html/2206.08219v2
  - v2 paper: https://arxiv.org/abs/2412.01508 , HTML
    https://arxiv.org/html/2412.01508v1
  - License file: https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf
  - Class list (repo): https://github.com/hukenovs/hagrid/blob/master/constants.py

- **Format / size:**
  - **v1:** 552,992 FullHD (1920×1080) RGB images, 18 gesture classes + a
    `no_gesture` class of 123,589 images; ~716 GB full, ~lightweight 512px
    version also published. 37,583 subjects.
    (https://arxiv.org/html/2206.08219v2 , repo README)
  - **v2:** 1,086,158 FullHD RGB images, 33 gesture classes + `no_gesture`,
    ~1.5 TB full / 119.4 GB at min-side 512px, 65,977 unique persons.
    (https://arxiv.org/html/2412.01508v1 , repo README)
  - Annotations: COCO-format hand bounding boxes + gesture label +
    `leading_hand` / `leading_conf`; HaGRIDv2 also ships MediaPipe-style hand
    landmarks in later releases. We do **not** need their annotations — we run
    our own MediaPipe Pose + Hands on the raw images.

- **v1 classes (18):** `call, dislike, fist, four, like, mute, ok, one, palm,
  peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up,
  two_up_inverted` (+ `no_gesture`). (https://arxiv.org/html/2206.08219v2)

- **v2 classes (33 + `no_gesture`):** `grabbing, grip, holy, point, call,
  three3, timeout, xsign, hand_heart, hand_heart2, little_finger, middle_finger,
  take_picture, dislike, fist, four, like, mute, ok, one, palm, peace,
  peace_inverted, rock, stop, stop_inverted, three, three2, two_up,
  two_up_inverted, three_gun, thumb_index, thumb_index2` (+ `no_gesture`).
  (https://github.com/hukenovs/hagrid/blob/master/constants.py)

- **Mapping to our 15 classes:**

  | HaGRID label | Our class | Notes |
  |---|---|---|
  | `ok` | `ok` | direct |
  | `rock` | `rock` | direct |
  | `peace` | `two_finger` | direct; `peace_inverted` (palm back) and `two_up` are extra useful positives/variants |
  | `like` | `thumb` | direct — thumbs-up. (**This is the only external source for `thumb`, which has no `s01` data.**) |
  | `no_gesture` | `idle` | partial — "hand visible, no gesture"; good negatives but not full `idle` coverage (no whole-body idle) |
  | `hand_heart` (v2) | `mini_heart` | two-hands-together heart |
  | `hand_heart2` (v2) | `mini_heart` | one-hand finger-heart (Korean) |
  | `call` (shaka: thumb+pinky) | — | **NOT** ILY (ILY = thumb+index+pinky). Pull as a **hard negative** for `i_love_you`. |
  | none | `i_love_you` | HaGRID has no ILY handshape |

- **License — read carefully:**
  - The v1 paper states plainly: *"License: CC BY-SA 4.0"*
    (https://arxiv.org/html/2206.08219v2).
  - The actual repo LICENSE file is **not** a Creative Commons license. Its own
    first footnote (verbatim from the PDF): *"This license is not a Creative
    Commons license. The text of this license is a reworking of a Creative
    Commons Corporation (Attribution-ShareAlike 4.0) license … under the terms
    of the CC0."* The operative terms mirror CC BY-SA 4.0: a *"worldwide,
    royalty-free, non-sublicensable, non-exclusive, irrevocable license"* to
    reproduce and to create Adapted Material, **conditioned on** attribution
    (§3(a): keep creator info, copyright notice, license notice, link, mark
    changes) and **share-alike** on Adapted Material (§3(b)(1): *"the Adapted
    Material License … must be a license with the same License Elements … or a
    BY-SA-compatible license"*). **There is no non-commercial clause.** §2(b)(1)
    notes personal/publicity/privacy/portrayal rights of the depicted people are
    **not** granted by the license.
    (https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf , text
    extracted 2026-08-30)
  - **Conflict on HaGRIDv2:** the HaGRIDv2 paper page has been summarized
    (secondary reading of https://arxiv.org/html/2412.01508v1) as *"modified
    CC-BY 4.0 … for non-commercial research use"*, and the HuggingFace mirror
    `testdummyvt/hagRIDv2_512px` lists license = "other" pointing back to the
    repo. **This could not be reconciled from a single primary source.** Treat
    HaGRIDv2's commercial status as **unresolved**; HaGRID v1's license text
    (above) contains no NC restriction.
  - **Practical read for us:** we are not redistributing their images — we
    extract normalized keypoints and train a classifier. Under the v1 license
    that is Adapted Material: permissible with attribution + a BY-SA-compatible
    license on any dataset/artifact we redistribute. Shipping the *trained
    classifier weights* in a commercial robot is the grey area (share-alike
    reach + subjects' publicity rights). **Get a lawyer's read before
    deployment; v1-derived features for `ok`/`rock`/`two_finger`/`thumb`/`idle`
    are low-risk for internal R&D now.**

- **Download:** direct per-class archives + `python download.py --save_path
  <PATH> --annotations --dataset` from the repo; 512p lightweight version
  recommended (our pipeline downscales anyway). No registration. Also
  unofficial HuggingFace mirrors (`cj-mills/hagrid-sample-30k-384p`,
  `testdummyvt/hagRIDv2_512px`) — convenient but verify against the official
  release.

- **Domain gap:** crowdsourced (Yandex.Toloka, ABC Elementary) — subjects record
  themselves on phones/laptops/tablets indoors, gesture held 0.5–4 m from
  camera, hand bbox ≤16 % of frame; frontal, roughly eye-level webcam framing,
  upper body / half-figure visible (https://arxiv.org/html/2206.08219v2). Our
  robot camera is lower and often further — expect a **camera-height / pitch**
  gap. Hand is usually large and unambiguous here, vs. `phase1_report.md` noting
  hands are ~a small fraction of the frame at robot distance. Mitigation: our
  wrist-anchored crop + hand-local normalization already removes most of the
  scale difference; rotation/scale augmentation covers the rest.

### 3.2 Jester / 20BN-Jester

- **Primary sources:** dataset page
  https://www.qualcomm.com/developer/software/jester-dataset ; paper (ICCVW
  2019) https://openaccess.thecvf.com/content_ICCVW_2019/papers/HANDS/Materzynska_The_Jester_Dataset_A_Large-Scale_Video_Dataset_of_Human_Gestures_ICCVW_2019_paper.pdf
- **Format / size:** ~148,092 short RGB **video** clips (3 s, 12 fps, 100 px
  height), 27 classes; total ~22.8 GB. **Dynamic** gestures, not single-frame.
- **Classes overlapping ours (loosely):** `Thumb Up` (→ `thumb`), `Stop Sign`
  (open palm), `Thumb Down` — but these are performed *as motions* toward a
  webcam, framed as **hand-only close-ups** (the body is barely in frame).
- **License:** Qualcomm "Data License Agreement – Research Use". Per Qualcomm's
  research-use agreements, use is *"solely for non-profit research purposes"* and
  *"not … for any Commercial Purpose"*
  (https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qvid-dataset-research-license-mar-27-2025.pdf).
  Must accept terms before download.
- **Verdict: low value.** Non-commercial, dynamic-only, hand-only framing (no
  usable body pose), and its static-pose overlap with us is thin. Skip unless we
  later add dynamic/temporal gestures.

### 3.3 Fall-detection datasets (for `laying`, some `sit`)

| Dataset | Format | Size | Primary source | License |
|---|---|---|---|---|
| **UR Fall Detection (URFD)** | RGB + depth PNG sequences, 2 cams, + accelerometer | 70 sequences (30 falls / 40 ADL), Kinect, 30 fps | https://fenix.ur.edu.pl/~mkepski/ds/uf.html | *"Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License … intended for non-commercial academic use"* (quoted from that page) |
| **Le2i Fall Detection** | RGB video (AVI), single cam | 191 videos across Home(60)/Coffee-room(70)/Office(64)/Lecture(27), 320×240, 25 fps, 9 subjects | http://le2i.cnrs.fr/Fall-detection-Dataset (lab page; now often mirrored, e.g. https://github.com/YifeiYang210/Fall_Detection_dataset) | **Not stated on the primary lab page.** Commonly redistributed "for research"; treat as research-only / unverified. |
| **UP-Fall Detection** | RGB + depth (2 Kinects) + wearables + EEG | 17 subjects, falls + ADL, ~21,499 images per cam, 320×240 | http://sites.google.com/up.edu.mx/har-up/ | **Not clearly stated**; access via the site. Treat as unverified. |
| **Multiple Cameras Fall (Montreal)** | 8 synchronized RGB cams | 24 scenarios (22 with a fall) | http://www.iro.umontreal.ca/~labimage/Dataset/ | Research use; verify. |

- **Mapping:** all contain a person **on the floor after a fall** → `laying`
  positives, and "sitting down" / "lying on a couch" ADL clips → `sit`. Cameras
  are wall-mounted, often high (UP-Fall ~2.4 m); this is actually a **closer
  match to a robot's downward view** than HaGRID's webcam framing, though these
  rooms are cluttered and low-res.
- **Verdict:** the realistic path to non-trivial `laying` training data, but
  **non-commercial / unverified licensing across the board** — development and
  cross-subject validation only, not shipped weights, until licenses are
  cleared. Le2i (plain RGB video, no depth dependency) is the easiest to run our
  pipeline on.

### 3.4 NTU RGB+D 60 / 120

- **Primary sources:** https://github.com/shahroudy/NTURGB-D ;
  https://rose1.ntu.edu.sg/dataset/actionRecognition/ (was unreachable at verify
  time — `ECONNREFUSED`; GitHub repo confirms the facts below).
- **Format / size:** **RGB videos + depth map sequences + IR + 3D skeletons.**
  NTU-60 = 56,880 clips / 60 actions / 40 subjects; NTU-120 = 114,480 clips /
  120 actions / 106 subjects. 3 camera heights/angles per action.
- **Skeleton format:** Kinect v2 = **25 body joints**, not MediaPipe-33 — so the
  shipped skeletons are category (b) with a real conversion cost (different joint
  set, no hands). **But the RGB videos are included**, so we ignore their
  skeletons and run our own MediaPipe → this makes NTU category (a) and high
  value.
- **Relevant classes:** A008 `sitting down`, A009 `standing up`, A043 `falling
  down`; NTU-120 adds `staggering`, `hopping` (one-foot — loosely `glico`-ish
  but not a real match). `sitting down` / `falling down` clips give many-subject,
  many-viewpoint `sit` and `laying` data.
- **License:** custom **ROSE Lab release agreement**, registration + signed
  agreement required, **research-only / non-commercial**
  (https://github.com/shahroudy/NTURGB-D — "Requesting the dataset" section).
- **Download:** register on the ROSE portal, accept the agreement; RGB videos
  are a very large separate download (hundreds of GB).
- **Verdict:** best available multi-subject source for `sit` and `laying`, but
  **non-commercial + heavyweight download + registration**. Use for the
  cross-person validation fold and for pre-training; do not ship weights derived
  from it without a commercial license.

### 3.5 Exercise / yoga / workout datasets (for `squat`, `t_pose`)

| Dataset | Format | Primary source | License |
|---|---|---|---|
| **Yoga-82** | image **URLs** (web-scraped) + train/test splits, 82 pose classes, 64–1133 imgs/class (~28k) | https://sites.google.com/view/yoga-82/home ; paper https://arxiv.org/abs/2004.10362 | *"non-commercial research and educational purposes only"*; redistribution "in original form only"; "Researcher's employer shall also be bound" (quoted from the site). Access via Google Form. |
| **Kaggle "Gym Workout/Exercises Video"** (philosopher0808) | YouTube-sourced clips, 22 exercises incl. `squat`, `push up`, `shoulder press` | https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video | **License field on the Kaggle page — could not be read remotely; must check in-browser.** Underlying clips are YouTube → copyright risk. |
| **Kaggle "Workout/Exercises Video"** (hasyimabdillah) | video, gym + bodyweight exercises | https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video | same caveat — verify on page. |
| **Fit3D** | multi-view RGB video + SMPL/3D, ~40 exercises incl. squats | https://fit3d.imar.ro/ | registration; research-only terms. |
| **MM-Fit** | smartphone/smartwatch IMU + Kinect skeleton + depth; 10 exercises incl. `squats` | https://mmfit.github.io/ | CC BY-NC 4.0 (non-commercial). Skeleton is Kinect, limited RGB. |

- **`squat`:** Yoga-82 has `Chair_Pose` (Utkatasana ≈ a squat-hold) but the
  cleanest is a gym-video dataset's `squat` class. All options are either
  non-commercial (MM-Fit, Fit3D, Yoga-82) or license-unverified + YouTube-based
  (Kaggle). Extract mid-rep frames → static `squat` samples.
- **`t_pose`:** no dataset targets "T-pose" by name. Yoga-82 is *reported* to
  include `Warrior_II_Pose` (arms horizontal, legs lunged) and a
  `Five-Pointed_Star` / star pose (arms + legs spread) — **verify against the
  actual class list in the repo/ReadMe**; both are only partial matches (legs
  differ). Better: filter a large in-the-wild pose dataset for "both wrists
  within ~15° of shoulder height, arms extended". This is a **data-curation**
  heuristic (allowed — Hard Constraint #1 is about the *classification path*,
  not label curation), but auto-labeled samples must be spot-checked.
- **Verdict:** partial. `squat` is gettable with a license caveat; `t_pose` is
  best synthesized by filtering COCO (below).

### 3.6 ASL / sign-language datasets (for `i_love_you`)

| Dataset | Format | Primary source | License | ILY? |
|---|---|---|---|---|
| **WLASL** | 2,000 glosses, ~21k YouTube video clips | https://github.com/dxli94/WLASL ; https://dxli94.github.io/WLASL/ | Computational Use of Data Agreement (C-UDA); repo states academic/computational use only; **videos are YouTube-hosted (fragile + third-party copyright)** | "i love you" *may* be a gloss — not confirmed from the gloss list |
| **MS-ASL** | 1,000 signs, YouTube URLs | https://www.microsoft.com/en-us/research/project/ms-asl/ | Microsoft research license; YouTube-sourced | not confirmed |
| **ASL Alphabet** (Kaggle, grassknoted) | 87k images, 29 classes = A–Z + space/del/nothing | https://www.kaggle.com/datasets/grassknoted/asl-alphabet | Kaggle page license (verify in-browser) | **No** — ILY is not a letter |
| **Sign Language MNIST** | 28×28 grayscale hand crops, 24 letters | https://www.kaggle.com/datasets/datamunge/sign-language-mnist | CC0 | **No**, and unusable anyway (tiny grayscale, no body) |
| **ASL-HG** | 36k JPG, 36 classes (A–Z, 0–9) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12877850/ | see article | **No** |

- **Reality:** the ASL "I love you" sign (ILY handshape: thumb + index + pinky
  extended) is a distinct handshape, but **no mainstream static-image dataset
  has it as a class** (alphabet datasets stop at letters; ILY is a lexical
  sign). WLASL/MS-ASL might contain it as a video gloss, but both are
  YouTube-sourced and research-licensed.
- **Verdict: no usable external data for `i_love_you`.** Keep it on `s01`.
  Mitigation: pull HaGRID `call` (thumb+pinky shaka) as an explicit **hard
  negative** so the classifier learns "index finger also extended" is the
  discriminator.

### 3.7 General pose datasets (for `raise_*_hand`, `t_pose`, `idle` negatives)

| Dataset | Format / size | Primary source | License |
|---|---|---|---|
| **COCO (keypoints)** | ~200k labelled images, ~250k person instances, 17-kpt annotations; **Flickr RGB images included** | https://cocodataset.org/ , terms https://cocodataset.org/#termsofuse | Annotations **CC BY 4.0**; images subject to Flickr terms. **Commercial use of the annotations is permitted under CC BY 4.0.** (Widely documented; the terms page itself could not be scraped remotely — verify at the link before relying on it commercially.) |
| **MPII Human Pose** | ~25k in-the-wild images, 40k people, 16 kpt, **410 activity labels** incl. sitting/sports | https://human-pose.mpi-inf.mpg.de/ | Annotations under **Simplified BSD**; *"commercial use is not allowed"* because the authors do not hold image copyright (stated on the site / annotation README). |

- **`raise_right_hand` / `raise_left_hand`:** no dedicated dataset worth using
  (classroom "hand-raising" sets — SCB-Dataset
  https://github.com/Whiffe/SCB-dataset , ActRec-Classroom — are far-field,
  bounding-box-only, many tiny people per frame, no per-person hand landmarks,
  and **do not distinguish left vs right** → **low value for us**). Instead:
  filter COCO/MPII for "one wrist above the nose/shoulder, the other down",
  label as `raise_right_hand` (mirror-augment gives `raise_left_hand` for free,
  per `schema.MIRROR_LABEL_SWAP`). Curation heuristic, spot-check required.
- **`t_pose`:** filter COCO for "both wrists near shoulder height, elbows
  extended, standing". COCO's CC BY 4.0 makes this the only commercial-safe
  option.
- **`idle` negatives:** COCO person crops of people standing/walking/talking
  with no target gesture are ideal `idle` data and are commercially licensed.
- **Verdict:** COCO is the workhorse for the "long tail" classes with a clean
  license; MPII is a non-commercial backup with ready-made activity labels.

### 3.8 `heart` (big two-arm overhead heart) and `glico_pose` (running-man)

- **`heart`:** HaGRID's `hand_heart` is a *chest-level, close-framed* two-hand
  heart — **not** the big two-arm overhead heart (arms raised, hands meeting
  above the head). No dataset found for the overhead version.
- **`glico_pose`:** the Glico / running-man pose (one arm up, opposite knee up,
  as in the Osaka Glico sign) has **no dataset**. Athletics "starting dash" /
  sprint-start pose datasets exist (AthleticsPose
  https://arxiv.org/abs/2507.12905 , AthletePose3D
  https://arxiv.org/abs/2503.07499) but the pose is a crouched block start, not
  the standing arms-up/knee-up Glico pose — **not a match**.
- **Verdict: nothing credible for either.** Both stay on `s01`.

---

## 4. Recommended plan

**Pull, in this order:**

1. **HaGRID v1, 512p, classes `ok` / `peace` / `rock` / `like` / `no_gesture`**
   (+ `peace_inverted`, `two_up` as extra `two_finger` variants, + `call` as an
   `i_love_you` hard negative). This is the single highest-value pull: five of
   our classes, clean-ish license, images that match our pipeline.
   - Download: `python download.py --dataset --annotations` filtered to those
     classes (or the per-class archives).
   - Feature work: run `backbone/pose.py` (IMAGE mode is fine for stills) +
     `backbone/hands.py` on each image; wrist-anchor the hand crops exactly as
     in the live pipeline; write normalized 152-dim vectors. Expect a
     meaningful fraction of images where MediaPipe Pose gets only a partial
     upper body — keep them if both shoulders + one wrist are visible, else
     drop. Budget: ~0.1–0.2 s/image on CPU → a 30–50k-image subset is a few
     CPU-hours; parallelize across cores.
   - Label `no_gesture` → `idle`; tag every HaGRID-derived row with
     `source=hagrid_v1`, `subject=hagrid_<worker_id>` so the
     subject-wise-split rule (`CLAUDE.md` #5) still holds and `s01`/`s02` stay
     pure test.

2. **HaGRIDv2 `hand_heart` + `hand_heart2` → `mini_heart`** — *only after* the
   HaGRIDv2 license question (§3.1) is resolved. Same extraction path.

3. **COCO 2017 train, person subset** — for `idle` negatives, `t_pose`, and the
   `raise_right_hand` seed set.
   - Filter with COCO's own 17-kpt annotations first (cheap), then re-extract
     with MediaPipe on the survivors.
   - `t_pose`: wrists within ~0.15 (normalized) of shoulder height, elbow angle
     > ~150°, standing. `raise_right_hand`: right wrist above nose, left wrist
     below shoulder. **Spot-check ≥100 auto-labeled samples per class.**
   - Mirror-augment `raise_right_hand` → `raise_left_hand` via
     `features/augment.py` (already relabels per `MIRROR_LABEL_SWAP`).

4. **Le2i Fall Detection** (plain RGB video, lightest to process) → `laying`
   (post-fall floor frames) and `sit` (ADL sitting clips). Then **NTU RGB+D
   A008/A043** if more subject diversity is needed for `sit`/`laying`.
   - Sample ~1 frame / 0.5 s from the relevant labeled intervals; run the full
     pose+hands pipeline (hands will mostly be `present=0` at that distance —
     that is correct and wanted).
   - **Flag every fall-dataset / NTU row `license=non-commercial`** in the
     dataset card so a future commercial build can exclude them and retrain.

5. **A gym-workout video dataset** (verify its Kaggle license first) → `squat`
   mid-rep frames. If no clean license, `squat` falls back to `s01` + heavy
   augmentation.

**Do not bother pulling:** Jester (NC + dynamic + hand-only), Sign Language
MNIST (unusable), classroom hand-raising datasets (far-field, no L/R, no
landmarks), athletics-start datasets (wrong pose).

**Extraction work summary:** one script, `data/extract_external.py`, that takes
`(image_or_video, label, source_tag, subject_tag)` and appends normalized
152-dim rows to a per-source `.npz`, reusing `backbone/` and `features/schema.py`
unchanged. Est. 1–2 days to write + a few CPU-days to run over HaGRID subset +
COCO subset + Le2i. Everything then flows into `pipeline/build_dataset.py` with
`s01`/`s02` still held out as the sole test set.

---

## 5. Risks / gaps

**Classes with no external data (stay on `s01`, single-subject leakage caveat
from `ARCHITECTURE.md` applies):**

- `i_love_you` — no dataset has the ASL ILY handshape as a class. Highest
  priority for extra in-house recording (multiple people).
- `heart` (big two-arm overhead) — HaGRID's heart is chest-level, not overhead.
- `glico_pose` — no dataset; niche pose.
- `thumb` currently has **zero** `s01` data (`CLAUDE.md`) — HaGRID `like` is
  therefore not just a supplement but the **only** training signal for it;
  cross-check carefully in Phase 4 confusion analysis (esp. `thumb` vs `ok` vs
  `rock`).

**License blockers for commercial / robotics deployment:**

- HaGRID: custom BY-SA-style license, **no NC clause in v1** but **share-alike
  reach** onto redistributed derivatives + subjects' publicity/privacy rights
  not granted → legal review before shipping weights. HaGRID**v2** commercial
  status **unresolved** (sources conflict).
- Jester, URFD, NTU RGB+D, MM-Fit, Fit3D, MPII: **explicitly non-commercial.**
  Usable for R&D and the cross-person validation fold; a deployment build must
  be retrainable without them.
- Le2i, UP-Fall, Kaggle workout datasets: license **not verifiable from the
  primary source** as of 2026-08-30 — must be confirmed before any use beyond
  local experimentation.
- COCO: annotations CC BY 4.0 (commercial OK) but the terms page could not be
  scraped here — confirm at https://cocodataset.org/#termsofuse.

**Domain-gap risks:**

- HaGRID and COCO are frontal/eye-level; the robot camera is lower and the
  person further. `phase1_report.md` already flags camera-height issues for
  `laying`. Fall datasets have the better (high/downward) angle but cluttered,
  low-res rooms. Net: external data will shift the classifier toward frontal
  framing — **Phase 6 field testing remains the real generalization gate**
  (`ARCHITECTURE.md`), and `s01`/`s02` staying as pure test is what will expose
  the shift.
- Mixing many sources risks the classifier learning "source style" instead of
  gesture. Mitigate: aggressive normalization (already in place), per-source
  subject tags in the split, and reporting per-source accuracy in Phase 4.
- Auto-labeled COCO subsets for `t_pose` / `raise_*_hand` carry label noise —
  every heuristic-labeled batch needs a manual spot-check before it enters the
  train set.

---

## 6. Sources

- HaGRID repo — https://github.com/hukenovs/hagrid
- HaGRID v1 paper — https://arxiv.org/abs/2206.08219 , https://arxiv.org/html/2206.08219v2
- HaGRIDv2 paper — https://arxiv.org/abs/2412.01508 , https://arxiv.org/html/2412.01508v1
- HaGRID license PDF — https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf
- HaGRID class constants — https://github.com/hukenovs/hagrid/blob/master/constants.py
- Jester dataset — https://www.qualcomm.com/developer/software/jester-dataset ; paper https://openaccess.thecvf.com/content_ICCVW_2019/papers/HANDS/Materzynska_The_Jester_Dataset_A_Large-Scale_Video_Dataset_of_Human_Gestures_ICCVW_2019_paper.pdf
- Qualcomm Research Use license — https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qvid-dataset-research-license-mar-27-2025.pdf
- UR Fall Detection — https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- Le2i Fall Detection — http://le2i.cnrs.fr/Fall-detection-Dataset (mirror: https://github.com/YifeiYang210/Fall_Detection_dataset)
- UP-Fall — http://sites.google.com/up.edu.mx/har-up/
- NTU RGB+D — https://github.com/shahroudy/NTURGB-D ; https://rose1.ntu.edu.sg/dataset/actionRecognition/
- Yoga-82 — https://sites.google.com/view/yoga-82/home ; https://arxiv.org/abs/2004.10362
- Kaggle Gym Workout/Exercises Video — https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video
- Kaggle Workout/Exercises Video — https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video
- Fit3D — https://fit3d.imar.ro/
- MM-Fit — https://mmfit.github.io/
- WLASL — https://github.com/dxli94/WLASL ; https://dxli94.github.io/WLASL/
- MS-ASL — https://www.microsoft.com/en-us/research/project/ms-asl/
- ASL Alphabet (Kaggle) — https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Sign Language MNIST — https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- COCO — https://cocodataset.org/ ; https://cocodataset.org/#termsofuse
- MPII Human Pose — https://human-pose.mpi-inf.mpg.de/
- SCB-Dataset (classroom behavior) — https://github.com/Whiffe/SCB-dataset
- AthleticsPose — https://arxiv.org/abs/2507.12905 ; AthletePose3D — https://arxiv.org/abs/2503.07499

---

## Round 2 — data for `i_love_you` / `heart` / `mini_heart`

Second, focused pass (2026-08-31) on the three classes Phase 3
(`docs/phase3_baseline.md`) left unmodelled. Task: find **training** data, or
prove there is none, or (for `mini_heart`) decide whether the gap is fixable
without new data.

**Method note / caveat.** Roboflow Universe (`universe.roboflow.com`) returns
HTTP 403 to the automated fetcher, and the browser tool needed an interactive
browser pick that a background agent cannot do. Every Roboflow fact below was
read through a text-reader proxy (`r.jina.ai`) rendering the real Roboflow
project/search pages on 2026-08-31. Treat the *class lists* as reliable and the
*per-class image counts* as "verify in-browser before pulling" — the numbers
quoted are whole-dataset totals; the target class is a fraction of that
(typically 100–500 images/class in these community sets). HaGRID / WLASL / arXiv
facts are from primary sources as cited.

### R2.0 One-paragraph verdict

- **`i_love_you` — NOW HAS DATA (verdict: yes).** Round 1 said "none clean".
  That was wrong about scope: it only checked ASL *video* corpora (WLASL/MS-ASL)
  and *alphabet* image sets. The ASL **"I Love You" lexical sign is a static
  one-hand handshape** (thumb+index+pinky extended) and it is a labelled class in
  **many Roboflow Universe sign-language datasets**, as RGB webcam images we can
  run MediaPipe on. Best single source: **`ananthu-s/asl-gvpbe`** (Roboflow,
  *classification*, whole set ~6,130 imgs, has an `IloveYou` class, CC BY 4.0).
  Aggregate 3–4 Roboflow sets → ~1–2k ILY images. Still supplement with HaGRID
  `call` as the hard negative (already wired in `extract_external.py` as
  `_ily_negative`).
- **`heart` (big overhead two-arm heart) — STILL NO PURPOSE-BUILT DATASET
  (verdict: no).** One thin lead: Roboflow **`shuindec/emblems-heart-detection`**
  has a `full-arm-heart` class (356 imgs total across 5 heart classes, CC BY 4.0)
  — unverified whether "full-arm-heart" is actually the overhead body heart.
  Real recommendation: `heart` is a **pose-only class** (arms make the shape,
  hand landmarks irrelevant) — treat it like `t_pose` and **mine it from COCO**
  by adding a filter branch, plus s01 augmentation.
- **`mini_heart` — DATA PROBLEM ALREADY SOLVED; the residual gap is body
  position, and it is fixable WITHOUT new data (verdict: yes, with an
  augmentation change).** HaGRIDv2 `hand_heart` + `hand_heart2` give the correct
  handshape, and our per-hand normalization (`features/normalize.py`
  `normalize_hand`) makes the hand-shape slice **100 % invariant to where the
  hand sits in the frame**. The chest-vs-overhead difference lives *only* in the
  body-pose part of the vector. Fix = an **arm-elevation augmentation** (R2.5).

### R2.1 Source → class table

| Source | Class(es) | Format | #samples (see caveat) | License | How to get it | MediaPipe-compatible? |
|---|---|---|---|---|---|---|
| Roboflow `ananthu-s/asl-gvpbe` | `i_love_you` (`IloveYou`) | RGB images, **classification** | whole set ~6,130; ILY class a fraction | CC BY 4.0 (stated on page) | Roboflow download API (free key) or "Download Dataset" curl on the page | **Yes** — frontal webcam images, run our pose+hands |
| Roboflow `signlanguageassistant/signlanguageai` | `i_love_you` (`Iloveyou`) | RGB images, object detection | whole set ~7,060 | not shown on page — verify | Roboflow API; use the bbox to crop | Yes |
| Roboflow `ml/…` "FSL" (Filipino SL) | `i_love_you` (`I LOVE YOU`) | RGB images, detection | whole set ~7,510 | verify | Roboflow API | Yes |
| Roboflow `asl-auyfj/asl-detection-lvx6a` | `i_love_you` (`I Love You`) | RGB images, detection | whole set 688, 36 classes | **CC BY 4.0** (stated) | Roboflow API | Yes |
| Roboflow `actions/actions-zqpb1`, `ece496-public-asl/…`, `indian-sign-language-detection/isl-yolov5`, `vraj-atah9/signlanguage-f0irs` | `i_love_you` | RGB images, detection | whole sets 1.4k–3.6k each | verify each | Roboflow API | Yes |
| **HaGRID v1 `call`** | `i_love_you` **hard negative** | RGB images | (already pulled) | HaGRID BY-SA-style (§3.1) | already in `extract_external.py` | Yes |
| **COCO 2017 train**, keypoint-filtered | `heart` (pose-only) | RGB images | unknown — est. low 100s (unverified) | annotations CC BY 4.0 | extend `extract_external.py` COCO filter | Yes (pose only; hands mostly absent, correct) |
| Roboflow `shuindec/emblems-heart-detection` | `heart`? (`full-arm-heart`) | RGB images, detection | 356 total / 5 classes | **CC BY 4.0** (stated) | Roboflow API | Yes, if the class is really the overhead body heart — **verify visually first** |
| MPII Human Pose (`cheerleading`/`dancing`/`aerobics` activity labels) | `heart` (pose-only, weak) | RGB images | a few hundred candidates | Simplified BSD, **non-commercial** (§3.7) | filter by activity id, then MediaPipe | Yes |
| **HaGRIDv2 `hand_heart` + `hand_heart2`** | `mini_heart` | RGB images (+ MediaPipe-21 landmarks shipped) | ~1,800 already extracted (`extract_external.py` cap) | "variant of CC BY-SA 4.0" (repo) / paper §8 says **non-commercial research** — conflict, see R2.4 | already wired: `python -m pipeline.extract_external hagrid_v2_heart` | **Yes** — HaGRIDv2 even ships MediaPipe 21-kpt hand landmarks |
| Roboflow finger-heart sets (`hand-gesture-rvwhe/hand-gesture-mebty` 295 imgs w/ `Finger Heart`; `annotation-duabw/my-first-project-av9ug` 88; others) | `mini_heart` (extra position variety) | RGB images, detection | ~30–300 each | mostly CC BY 4.0 | Roboflow API | Yes — but low volume; not needed if R2.5 augmentation is done |

### R2.2 `i_love_you` — detail

**The Round-1 gap was a scoping error.** Round 1 checked WLASL and MS-ASL (video,
YouTube-hosted, research-licensed) and the alphabet image sets (A–Z only). It did
not check the large population of **community sign-language object-detection /
classification datasets** on Roboflow, which routinely include the common
"survival signs" (`hello`, `yes`, `no`, `thanks`, `please`, `I love you`) as
static handshape classes photographed on webcams.

- The ASL **"I Love You"** sign = the **ILY handshape**: thumb + index + pinky
  extended, middle + ring folded (https://en.wikipedia.org/wiki/ILY_sign). This
  is exactly our `i_love_you` target and is distinct from `call`/shaka
  (thumb+pinky) and `rock`/horns (index+pinky). Our derived feature set already
  has the discriminators: `_hand_derived` computes per-finger curl for all five
  fingers plus a thumb-tip↔pinky-tip spread angle explicitly commented "(ILY)"
  (`features/derived.py` line ~90).
- **WLASL** (primary source, gloss list pulled 2026-08-31 from
  `code/I3D/preprocess/wlasl_class_list.txt`): contains `love` (gloss 561) and
  `fall in love` (gloss 1108) — **no `i love you` / `ily` gloss**. The `love`
  gloss = crossed-fists-over-chest sign, **not** the ILY handshape, and has only
  **12 video instances** (`WLASL_v0.3.json`), all scraped from signing-dictionary
  sites (handspeak, signingsavvy, aslpro, …). Dead end for our target.
- **MS-ASL**: 1,000-gloss list starts `hello`(0) … ends `challenge`(999)
  (https://arxiv.org/pdf/1812.01053). Could not open `MSASL_classes.json` from a
  primary mirror (repos 404'd); not verified to contain "I love you". Even if it
  does, it is YouTube-hosted research-licensed video — low value vs. the Roboflow
  images.
- **HaGRID v1 + v2**: full 34-class list re-verified from
  `github.com/hukenovs/hagrid/blob/master/constants.py` (fetched 2026-08-31):
  `grabbing, grip, holy, point, call, three3, timeout, xsign, hand_heart,
  hand_heart2, little_finger, middle_finger, take_picture, dislike, fist, four,
  like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted,
  three, three2, two_up, two_up_inverted, three_gun, thumb_index, thumb_index2,
  no_gesture`. **None is the ILY handshape.** The paper (arXiv:2412.01508 §… )
  gives only *functional* descriptions ("thumb_index used for ZOOM/SWIPE",
  "xsign = shut down system") and no finger-by-finger anatomy, but by their
  reference figures: `thumb_index`/`thumb_index2` = thumb+index only (L / pinch),
  `three_gun` = thumb+index+middle (finger-gun), `xsign` = crossed index fingers
  of two hands, `holy` = two hands / prayer-ish, `point` = index only. No
  thumb+index+pinky class. `call` (shaka) stays the best **hard negative**.
- **Best source: Roboflow Universe.** Verified (via proxy) to carry an ILY class:
  `ananthu-s/asl-gvpbe` (classification, CC BY 4.0, ~6.1k imgs),
  `asl-auyfj/asl-detection-lvx6a` (detection, CC BY 4.0, 688 imgs / 36 classes,
  class literally named "I Love You"), `signlanguageassistant/signlanguageai`
  (~7k), `ml/FSL`, `actions/actions-zqpb1`, `ece496-public-asl/ece496-public-asl`,
  `indian-sign-language-detection/isl-yolov5`, `vraj-atah9/signlanguage-f0irs`.
  These are frontal webcam RGB → straight into our pipeline. Domain gap same as
  HaGRID (frontal, eye-level, hand large) — the same mitigation applies
  (wrist-anchored crop + hand-local normalization + rotation/scale aug).
- **Still to verify in-browser (needs the human):** exact per-class ILY image
  counts and per-dataset licences for the sets marked "verify". The two
  CC BY 4.0-confirmed sets alone (`asl-gvpbe`, `asl-detection-lvx6a`) are enough
  to move `i_love_you` from unmodelled to trained.

### R2.3 `heart` (overhead two-arm heart) — detail

- **No purpose-built dataset.** Re-confirmed: Roboflow "heart" classes are almost
  all hand/finger hearts at chest level, or heart-shaped objects, or "heart"
  face-shape, or cardiac imaging. The one lead is
  `shuindec/emblems-heart-detection` (CC BY 4.0, 356 imgs, classes
  `2-finger-heart, full-arm-heart, traditional-heart, traditional-heart-2,
  upside_down_heart`). `full-arm-heart` *might* be the overhead body heart —
  Roboflow's page does not define it; **inspect the sample images before
  relying on it**. Even if good, ~50–70 images is seed-only.
- **Recommended path: mine COCO, treat `heart` as a pose-only class.** The
  overhead heart is defined entirely by arm geometry (both wrists above the head,
  wrists close together near the midline, both elbows bowed outboard). Hand shape
  is irrelevant — like `t_pose` and `raise_*_hand`, which we already mine from
  COCO. Add a branch to `_classify_coco` in `pipeline/extract_external.py`:
  ```python
  # inside _classify_coco, needs kpts 0 nose,1/2 eyes,3/4 ears,5/6 sho,7/8 elb,9/10 wri
  if ok(0, 5, 6, 7, 8, 9, 10):
      eye_y = min(xy[1][1], xy[2][1])
      both_overhead = lwri[1] < eye_y and rwri[1] < eye_y
      hands_together = abs(lwri[0] - rwri[0]) < 1.1 * sho_w
      elbows_out = lelb[0] > lsho[0] and relb[0] < rsho[0]   # elbows outboard of shoulders
      if both_overhead and hands_together and elbows_out:
          return "heart"
  ```
  **Yield: unknown, not verifiable without running the filter.** Order-of-
  magnitude guess: overhead "hands meet above head" poses (cheer, dance, "YMCA",
  stretching) are uncommon in COCO-train's ~260k person instances → **low
  hundreds** of candidates, of which MediaPipe will keep maybe half. If that
  proves too thin, fall back to s01 `heart_01/02` + the R2.5 arm-elevation
  augmentation applied to `t_pose` / `raise_*_hand` bodies to synthesise the
  arms-up geometry (then hand-curl left unconstrained). MPII
  `cheerleading`/`dancing` activity-labelled images are a non-commercial
  secondary source.
- **`heart` vs `mini_heart` confusion risk:** both end up "hands high / near
  midline". The separator is hand shape (heart = open/relaxed hands touching;
  mini_heart = finger-heart handshape) — which is exactly what our per-hand
  derived features encode, so keep both hands' `present` flags and curl features
  in the vector for these two.

### R2.4 `mini_heart` — is the chest↔overhead gap fixable without new data?

**Yes.** Root-cause analysis against the actual feature code:

1. `features/normalize.py::normalize_hand` centres each hand on its own wrist
   (landmark 0) and scales by wrist→middle-MCP distance. **Every one of the 42
   raw hand coords and all 13 per-hand derived features
   (`features/derived.py::_hand_derived`: 5 curls, 4 spreads, 3 tip-distance
   ratios) is therefore fully invariant to the hand's position, scale and
   in-plane translation in the frame.** A HaGRID chest-level finger-heart and an
   s01 overhead finger-heart produce the *same* hand slice.
2. The chest-vs-overhead difference is confined to the **body** part of the
   vector: the 66 normalized body coords, and 4 of the 12 body-derived features
   — `l_elbow`/`r_elbow` angle, `l_shoulder`/`r_shoulder` angle, and the two
   "wrist height relative to shoulders" values (`features/derived.py`
   `_body_derived`, lines ~73–74).
3. So the model failed on overhead `mini_heart` not because it lacks the
   handshape, but because every HaGRID training row has *arms-down* body
   geometry, and the classifier keyed on that.

**Recommendation (do this, in priority order):**

- **A. Add an arm-elevation augmentation** to `features/augment.py` and apply it
  (with a wide angle range) to the HaGRID `hand_heart`/`hand_heart2` rows — and
  to the Roboflow `i_love_you` rows, same chest-level problem. Mechanism, reusing
  the existing chain machinery (`_CHAINS`, `_TIP_FOLLOWS`):
  - pivot each arm chain `[shoulder → elbow → wrist]` about the **shoulder** by a
    random elevation angle θ (e.g. θ ∈ [0°, +150°] measured toward "straight up")
    so the wrist sweeps from its recorded near-hip position to above the head;
  - translate the hand landmark-followers (`15→17,19,21` / `16→18,20,22`) rigidly
    with their wrist, exactly as `_limb_jitter` already does;
  - **leave the `_LH` / `_RH` slices and the `LH_PRESENT`/`RH_PRESENT` flags
    untouched** — they are wrist-local, so the handshape rides along for free;
  - for two-hand `hand_heart`, after elevating, translate both arms so the two
    wrists sit close together near the head midline (that is the pose).
  This synthesises overhead finger-hearts from chest-level data with zero new
  recording, and is architecturally clean (it is data augmentation, not a
  hand-written rule in the classification path — Hard Constraint #1 OK).
- **B. Add one explicit body-frame feature: inter-wrist distance.** Because each
  hand is normalized independently, the vector currently has **no direct measure
  of how close the two hands are** — only implicitly via body coords. `mini_heart`
  (hands together) vs. two separate hands is a strong, position-robust signal.
  Add `‖pose_L_wrist(15) − pose_R_wrist(16)‖` in normalized body units to
  `_body_derived`. Cheap, helps `mini_heart`, `heart`, `t_pose`.
- **C. Do NOT** switch these classes to a body-position-invariant feature subset
  (dropping body coords / wrist-height). That information legitimately separates
  `raise_*_hand`, `t_pose`, `glico_pose` — throwing it away to help one class is
  a bad trade. A + B keep the full vector and add coverage.
- Roboflow finger-heart sets (R2.1) give a little real position variety but are
  tiny (30–300 imgs); pull them only if A+B underperform in Phase 4.

### R2.5 HaGRIDv2 licence (affects `mini_heart` and, if used, more)

Conflicting primary statements — unchanged from §3.1, now pinned:

- Repo README (fetched 2026-08-31): *"This work is licensed under a variant of
  Creative Commons Attribution-ShareAlike 4.0 International License."*
- HaGRIDv2 paper (arXiv:2412.01508), abstract/§8 (fetched 2026-08-31):
  *"the dataset … [is] publicly available under the modified Creative Commons
  CC-BY 4.0 license"* and *"We release the dataset under a public license for
  non-commercial use in research purposes."*
- HuggingFace mirror `testdummyvt/hagRIDv2_512px`: license field = "other",
  points back to the repo.

→ HaGRIDv2 (and therefore `hand_heart`/`hand_heart2` → `mini_heart`) carries a
**non-commercial** statement in the paper that the repo licence text does not.
Per `docs/phase2b_external_data.md` the project decision is that licences are not
a blocker for the RoboCup@Home use, so this is a **record-it-and-move-on** item:
tag every `mini_heart` row `source=hagrid_v2`, `license=non-commercial-disputed`
so a future commercial build can be retrained without it. Roboflow sets used for
`i_love_you` are mostly CC BY 4.0 (commercial-OK) — cleaner.

### R2.6 Extraction plan — concrete `extract_external.py` changes

1. **`i_love_you` via Roboflow.** Roboflow needs a (free) API key, like Kaggle
   does — this is a new auth dependency, note it for the user. Two integration
   options:
   - *Preferred:* add a `"kind": "roboflow"` branch. Config entry, e.g.:
     ```python
     "roboflow_ily": {
         "kind": "roboflow",
         "projects": [   # workspace/project/version, class-of-interest
             ("ananthu-s", "asl-gvpbe", 1, {"IloveYou": "i_love_you"}),
             ("asl-auyfj", "asl-detection-lvx6a", 1, {"I Love You": "i_love_you"}),
         ],
         "max_per_class": 1500,
     },
     ```
     Implementation: `pip install roboflow`; for each project
     `rf.workspace(ws).project(p).version(v).download("coco", location=TMP)`,
     iterate images, for detection sets crop to the ILY bbox first (falls back to
     full image for classification sets), run `Extractor.__call__`, write rows,
     delete the download. ~a few hundred MB per project, deleted after — fits the
     disk-safe pattern.
   - *Lighter:* manually download the 2 CC BY 4.0 sets once (COCO-JSON export),
     drop the folders in `data/_ext_tmp/`, and add a `"kind": "coco_folder"`
     reader that reuses the existing COCO image loop on a local dir. No new pip
     dep, no API key.
   Tag rows `source=roboflow_<project>`, `subject=roboflow_<project>_<n>` (these
   sets have no subject ids — cross-subject purity vs s01 still holds because
   s01 is a different domain entirely).
2. **`heart` via COCO.** Add the `heart` branch to `_classify_coco` (R2.3 code),
   add `"heart"` to the `classes` set in `run_coco`, and bump the candidate cap.
   No new source, no new dep. Spot-check ≥100 auto-labelled samples — the
   arms-overhead-hands-together filter will also catch some diving/reaching/
   celebration poses.
3. **`mini_heart`.** No extraction change — data is already pulled by
   `hagrid_v2_heart`. The work is in `features/augment.py` (R2.4 A) and
   `features/derived.py` (R2.4 B), plus wiring those classes back into
   `pipeline/build_dataset.py` `train.npz` (Phase 3 had excluded all three).

### R2.7 "Still nothing" list

- **`heart` (overhead two-arm)** — no purpose-built dataset exists anywhere
  (re-confirmed). It is now *seedable* from COCO pose-filtering (unverified
  yield) + the unverified `emblems-heart-detection/full-arm-heart` class +
  s01 augmentation, but there is no clean, sized, ready dataset. Lowest
  confidence of the three; Phase 6 field data still matters most here.
- Everything else has a path: `i_love_you` → Roboflow (real images, mostly
  CC BY 4.0), `mini_heart` → HaGRIDv2 + arm-elevation augmentation (no new data).

### R2.9 Outcome (2026-09-01, after implementing R2.4 + R2.6)

- **`mini_heart` — FIXED.** R2.4's arm-elevation augmentation
  (`features/augment.raise_arms`) + inter-wrist-distance derived feature:
  0.00 → **0.92 F1** (RF, cross-domain). No new data. `raise_right_hand` also
  rose 0.67 → 0.82. `mini_heart` removed from `PENDING_DATA`.
- **`i_love_you` — STILL UNMODELLED, and now we know why.** The `roboflow_ily`
  source was built and run on Colab — **606 real ILY images** from 5 Roboflow
  ASL detection datasets (ece496-public-asl, signlanguage-f0irs, actions-zqpb1,
  asl-detection-lvx6a, signlanguageai; `ananthu-s/asl-gvpbe` had no generated
  version). Training on them: `i_love_you` recall **0.00** (→ `rock`), and
  `rock` fell 0.78 → 0.48. **Root cause is the backbone, not the data**:
  MediaPipe Hands on s01's conversational-distance wrist-crop reports all
  fingers curled for `i_love_you` / `rock` / `two_finger` alike (measured:
  `idxCurl` ≈ 2.7 rad for all three, straight ≈ 0.4). The three classes are the
  same feature vector; no amount of ILY training data separates identical
  inputs. This is the reproduced, backbone-attributable failure CLAUDE.md #4
  asks for. Next: try upscaling the wrist crop before `hands.detect` (Phase 4,
  preprocessing only); failing that, re-record or accept the confusion cluster.
  The 606 rows are kept in `data/features_ext/roboflow_ily__*.npz`.
- **`heart` — pending.** `_classify_coco` heart branch wired; not yet extracted.

### R2.8 Round-2 sources

- ILY sign definition — https://en.wikipedia.org/wiki/ILY_sign
- WLASL gloss list — https://github.com/dxli94/WLASL/blob/master/code/I3D/preprocess/wlasl_class_list.txt ; instances — https://github.com/dxli94/WLASL/blob/master/start_kit/WLASL_v0.3.json
- MS-ASL paper — https://arxiv.org/pdf/1812.01053
- HaGRID constants (34 classes) — https://github.com/hukenovs/hagrid/blob/master/constants.py
- HaGRID README (annotations, licence) — https://github.com/hukenovs/hagrid/blob/master/README.md
- HaGRIDv2 paper (licence §8) — https://arxiv.org/abs/2412.01508 ; https://arxiv.org/html/2412.01508v1
- HaGRIDv2 512px mirror — https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px
- Roboflow `ananthu-s/asl-gvpbe` — https://universe.roboflow.com/ananthu-s/asl-gvpbe
- Roboflow `asl-auyfj/asl-detection-lvx6a` — https://universe.roboflow.com/asl-auyfj/asl-detection-lvx6a
- Roboflow `signlanguageassistant/signlanguageai` — https://universe.roboflow.com/signlanguageassistant/signlanguageai
- Roboflow `shuindec/emblems-heart-detection` — https://universe.roboflow.com/shuindec/emblems-heart-detection
- Roboflow `hand-gesture-rvwhe/hand-gesture-mebty` (finger heart) — https://universe.roboflow.com/hand-gesture-rvwhe/hand-gesture-mebty
- Roboflow download API docs — https://docs.roboflow.com/universe/find-a-dataset-on-universe
- Hand heart (gesture background) — https://en.wikipedia.org/wiki/Finger_heart
