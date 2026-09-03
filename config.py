"""Runtime knobs — the values you actually tweak between runs. Import from here,
don't hardcode them elsewhere.

Training/extraction code pins its own values (single-person, no smoothing) and
ignores this file — these only affect the live/demo pipeline.
"""

# --- multi-person (backbone/pose.py) ---
NUM_POSES = 2          # max people MediaPipe Pose tracks per frame. 1 = the
                       # original single-person config; higher = multi-person,
                       # latency grows ~linearly with the count.
MAX_CLASSIFY = 0       # 0 = classify everyone detected; N>0 = only the N
                       # largest (≈ closest) people, to cap the per-frame cost.

# --- temporal smoothing (pipeline/runtime.py) ---
SMOOTH_WINDOW = 7      # majority-vote window in frames (scripts/phase5_flicker.py)
CONF_FLOOR = 0.20      # window confidence below this -> "unknown"

# --- latency lever (pipeline/demo.py) ---
HAND_EVERY = 1         # run the hand landmarker every K frames (reuse between)

# --- person tracker (pipeline/multiperson.py) ---
TRACK_IOU = 0.30       # min IoU to keep a detection on the same track
TRACK_MAX_MISSED = 12  # drop a track after this many frames with no match
