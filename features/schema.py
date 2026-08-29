"""Feature-vector contract. Everything downstream (extraction, augmentation,
classifier) imports layout from here so there is one source of truth.

Backbone: MediaPipe Pose (33 landmarks) + MediaPipe Hands (21 landmarks/hand).
Chosen for the deployment target (shared-VRAM laptop) - runs on CPU.

Per-frame vector = body(66) + left_hand(42) + lh_present(1)
                              + right_hand(42) + rh_present(1)  = 152
"""

from __future__ import annotations

# ---- landmark counts ----
N_BODY = 33
N_HAND = 21

# ---- MediaPipe Pose indices we rely on ----
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_HIP, POSE_RIGHT_HIP = 23, 24
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16

# ---- MediaPipe Hands indices ----
HAND_WRIST = 0
HAND_MIDDLE_MCP = 9

# ---- vector layout (start offsets) ----
BODY_OFF = 0
LH_OFF = BODY_OFF + N_BODY * 2          # 66
LH_PRESENT = LH_OFF + N_HAND * 2        # 108
RH_OFF = LH_PRESENT + 1                 # 109
RH_PRESENT = RH_OFF + N_HAND * 2        # 151
FEATURE_DIM = RH_PRESENT + 1            # 152

# ---- classes ----
# `idle` must always be present (CLAUDE.md). Order is the label index order.
CLASSES = [
    "idle",
    "raise_right_hand", "raise_left_hand", "sit", "squat", "laying",
    "ok", "i_love_you", "rock", "two_finger", "thumb",
    "heart", "mini_heart", "t_pose", "glico_pose",
]

# classes that flip to another class under mirroring; the rest keep their label
MIRROR_LABEL_SWAP = {
    "raise_right_hand": "raise_left_hand",
    "raise_left_hand": "raise_right_hand",
}

# MediaPipe Pose left<->right landmark index pairs (for mirror augmentation).
# Non-paired indices (0 nose, 15-22 are paired below, etc.) keep position, only x flips.
POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def class_index(label: str) -> int:
    return CLASSES.index(label)
