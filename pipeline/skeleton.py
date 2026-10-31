"""Keypoint names and skeleton edges.

The pose models predict the 59 SUMediPose markers / joint centres in ``KEYPOINTS_59``
order (``all_63`` in the repo's keypoint_sets.yaml). The JSON files store all 63 Vicon
channels in ``SOURCE_LAYOUT`` order; ``SOURCE_TO_59`` picks the 59 out of them.
"""

SOURCE_LAYOUT = [
    "C7", "CLAV", "CentreOfMass", "CentreOfMassFloor", "LAJC", "LTOE", "LPSI",
    "LBHD", "LEJC", "LFRM", "LFHD", "RSHO", "LWRA", "RTHI", "LHJC", "LKJC",
    "LANK", "R_Foot_Out", "LTHI", "LSHO", "LSJC", "LTIB", "LHEE", "RKNE",
    "LELB", "LWJC", "LWRB", "LFIN", "AAAAAAAA", "MKNEL", "MKNER", "LMMED",
    "RMMED", "MElbowL", "MElbowR", "RAJC", "RTOE", "RPSI", "RBAK", "RBHD",
    "REJC", "RWRA", "RFHD", "RASI", "RWRB", "PelR", "RHJC", "RKJC", "RANK",
    "L_Foot_Out", "LKNE", "RELB", "RSJC", "RTIB", "RHEE", "PelL", "RFRM",
    "RWJC", "RFIN", "LASI", "BBBBBBB", "STRN", "T10",
]

KEYPOINTS_59 = [
    "C7", "CLAV", "LAJC", "LTOE", "LPSI", "LBHD", "LEJC", "LFRM", "LFHD", "RSHO",
    "LWRA", "RTHI", "LHJC", "LKJC", "LANK", "R_Foot_Out", "LTHI", "LSHO", "LSJC",
    "LTIB", "LHEE", "RKNE", "LELB", "LWJC", "LWRB", "LFIN", "MKNEL", "MKNER",
    "LMMED", "RMMED", "MElbowL", "MElbowR", "RAJC", "RTOE", "RPSI", "RBAK", "RBHD",
    "REJC", "RWRA", "RFHD", "RASI", "RWRB", "PelR", "RHJC", "RKJC", "RANK",
    "L_Foot_Out", "LKNE", "RELB", "RSJC", "RTIB", "RHEE", "PelL", "RFRM", "RWJC",
    "RFIN", "LASI", "STRN", "T10",
]

SOURCE_TO_59 = [SOURCE_LAYOUT.index(n) for n in KEYPOINTS_59]

# Main segments for drawing, as index pairs into KEYPOINTS_59.
_EDGE_NAMES = [
    ("LFHD", "RFHD"), ("LBHD", "RBHD"), ("LFHD", "LBHD"), ("RFHD", "RBHD"), ("C7", "LBHD"), ("C7", "RBHD"),
    ("C7", "CLAV"), ("CLAV", "STRN"), ("C7", "T10"), ("STRN", "T10"),
    ("C7", "LSJC"), ("C7", "RSJC"), ("LSJC", "LEJC"), ("LEJC", "LWJC"), ("LWJC", "LFIN"),
    ("RSJC", "REJC"), ("REJC", "RWJC"), ("RWJC", "RFIN"),
    ("LASI", "RASI"), ("LPSI", "RPSI"), ("LASI", "LPSI"), ("RASI", "RPSI"), ("T10", "LPSI"), ("T10", "RPSI"),
    ("LHJC", "RHJC"), ("LHJC", "LKJC"), ("LKJC", "LAJC"), ("LAJC", "LHEE"), ("LHEE", "LTOE"), ("LAJC", "LTOE"),
    ("RHJC", "RKJC"), ("RKJC", "RAJC"), ("RAJC", "RHEE"), ("RHEE", "RTOE"), ("RAJC", "RTOE"),
]
SKELETON_EDGES = [(KEYPOINTS_59.index(a), KEYPOINTS_59.index(b)) for a, b in _EDGE_NAMES]
