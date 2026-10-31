"""Non-angle feedback measures and static-trial checks for the live demo.

Every angle the demo shows comes from the real-time biomechanics stage (``stream_sim/biomech_stage.py``,
the Plug-in Gait model, continuous frame to frame). This module only adds the distance measures that
model does not output, and the checks shown after the static trial:

* pelvis drop       standing pelvis height (static trial) minus the current one (mm): squat depth
* lateral shift     mid-hip relative to mid-ankle along the body's left-right axis (mm, + = right)
"""
import numpy as np

from pipeline.skeleton import KEYPOINTS_59

UP = np.array([0.0, 1.0, 0.0])
IDX = {n: i for i, n in enumerate(KEYPOINTS_59)}

# name: (label, unit, group). Order = the order stored in the session files.
METRICS = {
    "pelvis_drop": ("Pelvis drop from standing", "mm", "depth"),
    "lateral_shift": ("Pelvis lateral shift (+ to the right)", "mm", "pelvis"),
}


def _p(xyz, name):
    return xyz[..., IDX[name], :]


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _reject(v, n):
    return v - np.sum(v * n, -1, keepdims=True) * n


def compute_metrics(xyz, static_xyz):
    """xyz (N, 59, 3) mm, static_xyz (59, 3) mm (the static trial) -> {name: (N,) array}."""
    xyz = np.asarray(xyz, float)
    right = _unit(_reject(_p(xyz, "RHJC") - _p(xyz, "LHJC"), UP))
    mid_hip = (_p(xyz, "RHJC") + _p(xyz, "LHJC")) / 2
    static_mid_hip = (static_xyz[IDX["RHJC"]] + static_xyz[IDX["LHJC"]]) / 2
    mid_ankle = (_p(xyz, "RAJC") + _p(xyz, "LAJC")) / 2
    return {"pelvis_drop": static_mid_hip[1] - mid_hip[..., 1],
            "lateral_shift": np.sum((mid_hip - mid_ankle) * right, -1)}


def static_check(static_xyz):
    """Numbers for the static-trial screen: segment lengths per side, symmetry, feet, height."""
    x = np.asarray(static_xyz, float)

    def seg(a, b):
        return float(np.linalg.norm(x[IDX[a]] - x[IDX[b]]))

    segments = {}
    for name, (a, b) in {"thigh": ("HJC", "KJC"), "shank": ("KJC", "AJC"), "upper arm": ("SJC", "EJC"),
                         "forearm": ("EJC", "WJC"), "foot": ("HEE", "TOE")}.items():
        r, l = seg("R" + a, "R" + b), seg("L" + a, "L" + b)
        segments[name] = {"r_mm": r, "l_mm": l, "asym_pct": 100 * abs(r - l) / ((r + l) / 2)}
    head_top = max(x[IDX[n]][1] for n in ("LFHD", "RFHD", "LBHD", "RBHD"))
    floor = min(x[IDX[n]][1] for n in ("LHEE", "RHEE", "LTOE", "RTOE"))
    return {
        "segments": segments,
        "heel_toe_height_r_mm": float(x[IDX["RTOE"]][1] - x[IDX["RHEE"]][1]),
        "heel_toe_height_l_mm": float(x[IDX["LTOE"]][1] - x[IDX["LHEE"]][1]),
        # head markers sit about 100 mm below the top of the head
        "marker_height_mm": float(head_top - floor + 100.0),
        "floor_y_mm": float(floor),
    }
