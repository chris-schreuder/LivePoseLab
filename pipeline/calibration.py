"""Stereo calibration of the 3-camera rig, read exactly as in error_3d/error_calc_3d.py.

external_calibration.csv lists, per subject session, the stereo calibration files that
link the main camera to each pair camera. Every .npz holds R, T (pair camera relative to
the main camera) and the intrinsics mtx_A (main) / mtx_B (pair). Subjects without their
own calibration use the most recent earlier session's.

Projection matrices (main camera = world origin):
    P1 = K1 [I | 0],   P2 = K2 [R12 | T12],   P3 = K3 [R13 | T13]
Coordinates come out in the calibration's units (mm / 60 for SUMediPose).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Rig:
    K: list                 # 3 intrinsic matrices (main, pair 1, pair 2)
    R: list                 # 3 rotations, main camera first (identity)
    T: list                 # 3 translations (3, 1)
    P: np.ndarray           # (3, 3, 4) projection matrices
    use_cams: list          # camera indices used for triangulation (repo rule)
    files: list             # calibration files used


def session_for(subject, sessions):
    """Map a subject to the calibration session it uses (latest session <= subject number)."""
    n = int(subject.lstrip("S"))
    numbers = sorted(int(s.lstrip("S")) for s in sessions)
    earlier = [s for s in numbers if s <= n]
    return f"S{earlier[-1] if earlier else numbers[0]}"


def load_rig(root, subject, main=1, pairs=(2, 3)):
    root = Path(root)
    df = pd.read_csv(root / "calibration" / "external_calibration.csv")
    session = session_for(subject, df["subject"].unique())
    rows = df[df["subject"] == session]

    files = [rows[(rows["main"] == main) & (rows["pair"] == p)]["path"].values[0] for p in pairs]
    pair1, pair2 = (np.load(root / f) for f in files)

    K = [pair1["mtx_A"], pair1["mtx_B"], pair2["mtx_B"]]
    R = [np.eye(3), pair1["R"], pair2["R"]]
    T = [np.zeros((3, 1)), pair1["T"].reshape(3, 1), pair2["T"].reshape(3, 1)]
    P = np.stack([k @ np.hstack([r, t]) for k, r, t in zip(K, R, T)])

    # Repo rule: calibration method 3 links all three cameras; otherwise triangulate with
    # the main camera and the pair it was directly linked to.
    link = rows[(rows["main_link"] == True) & (rows["cam"] == main)].iloc[0]   # noqa: E712
    if int(link["method"]) == 3:
        use = [0, 1, 2]
    else:
        use = [0, 1] if int(link["pair"]) == pairs[0] else [0, 2]
    return Rig(K=K, R=R, T=T, P=P, use_cams=use, files=[str(f) for f in files])
