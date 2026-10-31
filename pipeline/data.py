"""Reading the synchronised 3-camera recording.

The three camera JSON files (e.g. data/C1S1A3D1.json) have one entry per motion-capture
frame, and entry ``i`` refers to the same instant in all three cameras. Each entry holds
the image path, the person bounding box ``bbx`` (x, y, w, h), the 2D wand markers
``wand_xy``, and Vicon ground truth (``xy`` per camera, ``xyz`` in that camera's frame).
``<subject+action>_wcs.json`` holds the same frames in the Vicon world frame (mm).
Ground truth is only used for the visual checks and error numbers, never by the pipeline.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton import SOURCE_TO_59


@dataclass
class FrameSet:
    index: int
    image_paths: list          # 3 paths, main camera first
    bboxes: np.ndarray         # (3, 4) x, y, w, h
    wand_xy: np.ndarray        # (3, 5, 2)
    gt_xy: np.ndarray          # (3, 59, 2) ground-truth 2D (NaN = not available)
    gt_xyz_cam: np.ndarray     # (59, 3) ground truth in main-camera coordinates (calibration units)
    gt_xyz_wcs: np.ndarray     # (59, 3) ground truth in the world frame (mm), or NaN


class Recording:
    """One trial seen by three cameras, e.g. ``Recording("data", "S1A3D1", cams=("C1", "C2", "C3"))``."""

    def __init__(self, data_dir, trial, cams=("C1", "C2", "C3")):
        self.data_dir = Path(data_dir)
        self.trial, self.cams = trial, list(cams)
        self.subject = "S" + trial.split("S")[1].split("A")[0]
        self.frames = [json.load(open(self.data_dir / f"{c}{trial}.json")) for c in self.cams]
        if len({len(f) for f in self.frames}) != 1:
            raise ValueError("The camera JSON files have different frame counts.")
        wcs_path = self.data_dir / f"{trial}_wcs.json"
        self.wcs = json.load(open(wcs_path)) if wcs_path.exists() else None

    def __len__(self):
        return len(self.frames[0])

    def image_path(self, cam_idx, entry):
        # JSON paths look like frames/C1/S1/C1S1A3D1/C1S1A3D1_0355.jpg
        return self.data_dir / Path(entry["path"]).parent.name / Path(entry["path"]).name

    def wand_wcs(self):
        """(5, 3) wand marker positions in the world frame (mm); the wand is static."""
        return np.asarray(self.wcs[0]["xyz_wand"], float) if self.wcs else None

    def __getitem__(self, i):
        entries = [f[i] for f in self.frames]
        gt_wcs = (np.asarray(self.wcs[i]["xyz"], float)[SOURCE_TO_59] if self.wcs
                  else np.full((59, 3), np.nan))
        return FrameSet(
            index=i,
            image_paths=[self.image_path(c, e) for c, e in enumerate(entries)],
            bboxes=np.array([e["bbx"] for e in entries], float),
            wand_xy=np.array([e["wand_xy"] for e in entries], float),
            gt_xy=np.array([np.asarray(e["xy"], float)[SOURCE_TO_59, :2] for e in entries]),
            gt_xyz_cam=np.asarray(entries[0]["xyz"], float)[SOURCE_TO_59],
            gt_xyz_wcs=gt_wcs,
        )
