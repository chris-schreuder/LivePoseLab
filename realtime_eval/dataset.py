"""The full dataset layout on the evaluation machine:

    <data_root>/frames/S7/C1/C1S7A1D1/C1S7A1D1_0307.jpg     camera images
    <data_root>/camera_data/C1/S7A1D1.json                    per-camera frames: image path, person box,
                                                              wand 2D, Vicon 2D (``xy``) and 3D in camera frame
    <data_root>/wcs_data/S7A1D1.json                          Vicon markers and wand in the world frame (mm)

``FullDataRecording`` reads one trial (e.g. "S7A1D1") from it with the same interface as
``pipeline.Recording``, so the streaming engine and ``stream_sim`` run on it unchanged.
``discover`` lists the trials that have all three camera files, the world-frame file and frames.
"""
import json
import re
from pathlib import Path

import numpy as np

from pipeline.data import FrameSet, Recording
from pipeline.skeleton import SOURCE_TO_59

TRIAL_RE = re.compile(r"^(S\d+)(A\d+)(D\d+)$")


def parse_trial(trial):
    """'S7A1D1' -> ('S7', 'A1', 'D1')."""
    m = TRIAL_RE.match(trial)
    if not m:
        raise ValueError(f"trial name not of the form S<n>A<n>D<n>: {trial}")
    return m.groups()


class FullDataRecording(Recording):
    def __init__(self, data_root, trial, cams=("C1", "C2", "C3")):
        self.data_dir = Path(data_root)
        self.trial, self.cams = trial, list(cams)
        self.subject, self.action, self.take = parse_trial(trial)
        self.frames = [json.load(open(self.data_dir / "camera_data" / c / f"{trial}.json")) for c in self.cams]
        if len({len(f) for f in self.frames}) != 1:
            raise ValueError(f"{trial}: the camera JSON files have different frame counts")
        wcs = self.data_dir / "wcs_data" / f"{trial}.json"
        self.wcs = json.load(open(wcs)) if wcs.exists() else None

    def image_path(self, cam_idx, entry):
        """<root>/frames/<S>/<C>/<C+trial>/<file>; the JSON's own path (frames/C1/S1/...) is only used for
        the file name, so a different folder order in the JSON does not matter."""
        name = Path(entry["path"]).name
        cam = self.cams[cam_idx]
        p = self.data_dir / "frames" / self.subject / cam / f"{cam}{self.trial}" / name
        if not p.exists():                                    # fall back to the path as written in the JSON
            alt = self.data_dir / entry["path"]
            if alt.exists():
                return alt
        return p

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

    def wand_wcs(self):
        return np.asarray(self.wcs[0]["xyz_wand"], float) if self.wcs else None


def discover(data_root, cams=("C1", "C2", "C3")):
    """All trials with three camera files, a world-frame file and a frames folder per camera."""
    root = Path(data_root)
    out = []
    for w in sorted((root / "wcs_data").glob("*.json")):
        trial = w.stem
        if not TRIAL_RE.match(trial):
            continue
        subject = parse_trial(trial)[0]
        if all((root / "camera_data" / c / f"{trial}.json").exists() and
               (root / "frames" / subject / c / f"{c}{trial}").is_dir() for c in cams):
            out.append(trial)
    return sorted(out, key=lambda t: tuple(int(x[1:]) for x in parse_trial(t)))


def pick_spread(trials, n):
    """n trials spread over subjects and actions: round-robin over subjects, varying the action."""
    by_subject = {}
    for t in trials:
        by_subject.setdefault(parse_trial(t)[0], []).append(t)
    picked, used_actions = [], set()
    while len(picked) < min(n, len(trials)):
        progressed = False
        for subj in sorted(by_subject, key=lambda s: int(s[1:])):
            rest = [t for t in by_subject[subj] if t not in picked]
            if not rest or len(picked) >= n:
                continue
            fresh = [t for t in rest if parse_trial(t)[1] not in used_actions]
            t = (fresh or rest)[0]
            picked.append(t)
            used_actions.add(parse_trial(t)[1])
            progressed = True
        if not progressed:
            break
    return picked
