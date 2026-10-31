"""From main-camera coordinates to the Vicon world coordinate system (WCS).

The L-shaped calibration wand lies on the floor and its five markers are known in both
frames: triangulated from ``wand_xy`` in camera coordinates, and given in the WCS JSON
(mm). A rigid transform (rotation + translation, after converting camera units to mm)
is fitted to these five correspondences:

    X_wcs = R @ (s * X_cam) + t        s = calibration units -> mm (60 for SUMediPose)

Two details matter:
* The wand markers are ordered differently: the camera JSON lists the short-arm end first
  and the corner second, the WCS JSON lists the corner first. ``CAM_TO_WCS_WAND_ORDER``
  reorders them.
* The stored WCS is a mirror image of the OpenCV camera frame (x right, y down, z forward),
  so the best fit is a reflection (det R = -1). Because the five wand markers are coplanar,
  a proper rotation fits them almost as well, so the fit alone cannot tell the two apart.
  The physical rule used here: the camera's "down" axis (+y) must point down in the WCS,
  where +y is up.
"""
from dataclasses import dataclass

import numpy as np

CAM_TO_WCS_WAND_ORDER = [1, 0, 2, 3, 4]


@dataclass
class WorldTransform:
    R: np.ndarray        # (3, 3)
    t: np.ndarray        # (3,)
    scale: float         # camera units -> mm
    residual_mm: np.ndarray   # (5,) wand fit residual per marker

    def apply(self, points_cam):
        return (self.scale * points_cam) @ self.R.T + self.t


def _fit(A, B, reflect):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, -np.sign(d) if reflect else np.sign(d)])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def fit_world_transform(wand_cam, wand_wcs, scale, up_axis=1):
    """wand_cam: (5, 3) triangulated wand in camera units (camera JSON order); wand_wcs: (5, 3) mm."""
    A = scale * wand_cam[CAM_TO_WCS_WAND_ORDER]
    candidates = []
    for reflect in (False, True):
        R, t = _fit(A, wand_wcs, reflect)
        cam_down_in_wcs = R[:, 1]                     # where the camera's +y axis points in the WCS
        residual = np.linalg.norm(A @ R.T + t - wand_wcs, axis=1)
        candidates.append((cam_down_in_wcs[up_axis] < 0, -residual.mean(), R, t, residual))
    _, _, R, t, residual = max(candidates, key=lambda c: (c[0], c[1]))
    return WorldTransform(R=R, t=t, scale=scale, residual_mm=residual)
