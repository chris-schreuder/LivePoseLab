"""Optional last stage: real-time biomechanics (joint angles) from the world-frame 3D keypoints.

Uses the Plug-in Gait style model of the LivePoseFormer repository (``livepose/biomech``): segment
frames are built from the 59 keypoints and decomposed into flexion / adduction / rotation angles,
72 channels in all (pelvis, hip, knee, ankle, absolute ankle, foot progress, thorax, head, neck,
spine, shoulder, elbow; right and left).

**Static trial.** Done once before the camera clock starts, with the pipeline's own 3D result for
the first frame of the video as a placeholder static trial (the ground truth gets its own static
trial from its first frame, as in ``scripts/eval_joint_angles.py``). It sets two things:

* the lab alignment: the model assumes a right-handed lab frame with Y up and the subject's right
  along +Z (Plug-in Gait's progression frame). If the capture's world frame is left-handed (this one
  is: Y up, Z mirrored), Z is flipped; then the frame is turned about the vertical so the subject's
  right (left -> right hip centre) points along +Z. Joint angles (segment to segment) do not depend on
  this, but a mirrored frame flips the model's cross products and gave hip flexion ~180 deg standing,
  negative knee flexion and +-180 deg wraps; lab-relative angles (pelvis, thorax, head, foot
  progression) need the progression direction to mean anything;
* the ankle model offsets, with the repository's ``build_static_reference`` as it is.

**Per frame.** The repository's ``compute_angles`` takes about 5 ms per frame in pure Python and
scipy, and holds the GIL while doing it, which would slow every other thread of the pipeline. This
module computes the same 72 channels with a vectorised port: every segment frame of a frame (or a
batch of frames) is built with array operations, all relative rotations are stacked, and their
Euler angles are extracted in one go, following scipy's own path (matrix -> quaternion with
scipy's method, which also orthonormalises the slightly non-orthogonal radius frame -> Euler).
``calibrate`` checks the port against ``compute_angles`` on the static frame and on perturbed
copies of it and stores the largest difference in ``check_deg``.

**Continuity (real time).** An Euler decomposition is not unique: the same rotation is also
(a + 180, 180 - b, c + 180), and each angle is only defined modulo 360. Taking the textbook solution
every frame makes a channel jump by 180 or 360 deg when the rotation crosses a branch boundary (e.g.
shoulder flexion past 90 deg, or an angle near +-180), although nothing physical happened. Per frame
and per joint, ``angles`` therefore picks, of the two equivalent solutions, each shifted by the
nearest multiple of 360, the one closest to the previous frame: the same rotation, continuous
angles. Then a guard against pose glitches (mostly rotation about a bone's long axis, which a few
estimated keypoints define poorly): a change faster than ``max_step_deg`` per frame (default 25 deg
= 1500 deg/s at 60 fps, beyond human joint speeds) is not shown at once. The last good value is held
for up to ``hold_frames`` frames (default 3 = 50 ms); if the jump disappears in that time it was a
glitch and never reaches the output; if it persists, the angle follows at the rate limit until it
catches up. ``stats`` counts how often each correction was used.
"""
import sys
from pathlib import Path

import numpy as np

# channel order of the repository's ALL_H5_NAMES: (joint, side, plane) for every channel
JOINTS = ["pelvis", "hip", "knee", "ankle", "ankle_abs", "foot", "thorax", "head", "neck", "spine",
          "shoulder", "elbow"]
PLANES = ["flex", "add", "rot"]
PLANE_LABEL = {"flex": "flexion/extension (sagittal)", "add": "ab/adduction (frontal)",
               "rot": "rotation (transverse)"}


_I1, _I2 = [1, 2, 0], [2, 0, 1]


def _n(v):
    """Normalise along the last axis; leaves near-zero vectors unchanged (as the reference)."""
    m = np.sqrt((v * v).sum(-1, keepdims=True))
    return np.divide(v, m, out=v.copy(), where=m > 1e-9)


def _cross(a, b):
    """np.cross for (..., 3) arrays, without its argument handling (the hot path here)."""
    return a[..., _I1] * b[..., _I2] - a[..., _I2] * b[..., _I1]


def _rm(x, y, z):
    """Rotation matrix with the given column axes: (..., 3) x 3 -> (..., 3, 3)."""
    return np.stack([x, y, z], axis=-1)


def _t(m):
    return np.swapaxes(m, -1, -2)


def _rodrigues(axis, angle_deg):
    """Rotation of ``angle_deg`` around ``axis`` (..., 3) -> (..., 3, 3), as scipy's from_rotvec."""
    k = _n(axis)
    th = np.deg2rad(angle_deg)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([np.stack([z, -kz, ky], -1), np.stack([kz, z, -kx], -1), np.stack([-ky, kx, z], -1)], -2)
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _quat(m):
    """scipy's Rotation.from_matrix: quaternion (x, y, z, w) of (possibly non-orthogonal) matrices."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    dec = np.stack([m00, m11, m22, tr], -1)
    choice = np.argmax(dec, -1)
    q = np.empty(m.shape[:-2] + (4,))
    # choice 3 (trace largest)
    q3 = np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + tr], -1)
    q[...] = q3
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        qi = np.empty_like(q3)
        qi[..., i] = 1 - tr + 2 * m[..., i, i]
        qi[..., j] = m[..., j, i] + m[..., i, j]
        qi[..., k] = m[..., k, i] + m[..., i, k]
        qi[..., 3] = m[..., k, j] - m[..., j, k]
        q = np.where((choice == i)[..., None], qi, q)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _qmat(q):
    """Quaternion (x, y, z, w) -> orthonormal rotation matrix."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def lab_alignment(xyz, idx):
    """(3, 3) matrix applied to every point (x' = M @ x): right-handed frame (Z flipped if the frame is
    left-handed), turned about the vertical (Y) so the subject's right points along +Z."""
    xyz = np.asarray(xyz, float)
    up = np.array([0.0, 1.0, 0.0])
    right = xyz[idx["RHJC"]] - xyz[idx["LHJC"]]
    right[1] = 0
    right /= np.linalg.norm(right)
    fwd = (xyz[idx["LASI"]] + xyz[idx["RASI"]] - xyz[idx["LPSI"]] - xyz[idx["RPSI"]]) / 2
    fwd[1] = 0
    fwd -= right * fwd.dot(right)
    F = np.diag([1.0, 1.0, -1.0]) if fwd.dot(np.cross(right, up)) > 0 else np.eye(3)
    r = F @ right
    th = np.arctan2(-r[0], r[2])
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ F


def _euler(m, seq):
    """Intrinsic Euler angles in degrees, as scipy's as_euler for 'ZXY' and 'XZY'."""
    r = _qmat(_quat(m))
    if seq == "ZXY":       # R = Rz(a) Rx(b) Ry(c)
        a = np.arctan2(-r[..., 0, 1], r[..., 1, 1])
        b = np.arcsin(np.clip(r[..., 2, 1], -1, 1))
        c = np.arctan2(-r[..., 2, 0], r[..., 2, 2])
    else:                  # "XZY": R = Rx(a) Rz(b) Ry(c)
        a = np.arctan2(r[..., 2, 1], r[..., 1, 1])
        b = np.arcsin(np.clip(-r[..., 0, 1], -1, 1))
        c = np.arctan2(r[..., 0, 2], r[..., 0, 0])
    return np.degrees(np.stack([a, b, c], -1))


class OneEuro:
    """One Euro filter (Casiez et al., CHI 2012), vectorised over channels: an exponential low-pass
    whose cutoff rises with the signal's speed. Still or slow -> low cutoff, jitter removed; fast ->
    high cutoff, little lag. ``min_cutoff`` (Hz) sets the smoothing at rest, ``beta`` (per deg/s) how
    quickly the cutoff opens with speed. Causal, no lookahead, about 10 operations per channel."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


# (min_cutoff Hz, beta per deg/s) per plane, tuned against Vicon on S1A3D1 (HRNet and ViTPose, raw 3D).
# Ab/adduction and rotation get stronger smoothing: small ranges, the most pose noise, slow motion.
#   standard  high-frequency noise ~Vicon's level with HRNet (x1.1, raw x5.6), flexion lag ~33 ms,
#             knee peaks ~0.9 deg lower, angle error +0.1 deg
#   light     noise x3, lag ~17 ms, knee peaks ~0.4 deg lower, angle error -0.15 deg
# "auto" (in the pipelines) = standard on raw 3D, off when LivePoseFormer already smooths the 3D.
SMOOTHING = {"off": None,
             "light": {"flex": (2.0, 0.05), "other": (2.0, 0.05)},
             "standard": {"flex": (0.7, 0.01), "other": (0.4, 0.004)}}


class BiomechStage:
    def __init__(self, repo, max_step_deg=25.0, hold_frames=3, smoothing=None, fps=60.0):
        sys.path.insert(0, str(Path(repo).resolve()))
        from livepose.biomech import joint_angles as kja   # noqa: E402
        self.kja = kja
        self.channels = list(kja.ALL_H5_NAMES)
        self.channel_key = {name: key for key, name in kja.H5_NAMES.items()}   # name -> (joint, side, plane)
        self.idx = {name: i for i, name in enumerate(kja.KEYPOINT_NAMES_59)}
        self.max_step = max_step_deg
        self.hold_frames = hold_frames
        self.fps = fps
        self.set_smoothing(smoothing)
        self.static = None
        self.check_deg = None
        self.reset()

    def set_smoothing(self, smoothing):
        """None / "off", a preset name from ``SMOOTHING``, (min_cutoff Hz, beta) for every channel, or
        {"flex": (...), "other": (...)} for the flexion channels and the ab/adduction + rotation ones."""
        params = SMOOTHING.get(smoothing, smoothing) if isinstance(smoothing, str) or smoothing is None else smoothing
        self.smoothing = params
        self._smoother = None
        if params:
            if not isinstance(params, dict):
                params = {"flex": params, "other": params}
            flex = np.array([self.channel_key[c][2] == "flex" for c in self.channels])
            mc = np.where(flex, params["flex"][0], params["other"][0])
            beta = np.where(flex, params["flex"][1], params["other"][1])
            self._smoother = OneEuro(mc, beta)

    def reset(self):
        """Start a new stream: forget the previous frame used for continuity (and the smoothing state)."""
        self._prev = None
        self.last_unsmoothed = None
        if self._smoother is not None:
            self._smoother.reset()
        self.stats = {"frames": 0, "branch_switches": 0, "wraps_360": 0, "glitches_held": 0, "rate_limited": 0}

    # ------------------------------------------------------------------ static trial
    def calibrate(self, static_xyz):
        """Static trial (before the clock starts): lab alignment and ankle model offsets from one (59, 3) frame."""
        self.static = self.static_reference(static_xyz)
        self.check_deg = self._check(static_xyz)
        self.reset()

    def static_reference(self, static_xyz):
        M = lab_alignment(static_xyz, self.idx)
        x = np.asarray(static_xyz, float) @ M.T
        res = self.kja.build_static_reference(x, self.kja.KEYPOINT_NAMES_59)
        out = {s: (res[f"{s}_static_offsets"]["plantar_flexion_offset_deg"],
                   res[f"{s}_static_offsets"]["rotation_offset_deg"]) for s in ("R", "L")}
        out["M"] = M
        out["left_handed_input"] = bool(np.linalg.det(M) < 0)
        return out

    def _check(self, static_xyz, n=20):
        """Max difference (deg) between this port and the repository's compute_angles (same aligned input)."""
        rng = np.random.default_rng(0)
        frames = np.asarray(static_xyz, float)[None] + rng.normal(0, 30, (n, 59, 3))
        frames[0] = static_xyz
        M = self.static["M"]
        aligned = frames @ M.T
        ref_static = self.kja.build_static_reference(aligned[0], self.kja.KEYPOINT_NAMES_59)
        ref = np.array([[self.kja.compute_angles(self.kja.frame_to_marker_dict(f, self.kja.KEYPOINT_NAMES_59),
                                                 ref_static)[c] for c in self.channels] for f in aligned])
        d = _wrap(self.angles_batch(frames, continuous=False) - ref)
        return float(np.nanmax(np.abs(d)))

    # ------------------------------------------------------------------ per frame
    def angles(self, xyz, frames_since_last=1):
        """Real time: (59, 3) world-frame keypoints (mm) -> (72,) joint angles in degrees (channel order
        ``channels``), continuous with the previous call (see the module docstring). Pass the number of
        camera frames since the previous call if frames were dropped (scales the rate limit). With
        smoothing on, the One Euro filter runs after the continuity step (it must see unwrapped
        angles); the unsmoothed angles of the call are kept in ``last_unsmoothed``."""
        x = np.asarray(xyz, float)[None] @ self.static["M"].T
        E, self._prev = self._continuous(self._euler_all(x, self.static), self._prev, self.stats,
                                         [max(1, frames_since_last)])
        out = self.last_unsmoothed = self._channels(E)[0]
        if self._smoother is not None:
            out = self._smoother(out, max(1, frames_since_last) / self.fps)
        return out

    def angles_batch(self, xyz, static=None, continuous=True, steps=None):
        """(N, 59, 3) consecutive frames -> (N, 72). ``static`` defaults to the calibrated static trial.
        With ``continuous``, the frames are made continuous in order, as ``angles`` does live (``steps``:
        camera frames between consecutive rows, if they are not consecutive)."""
        static = static or self.static
        x = np.asarray(xyz, float) @ static["M"].T
        E = self._euler_all(x, static)
        if continuous:
            E, _ = self._continuous(E, None, None, steps)
        return self._channels(E)

    def _continuous(self, E, state, stats, steps=None):
        """Per frame and joint: of the two equivalent Euler solutions, each moved to the nearest 360 deg
        multiple, take the one closest to the previous frame; then hold glitches and limit the rate.
        ``state`` = (previous output (joints, 3), frames each angle has been held) or None."""
        names = list(E)
        arr = np.stack([E[n] for n in names], 1)                   # (N, joints, 3)
        out = np.empty_like(arr)
        prev, held = state if state is not None else (None, np.zeros(arr.shape[1:], int))
        for k in range(len(arr)):
            cur = arr[k]
            if prev is None:
                c = cur
            else:
                alt = cur + np.array([180.0, 0.0, 180.0])
                alt[:, 1] = 180.0 - cur[:, 1]
                c1, c2 = prev + _wrap(cur - prev), prev + _wrap(alt - prev)
                use_alt = np.abs(c2 - prev).sum(1) + 1e-9 < np.abs(c1 - prev).sum(1)
                c = np.where(use_alt[:, None], c2, c1)
                d = c - prev
                lim = self.max_step * (steps[k] if steps is not None else 1)
                fast = np.abs(d) > lim
                hold = fast & (held < self.hold_frames)               # sudden jump: keep the last good value
                limit = fast & ~hold                                   # still there after the hold: catch up
                if stats is not None:
                    raw = np.where(use_alt[:, None], alt, cur)
                    stats["branch_switches"] += int(use_alt.sum())
                    stats["wraps_360"] += int((np.abs(c - _wrap(raw)) > 1.0).sum())
                    stats["glitches_held"] += int((hold & (held == 0)).sum())
                    stats["rate_limited"] += int(limit.any(1).sum())
                c = np.where(hold, prev, np.where(limit, prev + np.clip(d, -lim, lim), c))
                held = np.where(hold, held + 1, np.where(limit, held, 0))
            out[k] = prev = c
            if stats is not None:
                stats["frames"] += 1
        return {n: out[:, j] for j, n in enumerate(names)}, (prev, held)

    def _euler_all(self, xyz, static):
        """Raw Euler triples (N, 3) of every segment pair (aligned input)."""
        p = {name: xyz[:, i] for name, i in self.idx.items()}

        # pelvis
        MASI = (p["LASI"] + p["RASI"]) / 2
        SACR = (p["LPSI"] + p["RPSI"]) / 2
        Z_P = _n(p["RHJC"] - p["LHJC"])
        Y_P = _n(_cross(_n(SACR - MASI), Z_P))
        R_P = _rm(_n(_cross(Y_P, Z_P)), Y_P, Z_P)
        # femurs
        Y_FR = _n(p["RHJC"] - p["RKJC"])
        X_FR = _n(_cross(_n(p["RKJC"] - p["RTHI"]), Y_FR))
        R_FR = _rm(X_FR, Y_FR, _n(_cross(X_FR, Y_FR)))
        Y_FL = _n(p["LHJC"] - p["LKJC"])
        X_FL = _n(_cross(_n(p["LTHI"] - p["LKJC"]), Y_FL))
        R_FL = _rm(X_FL, Y_FL, _n(_cross(X_FL, Y_FL)))
        # tibias
        Y_TR = _n(p["RKJC"] - p["RAJC"])
        X_TR = _n(_cross(_n(p["RAJC"] - p["RTIB"]), Y_TR))
        Z_TR = _n(_cross(X_TR, Y_TR))
        R_TR = _rm(X_TR, Y_TR, Z_TR)
        Y_TL = _n(p["LKJC"] - p["LAJC"])
        X_TL = _n(_cross(Y_TL, _n(p["LAJC"] - p["LTIB"])))
        Z_TL = _n(_cross(X_TL, Y_TL))
        R_TL = _rm(X_TL, Y_TL, Z_TL)
        # corrected (dynamic) feet, from the ankle model and the static offsets
        foot = {}
        for s in ("R", "L"):
            KJC, AJC, ANK, TOE = p[s + "KJC"], p[s + "AJC"], p[s + "TIB"], p[s + "TOE"]
            y = _n(KJC - AJC)
            zt = _n(ANK - AJC)
            x = _n(_cross(y, zt)) if s == "R" else _n(_cross(zt, y))
            z = _n(_cross(x, y))
            Rm = _rodrigues(y, -self.kja.TIBIAL_TORSION[s])
            z_un = _n((Rm @ z[..., None])[..., 0])
            fx = _n(TOE - AJC)
            fy = _n(_cross(z_un, fx))
            fz = _n(_cross(fx, fy))
            axes = _rm(fx, fy, fz)
            pf_off, rot_off = static[s]
            axes = _rodrigues(axes[..., :, 2], pf_off) @ axes
            axes = _rodrigues(axes[..., :, 1], rot_off) @ axes
            foot[s] = _rm(_n(axes[..., :, 0]), _n(axes[..., :, 1]), _n(axes[..., :, 2]))
        # absolute feet (HEE -> TOE axis in the lab frame)
        fa = {}
        for s, AF in (("R", Z_TR), ("L", Z_TL)):
            Z_FA = _n(p[s + "HEE"] - p[s + "TOE"])
            X_FA = _n(_cross(AF, Z_FA))
            fa[s] = _rm(Z_FA, X_FA, _n(_cross(Z_FA, X_FA)))
        # thorax
        lower, upper = (p["STRN"] + p["T10"]) / 2, (p["CLAV"] + p["C7"]) / 2
        back, front = (p["C7"] + p["T10"]) / 2, (p["CLAV"] + p["STRN"]) / 2
        Y_TH = _n(upper - lower)
        Z_TH = _n(_cross(_n(front - back), Y_TH))
        R_TH = _rm(_n(_cross(Y_TH, Z_TH)), Y_TH, Z_TH)
        # head
        X_HD = _n((p["LFHD"] + p["RFHD"]) / 2 - (p["LBHD"] + p["RBHD"]) / 2)
        Z_HDt = _n((p["LFHD"] + p["LBHD"]) / 2 - (p["RFHD"] + p["RBHD"]) / 2)
        Y_HD = _n(_cross(X_HD, Z_HDt))
        R_HD = _rm(X_HD, Y_HD, _n(_cross(X_HD, Y_HD)))
        # humeri and radii
        hum, rad = {}, {}
        for s in ("R", "L"):
            Y_H = _n(p[s + "SJC"] - p[s + "EJC"])
            Z_H = _n(_cross(_n(p[s + "WJC"] - p[s + "EJC"]), Y_H))
            hum[s] = _rm(_n(_cross(Y_H, Z_H)), Y_H, Z_H)
            Y_R = _n(p[s + "EJC"] - p[s + "WJC"])
            rad[s] = _rm(_n(_cross(Y_R, Z_H)), Y_R, Z_H)

        # every relative rotation decomposed with ZXY, stacked so the Euler step runs once
        zxy = [("pelvis", R_P), ("hip_r", _t(R_P) @ R_FR), ("hip_l", _t(R_P) @ R_FL),
               ("knee_r", _t(R_FR) @ R_TR), ("knee_l", _t(R_FL) @ R_TL),
               ("ankle_r", _t(R_TR) @ foot["R"]), ("ankle_l", _t(R_TL) @ foot["L"]),
               ("ankle_abs_r", fa["R"]), ("ankle_abs_l", fa["L"]),
               ("foot_r", foot["R"]), ("foot_l", foot["L"]), ("thorax", R_TH), ("head", R_HD),
               ("neck", _t(R_TH) @ R_HD), ("spine", _t(R_P) @ R_TH),
               ("elbow_r", _t(hum["R"]) @ rad["R"]), ("elbow_l", _t(hum["L"]) @ rad["L"])]
        e = _euler(np.stack([m for _, m in zxy], 1), "ZXY")
        E = {name: e[:, k] for k, (name, _) in enumerate(zxy)}
        sh = _euler(np.stack([_t(R_TH) @ hum["R"], _t(R_TH) @ hum["L"]], 1), "XZY")
        E["shoulder_r"], E["shoulder_l"] = sh[:, 0], sh[:, 1]
        return E

    def _channels(self, E):
        """Euler triples -> the 72 channels, with the model's sign conventions."""
        out = {}

        def put(joint, side, flex, add, rot):
            out[(joint, side, "flex")], out[(joint, side, "add")], out[(joint, side, "rot")] = flex, add, rot

        a = E["pelvis"]; put("pelvis", "r", -a[:, 0], -a[:, 1], a[:, 2]); put("pelvis", "l", -a[:, 0], a[:, 1], -a[:, 2])
        a = E["hip_r"]; put("hip", "r", a[:, 0], a[:, 1], a[:, 2])
        a = E["hip_l"]; put("hip", "l", a[:, 0], -a[:, 1], -a[:, 2])
        a = E["knee_r"]; put("knee", "r", -a[:, 0], a[:, 1], a[:, 2])
        a = E["knee_l"]; put("knee", "l", -a[:, 0], -a[:, 1], -a[:, 2])
        a = E["ankle_r"]; put("ankle", "r", a[:, 0], -a[:, 1], a[:, 2])
        a = E["ankle_l"]; put("ankle", "l", a[:, 0], a[:, 1], -a[:, 2])
        a = E["ankle_abs_r"]; put("ankle_abs", "r", a[:, 0], -a[:, 1], a[:, 2])
        a = E["ankle_abs_l"]; put("ankle_abs", "l", a[:, 0], a[:, 1], -a[:, 2])
        a = E["foot_r"]; put("foot", "r", -a[:, 0] - 90, a[:, 1], a[:, 2])
        a = E["foot_l"]; put("foot", "l", -a[:, 0] - 90, -a[:, 1], -a[:, 2])
        a = E["thorax"]; put("thorax", "r", -a[:, 0], -a[:, 1], a[:, 2]); put("thorax", "l", -a[:, 0], a[:, 1], -a[:, 2])
        a = E["head"]; put("head", "r", a[:, 0], -a[:, 1], a[:, 2]); put("head", "l", a[:, 0], a[:, 1], -a[:, 2])
        a = E["neck"]; put("neck", "r", -a[:, 0], a[:, 1], -a[:, 2]); put("neck", "l", -a[:, 0], -a[:, 1], a[:, 2])
        a = E["spine"]; put("spine", "r", -a[:, 0], a[:, 1], -a[:, 2]); put("spine", "l", -a[:, 0], -a[:, 1], a[:, 2])
        a = E["shoulder_r"]; put("shoulder", "r", a[:, 1], -a[:, 0], a[:, 2])
        a = E["shoulder_l"]; put("shoulder", "l", a[:, 1], a[:, 0], -a[:, 2])
        a = E["elbow_r"]; put("elbow", "r", a[:, 0], a[:, 1], a[:, 2])
        a = E["elbow_l"]; put("elbow", "l", a[:, 0], a[:, 1], a[:, 2])
        return np.stack([out[self.channel_key[c]] for c in self.channels], 1)
