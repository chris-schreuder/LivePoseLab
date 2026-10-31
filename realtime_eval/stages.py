"""Errors at every stage of the live pipeline, from the engine's per-frame records.

``stream_sim.report.summarize`` already measures timing, latency, frame rates, drops, the 3D error
before / after LivePoseFormer, filled frames and the joint-angle errors. This module adds what that
report does not cover, so the error can be followed through the whole chain:

    person detector  ->  2D keypoints (3 views)  ->  triangulated 3D  ->  LivePoseFormer 3D  ->  joint angles
    box IoU, misses      px error, PCK, % height     MPJPE by region      processed / filled      per plane / joint

and the per-frame arrays saved with every run (``frames.npz``), which the paper report pools across
recordings for exact percentiles.
"""
import numpy as np

from pipeline.skeleton import KEYPOINTS_59

# body regions of the 59 keypoints, for the 3D error breakdown
REGIONS = {
    "head": ("FHD", "BHD"),
    "trunk & pelvis": ("C7", "CLAV", "STRN", "T10", "BAK", "ASI", "PSI", "Pel", "HJC"),
    "upper limbs": ("SHO", "SJC", "ELB", "EJC", "MElbow", "FRM", "WRA", "WRB", "WJC", "FIN"),
    "lower limbs": ("THI", "KNE", "MKNE", "KJC", "TIB", "ANK", "MMED", "AJC", "HEE", "TOE", "Foot_Out"),
}
REGION_OF = []
for name in KEYPOINTS_59:
    REGION_OF.append(next((r for r, keys in REGIONS.items() if any(k in name for k in keys)), "other"))
REGION_OF = np.array(REGION_OF)


def _stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": None, "median": None, "p95": None, "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "n": int(len(a))}


def stage_errors(records, rec):
    """Detector, 2D and 3D-by-region errors of one run. ``rec`` gives the ground truth per frame."""
    processed = sorted((r for r in records.values() if "kp2d" in r), key=lambda r: r["seq"])
    out = {"processed_frames": len(processed)}
    if not processed:
        return out
    err2d, rel2d, err3d = [], [], []
    for r in processed:
        fs = rec[r["index"]]
        e = np.linalg.norm(np.asarray(r["kp2d"], float) - fs.gt_xy, axis=-1)            # (views, 59) px
        err2d.append(e)
        h = np.asarray(r.get("box", fs.bboxes), float)[:, 3]                           # person box height per view
        rel2d.append(100 * e / np.maximum(h, 1)[:, None])
        err3d.append(np.linalg.norm(np.asarray(r["xyz_wcs"], float) - fs.gt_xyz_wcs, axis=-1))
    err2d, rel2d, err3d = np.array(err2d), np.array(rel2d), np.array(err3d)            # (frames, views, 59), (frames, 59)
    views = {}
    for v in range(err2d.shape[1]):
        e = err2d[:, v]
        views[rec.cams[v]] = {"px": _stats(e.ravel()), "pct_height": _stats(rel2d[:, v].ravel()),
                              "pck10": float(np.nanmean(e < 10) * 100), "pck20": float(np.nanmean(e < 20) * 100)}
    out["keypoints_2d"] = {
        "px": _stats(err2d.ravel()), "pct_height": _stats(rel2d.ravel()),
        "pck5": float(np.nanmean(err2d < 5) * 100), "pck10": float(np.nanmean(err2d < 10) * 100),
        "pck20": float(np.nanmean(err2d < 20) * 100), "per_view": views,
    }
    out["triangulated_3d"] = {
        "mpjpe_mm": _stats(np.nanmean(err3d, axis=1)),
        "per_region_mm": {reg: _stats(err3d[:, REGION_OF == reg].ravel()) for reg in REGIONS},
    }
    return out


def replay_angles(bio, xyz, index, static_xyz):
    """The real-time biomechanics stage run again on the exact 3D stream it received live (the same frames,
    frame gaps, continuity, glitch hold and smoothing), but calibrated with another static trial.

    ``bio`` is a separate BiomechStage with the live settings (not the live one); ``xyz`` / ``index`` are the
    output frames in stream order (``out_xyz`` / ``out_index``). Returns (angles, angles_unsmoothed), (N, 72)."""
    bio.static = bio.static_reference(np.asarray(static_xyz, float))
    bio.reset()
    out, unsmoothed, last = [], [], None
    for x, i in zip(xyz, index):
        gap = 1 if last is None else max(1, int(round(i - last)))
        last = i
        out.append(bio.angles(x, gap))
        unsmoothed.append(bio.last_unsmoothed)
    return np.array(out, np.float32), np.array(unsmoothed, np.float32)


def frame_arrays(records, s):
    """Per-frame arrays of one run (saved as frames.npz), for pooling across recordings."""
    t0 = s.get("_t0", 0.0)
    rows = sorted(records.values(), key=lambda r: r["seq"])
    proc = [r for r in rows if "t_done" in r]
    outp = [r for r in rows if "angles" in r]
    arr = lambda rs, f: np.array([f(r) for r in rs], np.float32) if rs else np.zeros(0, np.float32)   # noqa: E731
    d = {
        # processed frames (the pose network ran on them)
        "proc_seq": arr(proc, lambda r: r["seq"]), "proc_index": arr(proc, lambda r: r["index"]),
        "proc_t_capture": arr(proc, lambda r: r["t_capture"] - t0),
        "proc_latency_3d_ms": arr(proc, lambda r: (r["t_done"] - r["t_capture"]) * 1000),
        "proc_mpjpe_raw_mm": arr(proc, lambda r: r.get("err_raw_mm", np.nan)),
        "proc_xyz_raw": arr(proc, lambda r: r["xyz_wcs"]),
        # output frames (joint angles produced: processed + filled)
        "out_seq": arr(outp, lambda r: r["seq"]), "out_index": arr(outp, lambda r: r["index"]),
        "out_filled": arr(outp, lambda r: 1.0 if r.get("filled") else 0.0),
        "out_latency_smoothed_3d_ms": arr(outp, lambda r: (r["t_lpf_out"] - r["t_capture"]) * 1000 if "t_lpf_out" in r else np.nan),
        "out_latency_angles_ms": arr(outp, lambda r: (r["t_bio_end"] - r["t_capture"]) * 1000),
        "out_mpjpe_mm": arr(outp, lambda r: r.get("err_lpf_mm", r.get("err_raw_mm", np.nan))),
        "out_xyz": arr(outp, lambda r: r.get("xyz_lpf", r.get("xyz_wcs"))),
        "out_angles": arr(outp, lambda r: r["angles"]),
        "out_angles_unsmoothed": arr(outp, lambda r: r.get("angles_unsmoothed", r["angles"])),
        "out_angles_gt": arr(outp, lambda r: r.get("angles_gt", np.full(len(r["angles"]), np.nan))),
        "out_angle_mae_deg": arr(outp, lambda r: r.get("angle_mae", np.nan)),
    }
    return d
