"""Export a recorded take: OpenSim (.trc markers, .mot joint angles), C3D, CSV, JSON and a PDF summary.

Everything is built from the session files and the take the user recorded in the app (frame range,
subject data, reps and feedback counts), and returned as one zip.
"""
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

PLANES = {"flex": "flexion/extension (sagittal)", "add": "ab/adduction (frontal)", "rot": "rotation (transverse)"}


def load_session(folder):
    folder = Path(folder)
    s = json.loads((folder / "session.json").read_text())
    frames = np.fromfile(folder / "frames.bin", np.float32).reshape(-1, s["stride"])
    return s, frames


def field(s, frames, name, shape=None):
    a, n = s["layout"][name]
    out = frames[:, a:a + n]
    return out.reshape((len(frames),) + shape) if shape else out


def _trc(path_name, markers, xyz, fps):
    n = len(xyz)
    buf = io.StringIO()
    buf.write(f"PathFileType\t4\t(X/Y/Z)\t{path_name}\n")
    buf.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
    buf.write(f"{fps:g}\t{fps:g}\t{n}\t{len(markers)}\tmm\t{fps:g}\t1\t{n}\n")
    buf.write("Frame#\tTime\t" + "\t\t\t".join(markers) + "\t\t\n")
    buf.write("\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(markers) + 1)) + "\n\n")
    for k in range(n):
        buf.write(f"{k + 1}\t{k / fps:.5f}\t" + "\t".join(f"{v:.3f}" for v in xyz[k].ravel()) + "\n")
    return buf.getvalue()


def _mot(name, channels, values, fps):
    n = len(values)
    buf = io.StringIO()
    buf.write(f"{name}\nversion=1\nnRows={n}\nnColumns={len(channels) + 1}\ninDegrees=yes\nendheader\n")
    buf.write("time\t" + "\t".join(channels) + "\n")
    for k in range(n):
        buf.write(f"{k / fps:.5f}\t" + "\t".join(f"{v:.4f}" for v in values[k]) + "\n")
    return buf.getvalue()


def _csv(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _c3d(markers, xyz, fps):
    import c3d
    w = c3d.Writer(point_rate=fps)
    w.set_point_labels([m.ljust(8)[:30] for m in markers])
    pts = np.zeros((len(markers), 5), np.float32)
    frames = []
    for k in range(len(xyz)):
        p = pts.copy()
        p[:, :3] = xyz[k]
        frames.append((p, np.zeros((0, 0), np.float32)))
    w.add_frames(frames)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def _pdf(s, req, t, met, met_gt, ang, ang_gt, has_gt):
    subj, recd, reps = req.get("subject", {}), req.get("recording", {}), req.get("reps", [])
    names = list(s["metrics"])
    chans = [c for c, u in zip(s["channels"], s["channel_used"]) if u]     # ang / ang_gt hold these channels

    def series(kind, name):
        """(live, vicon) of a joint angle ('ang', channel) or a distance measure ('met', name)."""
        if kind == "ang":
            i = chans.index(name)
            return ang[:, i], ang_gt[:, i]
        i = names.index(name)
        return met[:, i], met_gt[:, i]

    table = [("Knee flexion R", "ang", "knee_flexion_r", "deg"), ("Knee flexion L", "ang", "knee_flexion_l", "deg"),
             ("Hip flexion R", "ang", "hip_flexion_r", "deg"), ("Hip flexion L", "ang", "hip_flexion_l", "deg"),
             ("Ankle dorsiflexion R", "ang", "ankle_dorsiflexion_r", "deg"), ("Ankle dorsiflexion L", "ang", "ankle_dorsiflexion_l", "deg"),
             ("Knee adduction R (+ varus)", "ang", "knee_adduction_r", "deg"), ("Knee adduction L (+ varus)", "ang", "knee_adduction_l", "deg"),
             ("Hip adduction R", "ang", "hip_adduction_r", "deg"), ("Hip adduction L", "ang", "hip_adduction_l", "deg"),
             ("Trunk (thorax) flexion", "ang", "thorax_flexion_r", "deg"), ("Trunk side bending", "ang", "thorax_lateral_bending_r", "deg"),
             ("Pelvic tilt (anterior +)", "ang", "pelvis_anterior_tilt_r", "deg"), ("Pelvic obliquity", "ang", "pelvis_obliquity_r", "deg")]
    table += [(s["metrics"][m]["label"], "met", m, s["metrics"][m]["unit"]) for m in names]
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # page 1: subject, session, key results, reps
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "Markerless biomechanics - session summary", fontsize=16, weight="bold")
        fig.text(0.07, 0.925, f"{recd.get('name', s['name'])}   |   task: {recd.get('task', s['task'])}   |   "
                              f"{len(t) / s['fps']:.1f} s ({len(t)} frames)", fontsize=9)
        info = [("Subject", f"{subj.get('id', '')} {subj.get('name', '')}".strip() or "-"),
                ("Sex / age", f"{subj.get('sex', '-')} / {subj.get('age', '-')}"),
                ("Height / mass", f"{subj.get('height_cm', '-')} cm / {subj.get('mass_kg', '-')} kg"),
                ("Dominant side", subj.get("dominant", "-")),
                ("Pipeline", f"{s['pipeline']['detector']} -> {s['pipeline']['pose_model']} ({s['pipeline']['precision']})"
                             f" -> triangulation" + (f" -> LivePoseFormer" if s['pipeline']['lpf'] else "")
                             + " -> Plug-in Gait angles"),
                ("Recorded on", f"{s['pipeline']['gpu']}, {s['recorded_at']}"),
                ("Notes", recd.get("notes", "") or "-")]
        y = 0.89
        for k, v in info:
            fig.text(0.07, y, k, fontsize=9, weight="bold"); fig.text(0.28, y, str(v)[:95], fontsize=9)
            y -= 0.022
        y -= 0.015
        fig.text(0.07, y, "Real-time biomechanics over the take", fontsize=11, weight="bold"); y -= 0.025
        hdr = ["joint angle / measure", "min", "max", "mean"] + (["MAE vs Vicon"] if has_gt else [])
        rows = []
        for label, kind, name, unit in table:
            v, g = series(kind, name)
            row = [f"{label} ({unit})", f"{v.min():.1f}", f"{v.max():.1f}", f"{v.mean():.1f}"]
            if has_gt:
                row.append(f"{np.mean(np.abs(_wrap(v - g) if kind == 'ang' else v - g)):.1f}")
            rows.append(row)
        ax = fig.add_axes([0.07, y - 0.36, 0.86, 0.36]); ax.axis("off")
        widths = [0.44] + [0.56 / (len(hdr) - 1)] * (len(hdr) - 1)
        tb = ax.table(cellText=rows, colLabels=hdr, loc="upper left", cellLoc="left", colLoc="left", colWidths=widths)
        tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.25)
        y -= 0.40
        if reps:
            fig.text(0.07, y, f"Repetitions ({len(reps)})", fontsize=11, weight="bold"); y -= 0.02
            keys = [("n", "#"), ("duration_s", "time (s)"), ("peak_knee_r", "knee R"), ("peak_knee_l", "knee L"),
                    ("peak_hip", "hip"), ("max_drop_mm", "drop (mm)"), ("max_valgus_r", "valgus R"),
                    ("max_valgus_l", "valgus L"), ("max_trunk_lean", "trunk lean"), ("depth_reached", "depth")]
            rrows = [[(("yes" if r.get(k) else "no") if k == "depth_reached" else
                       (f"{r[k]:.1f}" if isinstance(r.get(k), (int, float)) and k != "n" else str(r.get(k, ""))))
                      for k, _ in keys] for r in reps]
            ax = fig.add_axes([0.07, max(0.03, y - 0.03 * (len(reps) + 1)), 0.86, 0.03 * (len(reps) + 1)]); ax.axis("off")
            tb = ax.table(cellText=rrows, colLabels=[h for _, h in keys], loc="upper left", cellLoc="center")
            tb.auto_set_font_size(False); tb.set_fontsize(7.5)
        pdf.savefig(fig); plt.close(fig)

        # page 2: the main curves
        tt = np.arange(len(t)) / s["fps"]
        panels = [("Knee flexion (deg)", [("knee_flexion_r", "R"), ("knee_flexion_l", "L")]),
                  ("Hip flexion (deg)", [("hip_flexion_r", "R"), ("hip_flexion_l", "L")]),
                  ("Ankle dorsiflexion (deg)", [("ankle_dorsiflexion_r", "R"), ("ankle_dorsiflexion_l", "L")]),
                  ("Knee adduction, + varus (deg)", [("knee_adduction_r", "R"), ("knee_adduction_l", "L")]),
                  ("Trunk / pelvis (deg)", [("thorax_flexion_r", "thorax flexion"), ("pelvis_anterior_tilt_r", "pelvic tilt")])]
        fig, axes = plt.subplots(len(panels), 1, figsize=(8.27, 11.69), sharex=True)
        for a, (title, ms) in zip(axes, panels):
            for (c, lab), col in zip(ms, ("#06b6d4", "#d946ef")):
                v, g = series("ang", c)
                a.plot(tt, v, color=col, lw=1.2, label=lab)
                if has_gt:
                    a.plot(tt, g, color=col, lw=0.8, ls="--", alpha=0.7)
            for r in reps:
                a.axvspan((r["start"] - req["range"][0]) / s["fps"], (r["end"] - req["range"][0]) / s["fps"],
                          color="0.92", zorder=0)
            a.set_ylabel(title, fontsize=8); a.legend(fontsize=7, loc="upper right"); a.grid(alpha=0.3)
        axes[-1].set_xlabel("time (s)" + ("   (dashed = Vicon)" if has_gt else ""))
        fig.suptitle("Real-time joint angles (grey = repetitions)", fontsize=11); fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # page 3: Plug-in Gait accuracy per plane and joint (simulation only)
        if has_gt:
            used = np.array(s["channel_used"])
            keys = [s["channel_keys"][c] for c in s["channels"]]
            err = np.abs(_wrap(ang - ang_gt))              # ang / ang_gt hold the used channels only
            planes = np.array([k[2] for k, u in zip(keys, used) if u])
            joints = np.array([k[0] for k, u in zip(keys, used) if u])
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.95, "Joint angles (Plug-in Gait) vs Vicon - mean absolute error (deg)", fontsize=12,
                     weight="bold")
            rows = [[PLANES[p], f"{err[:, planes == p].mean():.2f}", f"{np.median(err[:, planes == p]):.2f}"]
                    for p in PLANES]
            rows.append(["mean of the 3 planes", f"{np.mean([err[:, planes == p].mean() for p in PLANES]):.2f}", ""])
            ax = fig.add_axes([0.07, 0.75, 0.86, 0.17]); ax.axis("off")
            tb = ax.table(cellText=rows, colLabels=["plane", "MAE", "median"], loc="upper left", cellLoc="left")
            tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.4)
            jrows = []
            for j in dict.fromkeys(joints):
                jrows.append([j] + [f"{err[:, (joints == j) & (planes == p)].mean():.2f}"
                                    if ((joints == j) & (planes == p)).any() else "-" for p in PLANES])
            ax = fig.add_axes([0.07, 0.30, 0.86, 0.42]); ax.axis("off")
            tb = ax.table(cellText=jrows, colLabels=["joint (right + left)", "flex", "add", "rot"], loc="upper left",
                          cellLoc="left")
            tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.3)
            fig.text(0.07, 0.25, "Elbow ab/adduction and rotation are always 0 in this model and are left out.\n"
                                 "Vicon angles: the same model on the Vicon markers, with its own static trial.",
                     fontsize=8)
            pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()


def build_export(session_dir, req):
    """req: {subject, recording, range: [start, end), reps, thresholds, cue_counts} -> zip bytes."""
    s, frames = load_session(session_dir)
    a, b = req.get("range", [0, len(frames)])
    a, b = max(0, int(a)), min(len(frames), int(b))
    fr = frames[a:b]
    J = len(s["markers"])
    xyz, gt = field(s, fr, "xyz", (J, 3)), field(s, fr, "gt", (J, 3))
    use_unsmoothed = req.get("angle_smoothing") is False and "ang_unsmoothed" in s["layout"]
    ang, ang_gt = field(s, fr, "ang_unsmoothed" if use_unsmoothed else "ang"), field(s, fr, "ang_gt")
    met, met_gt = field(s, fr, "met"), field(s, fr, "met_gt")
    has_gt = bool(np.isfinite(gt).all() and np.abs(gt).sum() > 0)
    # The capture's world frame is left-handed (Y up, Z mirrored); OpenSim and C3D viewers expect a
    # right-handed frame, so the marker files get Z flipped (Y stays up).
    xyz_rh, gt_rh = xyz * [1, 1, -1], gt * [1, 1, -1]
    t = np.arange(b - a) / s["fps"]
    base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in req.get("recording", {}).get("name") or s["name"])
    used = np.array(s["channel_used"])
    channels = [c for c, u in zip(s["channels"], used) if u]
    mets = list(s["metrics"])

    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}/opensim/{base}_markers.trc", _trc(f"{base}_markers.trc", s["markers"], xyz_rh, s["fps"]))
        zf.writestr(f"{base}/opensim/{base}_joint_angles.mot", _mot(f"{base}_joint_angles", channels, ang[:, used], s["fps"]))
        zf.writestr(f"{base}/opensim/{base}_feedback_metrics.mot", _mot(f"{base}_feedback_metrics", mets, met, s["fps"]))
        zf.writestr(f"{base}/{base}_markers.c3d", _c3d(s["markers"], xyz_rh, s["fps"]))
        zf.writestr(f"{base}/csv/{base}_markers.csv", _csv(
            ["time"] + [f"{m}_{ax}" for m in s["markers"] for ax in "XYZ"],
            [[f"{t[k]:.5f}"] + [f"{v:.3f}" for v in xyz[k].ravel()] for k in range(len(t))]))
        zf.writestr(f"{base}/csv/{base}_joint_angles.csv", _csv(
            ["time"] + channels + ([f"vicon_{c}" for c in channels] if has_gt else []),
            [[f"{t[k]:.5f}"] + [f"{v:.3f}" for v in ang[k, used]] + ([f"{v:.3f}" for v in ang_gt[k, used]] if has_gt else [])
             for k in range(len(t))]))
        zf.writestr(f"{base}/csv/{base}_feedback_metrics.csv", _csv(
            ["time"] + mets + ([f"vicon_{m}" for m in mets] if has_gt else []),
            [[f"{t[k]:.5f}"] + [f"{v:.3f}" for v in met[k]] + ([f"{v:.3f}" for v in met_gt[k]] if has_gt else [])
             for k in range(len(t))]))
        if has_gt:
            zf.writestr(f"{base}/opensim/{base}_markers_vicon.trc",
                        _trc(f"{base}_markers_vicon.trc", s["markers"], gt_rh, s["fps"]))
        info = {"subject": req.get("subject", {}), "recording": req.get("recording", {}),
                "session": {k: s[k] for k in ("name", "trial", "recorded_at", "fps", "pipeline", "timing_ms")},
                "frames": [a, b], "recording_indices": s["indices"][a:b][:1] + s["indices"][a:b][-1:],
                "static_trial": s["static"], "reps": req.get("reps", []), "thresholds": req.get("thresholds", {}),
                "cue_counts": req.get("cue_counts", {}),
                "angle_smoothing": "off (unsmoothed angles exported)" if use_unsmoothed else s.get("angle_smoothing"),
                "units": {"markers": "mm, world frame (Y up); .trc/.c3d: Z flipped to make it right-handed",
                          "angles": "deg"}}
        zf.writestr(f"{base}/{base}_session.json", json.dumps(info, indent=2))
        zf.writestr(f"{base}/{base}_summary.pdf", _pdf(s, req, t, met, met_gt, ang[:, used], ang_gt[:, used], has_gt))
        zf.writestr(f"{base}/README.txt", (
            "OpenSim: load opensim/*_markers.trc as marker data (mm, Y up, right-handed as OpenSim expects; Z is\n"
            "flipped from the capture's left-handed world frame) for scaling and\n"
            "inverse kinematics with a model whose marker set matches (names are the 59 SUMediPose markers and\n"
            "joint centres). *_joint_angles.mot holds the Plug-in Gait angles (deg) of the live pipeline and\n"
            "*_feedback_metrics.mot the on-screen feedback metrics, for plotting in OpenSim.\n"
            "C3D: *_markers.c3d (mm). CSV: the same data as tables. summary.pdf: the session report.\n"))
    return z.getvalue(), f"{base}_export.zip"
