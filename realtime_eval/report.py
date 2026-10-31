"""Paper report of an end-to-end evaluation: everything in one folder, ready to quote.

    results/<run>/
        REPORT.md                    the full report (setup, real-time performance, accuracy along the
                                     pipeline, joint angles, per-recording results, validity summary)
        tables.tex                   the main tables as LaTeX (booktabs)
        results_per_recording.csv    one row per model x recording, every metric
        results_summary.csv          per model: mean, SD, min, max across recordings + pooled values
        figures/*.png|pdf            latency CDF, stage times, error by stage, joint-angle errors,
                                     per-recording errors, example joint-angle time series

Statistics: "mean ± SD" is across recordings (each recording is one observation); "pooled" weights
every frame equally across recordings; latency percentiles are exact, from all frames of all
recordings (frames.npz). Built by run_eval.py at the end, or alone with ``run_eval.py --report-only``.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PLANES = [("flex", "Sagittal (flexion/extension)"), ("add", "Frontal (ab/adduction)"), ("rot", "Transverse (rotation)")]
STAGES = [("DET (GPU: person detector)", "Person detector"), ("PRE (CPU)", "Crop + resize"), ("GPU", "2D pose network"),
          ("POST (CPU)", "Triangulation + world frame")]
COLORS = {"ViTPose": "#0891b2", "HRNet": "#c026d3"}
JOINT_LABEL = {"pelvis": "Pelvis", "hip": "Hip", "knee": "Knee", "ankle": "Ankle", "ankle_abs": "Foot (absolute)",
               "foot": "Foot progression", "thorax": "Thorax", "head": "Head", "neck": "Neck", "spine": "Spine",
               "shoulder": "Shoulder", "elbow": "Elbow"}
MIN_ROM_DEG = 10.0   # channels that move less than this in a recording (Vicon) are left out of the % ROM metrics


def g(d, *path, default=np.nan):
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def load_runs(out):
    runs = []
    for f in sorted(Path(out).glob("*/*/summary.json")):
        r = json.loads(f.read_text())
        npz = f.parent / "frames.npz"
        r["_frames"] = dict(np.load(npz)) if npz.exists() else {}
        r["_dir"] = f.parent
        runs.append(r)
    return runs


STATICS = {"pipeline": "out_angles", "vicon": "out_angles_gtstatic"}   # static trial -> live angles in frames.npz


def channel_rows(r, static="pipeline"):
    """Per angle channel of one run (final angles, all output frames):
    - range of motion of Vicon and of the live angles, MAE, MAE as % of the Vicon ROM and the ROM error in % of
      the Vicon ROM (> 0 = live overestimates);
    - offset (mean signed error, live − Vicon) and the errors after removing it, i.e. after aligning the mean of
      the live angle with the mean of the Vicon angle over the recording (how well the movement is tracked);
    - Pearson r between live and Vicon (unchanged by the alignment).
    ``static``: "pipeline" = the live angles (static trial from the pipeline's 3D of frame 0), "vicon" = the same
    stream with the static trial from the Vicon markers of frame 0 (add_gt_static.py / run_eval.py)."""
    fr, ch = r["_frames"], r["summary"].get("biomech_channels", [])
    if fr.get(STATICS[static]) is None or not ch:
        return []
    live, gt = fr[STATICS[static]].astype(float), fr["out_angles_gt"].astype(float)
    excluded = set(g(r["summary"], "biomech", "excluded_channels", default=[]) or [])
    joints = list(JOINT_LABEL)                       # channels come as joint x side (r, l) x plane (flex, add, rot)
    rows = []
    for k, c in enumerate(ch):
        ok = np.isfinite(live[:, k]) & np.isfinite(gt[:, k])
        if c in excluded or ok.sum() < 10:
            continue
        a, b = live[ok, k], gt[ok, k]
        rom_gt, rom_live = float(np.ptp(b)), float(np.ptp(a))
        e = (a - b + 180) % 360 - 180
        offset = float(e.mean())
        mae, mae_al = float(np.mean(np.abs(e))), float(np.mean(np.abs(e - offset)))
        pct = (lambda v: 100 * v / rom_gt) if rom_gt > 0 else (lambda v: np.nan)   # noqa: E731
        b_ = a - e                                   # Vicon on the same branch as the live angle
        rr = float(np.corrcoef(a, b_)[0, 1]) if a.std() > 1e-9 and b_.std() > 1e-9 else np.nan
        rows.append({"model": r["model_name"], "static": static, "trial": r["trial"], "subject": r["subject"],
                     "action": r["action"], "channel": c, "joint": joints[k // 6], "side": c[-1], "plane": PLANES[k % 3][0],
                     "rom_vicon_deg": rom_gt, "rom_live_deg": rom_live, "mae_deg": mae, "mae_pct_rom": pct(mae),
                     "rom_error_pct": pct(rom_live - rom_gt), "offset_deg": offset, "abs_offset_deg": abs(offset),
                     "mae_aligned_deg": mae_al,
                     "mae_aligned_pct_rom": pct(mae_al), "rmse_deg": float(np.sqrt(np.mean(e ** 2))),
                     "rmse_aligned_deg": float(e.std()), "pearson_r": rr, "included": rom_gt >= MIN_ROM_DEG})
    return rows


def _sel(rows, included, match):
    return [x for x in rows if (x["included"] or not included) and all(x[k] == want for k, want in match.items())]


def rom_median(rows, key, **match):
    """Median over the channels that move at least MIN_ROM_DEG (for % ROM metrics)."""
    v = [x[key] for x in _sel(rows, True, match)]
    return (float(np.nanmedian(v)), len(v)) if v else (np.nan, 0)


def ch_mean(rows, key, **match):
    """Mean over all channels (for errors in degrees)."""
    v = np.array([x[key] for x in _sel(rows, False, match)], float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def r_mean(rows, **match):
    """Mean Pearson r (through Fisher z) over the channels that move at least MIN_ROM_DEG."""
    v = np.array([x["pearson_r"] for x in _sel(rows, True, match)], float)
    v = v[np.isfinite(v)]
    return float(np.tanh(np.arctanh(np.clip(v, -0.9999, 0.9999)).mean())) if len(v) else np.nan


def run_row(r):
    """Every scalar metric of one run (one row of results_per_recording.csv)."""
    s, st = r["summary"], r.get("stages", {})
    lpf, bio, det = s.get("lpf", {}), s.get("biomech", {}), s.get("detector", {})
    mae = g(bio, "mae_deg", default={})
    acc = g(lpf, "accuracy", default={})
    filled = g(lpf, "filled_frames", default=0) or 0
    row = {
        "model": r["model_name"], "trial": r["trial"], "subject": r["subject"], "action": r["action"],
        "duration_s": r["duration_s"], "frames_captured": s["captured"], "frames_processed": s["processed"],
        "dropped_pct": s["dropped_pct"], "filled_pct": 100 * filled / max(s["captured"], 1),
        "pose_rate_fps": s["output_rate_fps"], "output_rate_fps": g(lpf, "output_rate_fps", default=g(bio, "output_rate_fps")),
        "max_sustainable_fps": s["max_sustainable_fps"], "bottleneck": s["bottleneck"],
        "latency_3d_mean_ms": g(s, "delay_ms", "mean"), "latency_3d_p95_ms": g(s, "delay_ms", "p95"),
        "latency_smoothed3d_mean_ms": g(lpf, "delay_ms_mean"), "latency_smoothed3d_p95_ms": g(lpf, "delay_ms_p95"),
        "latency_angles_mean_ms": g(bio, "delay_ms_mean"), "latency_angles_p95_ms": g(bio, "delay_ms_p95"),
        "delay_trend_ms_per_s": s["delay_trend_ms_per_s"], "clock_late_p99_ms": s["clock_late_ms_p99"],
        "reader_underruns": s.get("reader_underruns", 0),
    }
    for key, label in STAGES:
        row[f"t_{label}_ms"] = g(s, "stage_service_ms", key, "mean")
        row[f"t_{label}_p95_ms"] = g(s, "stage_service_ms", key, "p95")
    row.update({
        "t_LivePoseFormer_ms": g(lpf, "compute_ms_mean"), "t_LivePoseFormer_p95_ms": g(lpf, "compute_ms_p95"),
        "t_Biomechanics_ms": g(bio, "compute_ms_mean"), "t_Biomechanics_p95_ms": g(bio, "compute_ms_p95"),
        "lpf_busy_pct": g(lpf, "busy_pct"), "lpf_keeps_up": g(lpf, "keeps_up", default=""),
        "bio_busy_pct": g(bio, "busy_pct"), "bio_keeps_up": g(bio, "keeps_up", default=""),
        "det_iou_mean": g(det, "iou_json_mean"), "det_iou_p5": g(det, "iou_json_p5"), "det_conf_mean": g(det, "conf_mean"),
        "det_misses": g(det, "misses", default=0),
        "kp2d_px_mean": g(st, "keypoints_2d", "px", "mean"), "kp2d_px_median": g(st, "keypoints_2d", "px", "median"),
        "kp2d_pct_height_mean": g(st, "keypoints_2d", "pct_height", "mean"),
        "kp2d_pck5": g(st, "keypoints_2d", "pck5"), "kp2d_pck10": g(st, "keypoints_2d", "pck10"),
        "kp2d_pck20": g(st, "keypoints_2d", "pck20"),
        "mpjpe_raw_mm": g(s, "mpjpe_world_mm"), "mpjpe_raw_p95_mm": g(st, "triangulated_3d", "mpjpe_mm", "p95"),
        "mpjpe_lpf_processed_mm": g(acc, "mpjpe_after_mm"), "mpjpe_filled_mm": g(lpf, "filled_mpjpe_mm"),
        "mpjpe_all_output_mm": g(lpf, "all_output_mpjpe_mm", default=g(acc, "mpjpe_after_mm")),
        "lpf_frames_improved_pct": g(acc, "frames_improved_pct"),
        "jitter_raw_mm": g(acc, "accel_error_before"), "jitter_lpf_mm": g(acc, "accel_error_after"),
        "jitter_all_output_mm": g(lpf, "all_output_accel_error", default=g(acc, "accel_error_after")),
    })
    for reg in ("head", "trunk & pelvis", "upper limbs", "lower limbs"):
        row[f"mpjpe_raw_{reg}_mm"] = g(st, "triangulated_3d", "per_region_mm", reg, "mean")
    for grp, name in (("before LPF", "raw3d"), ("unsmoothed", "lpf_unsmoothed"), ("all output", "final"),
                      ("measured", "measured"), ("filled", "filled")):
        for p, _ in PLANES:
            row[f"angle_{name}_{p}_deg"] = g(mae, grp, p)
        row[f"angle_{name}_mean_deg"] = g(mae, grp, "mean")
    row.update({"angle_jitter_final": g(bio, "jitter_deg"), "angle_jitter_unsmoothed": g(bio, "jitter_unsmoothed_deg"),
                "angle_jitter_raw3d": g(bio, "jitter_raw_deg"),
                "continuity_glitches_held": g(bio, "continuity", "glitches_held", default=0),
                "continuity_rate_limited": g(bio, "continuity", "rate_limited", default=0),
                "continuity_branch_switches": g(bio, "continuity", "branch_switches", default=0),
                "angle_frames": g(bio, "frames", default=0)})
    for static, sfx in (("pipeline", ""), ("vicon", "_vicon_static")):
        cr = channel_rows(r, static)
        for p, _ in PLANES:
            if static != "pipeline":
                row[f"angle_final_{p}_deg{sfx}"] = ch_mean(cr, "mae_deg", plane=p)
            row[f"angle_final_{p}_pct_rom{sfx}"] = rom_median(cr, "mae_pct_rom", plane=p)[0]
            row[f"rom_error_{p}_pct{sfx}"] = rom_median(cr, "rom_error_pct", plane=p)[0]
            row[f"angle_aligned_{p}_deg{sfx}"] = ch_mean(cr, "mae_aligned_deg", plane=p)
            row[f"angle_aligned_{p}_pct_rom{sfx}"] = rom_median(cr, "mae_aligned_pct_rom", plane=p)[0]
            row[f"angle_offset_abs_{p}_deg{sfx}"] = ch_mean(cr, "abs_offset_deg", plane=p)
            row[f"angle_pearson_r_{p}{sfx}"] = r_mean(cr, plane=p)
        if static != "pipeline":
            row[f"angle_final_mean_deg{sfx}"] = ch_mean(cr, "mae_deg")
        row[f"angle_final_pct_rom{sfx}"] = rom_median(cr, "mae_pct_rom")[0]
        row[f"rom_error_pct{sfx}"] = rom_median(cr, "rom_error_pct")[0]
        row[f"angle_aligned_mean_deg{sfx}"] = ch_mean(cr, "mae_aligned_deg")
        row[f"angle_aligned_pct_rom{sfx}"] = rom_median(cr, "mae_aligned_pct_rom")[0]
        row[f"angle_pearson_r{sfx}"] = r_mean(cr)
    return row


def _num(v):
    return isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and np.isfinite(v)


def model_summary(rows):
    """Per model: mean, SD, min, max across recordings of every numeric column."""
    out = {}
    for m in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == m]
        d = {}
        for k in rs[0]:
            vals = np.array([r[k] for r in rs if _num(r.get(k))], float)
            if len(vals):
                d[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "min": float(vals.min()), "max": float(vals.max()), "n": int(len(vals))}
        out[m] = d
    return out


def pooled(runs, model):
    """Frame-level pooling over the recordings of one model."""
    rs = [r for r in runs if r["model_name"] == model]
    fr = [r["_frames"] for r in rs if r["_frames"]]
    cat = lambda k: np.concatenate([f[k] for f in fr if k in f and len(f[k])]) if fr else np.zeros(0)   # noqa: E731
    lat_a, lat_s, lat_3d = cat("out_latency_angles_ms"), cat("out_latency_smoothed_3d_ms"), cat("proc_latency_3d_ms")
    q = lambda a, p: float(np.nanpercentile(a, p)) if len(a) else np.nan   # noqa: E731
    d = {"output_frames": int(len(lat_a)), "processed_frames": int(len(lat_3d)),
         "captured_frames": int(sum(r["summary"]["captured"] for r in rs))}
    for name, a in (("angles", lat_a), ("smoothed3d", lat_s), ("3d", lat_3d)):
        d[f"lat_{name}"] = {"mean": float(np.nanmean(a)) if len(a) else np.nan, "p50": q(a, 50), "p95": q(a, 95),
                            "p99": q(a, 99), "max": float(np.nanmax(a)) if len(a) else np.nan}
    for thr in (50, 100, 150, 200):
        d[f"angles_within_{thr}ms_pct"] = float(np.mean(lat_a <= thr) * 100) if len(lat_a) else np.nan
    mp_raw, mp_out = cat("proc_mpjpe_raw_mm"), cat("out_mpjpe_mm")
    d["mpjpe_raw_mm"] = {"mean": float(np.nanmean(mp_raw)), "median": q(mp_raw, 50), "p95": q(mp_raw, 95)} if len(mp_raw) else {}
    d["mpjpe_output_mm"] = {"mean": float(np.nanmean(mp_out)), "median": q(mp_out, 50), "p95": q(mp_out, 95)} if len(mp_out) else {}
    # frame-weighted joint-angle errors per plane and per joint
    w = np.array([g(r["summary"], "biomech", "frames", default=0) for r in rs], float)
    for grp in ("all output", "before LPF", "unsmoothed", "measured", "filled"):
        for p, _ in PLANES:
            v = np.array([g(r["summary"], "biomech", "mae_deg", grp, p) for r in rs], float)
            ok = np.isfinite(v) & (w > 0)
            d[f"angle_{grp}_{p}"] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    joints = {}
    for r, wi in zip(rs, w):
        for j, planes in (g(r["summary"], "biomech", "per_joint_deg", default={}) or {}).items():
            for p, v in planes.items():
                if v is not None and np.isfinite(v):
                    joints.setdefault(j, {}).setdefault(p, []).append((v, wi))
    d["per_joint"] = {j: {p: float(sum(v * x for v, x in vs) / max(sum(x for _, x in vs), 1)) for p, vs in ps.items()}
                      for j, ps in joints.items()}
    return d


def ms(summary, model, key, d=1, unit=""):
    v = summary.get(model, {}).get(key)
    if not v:
        return "–"
    return f"{v['mean']:.{d}f} ± {v['sd']:.{d}f}{unit}" if v["n"] > 1 else f"{v['mean']:.{d}f}{unit}"


def f(v, d=1, unit=""):
    return f"{v:.{d}f}{unit}" if _num(v) else "–"


# ---------------------------------------------------------------------- figures
def figures(runs, models, pool, fig_dir):
    fig_dir.mkdir(exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{name}.{ext}", dpi=160)
        plt.close(fig)
        made.append(name)

    # latency CDF
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in models:
        for key, ls, lab in (("out_latency_angles_ms", "-", "capture → joint angles"), ("proc_latency_3d_ms", "--", "capture → 3D")):
            a = np.concatenate([r["_frames"][key] for r in runs if r["model_name"] == m and r["_frames"].get(key) is not None])
            a = np.sort(a[np.isfinite(a)])
            if len(a):
                ax.plot(a, np.arange(1, len(a) + 1) / len(a) * 100, ls, color=COLORS.get(m), label=f"{m}: {lab}")
    ax.set_xlabel("latency (ms)"); ax.set_ylabel("frames (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("End-to-end latency per frame (all recordings pooled)", fontsize=10)
    save(fig, "latency_cdf")

    # stage times
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [lab for _, lab in STAGES] + ["LivePoseFormer", "Biomechanics"]
    x = np.arange(len(labels))
    wdt = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        rows = [run_row(r) for r in runs if r["model_name"] == m]
        means = [np.nanmean([rw[f"t_{lab}_ms"] for rw in rows]) for lab in labels]
        sds = [np.nanstd([rw[f"t_{lab}_ms"] for rw in rows]) for lab in labels]
        ax.bar(x + i * wdt, means, wdt, yerr=sds, color=COLORS.get(m), label=m, capsize=2)
    ax.axhline(1000 / 60, color="0.4", ls=":", lw=1, label="60 fps budget (16.7 ms)")
    ax.set_xticks(x + wdt * (len(models) - 1) / 2); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("time per frame (ms)"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Processing time per stage (mean ± SD across recordings)", fontsize=10)
    save(fig, "stage_times")

    # 3D error by stage
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cats = [("mpjpe_raw_mm", "Triangulated 3D\n(processed frames)"), ("mpjpe_lpf_processed_mm", "After LivePoseFormer\n(processed frames)"),
            ("mpjpe_filled_mm", "Filled by\nLivePoseFormer"), ("mpjpe_all_output_mm", "All output\n(60 fps)")]
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        rows = [run_row(r) for r in runs if r["model_name"] == m]
        means = [np.nanmean([rw[k] for rw in rows]) if any(_num(rw[k]) for rw in rows) else 0 for k, _ in cats]
        sds = [np.nanstd([rw[k] for rw in rows]) if any(_num(rw[k]) for rw in rows) else 0 for k, _ in cats]
        ax.bar(x + i * wdt, means, wdt, yerr=sds, color=COLORS.get(m), label=m, capsize=2)
    ax.set_xticks(x + wdt * (len(models) - 1) / 2); ax.set_xticklabels([c for _, c in cats], fontsize=8)
    ax.set_ylabel("MPJPE vs Vicon (mm)"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    ax.set_title("3D marker error along the pipeline", fontsize=10)
    save(fig, "error_by_stage_3d")

    # joint-angle error by stage and plane
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        grps = [("before LPF", "raw 3D"), ("unsmoothed", "after LPF"), ("all output", "final (smoothed)")]
        for gi, (grp, lab) in enumerate(grps):
            vals = [pool[m].get(f"angle_{grp}_{p}", np.nan) for p, _ in PLANES]
            ax.bar(np.arange(3) + gi * 0.27, np.nan_to_num(vals), 0.27, label=lab, alpha=0.6 + 0.2 * gi, color=COLORS.get(m))
        ax.set_xticks(np.arange(3) + 0.27); ax.set_xticklabels(["sagittal", "frontal", "transverse"], fontsize=8)
        ax.set_title(m, fontsize=10); ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("joint-angle MAE vs Vicon (deg)"); axes[0][-1].legend(fontsize=7)
    fig.suptitle("Joint-angle error along the pipeline (frame-weighted, all recordings)", fontsize=10)
    save(fig, "error_by_stage_angles")

    # per recording
    fig, ax = plt.subplots(figsize=(7, 3.4))
    trials = list(dict.fromkeys(r["trial"] for r in runs))
    x = np.arange(len(trials))
    for i, m in enumerate(models):
        vals = []
        for t in trials:
            rr = [run_row(r) for r in runs if r["model_name"] == m and r["trial"] == t]
            vals.append(rr[0]["mpjpe_all_output_mm"] if rr and _num(rr[0]["mpjpe_all_output_mm"]) else 0)
        ax.bar(x + i * wdt, vals, wdt, color=COLORS.get(m), label=m)
    ax.set_xticks(x + wdt * (len(models) - 1) / 2); ax.set_xticklabels(trials, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE, all output (mm)"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    ax.set_title("3D error of the 60 fps output per recording", fontsize=10)
    save(fig, "per_recording_mpjpe")

    # example time series: knee flexion, first recording of each model
    for m in models:
        rr = [r for r in runs if r["model_name"] == m and r["_frames"].get("out_angles") is not None]
        if not rr:
            continue
        r = rr[0]
        ch = r["summary"].get("biomech_channels", [])
        if "knee_flexion_r" not in ch:
            continue
        k = ch.index("knee_flexion_r")
        fr = r["_frames"]
        t = fr["out_seq"] / r["config"]["fps"]
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.plot(t, fr["out_angles_gt"][:, k], "k-", lw=1, label="Vicon")
        live = fr["out_angles"][:, k]
        filled = fr["out_filled"] > 0
        ax.plot(t[~filled], live[~filled], ".", ms=2, color=COLORS.get(m), label="live (measured frame)")
        if filled.any():
            ax.plot(t[filled], live[filled], ".", ms=2, color="#f97316", label="live (filled by LivePoseFormer)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("knee flexion R (deg)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_title(f"{m}, {r['trial']}: real-time knee flexion vs Vicon", fontsize=10)
        save(fig, f"timeseries_knee_{m.lower()}")
    return made


# ---------------------------------------------------------------------- report
def build(out):
    out = Path(out)
    runs = load_runs(out)
    if not runs:
        print(f"no finished runs in {out}")
        return
    plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else {}
    rows = [run_row(r) for r in runs]
    names = {"vitpose": "ViTPose", "hrnet": "HRNet"}
    present = list(dict.fromkeys(r["model"] for r in rows))
    models = [names[m] for m in plan.get("models", []) if names.get(m) in present] +         [m for m in present if m not in [names.get(x) for x in plan.get("models", [])]]   # in the order they were run
    summary = model_summary(rows)
    pool = {m: pooled(runs, m) for m in models}
    keys = list(rows[0])
    with open(out / "results_per_recording.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if _num(v) else v) for k, v in r.items()} for r in rows])
    with open(out / "results_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "metric", "mean", "sd", "min", "max", "n_recordings"])
        for m in models:
            for k, v in summary[m].items():
                w.writerow([m, k, f"{v['mean']:.4f}", f"{v['sd']:.4f}", f"{v['min']:.4f}", f"{v['max']:.4f}", v["n"]])
    rom = [x for r in runs for x in channel_rows(r)]                 # live angles (pipeline static trial)
    rom_v = [x for r in runs for x in channel_rows(r, "vicon")]      # same stream, Vicon static trial
    fmt = lambda x: {k: (f"{v:.4f}" if _num(v) else v) for k, v in x.items()}   # noqa: E731
    if rom:
        with open(out / "angles_per_channel.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rom[0]))
            w.writeheader()
            w.writerows([fmt(x) for x in rom + rom_v])
        # static x subject x action x joint x plane, right and left pooled (one recording per subject and action)
        groups = {}
        cols = ("model", "static", "subject", "action", "trial", "joint", "plane")
        for x in rom + rom_v:
            groups.setdefault(tuple(x[c] for c in cols), []).append(x)
        detail = []
        for key, xs in groups.items():
            mean = lambda k: float(np.nanmean([x[k] for x in xs]))   # noqa: E731
            rv = np.array([x["pearson_r"] for x in xs], float)
            rv = rv[np.isfinite(rv)]
            detail.append(dict(zip(cols, key)) | {
                "sides": len(xs), "rom_vicon_deg": mean("rom_vicon_deg"), "min_rom_vicon_deg": min(x["rom_vicon_deg"] for x in xs),
                "mae_deg": mean("mae_deg"), "mae_aligned_deg": mean("mae_aligned_deg"),
                "abs_offset_deg": mean("abs_offset_deg"),          # signed offsets of r / l can cancel
                "rmse_deg": mean("rmse_deg"), "rmse_aligned_deg": mean("rmse_aligned_deg"),
                "mae_pct_rom": mean("mae_pct_rom"), "mae_aligned_pct_rom": mean("mae_aligned_pct_rom"),
                "pearson_r": float(np.tanh(np.arctanh(np.clip(rv, -0.9999, 0.9999)).mean())) if len(rv) else np.nan})
        with open(out / "angles_subject_action_joint_plane.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(detail[0]))
            w.writeheader()
            w.writerows([fmt(x) for x in detail])
        (out / "rom_per_channel.csv").unlink(missing_ok=True)       # replaced by angles_per_channel.csv
    (out / "pooled.json").write_text(json.dumps(pool, indent=1, default=float))
    figs = figures(runs, models, pool, out / "figures")

    r0 = runs[0]
    cfg, sysi = r0["config"], r0["system"]
    trials = list(dict.fromkeys(r["trial"] for r in runs))
    L = []
    A = L.append
    A("# Real-time markerless biomechanics pipeline: end-to-end evaluation\n")
    A(f"Generated {datetime.now():%Y-%m-%d %H:%M} from {len(runs)} runs ({len(models)} pose models × {len(trials)} recordings).\n")
    A("## 1. Setup\n")
    A("**Pipeline (per camera frame set, 3 synchronised views):** YOLO person detector → 2D pose network "
      "(HRNet-W48 or ViTPose-B, 59 keypoints) → DLT triangulation → world frame (wand) → LivePoseFormer "
      "(streaming spatio-temporal transformer, 3-frame lookahead; fills frames the pose network could not "
      "process) → real-time biomechanics (Plug-in Gait joint angles, 72 channels, continuity and glitch hold, "
      "One Euro smoothing). Every stage runs in its own thread; LivePoseFormer and the biomechanics stage are "
      "decoupled through queues and never hold up the pose stages.\n")
    A("**Simulation:** three hardware-synchronised 60 fps cameras are simulated by releasing each recorded frame "
      "set on a fixed 60 Hz clock, read from disk and decoded just ahead of the clock. A frame the pipeline cannot "
      "take when the next one arrives is dropped (policy *latest*), as in a live system, and LivePoseFormer fills it. "
      "The static trial (ankle model calibration, lab alignment) is the first frame of each recording, taken before "
      "the clock starts. Errors are against Vicon ground truth; Vicon joint angles come from the same biomechanical "
      "model applied to the Vicon markers, with its own static trial.\n")
    A("| Setting | Value |\n|---|---|")
    for k, v in [("GPU", f"{sysi.get('gpu', '–')} ({sysi.get('gpu_memory_gb', '–')} GB)"),
                 ("CPU", f"{sysi.get('cpu', '–')} ({sysi.get('cpu_count', '–')} logical cores)"),
                 ("Software", f"PyTorch {sysi.get('torch')}, CUDA {sysi.get('cuda')}, Python {sysi.get('python')}, {sysi.get('platform', '')}"),
                 ("Camera rate", f"{cfg['fps']:g} fps × 3 views"), ("Pose precision", cfg["precision"] + (" + torch.compile" if cfg["compile"] else "")),
                 ("Person detector", cfg["detector"]), ("LivePoseFormer", f"{cfg['lpf_checkpoint']} (fills dropped frames: {cfg['fill_dropped']})"),
                 ("Joint-angle smoothing", f"One Euro, preset '{cfg['angle_smoothing']}'"),
                 ("Frame policy", "latest (drop frames that cannot be processed in time)")]:
        A(f"| {k} | {v} |")
    A("\n**Recordings**\n\n| Recording | Subject | Action | Frames | Duration (s) |\n|---|---|---|---:|---:|")
    for t in trials:
        r = next(x for x in runs if x["trial"] == t)
        A(f"| {t} | {r['subject']} | {r['action']} | {r['n_frames']} | {r['duration_s']:.1f} |")
    tot = sum(next(x for x in runs if x["trial"] == t)["duration_s"] for t in trials)
    A(f"\nTotal: {tot:.0f} s of 60 fps video per model ({tot * 60 * 3:.0f} camera images).\n")

    # 2. real-time performance
    A("## 2. Real-time performance\n")
    A("Mean ± SD across recordings; latency percentiles pooled over all frames.\n")
    hdr = "| Metric | " + " | ".join(models) + " |\n|---|" + "---:|" * len(models)
    A(hdr)
    perf = [
        ("Pose network rate (frames/s)", lambda m: ms(summary, m, "pose_rate_fps")),
        ("Frames dropped by the pose stages (%)", lambda m: ms(summary, m, "dropped_pct")),
        ("Frames filled by LivePoseFormer (%)", lambda m: ms(summary, m, "filled_pct")),
        ("Output rate, joint angles (frames/s)", lambda m: ms(summary, m, "output_rate_fps")),
        ("Max sustainable pose rate (frames/s)", lambda m: ms(summary, m, "max_sustainable_fps")),
        ("Latency capture → 3D, mean (ms)", lambda m: ms(summary, m, "latency_3d_mean_ms")),
        ("Latency capture → smoothed 3D, mean (ms)", lambda m: ms(summary, m, "latency_smoothed3d_mean_ms")),
        ("Latency capture → joint angles, mean (ms)", lambda m: ms(summary, m, "latency_angles_mean_ms")),
        ("Latency capture → joint angles, p50 / p95 / p99 (ms, pooled)",
         lambda m: f"{f(pool[m]['lat_angles']['p50'], 0)} / {f(pool[m]['lat_angles']['p95'], 0)} / {f(pool[m]['lat_angles']['p99'], 0)}"),
        ("Joint-angle frames within 100 / 150 / 200 ms (%)",
         lambda m: f"{f(pool[m]['angles_within_100ms_pct'])} / {f(pool[m]['angles_within_150ms_pct'])} / {f(pool[m]['angles_within_200ms_pct'])}"),
        ("Delay trend (ms per s; > 0 = falling behind)", lambda m: ms(summary, m, "delay_trend_ms_per_s", 2)),
        ("LivePoseFormer keeps up / busy (%)", lambda m: f"{sum(1 for r in rows if r['model'] == m and r['lpf_keeps_up'] is True)}/"
                                                      f"{sum(1 for r in rows if r['model'] == m)} runs, {ms(summary, m, 'lpf_busy_pct', 0)}"),
        ("Biomechanics keeps up / busy (%)", lambda m: f"{sum(1 for r in rows if r['model'] == m and r['bio_keeps_up'] is True)}/"
                                                    f"{sum(1 for r in rows if r['model'] == m)} runs, {ms(summary, m, 'bio_busy_pct', 0)}"),
        ("Camera clock lateness p99 (ms) / reader underruns", lambda m: f"{ms(summary, m, 'clock_late_p99_ms', 2)} / {int(np.nansum([r['reader_underruns'] for r in rows if r['model'] == m]))}"),
    ]
    for lab, fn in perf:
        A(f"| {lab} | " + " | ".join(fn(m) for m in models) + " |")
    A("\n**Processing time per frame and stage (ms, mean ± SD across recordings; p95 in brackets)**\n")
    A(hdr.replace("Metric", "Stage"))
    for lab in [lab for _, lab in STAGES] + ["LivePoseFormer", "Biomechanics"]:
        A(f"| {lab} | " + " | ".join(f"{ms(summary, m, f't_{lab}_ms', 2)} ({ms(summary, m, f't_{lab}_p95_ms', 1)})" for m in models) + " |")
    A("\nThe pose stages form a pipeline, so the pose rate is set by the slowest stage (bottleneck), while the "
      "latency is roughly the sum of the stages plus waiting. LivePoseFormer adds its 3-frame lookahead "
      "(50 ms at 60 fps) to the latency of the smoothed 3D and the joint angles.\n")

    # 3. accuracy along the pipeline
    A("## 3. Accuracy along the pipeline\n")
    A("Errors against Vicon at the output of every stage (mean ± SD across recordings).\n")
    A(hdr.replace("Metric", "Stage · output · metric"))
    acc_rows = [
        ("Person detector · box IoU with reference box", lambda m: ms(summary, m, "det_iou_mean", 3)),
        ("Person detector · missed detections (views)", lambda m: f"{int(np.nansum([r['det_misses'] for r in rows if r['model'] == m]))}"),
        ("2D pose · keypoint error (px)", lambda m: ms(summary, m, "kp2d_px_mean")),
        ("2D pose · keypoint error (% of person height)", lambda m: ms(summary, m, "kp2d_pct_height_mean", 2)),
        ("2D pose · PCK@5 / @10 / @20 px (%)", lambda m: f"{ms(summary, m, 'kp2d_pck5', 0)} / {ms(summary, m, 'kp2d_pck10', 0)} / {ms(summary, m, 'kp2d_pck20', 0)}"),
        ("Triangulation · MPJPE, processed frames (mm)", lambda m: ms(summary, m, "mpjpe_raw_mm")),
        ("Triangulation · MPJPE p95 (mm)", lambda m: ms(summary, m, "mpjpe_raw_p95_mm")),
        ("LivePoseFormer · MPJPE, processed frames (mm)", lambda m: ms(summary, m, "mpjpe_lpf_processed_mm")),
        ("LivePoseFormer · MPJPE, filled frames (mm)", lambda m: ms(summary, m, "mpjpe_filled_mm")),
        ("LivePoseFormer · MPJPE, all 60 fps output (mm)", lambda m: ms(summary, m, "mpjpe_all_output_mm")),
        ("3D jitter (acceleration error, mm/frame²): raw → LivePoseFormer",
         lambda m: f"{ms(summary, m, 'jitter_raw_mm', 2)} → {ms(summary, m, 'jitter_all_output_mm', 2)}"),
        ("Joint angles · MAE, mean of 3 planes: raw 3D → after LPF → final (deg)",
         lambda m: f"{f(np.nanmean([pool[m][f'angle_before LPF_{p}'] for p, _ in PLANES]), 2)} → "
                   f"{f(np.nanmean([pool[m][f'angle_unsmoothed_{p}'] for p, _ in PLANES]), 2)} → "
                   f"{f(np.nanmean([pool[m][f'angle_all output_{p}'] for p, _ in PLANES]), 2)}"),
        ("Joint angles · jitter (deg/frame²): raw 3D → after LPF → final",
         lambda m: f"{ms(summary, m, 'angle_jitter_raw3d', 2)} → {ms(summary, m, 'angle_jitter_unsmoothed', 2)} → {ms(summary, m, 'angle_jitter_final', 2)}"),
    ]
    for lab, fn in acc_rows:
        A(f"| {lab} | " + " | ".join(fn(m) for m in models) + " |")
    A("\n**3D error of the triangulated markers by body region (mm, mean across recordings)**\n")
    A(hdr.replace("Metric", "Region"))
    for reg in ("head", "trunk & pelvis", "upper limbs", "lower limbs"):
        A(f"| {reg} | " + " | ".join(ms(summary, m, f"mpjpe_raw_{reg}_mm") for m in models) + " |")
    A("\n**2D keypoint error per camera (px, mean across recordings)**\n")
    cams = cfg["cams"].split(",")
    A(hdr.replace("Metric", "Camera"))
    for c in cams:
        vals = []
        for m in models:
            v = [g(r.get("stages", {}), "keypoints_2d", "per_view", c, "px", "mean") for r in runs if r["model_name"] == m]
            vals.append(f(np.nanmean([x for x in v if _num(x)]) if any(_num(x) for x in v) else np.nan))
        A(f"| {c} | " + " | ".join(vals) + " |")

    # 4. joint angles
    A("\n## 4. Joint angles (real-time biomechanics output)\n")
    A("Mean absolute error against Vicon, frame-weighted over all recordings (deg). Elbow ab/adduction and rotation "
      "are always 0 in the model and are excluded.\n")
    A("| Plane | " + " | ".join(f"{m} raw 3D | {m} after LPF | {m} final" for m in models) + " |\n|---|" + "---:|" * (3 * len(models)))
    for p, lab in PLANES + [("mean", "Mean of 3 planes")]:
        cells = []
        for m in models:
            for grp in ("before LPF", "unsmoothed", "all output"):
                v = np.nanmean([pool[m][f"angle_{grp}_{q}"] for q, _ in PLANES]) if p == "mean" else pool[m][f"angle_{grp}_{p}"]
                cells.append(f(v, 2))
        A(f"| {lab} | " + " | ".join(cells) + " |")
    A("\n**Measured vs filled frames (final angles, MAE mean of 3 planes, deg)**\n")
    A(hdr.replace("Metric", "Frames"))
    for grp, lab in (("measured", "Measured (pose network ran)"), ("filled", "Filled by LivePoseFormer")):
        A(f"| {lab} | " + " | ".join(f(np.nanmean([pool[m][f'angle_{grp}_{p}'] for p, _ in PLANES]), 2) for m in models) + " |")
    A("\n**Per joint (final angles, MAE, deg; right and left pooled)**\n")
    A("| Joint | " + " | ".join(f"{m} sag. | {m} front. | {m} transv." for m in models) + " |\n|---|" + "---:|" * (3 * len(models)))
    for j in JOINT_LABEL:
        cells = []
        for m in models:
            pj = pool[m]["per_joint"].get(j, {})
            cells += [f(pj.get(p), 2) for p, _ in PLANES]
        A(f"| {JOINT_LABEL[j]} | " + " | ".join(cells) + " |")
    if rom:
        A("\n**Error relative to the range of motion (final angles)**\n")
        A(f"Per angle channel and recording: MAE as % of the Vicon range of motion (max − min of the Vicon angle in "
          f"that recording), and the ROM error, (live ROM − Vicon ROM) / Vicon ROM × 100 (> 0 = the live angles "
          f"overestimate the ROM). Channels with a Vicon ROM below {MIN_ROM_DEG:g}° in a recording are left out (a "
          f"percentage of an almost still angle is meaningless). Values are medians over recordings and sides; n = "
          f"channel-recordings included. Per channel: `angles_per_channel.csv`.\n")
        A("| Plane | " + " | ".join(f"{m} MAE (% ROM) | {m} ROM error (%) | {m} n" for m in models) + " |\n|---|" + "---:|" * (3 * len(models)))
        for p, lab in PLANES + [(None, "All planes")]:
            cells = []
            for m in models:
                sel = {"model": m} if p is None else {"model": m, "plane": p}
                v, n = rom_median(rom, "mae_pct_rom", **sel)
                cells += [f(v, 1), f(rom_median(rom, "rom_error_pct", **sel)[0], 1), str(n)]
            A(f"| {lab} | " + " | ".join(cells) + " |")
        A("\n*Per joint: MAE as % of ROM (median Vicon ROM in brackets), right and left pooled*\n")
        A("| Joint | " + " | ".join(f"{m} sag. | {m} front. | {m} transv." for m in models) + " |\n|---|" + "---:|" * (3 * len(models)))
        for j in JOINT_LABEL:
            cells = []
            for m in models:
                for p, _ in PLANES:
                    v, n = rom_median(rom, "mae_pct_rom", model=m, joint=j, plane=p)
                    cells.append(f"{f(v, 0)} ({f(rom_median(rom, 'rom_vicon_deg', model=m, joint=j, plane=p)[0], 0)}°)" if n else "–")
            A(f"| {JOINT_LABEL[j]} | " + " | ".join(cells) + " |")
        A("\n*Per action and per subject: MAE as % of ROM, all planes*\n")
        A("| Group | " + " | ".join(f"{m} sag. | {m} front. | {m} transv. | {m} all" for m in models) + " |\n|---|" + "---:|" * (4 * len(models)))
        for key, label in (("action", "Action"), ("subject", "Subject")):
            for val in sorted(dict.fromkeys(x[key] for x in rom), key=lambda s: (len(s), s)):
                cells = []
                for m in models:
                    cells += [f(rom_median(rom, "mae_pct_rom", model=m, plane=p, **{key: val})[0], 1) for p, _ in PLANES]
                    cells.append(f(rom_median(rom, "mae_pct_rom", model=m, **{key: val})[0], 1))
                A(f"| {label} {val} | " + " | ".join(cells) + " |")

        A("\n**Tracking after offset alignment (final angles)**\n")
        A("Each live angle channel is shifted so that its mean over the recording equals the Vicon mean (the offset "
          "= mean signed error, live − Vicon, is removed). The aligned error measures how well the movement itself is "
          "tracked, independent of a constant offset (static trial, joint-centre and segment-axis definitions). The "
          "alignment uses the whole recording, so it is an offline analysis; live, the offset would come from a "
          "calibration. Pearson r is the same before and after alignment. Errors in degrees: mean over all channels "
          f"and recordings; % ROM: median, and r: mean through Fisher z, over channels with a Vicon ROM ≥ {MIN_ROM_DEG:g}°. "
          "Every subject × action × joint × plane: `angles_subject_action_joint_plane.csv`; per channel: "
          "`angles_per_channel.csv`.\n")
        A("| Plane | " + " | ".join(f"{m} MAE (deg) raw → aligned | {m} MAE (% ROM) raw → aligned | {m} mean abs. offset (deg) | {m} r"
                                    for m in models) + " |\n|---|" + "---:|" * (4 * len(models)))
        for p, lab in PLANES + [(None, "All planes")]:
            cells = []
            for m in models:
                sel = {"model": m} if p is None else {"model": m, "plane": p}
                off = np.nanmean([abs(x["offset_deg"]) for x in _sel(rom, False, sel)])
                cells += [f"{f(ch_mean(rom, 'mae_deg', **sel), 2)} → {f(ch_mean(rom, 'mae_aligned_deg', **sel), 2)}",
                          f"{f(rom_median(rom, 'mae_pct_rom', **sel)[0], 1)} → {f(rom_median(rom, 'mae_aligned_pct_rom', **sel)[0], 1)}",
                          f(off, 2), f(r_mean(rom, **sel), 3)]
            A(f"| {lab} | " + " | ".join(cells) + " |")
        A("\n*Per joint: MAE (deg) raw → aligned, Pearson r in brackets; right and left pooled*\n")
        A("| Joint | " + " | ".join(f"{m} sag. | {m} front. | {m} transv." for m in models) + " |\n|---|" + "---:|" * (3 * len(models)))
        for j in JOINT_LABEL:
            cells = []
            for m in models:
                for p, _ in PLANES:
                    sel = {"model": m, "joint": j, "plane": p}
                    cells.append(f"{f(ch_mean(rom, 'mae_deg', **sel), 1)} → {f(ch_mean(rom, 'mae_aligned_deg', **sel), 1)} "
                                 f"({f(r_mean(rom, **sel), 2)})" if _sel(rom, False, sel) else "–")
            A(f"| {JOINT_LABEL[j]} | " + " | ".join(cells) + " |")
        A("\n*Per action and per subject: MAE (deg) raw → aligned, Pearson r in brackets*\n")
        A("| Group | " + " | ".join(f"{m} sag. | {m} front. | {m} transv. | {m} all" for m in models) + " |\n|---|" + "---:|" * (4 * len(models)))
        for key, label in (("action", "Action"), ("subject", "Subject")):
            for val in sorted(dict.fromkeys(x[key] for x in rom), key=lambda s: (len(s), s)):
                cells = []
                for m in models:
                    for p in [p for p, _ in PLANES] + [None]:
                        sel = {"model": m, key: val} | ({} if p is None else {"plane": p})
                        cells.append(f"{f(ch_mean(rom, 'mae_deg', **sel), 1)} → {f(ch_mean(rom, 'mae_aligned_deg', **sel), 1)} "
                                     f"({f(r_mean(rom, **sel), 2)})")
                A(f"| {label} {val} | " + " | ".join(cells) + " |")

    if rom_v:
        A("\n**Static trial from Vicon instead of the pipeline (final angles)**\n")
        A("Live, the static trial (lab alignment, ankle plantar-flexion and rotation offsets) is the pipeline's own 3D "
          "of the first frame. Here the real-time biomechanics stage is replayed on exactly the 3D stream it received "
          "in each run (same frames and frame gaps, continuity, glitch hold and smoothing), calibrated with the Vicon "
          "markers of the first frame instead, the same static trial the Vicon reference angles use. This shows how "
          "much of the angle error comes from the static trial. Pipeline → Vicon static; definitions as in the tables "
          "above (aligned = after removing the mean offset). Every subject × action × joint × plane for both: "
          "`angles_subject_action_joint_plane.csv` (column `static`).\n")
        A("| Plane | " + " | ".join(f"{m} MAE (deg) | {m} aligned MAE (deg) | {m} MAE (% ROM) | {m} mean abs. offset (deg) | {m} r"
                                    for m in models) + " |\n|---|" + "---:|" * (5 * len(models)))
        for p, lab in PLANES + [(None, "All planes")]:
            cells = []
            for m in models:
                sel = {"model": m} if p is None else {"model": m, "plane": p}
                cells += [f"{f(ch_mean(rom, 'mae_deg', **sel), 2)} → {f(ch_mean(rom_v, 'mae_deg', **sel), 2)}",
                          f"{f(ch_mean(rom, 'mae_aligned_deg', **sel), 2)} → {f(ch_mean(rom_v, 'mae_aligned_deg', **sel), 2)}",
                          f"{f(rom_median(rom, 'mae_pct_rom', **sel)[0], 1)} → {f(rom_median(rom_v, 'mae_pct_rom', **sel)[0], 1)}",
                          f"{f(ch_mean(rom, 'abs_offset_deg', **sel), 2)} → {f(ch_mean(rom_v, 'abs_offset_deg', **sel), 2)}",
                          f"{f(r_mean(rom, **sel), 3)} → {f(r_mean(rom_v, **sel), 3)}"]
            A(f"| {lab} | " + " | ".join(cells) + " |")
        A("\n*Per joint: MAE (deg) pipeline → Vicon static; in brackets, with the Vicon static: aligned MAE and Pearson r*\n")
        A("| Joint | " + " | ".join(f"{m} sag. | {m} front. | {m} transv." for m in models) + " |\n|---|" + "---:|" * (3 * len(models)))
        for j in JOINT_LABEL:
            cells = []
            for m in models:
                for p, _ in PLANES:
                    sel = {"model": m, "joint": j, "plane": p}
                    cells.append(f"{f(ch_mean(rom, 'mae_deg', **sel), 1)} → {f(ch_mean(rom_v, 'mae_deg', **sel), 1)} "
                                 f"({f(ch_mean(rom_v, 'mae_aligned_deg', **sel), 1)}, {f(r_mean(rom_v, **sel), 2)})"
                                 if _sel(rom_v, False, sel) else "–")
            A(f"| {JOINT_LABEL[j]} | " + " | ".join(cells) + " |")
        A("\n*Per action and per subject: MAE (deg) pipeline → Vicon static; in brackets, with the Vicon static: aligned MAE and Pearson r*\n")
        A("| Group | " + " | ".join(f"{m} sag. | {m} front. | {m} transv. | {m} all" for m in models) + " |\n|---|" + "---:|" * (4 * len(models)))
        for key, label in (("action", "Action"), ("subject", "Subject")):
            for val in sorted(dict.fromkeys(x[key] for x in rom_v), key=lambda s: (len(s), s)):
                cells = []
                for m in models:
                    for p in [p for p, _ in PLANES] + [None]:
                        sel = {"model": m, key: val} | ({} if p is None else {"plane": p})
                        cells.append(f"{f(ch_mean(rom, 'mae_deg', **sel), 1)} → {f(ch_mean(rom_v, 'mae_deg', **sel), 1)} "
                                     f"({f(ch_mean(rom_v, 'mae_aligned_deg', **sel), 1)}, {f(r_mean(rom_v, **sel), 2)})")
                A(f"| {label} {val} | " + " | ".join(cells) + " |")
    elif rom:
        A("\n*Joint angles with a Vicon static trial: not in these runs yet (run `add_gt_static.py`, then `--report-only`).*\n")
    A("\n**Continuity of the real-time angles** (corrections applied by the biomechanics stage, total over all recordings)\n")
    A(hdr.replace("Metric", "Correction"))
    for k, lab in (("continuity_glitches_held", "Pose glitches held (> 25°/frame, ≤ 3 frames)"),
                   ("continuity_rate_limited", "Rate-limited frames"), ("continuity_branch_switches", "Euler branch switches")):
        A(f"| {lab} | " + " | ".join(str(int(np.nansum([r[k] for r in rows if r['model'] == m]))) for m in models) + " |")

    # 5. per recording
    A("\n## 5. Per-recording results\n")
    A("| Model | Recording | Pose fps | Dropped % | Output fps | Latency → angles (ms) | MPJPE raw (mm) | MPJPE output (mm) | Angles MAE (deg) | Angles MAE (% ROM) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        A(f"| {r['model']} | {r['trial']} | {f(r['pose_rate_fps'])} | {f(r['dropped_pct'])} | {f(r['output_rate_fps'])} | "
          f"{f(r['latency_angles_mean_ms'], 0)} | {f(r['mpjpe_raw_mm'])} | {f(r['mpjpe_all_output_mm'])} | {f(r['angle_final_mean_deg'], 2)} | "
          f"{f(r['angle_final_pct_rom'], 1)} |")

    # 6. validity summary
    A("\n## 6. Summary for real-time feedback\n")
    for m in models:
        p_ = pool[m]
        A(f"- **{m}:** joint angles at {ms(summary, m, 'output_rate_fps')} frames/s from a {cfg['fps']:g} fps camera feed "
          f"(pose network {ms(summary, m, 'pose_rate_fps')} frames/s, {ms(summary, m, 'filled_pct', 0)} % of frames filled); "
          f"latency from capture to joint angles {f(p_['lat_angles']['p50'], 0)} ms median, {f(p_['lat_angles']['p95'], 0)} ms p95 "
          f"({f(p_['angles_within_100ms_pct'], 0)} % of frames within 100 ms, {f(p_['angles_within_150ms_pct'], 0)} % within 150 ms); "
          f"3D marker error {ms(summary, m, 'mpjpe_all_output_mm')} mm; joint-angle error "
          + ", ".join(f"{lab.split(' ')[0].lower()} {f(p_[f'angle_all output_{p}'], 1)}°" for p, lab in PLANES) + ".")
    A("\n## 7. Figures\n")
    for name in figs:
        A(f"- `figures/{name}.png` (and .pdf)")
        A(f"  ![{name}](figures/{name}.png)")
    A("\n## 8. Notes\n")
    A("- Camera feeds are simulated from recorded frames on a fixed 60 Hz clock; image capture, transfer and decoding "
      "by real cameras are not included in the latency.")
    A("- Person boxes come from the YOLO detector; the reference box for the IoU is the box stored with the dataset.")
    A("- The static trial is the first frame of each recording (a separate static recording in practice).")
    A("- Vicon joint angles are computed with the same biomechanical model from the Vicon markers, so the angle errors "
      "measure the markerless pipeline, not the model.")
    A(f"- Per-run details (full text report, timing / accuracy / joint-angle plots, per-frame arrays) are in "
      f"`<model>/<recording>/`. Raw tables: `results_per_recording.csv`, `results_summary.csv`, `pooled.json`, "
      f"`angles_per_channel.csv`, `angles_subject_action_joint_plane.csv`.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_latex(out, models, summary, pool)
    print(f"report written: {out / 'REPORT.md'}")


def write_latex(out, models, summary, pool):
    def ms_tex(m, k, d=1):
        v = summary.get(m, {}).get(k)
        return "--" if not v else (f"${v['mean']:.{d}f} \\pm {v['sd']:.{d}f}$" if v["n"] > 1 else f"${v['mean']:.{d}f}$")

    def fv(v, d=1):
        return f"${v:.{d}f}$" if _num(v) else "--"

    cols = "l" + "r" * len(models)
    head = " & ".join(["Metric"] + models) + r" \\"
    T = []
    T.append("% Generated by realtime_eval/report.py. Needs \\usepackage{booktabs}.\n")
    T.append(r"\begin{table}[t]\centering\caption{Real-time performance of the full pipeline (60\,fps, 3 views).}\label{tab:rt_performance}")
    T.append(rf"\begin{{tabular}}{{{cols}}}\toprule {head}\midrule")
    for lab, k, d in (("Pose network rate (fps)", "pose_rate_fps", 1), ("Dropped frames (\\%)", "dropped_pct", 1),
                      ("Filled by LivePoseFormer (\\%)", "filled_pct", 1), ("Output rate (fps)", "output_rate_fps", 1),
                      ("Latency to 3D (ms)", "latency_3d_mean_ms", 1), ("Latency to joint angles (ms)", "latency_angles_mean_ms", 1)):
        T.append(f"{lab} & " + " & ".join(ms_tex(m, k, d) for m in models) + r" \\")
    T.append("Latency to angles p95 (ms, pooled) & " + " & ".join(fv(pool[m]["lat_angles"]["p95"], 0) for m in models) + r" \\")
    T.append(r"\midrule")
    for _, lab in STAGES:
        T.append(f"{lab} (ms) & " + " & ".join(ms_tex(m, f"t_{lab}_ms", 2) for m in models) + r" \\")
    T.append("LivePoseFormer (ms) & " + " & ".join(ms_tex(m, "t_LivePoseFormer_ms", 2) for m in models) + r" \\")
    T.append("Biomechanics (ms) & " + " & ".join(ms_tex(m, "t_Biomechanics_ms", 2) for m in models) + r" \\")
    T.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

    T.append(r"\begin{table}[t]\centering\caption{Error along the pipeline against Vicon.}\label{tab:rt_accuracy}")
    T.append(rf"\begin{{tabular}}{{{cols}}}\toprule {head}\midrule")
    for lab, k, d in (("Detector box IoU", "det_iou_mean", 3), ("2D keypoint error (px)", "kp2d_px_mean", 1),
                      ("2D PCK@10\\,px (\\%)", "kp2d_pck10", 1), ("Triangulated 3D MPJPE (mm)", "mpjpe_raw_mm", 1),
                      ("After LivePoseFormer, processed (mm)", "mpjpe_lpf_processed_mm", 1),
                      ("Filled frames (mm)", "mpjpe_filled_mm", 1), ("All 60\\,fps output (mm)", "mpjpe_all_output_mm", 1),
                      ("3D jitter raw (mm/frame$^2$)", "jitter_raw_mm", 2), ("3D jitter output (mm/frame$^2$)", "jitter_all_output_mm", 2)):
        T.append(f"{lab} & " + " & ".join(ms_tex(m, k, d) for m in models) + r" \\")
    T.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

    T.append(r"\begin{table}[t]\centering\caption{Joint-angle mean absolute error against Vicon (deg, frame-weighted).}\label{tab:rt_angles}")
    c2 = "l" + "rrr" * len(models)
    T.append(rf"\begin{{tabular}}{{{c2}}}\toprule")
    T.append("Plane & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in models) + r" \\")
    T.append(" & " + " & ".join(["raw 3D & after LPF & final"] * len(models)) + r" \\\midrule")
    for p, lab in PLANES + [("mean", "Mean")]:
        cells = []
        for m in models:
            for grp in ("before LPF", "unsmoothed", "all output"):
                v = np.nanmean([pool[m][f"angle_{grp}_{q}"] for q, _ in PLANES]) if p == "mean" else pool[m][f"angle_{grp}_{p}"]
                cells.append(fv(v, 2))
        T.append(f"{lab.split(' (')[0]} & " + " & ".join(cells) + r" \\")
    T.append(r"\bottomrule\end{tabular}\end{table}")
    (out / "tables.tex").write_text("\n".join(T) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    build(Path(sys.argv[1]))
