"""Turning the per-frame timestamps into the real-time verdict, a text report and plots.

Definitions (all per frame set of 3 images):
* delay (latency)   finished - captured: how old a result is when it becomes available
* queue wait        taken by the pipeline - captured
* service time      time a stage spends on one frame (PRE, GPU, POST)
* output rate       results per second in steady state
* budget            1000 / fps ms: the time available per frame at the camera rate
"""
import json
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STAGES = [("DET (GPU: person detector)", "t_det_start", "t_det_end"), ("PRE (CPU)", "t_pre_start", "t_pre_end"),
          ("GPU", "t_gpu_start", "t_gpu_end"), ("POST (CPU)", "t_post_start", "t_done")]


def system_info():
    import torch
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor(),
            "torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarize(records, n_captured, fps, mode, policy, source, gt_by_index=None, lpf=None, bio=None):
    done = sorted((r for r in records.values() if "t_done" in r), key=lambda r: r["seq"])
    t0 = source.t0
    budget = 1000.0 / fps
    cap = np.array([r["t_capture"] for r in done])
    fin = np.array([r["t_done"] for r in done])
    lat = (fin - cap) * 1000
    wait = np.array([r["t_pre_start"] - r["t_capture"] for r in done]) * 1000
    # Pipelined: LivePoseFormer reads an unbounded queue in its own thread, so it is not a pose stage
    # (it cannot slow the pose pipeline); whether it keeps up is reported separately below.
    # Sequential: it runs inline after each frame, so its time is part of every frame's work.
    stages = [st for st in STAGES if done and st[1] in done[0]]     # DET only with a detector
    stages += ([("LivePoseFormer (inline)", "t_lpf_start", "t_lpf_done")]
               if lpf is not None and mode == "sequential" else [])
    stages += ([("Biomechanics (inline)", "t_bio_drain_start", "t_bio_drain_done")]
               if bio is not None and mode == "sequential" else [])
    service = {name: np.array([r[b] - r[a] for r in done if a in r and b in r]) * 1000 for name, a, b in stages}

    # Steady-state output rate: skip the first second (start-up transient) when there is enough data.
    steady = fin[cap - t0 >= 1.0] if (cap - t0 >= 1.0).sum() > 10 else fin
    out_rate = (len(steady) - 1) / (steady[-1] - steady[0]) if len(steady) > 1 else float("nan")
    gaps = np.diff(fin) * 1000
    # Delay trend: a positive slope means the pipeline is falling further behind every second.
    slope = float(np.polyfit(cap - t0, lat, 1)[0]) if len(done) > 10 else float("nan")

    stage_means = {k: float(v.mean()) for k, v in service.items()}
    per_frame_work = sum(stage_means.values())
    if mode == "pipelined":          # every stage has its own thread and works in parallel
        bottleneck = max(stage_means, key=stage_means.get)
        capacity_ms = stage_means[bottleneck]
    else:
        bottleneck = max(stage_means, key=stage_means.get)
        capacity_ms = per_frame_work
    dropped = n_captured - len(done)

    if policy == "latest":
        realtime = dropped / n_captured < 0.01
    else:
        realtime = slope < 1.0 and lat[-1] < 3 * max(budget, lat[:20].mean() if len(lat) else budget)

    s = {
        "fps_in": fps, "mode": mode, "policy": policy, "budget_ms": budget,
        "captured": n_captured, "processed": len(done), "dropped": dropped,
        "dropped_pct": 100 * dropped / n_captured,
        "output_rate_fps": out_rate,
        "delay_ms": {"mean": float(lat.mean()), "p50": _pct(lat, 50), "p95": _pct(lat, 95), "p99": _pct(lat, 99),
                     "max": float(lat.max())},
        "delay_trend_ms_per_s": slope,
        "queue_wait_ms_mean": float(wait.mean()),
        "output_gap_ms": {"p50": _pct(gaps, 50), "p95": _pct(gaps, 95)},
        "stage_service_ms": {k: {"mean": float(v.mean()), "p95": _pct(v, 95)} for k, v in service.items()},
        "work_per_frame_ms": per_frame_work,
        "capacity_ms_per_frame": capacity_ms,
        "max_sustainable_fps": 1000 / capacity_ms,
        "bottleneck": bottleneck,
        "speedup_needed": capacity_ms / budget,
        "realtime": bool(realtime),
        "clock_late_ms_p99": _pct(np.array(source.late) * 1000, 99),
        "reader_underruns": len(source.video.underruns),
        "stopped_early_s": None if source.stopped_early is None else source.stopped_early / fps,
    }
    if gt_by_index is not None:
        for r in done:
            gt = gt_by_index[r["index"]]
            r["err_raw_mm"] = float(np.nanmean(np.linalg.norm(r["xyz_wcs"] - gt, axis=-1)))
            if "xyz_lpf" in r:
                r["err_lpf_mm"] = float(np.nanmean(np.linalg.norm(r["xyz_lpf"] - gt, axis=-1)))
        s["mpjpe_world_mm"] = float(np.mean([r["err_raw_mm"] for r in done]))
    det = [r for r in done if "det_conf" in r]
    if det:
        conf = np.concatenate([r["det_conf"] for r in det])
        iou = np.array([r["det_iou_json"] for r in det])
        s["detector"] = {"frames": len(det), "views": len(conf), "conf_mean": float(conf.mean()),
                         "conf_min": float(conf.min()), "misses": int(sum(r["det_misses"] for r in det)),
                         "iou_json_mean": float(iou.mean()), "iou_json_p5": _pct(iou, 5)}
    if lpf is not None:
        lpf_rows = [r for r in done if "t_lpf_out" in r]
        lpf_lat = np.array([r["t_lpf_out"] - r["t_capture"] for r in lpf_rows]) * 1000
        pushed = sorted((r for r in records.values() if "t_push_end" in r), key=lambda r: r["t_push_start"])
        cost = np.array([r["t_push_end"] - r["t_push_start"] for r in pushed]) * 1000
        wait = np.array([r["t_push_start"] - r["t_lpf_known"] for r in pushed]) * 1000
        tp = np.array([r["t_capture"] - t0 for r in pushed])
        span = pushed[-1]["t_push_end"] - pushed[0]["t_push_start"] if len(pushed) > 1 else float("nan")
        # Wait = from the moment a result (or a drop) is known to the start of its push. It includes
        # waiting for stream order: a dropped frame waits for the frame ahead of it to leave the pose
        # network. LivePoseFormer keeps up when this wait does not grow (trend) and it has idle time.
        wait_slope = float(np.polyfit(tp, wait, 1)[0]) if len(pushed) > 10 else float("nan")
        s["lpf"] = {"weights": "checkpoint" if lpf.has_weights else "random (timing only)",
                    "cuda_graph": bool(lpf.use_graph), "graph_check_mm": lpf.graph_check_mm,
                    "released": len(lpf_rows), "delay_ms_mean": float(lpf_lat.mean()) if len(lpf_lat) else None,
                    "delay_ms_p95": _pct(lpf_lat, 95),
                    "pushed": len(pushed), "pushed_filled": sum(bool(r.get("filled")) for r in pushed),
                    "compute_ms_mean": float(cost.mean()) if len(cost) else None, "compute_ms_p95": _pct(cost, 95),
                    "wait_ms_mean": float(wait.mean()) if len(wait) else None, "wait_ms_p95": _pct(wait, 95),
                    "wait_trend_ms_per_s": wait_slope,
                    "busy_pct": float(100 * cost.sum() / 1000 / span) if len(pushed) > 1 else None,
                    "keeps_up": bool(len(pushed) > 10 and wait_slope < 1.0 and cost.sum() / 1000 < 0.95 * span)}
        if gt_by_index is not None:
            s["lpf"]["accuracy"] = lpf_accuracy([r for r in done if "err_lpf_mm" in r], gt_by_index)

        # Smoothed output rate, and the dropped frames LivePoseFormer filled (--lpf-fill-dropped).
        filled = sorted((r for r in records.values() if r.get("filled") and "xyz_lpf" in r), key=lambda r: r["seq"])
        outs = sorted(r["t_lpf_out"] for r in list(done) + filled if "t_lpf_out" in r)
        steady_out = [t for t in outs if t - t0 >= 1.0] if sum(t - t0 >= 1.0 for t in outs) > 10 else outs
        s["lpf"]["output_frames"] = len(outs)
        s["lpf"]["output_rate_fps"] = ((len(steady_out) - 1) / (steady_out[-1] - steady_out[0])
                                       if len(steady_out) > 1 else float("nan"))
        s["lpf"]["filled_frames"] = len(filled)
        if filled:
            delay = np.array([r["t_lpf_out"] - r["t_capture"] for r in filled]) * 1000
            s["lpf"]["filled_delay_ms_mean"] = float(delay.mean())
            if gt_by_index is not None:
                for r in filled:
                    r["err_lpf_mm"] = float(np.nanmean(np.linalg.norm(r["xyz_lpf"] - gt_by_index[r["index"]], axis=-1)))
                err = np.array([r["err_lpf_mm"] for r in filled])
                s["lpf"]["filled_mpjpe_mm"] = float(err.mean())
                s["lpf"]["filled_median_mm"] = float(np.median(err))
                s["lpf"]["filled_p95_mm"] = _pct(err, 95)
                both = sorted([r for r in done if "err_lpf_mm" in r] + filled, key=lambda r: r["seq"])
                s["lpf"]["all_output_mpjpe_mm"] = float(np.mean([r["err_lpf_mm"] for r in both]))
                s["lpf"]["all_output_accel_error"] = _accel_error(both, "xyz_lpf", gt_by_index)
    if bio is not None:
        s["biomech"] = biomech_summary(records, t0, budget, bio, gt_by_index, lpf_on=lpf is not None)
        s["biomech_channels"] = bio.channels
    return s, done


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def _angle_jitter(idx, pred, gt):
    """Mean |second difference of the angle error| (deg/frame^2) over runs of 3 consecutive
    recording frames: how much the angles shake compared with the ground truth."""
    idx = np.asarray(idx)
    if len(idx) < 3:
        return float("nan")
    ok = (idx[1:-1] == idx[:-2] + 1) & (idx[2:] == idx[1:-1] + 1)
    d1 = _wrap(np.diff(_wrap(pred - gt), axis=0))
    acc = np.abs(d1[1:] - d1[:-1])[ok]
    return float(np.nanmean(acc)) if len(acc) else float("nan")


def biomech_summary(records, t0, budget, bio, gt_by_index=None, lpf_on=False):
    """Timing of the biomechanics stage and, with ground truth, the joint-angle errors per plane."""
    from .biomech_stage import JOINTS, PLANES
    rows = sorted((r for r in records.values() if "t_bio_end" in r), key=lambda r: r["seq"])
    if not rows:
        return {"frames": 0}
    cost = np.array([r["t_bio_end"] - r["t_bio_start"] for r in rows]) * 1000
    wait = np.array([r["t_bio_start"] - r["t_bio_known"] for r in rows]) * 1000
    delay = np.array([r["t_bio_end"] - r["t_capture"] for r in rows]) * 1000
    tc = np.array([r["t_capture"] - t0 for r in rows])
    starts = min(r["t_bio_start"] for r in rows)
    span = max(r["t_bio_end"] for r in rows) - starts
    slope = float(np.polyfit(tc, wait, 1)[0]) if len(rows) > 10 else float("nan")
    ends = np.sort([r["t_bio_end"] for r in rows])
    steady = ends[ends - t0 >= 1.0] if (ends - t0 >= 1.0).sum() > 10 else ends
    b = {"source": "LivePoseFormer output" if lpf_on else "raw 3D", "frames": len(rows),
         "check_deg": bio.check_deg, "compute_ms_mean": float(cost.mean()), "compute_ms_p95": _pct(cost, 95),
         "busy_pct": float(100 * cost.sum() / 1000 / span) if span > 0 else None,
         "wait_ms_mean": float(wait.mean()), "wait_ms_p95": _pct(wait, 95), "wait_trend_ms_per_s": slope,
         "keeps_up": bool(len(rows) > 10 and slope < 1.0 and cost.sum() / 1000 < 0.95 * span),
         "delay_ms_mean": float(delay.mean()), "delay_ms_p95": _pct(delay, 95),
         "output_rate_fps": (len(steady) - 1) / (steady[-1] - steady[0]) if len(steady) > 1 else float("nan"),
         "continuity": dict(bio.stats), "left_handed_input": bio.static["left_handed_input"],
         "smoothing": getattr(bio, "smoothing_name", "off") if bio.smoothing else "off"}
    if not gt_by_index:
        return b

    # Ground-truth angles: the same model on the Vicon keypoints, with its own static trial (first frame).
    gt_static = bio.static_reference(gt_by_index[min(gt_by_index)])
    idx = [r["index"] for r in rows]
    steps = [1] + [max(1, b_ - a_) for a_, b_ in zip(idx, idx[1:])]
    gt_ang = bio.angles_batch(np.stack([gt_by_index[i] for i in idx]), static=gt_static, steps=steps)
    pred = np.stack([r["angles"] for r in rows])
    err = np.abs(_wrap(pred - gt_ang))
    for r, g, e in zip(rows, gt_ang, err):
        r["angles_gt"], r["angle_err"] = g, e
    planes = np.array([bio.channel_key[c][2] for c in bio.channels])
    joints = np.array([bio.channel_key[c][0] for c in bio.channels])
    # Channels the model always sets to 0 (elbow ab/adduction and rotation: the forearm frame is built
    # on the upper arm's axis) carry no information, so they are left out of every metric.
    used = (np.nanmax(np.abs(gt_ang), 0) > 1e-6) | (np.nanmax(np.abs(pred), 0) > 1e-6)
    b["excluded_channels"] = [c for c, u in zip(bio.channels, used) if not u]
    planes = np.where(used, planes, "")
    for r, e in zip(rows, err):
        r["angle_mae"] = float(np.nanmean(e[used]))

    def per_plane(e):
        d = {p: float(np.nanmean(e[:, planes == p])) for p in PLANES}
        d["mean"] = float(np.mean([d[p] for p in PLANES]))
        return d

    filled = np.array([bool(r.get("filled")) for r in rows])
    groups = {"all output": err}
    if all("angles_unsmoothed" in r for r in rows):          # the same output without the One Euro filter
        unsm = np.stack([r["angles_unsmoothed"] for r in rows])
        groups["unsmoothed"] = np.abs(_wrap(unsm - gt_ang))
        b["jitter_unsmoothed_deg"] = _angle_jitter(idx, unsm[:, used], gt_ang[:, used])
    if filled.any():
        groups["measured"], groups["filled"] = err[~filled], err[filled]
    if lpf_on:                    # the same measured frames, from the raw 3D (before LivePoseFormer)
        meas = [k for k, r in enumerate(rows) if "xyz_wcs" in r]
        if meas:
            mi = [idx[k] for k in meas]
            raw = bio.angles_batch(np.stack([rows[k]["xyz_wcs"] for k in meas]),
                                   steps=[1] + [max(1, b_ - a_) for a_, b_ in zip(mi, mi[1:])])
            groups["before LPF"] = np.abs(_wrap(raw - gt_ang[meas]))
            b["jitter_raw_deg"] = _angle_jitter([idx[k] for k in meas], raw[:, used], gt_ang[meas][:, used])
    b["mae_deg"] = {g: per_plane(e) for g, e in groups.items()}
    b["median_deg"] = {p: float(np.nanmedian(err[:, planes == p])) for p in PLANES}
    b["p95_deg"] = {p: _pct(err[:, planes == p].ravel(), 95) for p in PLANES}
    b["per_joint_deg"] = {j: {p: float(np.nanmean(err[:, (joints == j) & (planes == p)]))
                              if ((joints == j) & (planes == p)).any() else None for p in PLANES}
                          for j in JOINTS}
    b["jitter_deg"] = _angle_jitter(idx, pred[:, used], gt_ang[:, used])
    return b


def _accel_error(rows, key, gt_by_index):
    """Mean |second difference of prediction - second difference of GT| (mm/frame^2), over runs of
    3 consecutive recording frames. A measure of jitter."""
    errs = []
    for a, b, c in zip(rows, rows[1:], rows[2:]):
        if not (b["index"] == a["index"] + 1 and c["index"] == b["index"] + 1):
            continue
        pred = c[key] - 2 * b[key] + a[key]
        gt = gt_by_index[c["index"]] - 2 * gt_by_index[b["index"]] + gt_by_index[a["index"]]
        errs.append(np.nanmean(np.linalg.norm(pred - gt, axis=-1)))
    return float(np.mean(errs)) if errs else float("nan")


def lpf_accuracy(rows, gt_by_index):
    """Error before (raw triangulated 3D) and after LivePoseFormer, on the same frames."""
    before = np.array([r["err_raw_mm"] for r in rows])
    after = np.array([r["err_lpf_mm"] for r in rows])
    out = {
        "frames": len(rows),
        "mpjpe_before_mm": float(before.mean()), "mpjpe_after_mm": float(after.mean()),
        "median_before_mm": float(np.median(before)), "median_after_mm": float(np.median(after)),
        "p95_before_mm": _pct(before, 95), "p95_after_mm": _pct(after, 95),
        "frames_improved_pct": float(100 * (after < before).mean()),
        "accel_error_before": _accel_error(rows, "xyz_wcs", gt_by_index),
        "accel_error_after": _accel_error(rows, "xyz_lpf", gt_by_index),
    }
    out["mpjpe_change_pct"] = 100 * (out["mpjpe_after_mm"] - out["mpjpe_before_mm"]) / out["mpjpe_before_mm"]
    return out


def text_report(s, title, info):
    st = s["stage_service_ms"]
    lines = [
        title, "=" * 78,
        f"GPU {info.get('gpu', 'none')} | torch {info['torch']} (CUDA {info['cuda']}) | {info['cpu']}",
        f"input {s['fps_in']:.0f} fps (budget {s['budget_ms']:.2f} ms/frame) | mode {s['mode']} | policy {s['policy']}",
        "",
        f"VERDICT: {'REAL TIME' if s['realtime'] else 'NOT real time'} at {s['fps_in']:.0f} fps"
        f"  ->  max sustainable {s['max_sustainable_fps']:.1f} fps, needs {s['speedup_needed']:.2f}x "
        f"{'(already fast enough)' if s['speedup_needed'] <= 1 else 'speed-up'} (bottleneck: {s['bottleneck']})",
        "",
        f"frames captured / processed / dropped : {s['captured']} / {s['processed']} / {s['dropped']} "
        f"({s['dropped_pct']:.1f}% dropped)",
        f"output rate (steady state)            : {s['output_rate_fps']:.1f} results/s",
        f"delay capture -> 3D result (ms)       : mean {s['delay_ms']['mean']:.1f}  p50 {s['delay_ms']['p50']:.1f}  "
        f"p95 {s['delay_ms']['p95']:.1f}  p99 {s['delay_ms']['p99']:.1f}  max {s['delay_ms']['max']:.1f}",
        f"delay trend                           : {s['delay_trend_ms_per_s']:+.1f} ms per second of stream "
        f"({'falling behind' if s['delay_trend_ms_per_s'] > 1 else 'stable'})",
        f"mean wait before processing starts    : {s['queue_wait_ms_mean']:.1f} ms",
        f"gap between results (ms)              : p50 {s['output_gap_ms']['p50']:.1f}  p95 {s['output_gap_ms']['p95']:.1f}",
        "",
        "service time per stage (ms)            mean     p95",
    ]
    for k, v in st.items():
        flag = "  <- bottleneck" if s["bottleneck"] == k else ""
        lines.append(f"  {k:<36}{v['mean']:>7.2f}{v['p95']:>8.2f}{flag}")
    lines.append(f"  {'total work per frame':<36}{s['work_per_frame_ms']:>7.2f}")
    if "detector" in s:
        d = s["detector"]
        lines.append(f"\nperson detector                       : {d['views']} views, confidence mean {d['conf_mean']:.2f} "
                     f"min {d['conf_min']:.2f}, {d['misses']} misses (previous box kept); box IoU with the JSON "
                     f"boxes mean {d['iou_json_mean']:.2f}, 5th percentile {d['iou_json_p5']:.2f}")
    if "mpjpe_world_mm" in s:
        lines.append(f"{'' if 'detector' in s else chr(10)}accuracy of processed frames          : "
                     f"MPJPE {s['mpjpe_world_mm']:.1f} mm (world frame, "
                     f"raw triangulated 3D{', before LivePoseFormer' if 'lpf' in s else ''})")
    if "lpf" in s:
        l = s["lpf"]
        graph = ("CUDA graph" + (f", graph vs eager max diff {l['graph_check_mm']:.4f} mm"
                                 if l["graph_check_mm"] is not None else "")) if l["cuda_graph"] else "eager"
        lines += ["", f"LivePoseFormer stage ({l['weights']}, {graph}):",
                  f"  frames pushed {l['pushed']} ({l['pushed_filled']} of them filled drops), "
                  f"compute per pushed frame mean {l['compute_ms_mean']:.2f} ms, p95 {l['compute_ms_p95']:.2f} ms, "
                  f"busy {l['busy_pct']:.0f}% of the time",
                  f"  wait for its turn (stream order + queue) mean {l['wait_ms_mean']:.1f} ms, p95 {l['wait_ms_p95']:.1f} ms, trend "
                  f"{l['wait_trend_ms_per_s']:+.1f} ms/s -> "
                  + ("keeps up with the stream" if l["keeps_up"] else "FALLS BEHIND (its queue keeps growing)"),
                  f"  delay capture -> smoothed output mean {l['delay_ms_mean']:.1f} ms, p95 {l['delay_ms_p95']:.1f} ms "
                  f"(includes waiting for 3 future frames)"]
        if "accuracy" in l:
            a = l["accuracy"]
            lines += [
                f"  error vs Vicon GT, world frame, same {a['frames']} frames:   before LPF    after LPF",
                f"    MPJPE mean (mm)                               {a['mpjpe_before_mm']:>9.1f}    {a['mpjpe_after_mm']:>9.1f}"
                f"   ({a['mpjpe_change_pct']:+.1f}%)",
                f"    MPJPE median (mm)                             {a['median_before_mm']:>9.1f}    {a['median_after_mm']:>9.1f}",
                f"    MPJPE p95 (mm)                                {a['p95_before_mm']:>9.1f}    {a['p95_after_mm']:>9.1f}",
                (f"    acceleration error / jitter (mm/frame^2)      {a['accel_error_before']:>9.2f}    {a['accel_error_after']:>9.2f}"
                 if np.isfinite(a["accel_error_before"]) else
                 "    acceleration error / jitter (mm/frame^2)            n/a (needs consecutive frames: no drops)"),
                f"    frames where LPF lowered the error: {a['frames_improved_pct']:.0f}%",
            ]
        lines.append(f"  smoothed output: {l['output_frames']} frames, {l['output_rate_fps']:.1f} frames/s")
        if l.get("filled_frames"):
            lines.append(f"  dropped frames filled by LivePoseFormer (pushed as all-NaN): {l['filled_frames']}, "
                         f"delay capture -> filled output {l['filled_delay_ms_mean']:.1f} ms")
            if "filled_mpjpe_mm" in l:
                lines += [
                    f"    MPJPE of filled frames (mm): mean {l['filled_mpjpe_mm']:.1f}  median {l['filled_median_mm']:.1f}"
                    f"  p95 {l['filled_p95_mm']:.1f}   (no raw 3D exists for these frames)",
                    f"    MPJPE of every smoothed output, processed + filled (mm): {l['all_output_mpjpe_mm']:.1f}",
                    (f"    jitter of the full-rate smoothed output (mm/frame^2): {l['all_output_accel_error']:.2f}"
                     if np.isfinite(l["all_output_accel_error"]) else ""),
                ]
        if l["weights"].startswith("random"):
            lines.append("    (random weights: the 'after' numbers are meaningless; pass --lpf-checkpoint)")
    if "biomech" in s:
        b = s["biomech"]
        lines += ["", f"Biomechanics stage (Plug-in Gait joint angles, 72 channels, from the {b['source']}; "
                      f"static trial = first frame, before the clock; angle smoothing: {b['smoothing']}):"]
        if b["frames"]:
            lines += [
                f"  frames {b['frames']}, compute per frame mean {b['compute_ms_mean']:.2f} ms, p95 "
                f"{b['compute_ms_p95']:.2f} ms, busy {b['busy_pct']:.0f}% of the time "
                f"(fast port vs reference model max diff {b['check_deg']:.1e} deg)",
                f"  wait for its turn mean {b['wait_ms_mean']:.1f} ms, p95 {b['wait_ms_p95']:.1f} ms, trend "
                f"{b['wait_trend_ms_per_s']:+.1f} ms/s -> "
                + ("keeps up with the stream" if b["keeps_up"] else "FALLS BEHIND (its queue keeps growing)"),
                f"  delay capture -> joint angles mean {b['delay_ms_mean']:.1f} ms, p95 {b['delay_ms_p95']:.1f} ms; "
                f"output {b['output_rate_fps']:.1f} frames/s",
                f"  continuity: {b['continuity']['branch_switches']} Euler branch switches, "
                f"{b['continuity']['wraps_360']} 360-deg wraps, {b['continuity']['glitches_held']} glitches held, "
                f"{b['continuity']['rate_limited']} rate-limited frames "
                f"(of {b['continuity']['frames']}); world frame "
                + ("left-handed -> Z flipped for the model" if b["left_handed_input"] else "right-handed")]
        if "mae_deg" in b:
            from .biomech_stage import JOINTS, PLANE_LABEL, PLANES
            groups = list(b["mae_deg"])
            lines.append("  joint-angle error vs Vicon GT (mean absolute error, deg; GT angles from the same model "
                         "with its own static trial):")
            lines.append(f"    {'plane':<34}" + "".join(f"{g:>14}" for g in groups))
            for p in PLANES + ["mean"]:
                label = PLANE_LABEL.get(p, "mean of the 3 planes")
                lines.append(f"    {label:<34}" + "".join(f"{b['mae_deg'][g][p]:>14.2f}" for g in groups))
            lines.append("    all output, median / p95 (deg): " + ",  ".join(
                f"{p} {b['median_deg'][p]:.2f} / {b['p95_deg'][p]:.2f}" for p in PLANES))
            lines.append(f"  per joint, all output (deg, right + left):  {'flex':>7}{'add':>7}{'rot':>7}")
            for j in JOINTS:
                v = b["per_joint_deg"][j]
                lines.append(f"    {j:<40}" + "".join("      -" if v[p] is None else f"{v[p]:>7.2f}" for p in PLANES))
            if b["excluded_channels"]:
                lines.append(f"    ('-' and left out of all angle metrics: {', '.join(b['excluded_channels'])}, "
                             f"always 0 in this model)")
            jit = f"  joint-angle jitter (deg/frame^2): output {b['jitter_deg']:.2f}"
            if "jitter_unsmoothed_deg" in b:
                jit += f", unsmoothed {b['jitter_unsmoothed_deg']:.2f}"
            if "jitter_raw_deg" in b:
                jit += f", before LPF {b['jitter_raw_deg']:.2f}"
            lines.append(jit + ("" if np.isfinite(b["jitter_deg"]) else "  (n/a: needs consecutive frames)"))
            if s.get("lpf", {}).get("weights", "").startswith("random"):
                lines.append("    (LivePoseFormer has random weights: only the 'before LPF' angles are meaningful; "
                             "pass --lpf-checkpoint)")
    lines.append(f"\nsource clock check: releases were at most {s['clock_late_ms_p99']:.2f} ms late (p99); "
                 + ("the disk reader always had the next frame ready" if not s["reader_underruns"] else
                    f"the disk reader was late for {s['reader_underruns']} frames (raise --reader-threads; "
                    f"those frames were released late, which is the simulator, not the pipeline)"))
    if s["stopped_early_s"] is not None:
        lines.append(f"stopped early after {s['stopped_early_s']:.1f} s: the queue reached --max-backlog frames "
                     f"and would only keep growing")
    return "\n".join(lines)


def plot(s, done, dropped_seqs, source, path, title):
    t0 = source.t0
    cap = np.array([r["t_capture"] - t0 for r in done])
    lat = np.array([r["t_done"] - r["t_capture"] for r in done]) * 1000
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    a = ax[0, 0]
    a.plot(cap, lat, ".", ms=3, label="3D result")
    lpf = [(r["t_capture"] - t0, (r["t_lpf_out"] - r["t_capture"]) * 1000) for r in done if "t_lpf_out" in r]
    if lpf:
        a.plot(*zip(*lpf), ".", ms=3, label="after LivePoseFormer")
    a.axhline(s["budget_ms"], color="r", ls="--", lw=1, label=f"1 frame = {s['budget_ms']:.1f} ms")
    a.set_xlabel("capture time (s)"); a.set_ylabel("delay (ms)"); a.set_title("delay per frame (capture -> result)")
    a.legend(fontsize=8)

    a = ax[0, 1]
    seqs = np.array([r["seq"] for r in done])
    a.plot(seqs / s["fps_in"], (np.array([r["t_done"] for r in done]) - t0), ".", ms=3, color="g", label="processed")
    if dropped_seqs:
        d = np.array(dropped_seqs) / s["fps_in"]
        a.plot(d, d, "|", color="r", ms=6, alpha=0.4, label=f"dropped ({len(dropped_seqs)})")
    lim = max(s["captured"] / s["fps_in"], cap.max() if len(cap) else 1)
    a.plot([0, lim], [0, lim], "k--", lw=0.8, label="zero delay")
    a.set_xlabel("capture time (s)"); a.set_ylabel("finish time (s)")
    a.set_title(f"stream timeline: {s['processed']} processed, {s['dropped']} dropped"); a.legend(fontsize=8)

    a = ax[1, 0]
    names = list(s["stage_service_ms"])
    means = [s["stage_service_ms"][n]["mean"] for n in names]
    p95 = [s["stage_service_ms"][n]["p95"] for n in names]
    a.barh(names, means, xerr=[np.zeros(len(means)), np.clip(np.array(p95) - means, 0, None)], color="#4c72b0", capsize=3)
    for i, m in enumerate(means):
        a.text(m, i, f" {m:.1f} ms", va="center")
    a.axvline(s["budget_ms"], color="r", ls="--", lw=1, label="frame budget")
    a.invert_yaxis(); a.set_xlabel("ms per frame set (whisker = p95)")
    a.set_title(f"service time per stage - bottleneck {s['bottleneck']}"); a.legend(fontsize=8)

    a = ax[1, 1]
    a.hist(lat, bins=40, color="#55a868")
    a.axvline(s["delay_ms"]["p95"], color="k", ls=":", label=f"p95 {s['delay_ms']['p95']:.1f} ms")
    a.set_xlabel("delay (ms)"); a.set_ylabel("frames"); a.set_title("delay distribution"); a.legend(fontsize=8)

    verdict = "REAL TIME" if s["realtime"] else "NOT real time"
    fig.suptitle(f"{title}\n{verdict} at {s['fps_in']:.0f} fps | output {s['output_rate_fps']:.1f}/s | "
                 f"max sustainable {s['max_sustainable_fps']:.1f} fps | needs {s['speedup_needed']:.2f}x")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_lpf_accuracy(s, done, source, path, title, filled=()):
    """Per-frame MPJPE before and after LivePoseFormer, over time and as distributions."""
    rows = [r for r in done if "err_lpf_mm" in r]
    filled = [r for r in filled if "err_lpf_mm" in r]
    if not rows:
        return
    a = s["lpf"]["accuracy"]
    t = np.array([r["t_capture"] - source.t0 for r in rows])
    before = np.array([r["err_raw_mm"] for r in rows])
    after = np.array([r["err_lpf_mm"] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    a1.plot(t, before, lw=0.8, label=f"before LivePoseFormer (mean {a['mpjpe_before_mm']:.1f} mm)")
    a1.plot(t, after, lw=0.8, label=f"after LivePoseFormer (mean {a['mpjpe_after_mm']:.1f} mm)")
    if filled:
        a1.plot([r["t_capture"] - source.t0 for r in filled], [r["err_lpf_mm"] for r in filled], ".", ms=3,
                color="#d62728", label=f"dropped frame filled by LivePoseFormer (mean {s['lpf']['filled_mpjpe_mm']:.1f} mm)")
    a1.set_xlabel("capture time (s)"); a1.set_ylabel("MPJPE vs Vicon GT (mm)")
    a1.set_title("error per frame (world frame)"); a1.legend(fontsize=8)
    bins = np.linspace(0, np.percentile(np.r_[before, after], 99), 40)
    a2.hist(before, bins=bins, alpha=0.6, label="before")
    a2.hist(after, bins=bins, alpha=0.6, label="after")
    a2.set_xlabel("MPJPE (mm)"); a2.set_ylabel("frames"); a2.legend(fontsize=8)
    a2.set_title(f"LPF lowered the error in {a['frames_improved_pct']:.0f}% of frames")
    fig.suptitle(f"{title}\nMPJPE {a['mpjpe_before_mm']:.1f} -> {a['mpjpe_after_mm']:.1f} mm "
                 f"({a['mpjpe_change_pct']:+.1f}%), jitter {a['accel_error_before']:.2f} -> "
                 f"{a['accel_error_after']:.2f} mm/frame^2")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_biomech(s, records, source, path, title,
                 channels=("hip_flexion_r", "knee_flexion_r", "ankle_dorsiflexion_r")):
    """A few joint angles over time (prediction vs GT) and the mean error per plane."""
    from .biomech_stage import PLANES
    rows = sorted((r for r in records.values() if "angles_gt" in r), key=lambda r: r["seq"])
    if not rows or "mae_deg" not in s.get("biomech", {}):
        return
    names = list(s["biomech_channels"])
    t = np.array([r["t_capture"] - source.t0 for r in rows])
    filled = np.array([bool(r.get("filled")) for r in rows])
    fig, axes = plt.subplots(len(channels) + 1, 1, figsize=(12, 2.6 * (len(channels) + 1)))
    for a, ch in zip(axes, channels):
        k = names.index(ch)
        gt = np.array([r["angles_gt"][k] for r in rows])
        a.plot(t, gt, "k-", lw=1, label="Vicon GT")
        pred = gt + _wrap(np.array([r["angles"][k] for r in rows]) - gt)   # no +-180 deg jumps in the plot
        a.plot(t[~filled], pred[~filled], ".", ms=2.5, label="live (measured frame)")
        if filled.any():
            a.plot(t[filled], pred[filled], ".", ms=2.5, color="#d62728", label="live (filled by LivePoseFormer)")
        a.set_ylabel(f"{ch} (deg)"); a.legend(fontsize=7, loc="upper right")
    axes[-2].set_xlabel("stream time (s)")
    b = s["biomech"]["mae_deg"]
    groups = list(b)
    x = np.arange(len(PLANES))
    for g_i, g in enumerate(groups):
        axes[-1].bar(x + g_i * 0.8 / len(groups), [b[g][p] for p in PLANES], 0.8 / len(groups), label=g)
    axes[-1].set_xticks(x + 0.4 - 0.4 / len(groups)); axes[-1].set_xticklabels(["sagittal", "frontal", "transverse"])
    axes[-1].set_ylabel("mean abs error (deg)"); axes[-1].legend(fontsize=8)
    fig.suptitle(f"{title}\njoint angles: mean abs error {b['all output']['mean']:.2f} deg over the three planes")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_json(s, info, args, path):
    with open(path, "w") as f:
        json.dump({"summary": s, "system": info, "args": vars(args)}, f, indent=2, default=str)
