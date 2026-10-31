"""Record a replayable session for the live demo.

Runs the full live pipeline on every frame of a trial, in order, the way the live system would
process it: person detector -> 2D pose (3 views) -> triangulation -> world frame -> optional
LivePoseFormer -> joint angles (Plug-in Gait) -> feedback metrics. Every stage is timed. The result
is written as a session folder that the demo app plays back in real time:

    live_demo/sessions/<name>/
        session.json     everything that is not per frame: trial, pipeline, cameras (frustums in the
                         world frame), skeleton, channel and metric names, static trial checks,
                         timing summary, and the layout of frames.bin
        frames.bin       float32 [frames, fields]: 2D keypoints, confidences and boxes per view,
                         3D markers (live, raw, Vicon), 72 joint angles (live, Vicon), feedback
                         metrics (live, Vicon), stage times, processing latency
        video_c1/2/3.bin the camera frames as small JPEGs, back to back (offsets in session.json)

    # from the repository folder:
    python live_demo/record_session.py --model hrnet --name S1A3D1_squat
    python live_demo/record_session.py --model vitpose --precision bf16 --lpf-repo ../LivePoseFormer \\
        --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59.pth --name S1A3D1_squat_lpf
    python live_demo/record_session.py --refresh S1A3D1_squat [repo] [--angle-smoothing=standard]   # recompute angles

The static trial is taken from the first frame, before the recording starts (as a real session
records it before going live). Ground truth (Vicon) is stored alongside for the validation overlay.
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import METRICS, compute_metrics, static_check  # noqa: E402
from pipeline import LivePipeline, Recording  # noqa: E402
from pipeline.skeleton import KEYPOINTS_59, SKELETON_EDGES  # noqa: E402
from pipeline.triangulation import triangulate  # noqa: E402
from stream_sim.biomech_stage import BiomechStage  # noqa: E402
from stream_sim.lpf_repo import HELP, find_lpf_repo  # noqa: E402

MODELS = {"hrnet": "w48_768x576_all_63", "vitpose": "ViT_B_simple_all_63"}
PRECISION = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
STAGES = ["det", "pre", "gpu", "post", "lpf", "bio"]


def camera_frustums(pipe, image_hw, depth_mm=450.0):
    """Each camera's centre and image-corner rays at ``depth_mm``, in the world frame (mm)."""
    h, w = image_hw
    depth = depth_mm / pipe.units_to_mm
    cams = []
    for k, (K, R, T) in enumerate(zip(pipe.rig.K, pipe.rig.R, pipe.rig.T)):
        T = np.asarray(T).reshape(3)
        to_world = lambda x_k: pipe.world.apply(((x_k - T) @ R)[None])[0]   # cam k -> cam 1 -> world
        corners = []
        for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
            ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
            corners.append(to_world(ray / ray[2] * depth).tolist())
        cams.append({"name": f"C{k + 1}", "center": to_world(np.zeros(3)).tolist(), "corners": corners})
    return cams


def biomech_outputs(bio, out_xyz, static_xyz, gt):
    """Real-time joint angles (frame by frame, continuous), Vicon angles (same model, own static trial),
    per-frame biomech time, the non-angle measures and the static-trial checks."""
    bio.calibrate(static_xyz)                             # static trial: before the recording starts
    n = len(out_xyz)
    angles = np.zeros((n, len(bio.channels)), np.float32)
    unsmoothed = np.zeros_like(angles)
    t_bio = np.zeros(n, np.float32)
    for k in range(n):
        t0 = time.perf_counter()
        angles[k] = bio.angles(out_xyz[k])
        t_bio[k] = (time.perf_counter() - t0) * 1000
        unsmoothed[k] = bio.last_unsmoothed
    angles_gt = bio.angles_batch(gt.astype(np.float64), static=bio.static_reference(gt[0])).astype(np.float32)
    used = (np.abs(angles_gt).max(0) > 1e-6) | (np.abs(angles).max(0) > 1e-6)
    met, met_gt = compute_metrics(out_xyz, static_xyz), compute_metrics(gt, gt[0])
    info = {"channels": bio.channels, "channel_used": used.tolist(),
            "channel_keys": {c: list(bio.channel_key[c]) for c in bio.channels},
            "metrics": {m: {"label": v[0], "unit": v[1], "group": v[2]} for m, v in METRICS.items()},
            "angle_smoothing": {"preset": getattr(bio, "smoothing_name", "off"),
                                "one_euro": bio.smoothing or None},
            "static": {"live": static_check(static_xyz), "vicon": static_check(gt[0]),
                       "biomech_check_deg": bio.check_deg, "continuity": dict(bio.stats),
                       "left_handed_input": bio.static["left_handed_input"]}}
    return (angles, unsmoothed, angles_gt, t_bio, np.stack([met[m] for m in METRICS], 1),
            np.stack([met_gt[m] for m in METRICS], 1), info)


def make_biomech(repo, smoothing, with_lpf):
    """The biomech stage with the angle smoothing preset; auto = standard on raw 3D, off with LivePoseFormer."""
    name = ("off" if with_lpf else "standard") if smoothing == "auto" else smoothing
    bio = BiomechStage(repo, smoothing=name)
    bio.smoothing_name = name
    return bio


def write_frames(folder, fields):
    """Write the per-frame float32 table; returns (layout, stride)."""
    layout, col = {}, 0
    for fname, arr in fields:
        layout[fname] = [col, arr.shape[1]]
        col += arr.shape[1]
    np.concatenate([a.astype(np.float32) for _, a in fields], 1).tofile(Path(folder) / "frames.bin")
    return layout, col


def refresh(folder, biomech_repo, smoothing="auto"):
    """Recompute the joint angles and measures of a recorded session from its stored 3D."""
    folder = Path(folder)
    s = json.loads((folder / "session.json").read_text())
    frames = np.fromfile(folder / "frames.bin", np.float32).reshape(-1, s["stride"])
    old = {k: frames[:, a:a + n] for k, (a, n) in s["layout"].items()}
    J = len(s["markers"])
    xyz = {k: old[k].reshape(len(frames), J, 3).astype(np.float64) for k in ("xyz", "xyz_raw", "gt")}
    bio = make_biomech(biomech_repo, smoothing, bool(s["pipeline"]["lpf"]))
    ang, ang_un, ang_gt, t_bio, met, met_gt, info = biomech_outputs(bio, xyz["xyz"], xyz["xyz_raw"][0], xyz["gt"])
    ti = s["stages"].index("bio")
    old["latency"] = old["latency"] - old["t"][:, ti:ti + 1] + t_bio[:, None]
    old["t"] = old["t"].copy()
    old["t"][:, ti] = t_bio
    old.update(ang=ang, ang_unsmoothed=ang_un, ang_gt=ang_gt, met=met, met_gt=met_gt)
    order = ["kp2d", "conf2d", "box", "xyz", "xyz_raw", "filled", "gt", "ang", "ang_unsmoothed", "ang_gt", "met", "met_gt",
             "t", "latency"]
    s["layout"], s["stride"] = write_frames(folder, [(k, old[k]) for k in order])
    s.update(info)
    s["timing_ms"]["bio"] = {"mean": float(t_bio.mean()), "p95": float(np.percentile(t_bio, 95))}
    (folder / "session.json").write_text(json.dumps(s))
    print(f"refreshed {folder.name}: smoothing {info['angle_smoothing']['preset']}, continuity {info['static']['continuity']}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] in ("--refresh", "--refresh-metrics"):
        rest = sys.argv[3:]
        smoothing = next((a.split("=", 1)[1] for a in rest if a.startswith("--angle-smoothing=")), "auto")
        repos = [a for a in rest if not a.startswith("--")]
        repo = find_lpf_repo(repos[0] if repos else None, required=True)
        return refresh(Path(__file__).resolve().parent / "sessions" / sys.argv[2], repo, smoothing)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="hrnet", choices=list(MODELS))
    p.add_argument("--checkpoint", default="best_nmae_model.pth")
    p.add_argument("--trial", default="S1A3D1")
    p.add_argument("--cams", default="C1,C2,C3")
    p.add_argument("--start", type=int, default=200, help="first recorded frame (the static trial)")
    p.add_argument("--frames", type=int, default=None, help="number of frames (default: to the end)")
    p.add_argument("--precision", default="fp32", choices=list(PRECISION))
    p.add_argument("--detector", default="yolo26s", help="YOLO model name or .pt, or 'json' for the JSON boxes")
    p.add_argument("--lpf-repo", help="LivePoseFormer repository (adds the LivePoseFormer stage)")
    p.add_argument("--lpf-checkpoint", help="LivePoseFormer 59-keypoint checkpoint")
    p.add_argument("--biomech-repo", help="for the joint-angle model; default: --lpf-repo, else the " + HELP)
    p.add_argument("--angle-smoothing", default="auto", choices=["auto", "off", "light", "standard"],
                   help="One Euro smoothing of the joint angles; auto = standard on raw 3D, off with LivePoseFormer")
    p.add_argument("--video-height", type=int, default=600, help="height of the stored camera frames (px)")
    p.add_argument("--name", help="session name (default: <trial>_<model>)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    fps = 60.0
    name = args.name or f"{args.trial}_{args.model}"
    out = Path(__file__).resolve().parent / "sessions" / name
    out.mkdir(parents=True, exist_ok=True)

    rec = Recording(ROOT / "data", args.trial, args.cams.split(","))
    end = len(rec) if args.frames is None else min(len(rec), args.start + args.frames)
    indices = [i for i in range(args.start, end) if all(pth.exists() for pth in rec[i].image_paths)]
    indices = indices[:next((k for k in range(1, len(indices)) if indices[k] != indices[k - 1] + 1), len(indices))]
    print(f"{args.trial}: frames {indices[0]}-{indices[-1]} ({len(indices)} frames, {len(indices) / fps:.1f} s)")

    pipe = LivePipeline(ROOT, MODELS[args.model], rec.subject, args.checkpoint, args.device)
    pipe.setup_world(rec[args.start].wand_xy, rec.wand_wcs())
    pose = pipe.pose
    amp = PRECISION[args.precision] if pose.device.type == "cuda" else None
    detector = None
    if args.detector != "json":
        from stream_sim.detector import PersonDetector
        weights = args.detector if args.detector.endswith(".pt") else str(ROOT / f"{args.detector}.pt")
        detector = PersonDetector(weights, args.device, precision=args.precision)
    lpf = None
    if args.lpf_repo:
        from stream_sim.lpf_stage import LpfStage
        lpf = LpfStage(args.lpf_repo, args.lpf_checkpoint, args.device)
    bio_repo = find_lpf_repo(args.biomech_repo or args.lpf_repo, required=True)
    bio = make_biomech(bio_repo, args.angle_smoothing, lpf is not None)

    def read(k):
        return [cv2.imdecode(np.fromfile(str(pth), np.uint8), cv2.IMREAD_COLOR_RGB) for pth in rec[k].image_paths]

    @torch.inference_mode()
    def process(images, json_boxes):
        t = {}
        t0 = time.perf_counter()
        boxes = detector(images)[0] if detector is not None else json_boxes
        t1 = time.perf_counter()
        crops, adj = pose.crop(images, boxes)
        batch = pose.resize(crops)
        t2 = time.perf_counter()
        x = pose.upload(batch)
        with (torch.autocast("cuda", dtype=amp) if amp else nullcontext()):
            heat = pose.forward(x)
        kp, conf = pose.decode_on_device(heat)
        kp, conf = kp.cpu(), conf.cpu()
        t3 = time.perf_counter()
        kp2d, conf = pose.to_image(kp, conf, adj)
        xyz = pipe.world.apply(triangulate(kp2d[pipe.rig.use_cams], pipe.P))
        t4 = time.perf_counter()
        t.update(det=(t1 - t0) * 1000 if detector is not None else 0.0, pre=(t2 - t1) * 1000,
                 gpu=(t3 - t2) * 1000, post=(t4 - t3) * 1000)
        return kp2d, conf, np.asarray(boxes, float), xyz, t

    # warm-up (cuDNN autotuning, detector, allocator) on the first frame
    first = read(indices[0])
    for _ in range(3):
        process(first, rec[indices[0]].bboxes)
    image_hw = first[0].shape[:2]

    n = len(indices)
    J = len(KEYPOINTS_59)
    kp2d_all = np.zeros((n, 3, J, 2), np.float32)
    conf_all = np.zeros((n, 3, J), np.float32)
    box_all = np.zeros((n, 3, 4), np.float32)
    raw_all = np.zeros((n, J, 3), np.float32)
    times = {s: np.zeros(n, np.float32) for s in STAGES}
    scale = args.video_height / image_hw[0]
    vw = round(image_hw[1] * scale)
    video = [bytearray() for _ in range(3)]
    offsets = [[] for _ in range(3)]
    last_path = [None] * 3

    t_start = time.perf_counter()
    for k, idx in enumerate(indices):
        fs = rec[idx]
        images = first if k == 0 else read(idx)
        kp2d, conf, boxes, xyz, t = process(images, fs.bboxes)
        kp2d_all[k], conf_all[k], box_all[k], raw_all[k] = kp2d, conf, boxes, xyz
        for s, v in t.items():
            times[s][k] = v
        for c in range(3):                                # small JPEGs for the replay (shared files once)
            if fs.image_paths[c] != last_path[c]:
                small = cv2.resize(images[c], (vw, args.video_height), interpolation=cv2.INTER_AREA)
                jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])[1]
                offsets[c].append([len(video[c]), len(jpg)])
                video[c] += jpg.tobytes()
                last_path[c] = fs.image_paths[c]
            else:
                offsets[c].append(offsets[c][-1])
        if k % 100 == 0:
            print(f"  frame {k}/{n}  ({(time.perf_counter() - t_start) / (k + 1) * 1000:.0f} ms per frame)")

    # LivePoseFormer (streaming, frame by frame), then the joint angles and metrics of the output
    out_xyz = raw_all.copy()
    if lpf is not None:
        lpf.reset()
        released = []
        for k in range(n):
            t0 = time.perf_counter()
            released += lpf.push(k, raw_all[k].astype(np.float64))
            times["lpf"][k] = (time.perf_counter() - t0) * 1000
        released += lpf.flush()
        for seq, xyz in released:
            out_xyz[seq] = xyz
    gt = np.stack([rec[i].gt_xyz_wcs for i in indices]).astype(np.float32)
    angles, angles_un, angles_gt, times["bio"], met, met_gt, bio_info = biomech_outputs(bio, out_xyz, raw_all[0], gt)
    lookahead_ms = 3 * 1000 / fps if lpf is not None else 0.0
    latency = times["det"] + times["pre"] + times["gpu"] + times["post"] + lookahead_ms + times["lpf"] + times["bio"]

    fields = [("kp2d", kp2d_all.reshape(n, -1)), ("conf2d", conf_all.reshape(n, -1)), ("box", box_all.reshape(n, -1)),
              ("xyz", out_xyz.reshape(n, -1)), ("xyz_raw", raw_all.reshape(n, -1)),
              ("filled", np.zeros((n, 1), np.float32)), ("gt", gt.reshape(n, -1)),
              ("ang", angles), ("ang_unsmoothed", angles_un), ("ang_gt", angles_gt), ("met", met), ("met_gt", met_gt),
              ("t", np.stack([times[s] for s in STAGES], 1)), ("latency", latency[:, None])]
    layout, col = write_frames(out, fields)
    for c in range(3):
        (out / f"video_c{c + 1}.bin").write_bytes(bytes(video[c]))

    def stat(a):
        return {"mean": float(np.mean(a)), "p95": float(np.percentile(a, 95))}

    work = times["det"] + times["pre"] + times["gpu"] + times["post"]
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    session = {
        "name": name, "version": 1, "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "trial": args.trial, "subject_id": rec.subject, "task": "squat", "fps": fps,
        "start_index": indices[0], "n_frames": n, "indices": indices,
        "pipeline": {"pose_model": pose.name, "precision": args.precision,
                     "detector": detector.name if detector is not None else "JSON boxes",
                     "lpf": (Path(args.lpf_checkpoint).name if args.lpf_checkpoint else "random weights")
                     if lpf is not None else None,
                     "biomech": "Plug-in Gait (LivePoseFormer repo), vectorised", "gpu": gpu_name,
                     "torch": torch.__version__,
                     "mode": "offline: every frame processed in order; latency = processing time per frame"
                             + (" + LivePoseFormer's 3-frame lookahead" if lpf is not None else "")},
        "timing_ms": {s: stat(times[s]) for s in STAGES} | {"work_per_frame": stat(work), "latency": stat(latency)},
        "capacity_fps": float(1000 / np.mean(work)),
        "image_size": [int(image_hw[1]), int(image_hw[0])], "video_size": [vw, args.video_height],
        "video_offsets": offsets,
        "markers": KEYPOINTS_59, "skeleton": [list(e) for e in SKELETON_EDGES],
        "stages": STAGES, "cameras": camera_frustums(pipe, image_hw),
        "layout": layout, "stride": col,
    } | bio_info
    (out / "session.json").write_text(json.dumps(session))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"session written to {out} ({size:.1f} MB): work per frame {np.mean(work):.1f} ms, "
          f"latency {np.mean(latency):.1f} ms, MPJPE live {np.nanmean(np.linalg.norm(out_xyz - gt, axis=-1)):.1f} mm")


if __name__ == "__main__":
    main()
