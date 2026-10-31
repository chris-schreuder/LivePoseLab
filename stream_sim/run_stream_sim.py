"""Simulate a live 60 fps stream from 3 synchronised cameras and test whether the pipeline keeps up.

    # from the repository folder:
    python stream_sim/run_stream_sim.py --model vitpose                      # the whole video at 60 fps, both policies
    python stream_sim/run_stream_sim.py --model hrnet --duration 10          # only the first 10 s
    python stream_sim/run_stream_sim.py --model vitpose --mode both          # pipelined vs sequential
    python stream_sim/run_stream_sim.py --model vitpose --fake-network-ms 5,10,15   # what if the network were faster?
    python stream_sim/run_stream_sim.py --model vitpose --precision bf16 --compile  # RTX 30xx options
    python stream_sim/run_stream_sim.py --model vitpose --lpf-repo ../LivePoseFormer   # + LivePoseFormer

Every scenario (mode x policy x what-if) gets a folder in stream_sim/results/<run>/ with
report.txt, report.png, frames.csv and summary.json; the run folder gets comparison.txt/.csv.
See stream_sim/README.md for what the numbers mean.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pipeline import LivePipeline, Recording  # noqa: E402
from stream_sim.engine import StreamEngine  # noqa: E402
from stream_sim.lpf_repo import HELP, find_lpf_repo  # noqa: E402
from stream_sim.report import (plot, plot_biomech, plot_lpf_accuracy, save_json, summarize, system_info,  # noqa: E402
                               text_report)  # noqa: E402
from stream_sim.source import CameraSource, LatestSink, LiveVideo, QueueSink  # noqa: E402

MODELS = {"hrnet": "w48_768x576_all_63", "vitpose": "ViT_B_simple_all_63"}


def build_pipeline(args, rec):
    pipe = LivePipeline(ROOT, MODELS[args.model], rec.subject, args.checkpoint, args.device)
    pipe.setup_world(rec[args.start].wand_xy, rec.wand_wcs())
    if args.compile:
        try:
            compiled = torch.compile(pipe.pose.model)
            pipe.pose.model = compiled
            pipe.pose.warmup()                       # compilation happens here; fails fast if unsupported
            print("torch.compile: enabled")
        except Exception as e:                       # e.g. no Triton on Windows
            pipe.pose.model = getattr(pipe.pose.model, "_orig_mod", pipe.pose.model)
            print(f"torch.compile not available ({type(e).__name__}); running uncompiled")
    return pipe


def run_scenario(engine, video, fps, mode, policy, max_backlog):
    sink = LatestSink(on_drop=engine.on_drop) if policy == "latest" else QueueSink()
    source = CameraSource(video, sink, fps, max_backlog)
    video.open()                       # restart the video and fill the read-ahead buffer
    # For frames that were dropped: which recording frame they were, and when they were captured.
    engine.start_stream(lambda s: (video.indices[s], source.t0 + s / fps))
    if engine.cuda:
        torch.cuda.synchronize()
    # The camera clock starts only after the engine's threads have warmed up (on_ready).
    if mode == "pipelined":
        engine.run_pipelined(sink, policy, on_ready=source.start)
    else:
        engine.run_sequential(sink, on_ready=source.start)
    source.join()
    video.close()
    if source.error is not None:
        raise source.error
    if engine.errors:
        raise engine.errors[0]
    return source, sink


def write_frames_csv(done, source, path):
    keys = ["seq", "index", "t_capture", "t_pre_start", "t_pre_end", "t_gpu_start", "t_gpu_end",
            "t_post_start", "t_done", "t_lpf_known", "t_push_start", "t_push_end", "t_lpf_out",
            "t_bio_known", "t_bio_start", "t_bio_end"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([k if k in ("seq", "index") else k + "_ms" for k in keys]
                   + ["delay_ms", "mpjpe_raw_mm", "mpjpe_after_lpf_mm", "filled_by_lpf", "angle_mae_deg"])
        for r in done:
            row = [r["seq"], r["index"]] + [
                round((r[k] - source.t0) * 1000, 3) if k in r else "" for k in keys[2:]]
            errs = [round(r[k], 2) if k in r else "" for k in ("err_raw_mm", "err_lpf_mm")]
            delay = round((r["t_done"] - r["t_capture"]) * 1000, 3) if "t_done" in r else ""
            ang = round(r["angle_mae"], 3) if "angle_mae" in r else ""
            w.writerow(row + [delay] + errs + [int(bool(r.get("filled"))), ang])


def write_angles_csv(records, channels, source, path):
    """Every output frame's 72 joint angles (live and Vicon GT), in degrees."""
    rows = sorted((r for r in records.values() if "angles" in r), key=lambda r: r["seq"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "index", "filled_by_lpf", "t_capture_ms", "delay_to_angles_ms"]
                   + [f"live_{c}" for c in channels] + [f"gt_{c}" for c in channels])
        for r in rows:
            gt = list(np.round(r["angles_gt"], 3)) if "angles_gt" in r else [""] * len(channels)
            w.writerow([r["seq"], r["index"], int(bool(r.get("filled"))), round((r["t_capture"] - source.t0) * 1000, 3),
                        round((r["t_bio_end"] - r["t_capture"]) * 1000, 3)] + list(np.round(r["angles"], 3)) + gt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vitpose", choices=list(MODELS))
    p.add_argument("--checkpoint", default="best_nmae_model.pth")
    p.add_argument("--trial", default="S1A3D1")
    p.add_argument("--cams", default="C1,C2,C3")
    p.add_argument("--fps", type=float, default=60.0, help="camera frame rate to simulate")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds of stream per scenario (default: from --start to the end of the video)")
    p.add_argument("--start", type=int, default=200, help="first recorded frame index to use")
    p.add_argument("--prefetch", type=int, default=120,
                   help="frame sets the live reader keeps decoded ahead of the clock (~19 MB each, ~0.6 MB as JPEG)")
    p.add_argument("--reader-threads", type=int, default=3,
                   help="threads that read and decode frames from disk ahead of the clock")
    p.add_argument("--max-backlog", type=int, default=300,
                   help="policy all: stop the run once this many frames are queued (it would only keep growing)")
    p.add_argument("--mode", default="pipelined", choices=["pipelined", "sequential", "both"])
    p.add_argument("--policy", default="both", choices=["latest", "all", "both"],
                   help="latest = drop frames the pipeline cannot take (live); all = queue everything")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="network precision (fp16/bf16 need a GPU with tensor cores, e.g. RTX 30xx)")
    p.add_argument("--compile", action="store_true", help="try torch.compile (needs Triton, i.e. Linux)")
    p.add_argument("--decode-jpeg", action="store_true", help="cameras send JPEG; decoding is part of the pipeline")
    p.add_argument("--detector", default="yolo26s",
                   help="person detector giving the crop boxes: a YOLO model name (yolo26n/s/m/l/x; weights are "
                        "downloaded on first use) or a .pt path, or 'json' to use the boxes in the camera JSON files")
    p.add_argument("--detector-conf", type=float, default=0.25,
                   help="minimum person confidence; below it the previous box is kept and a miss is counted")
    p.add_argument("--fake-network-ms", default="",
                   help="comma list: replace the network by a fixed time to see what a faster network would give")
    p.add_argument("--lpf-repo", help="LivePoseFormer repository folder: adds the streaming smoothing stage")
    p.add_argument("--lpf-checkpoint", help="LivePoseFormer 59-keypoint checkpoint (random weights if omitted)")
    p.add_argument("--lpf-no-graph", action="store_true",
                   help="run LivePoseFormer eagerly instead of replaying it as a CUDA graph (for comparison)")
    p.add_argument("--lpf-fill-dropped", action="store_true",
                   help="push every frame the pose pipeline dropped to LivePoseFormer as an all-NaN frame, "
                        "so it imputes the missing frames and its output keeps the full camera rate")
    p.add_argument("--biomech", action="store_true",
                   help="add the real-time biomechanics stage (joint angles) after the 3D / LivePoseFormer")
    p.add_argument("--biomech-repo", help="for the joint-angle model; default: --lpf-repo, else the " + HELP)
    p.add_argument("--angle-smoothing", default="auto", choices=["auto", "off", "light", "standard"],
                   help="One Euro smoothing of the joint angles; auto = standard on raw 3D, off with LivePoseFormer")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(ROOT / "stream_sim" / "results"))
    args = p.parse_args()
    if args.lpf_fill_dropped and not args.lpf_repo:
        p.error("--lpf-fill-dropped needs the LivePoseFormer stage (--lpf-repo)")

    info = system_info()
    gpu_tag = info.get("gpu", "cpu").replace("NVIDIA GeForce ", "").replace(" ", "")
    run_dir = Path(args.out) / f"{args.model}_{gpu_tag}_{int(args.fps)}fps_{args.precision}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {info.get('gpu', 'none')} | torch {info['torch']} | results -> {run_dir}")

    rec = Recording(ROOT / "data", args.trial, args.cams.split(","))
    n_wanted = None if args.duration is None else int(round(args.duration * args.fps))
    video = LiveVideo(rec, args.start, n_wanted, keep_jpeg=args.decode_jpeg,
                      prefetch=args.prefetch, threads=args.reader_threads)
    n_frames = len(video)
    print(f"live video: frames {video.indices[0]}-{video.indices[-1]} of {args.trial} ({n_frames} frame sets, "
          f"{n_frames / args.fps:.1f} s at {args.fps:g} fps), read from disk {args.prefetch} frame sets ahead "
          f"by {args.reader_threads} threads{' (JPEG bytes only)' if args.decode_jpeg else ''}")

    pipe = build_pipeline(args, rec)
    detector = None
    if args.detector != "json":
        from stream_sim.detector import PersonDetector
        weights = args.detector if args.detector.endswith(".pt") else str(ROOT / f"{args.detector}.pt")
        detector = PersonDetector(weights, args.device, precision=args.precision, conf=args.detector_conf)
        print(f"detector: {detector.name} ({weights})")
    lpf = None
    if args.lpf_repo:
        from stream_sim.lpf_stage import LpfStage
        lpf = LpfStage(args.lpf_repo, args.lpf_checkpoint, args.device, use_graph=not args.lpf_no_graph)
    bio = None
    if args.biomech:
        from stream_sim.biomech_stage import BiomechStage
        smoothing = ("off" if args.lpf_repo else "standard") if args.angle_smoothing == "auto" else args.angle_smoothing
        bio = BiomechStage(find_lpf_repo(args.biomech_repo or args.lpf_repo, required=True), smoothing=smoothing)
        bio.smoothing_name = smoothing
    gt = dict(zip(video.indices, video.gt_wcs))

    modes = ["pipelined", "sequential"] if args.mode == "both" else [args.mode]
    policies = ["latest", "all"] if args.policy == "both" else [args.policy]
    fakes = [None] + [float(x) for x in args.fake_network_ms.split(",") if x.strip()]

    rows = []
    for fake in fakes:
        engine = StreamEngine(pipe, args.precision, fake, detector, lpf, args.decode_jpeg,
                              args.lpf_fill_dropped, bio)
        engine.warmup(video.warm_frames(5))
        for mode in modes:
            for policy in policies:
                name = f"{mode}_{policy}" + (f"_network{fake:g}ms" if fake is not None else "")
                print(f"\n--- {name}: {n_frames} frame sets at {args.fps:g} fps ({n_frames / args.fps:.1f} s) ---")
                source, sink = run_scenario(engine, video, args.fps, mode, policy, args.max_backlog)
                s, done = summarize(engine.records, len(source.late), args.fps, mode, policy, source,
                                    gt if fake is None else None, lpf, bio)
                title = (f"{pipe.pose.name} | {name} | {info.get('gpu', 'cpu')} | {args.precision}"
                         + (f" | {detector.name} boxes" if detector is not None else " | JSON boxes")
                         + (" | JPEG decode" if args.decode_jpeg else "")
                         + (" | LPF fills dropped frames" if args.lpf_fill_dropped else "")
                         + (" | biomech" if bio is not None else ""))
                report = text_report(s, title, info)
                print(report)
                out = run_dir / name
                out.mkdir(exist_ok=True)
                (out / "report.txt").write_text(report + "\n")
                dropped = sorted(getattr(sink, "dropped_seqs", []) + list(engine.handoff_drops))
                plot(s, done, dropped, source, out / "report.png", title)
                filled = sorted((r for r in engine.records.values() if r.get("filled") and "xyz_lpf" in r),
                                key=lambda r: r["seq"])
                if "accuracy" in s.get("lpf", {}):
                    plot_lpf_accuracy(s, done, source, out / "lpf_accuracy.png", title, filled)
                write_frames_csv(sorted(done + filled, key=lambda r: r["seq"]), source, out / "frames.csv")
                if bio is not None:
                    write_angles_csv(engine.records, bio.channels, source, out / "angles.csv")
                    plot_biomech(s, engine.records, source, out / "biomech.png", title)
                save_json(s, info, args, out / "summary.json")
                rows.append({"scenario": name, "realtime": s["realtime"], "output_fps": round(s["output_rate_fps"], 1),
                             "dropped_pct": round(s["dropped_pct"], 1), "delay_mean_ms": round(s["delay_ms"]["mean"], 1),
                             "delay_p95_ms": round(s["delay_ms"]["p95"], 1),
                             "delay_trend_ms_per_s": round(s["delay_trend_ms_per_s"], 1),
                             "max_sustainable_fps": round(s["max_sustainable_fps"], 1),
                             "speedup_needed": round(s["speedup_needed"], 2), "bottleneck": s["bottleneck"],
                             "mpjpe_raw_mm": round(s.get("mpjpe_world_mm", float("nan")), 1),
                             "mpjpe_after_lpf_mm": round(s.get("lpf", {}).get("accuracy", {}).get(
                                 "mpjpe_after_mm", float("nan")), 1),
                             "mpjpe_filled_mm": round(s.get("lpf", {}).get("filled_mpjpe_mm", float("nan")), 1),
                             "lpf_output_fps": round(s.get("lpf", {}).get("output_rate_fps", float("nan")), 1),
                             "lpf_keeps_up": s.get("lpf", {}).get("keeps_up", ""),
                             "angle_err_deg": round(s.get("biomech", {}).get("mae_deg", {}).get(
                                 "all output", {}).get("mean", float("nan")), 2),
                             "bio_keeps_up": s.get("biomech", {}).get("keeps_up", "")})

    with open(run_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    header = (f"{'scenario':<34}{'real time':>10}{'out fps':>9}{'drop %':>8}{'delay':>8}{'p95':>8}{'trend':>8}"
              f"{'max fps':>9}{'need x':>8}{'mm raw':>8}{'mm LPF':>8}{'mm fill':>8}{'LPF fps':>9}{'LPF ok':>8}"
              f"{'deg':>7}{'BIO ok':>8}")
    lines = [f"{pipe.pose.name} on {info.get('gpu', 'cpu')}, input {args.fps:g} fps, {args.precision}", header,
             "-" * len(header)]
    for r in rows:
        lines.append(f"{r['scenario']:<34}{'YES' if r['realtime'] else 'no':>10}{r['output_fps']:>9}"
                     f"{r['dropped_pct']:>8}{r['delay_mean_ms']:>8}{r['delay_p95_ms']:>8}{r['delay_trend_ms_per_s']:>8}"
                     f"{r['max_sustainable_fps']:>9}{r['speedup_needed']:>8}"
                     f"{r['mpjpe_raw_mm']:>8}{r['mpjpe_after_lpf_mm']:>8}{r['mpjpe_filled_mm']:>8}{r['lpf_output_fps']:>9}"
                     f"{'' if r['lpf_keeps_up'] == '' else ('yes' if r['lpf_keeps_up'] else 'NO'):>8}"
                     f"{r['angle_err_deg']:>7}{'' if r['bio_keeps_up'] == '' else ('yes' if r['bio_keeps_up'] else 'NO'):>8}")
    lines.append("delay = capture -> 3D result (ms); trend = ms the delay grows per second; need x = speed-up for real time;\n"
                 "mm raw / mm LPF = MPJPE vs Vicon GT (world frame) before / after LivePoseFormer (nan = not run);\n"
                 "mm fill = MPJPE of dropped frames filled by LivePoseFormer (--lpf-fill-dropped);\n"
                 "LPF fps = smoothed output frames/s; LPF ok = LivePoseFormer keeps up (its queue does not grow);\n"
                 "deg = joint-angle mean abs error vs GT over the 3 planes (--biomech); BIO ok = biomechanics keeps up")
    (run_dir / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n\nResults in {run_dir}")


if __name__ == "__main__":
    main()
