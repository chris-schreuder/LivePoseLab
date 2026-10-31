"""End-to-end evaluation of the real-time pipeline on simulated 60 fps camera feeds.

For every model and every selected recording, the three cameras are simulated at 60 fps (frames
read from disk just ahead of a fixed clock) and the full live pipeline processes them exactly as the
stream simulator / demo does:

    YOLO person detector -> 2D pose (ViTPose or HRNet, bf16, torch.compile) -> triangulation ->
    world frame -> LivePoseFormer (fills dropped frames) -> real-time biomechanics (joint angles,
    continuity, One Euro smoothing)

Frames the pipeline cannot take are dropped (policy "latest"), as live. Every run records timing per
stage, latency per frame, frame rates, drops and fills, and the errors at every stage against Vicon.
At the end ``report.py`` writes the paper report (REPORT.md, LaTeX tables, figures, CSVs).

    # from the repository folder:
    python realtime_eval/run_eval.py --data-root /path/to/data --list                      # what is there
    python realtime_eval/run_eval.py --data-root /path/to/data --auto 6 \\
        --lpf-repo ../LivePoseFormer --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59.pth
    python realtime_eval/run_eval.py --data-root /path/to/data --recordings S7A1D1 S8A2D1 ...
    python realtime_eval/run_eval.py --data-root /path/to/data --all ...                  # every recording
    python realtime_eval/run_eval.py --out realtime_eval/results/<run> --report-only           # rebuild the report

Runs already finished are skipped, so an interrupted evaluation continues where it stopped (``--force``
redoes them). A progress line shows the current run and the estimated time left.
"""
import argparse
import json
import platform
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset import FullDataRecording, discover, pick_spread  # noqa: E402
from pipeline import LivePipeline  # noqa: E402
from pipeline.calibration import load_rig  # noqa: E402
from stages import frame_arrays, replay_angles, stage_errors  # noqa: E402
from stream_sim.biomech_stage import BiomechStage  # noqa: E402
from stream_sim.engine import StreamEngine  # noqa: E402
from stream_sim.lpf_repo import HELP, find_lpf_repo  # noqa: E402
from stream_sim.report import plot, plot_biomech, plot_lpf_accuracy, summarize, system_info, text_report  # noqa: E402
from stream_sim.source import CameraSource, LatestSink, LiveVideo  # noqa: E402

MODELS = {"vitpose": ("ViT_B_simple_all_63", "ViTPose"), "hrnet": ("w48_768x576_all_63", "HRNet")}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def fmt_time(s):
    return str(timedelta(seconds=int(max(0, s))))


class Progress:
    """Progress over all runs with an estimate of the time left (stream time + measured overheads)."""

    def __init__(self, plan, fps, out):
        self.plan, self.fps, self.out = plan, fps, out          # plan: [(model, trial, n_frames, done?)]
        self.t_start = time.time()
        self.overheads = []                                     # setup + warm-up + analysis per finished run
        self.i, self.frames = 0, 0
        self.todo_total = sum(1 for p in plan if not p[3])     # runs to do in this session
        self.done_count = 0
        self.tty = sys.stdout.isatty()
        self._last_print = 0

    def est_overhead(self):
        return float(np.mean(self.overheads)) if self.overheads else 40.0

    def remaining(self):
        cur = self.plan[self.i]
        left = max(0, cur[2] - self.frames) / self.fps + (0 if self.frames else self.est_overhead())
        for m, t, n, done in self.plan[self.i + 1:]:
            if not done:
                left += n / self.fps + self.est_overhead()
        return left

    def show(self, status="", force=False):
        now = time.time()
        if not force and now - self._last_print < (1.0 if self.tty else 15.0):
            return
        self._last_print = now
        m, t, n, _ = self.plan[self.i]
        k, todo_n = self.done_count + 1, self.todo_total
        bar = int(20 * self.frames / max(n, 1))
        line = (f"[run {k}/{todo_n}] {MODELS[m][1]:<8} {t:<9} [{'#' * bar}{'.' * (20 - bar)}] "
                f"{self.frames:>5}/{n} fr  {status:<22} elapsed {fmt_time(now - self.t_start)}  "
                f"left ~{fmt_time(self.remaining())}  (done ~{(datetime.now() + timedelta(seconds=self.remaining())).strftime('%H:%M')})")
        print(("\r" + line) if self.tty else line, end="" if self.tty else "\n", flush=True)
        (self.out / "progress.json").write_text(json.dumps({
            "run": k, "runs": todo_n, "model": m, "trial": t, "frames": self.frames, "n_frames": n,
            "status": status, "elapsed_s": now - self.t_start, "left_s": self.remaining()}))


def build_pipeline(model, subject, args, log):
    model_dir, name = MODELS[model]
    pipe = LivePipeline(ROOT, model_dir, subject, "best_nmae_model.pth", args.device)
    if args.compile:
        try:
            pipe.pose.model = torch.compile(pipe.pose.model)
            pipe.pose.warmup()
            log(f"{name}: torch.compile enabled")
        except Exception as e:                                   # e.g. no Triton
            pipe.pose.model = getattr(pipe.pose.model, "_orig_mod", pipe.pose.model)
            log(f"{name}: torch.compile not available ({type(e).__name__}: {e}); running uncompiled")
    return pipe


def run_one(model, trial, pipe, detector, lpf, bio, args, prog, out_dir, info):
    """Stream one recording through the full pipeline and save everything about it."""
    t_setup = time.time()
    rec = FullDataRecording(args.data_root, trial, args.cams.split(","))
    pipe.rig = load_rig(ROOT, rec.subject)                     # calibration of this subject's session
    pipe.P = pipe.rig.P[pipe.rig.use_cams]
    pipe.setup_world(rec[args.start].wand_xy, rec.wand_wcs())
    n_cap = None if args.max_seconds is None else int(round(args.max_seconds * args.fps))
    video = LiveVideo(rec, args.start, n_cap, prefetch=args.prefetch, threads=args.reader_threads)
    engine = StreamEngine(pipe, args.precision, None, detector, lpf, False, args.fill and lpf is not None, bio)
    prog.show("warm-up, static trial", force=True)
    engine.warmup(video.warm_frames(5))                        # static trial (first frame) + GPU warm-up
    sink = LatestSink(on_drop=engine.on_drop)
    source = CameraSource(video, sink, args.fps)
    video.open()
    engine.start_stream(lambda q: (video.indices[q], source.t0 + q / args.fps))
    overhead = time.time() - t_setup
    running = threading.Event()
    running.set()

    def monitor():
        while running.is_set():
            prog.frames = len(source.late)
            prog.show("streaming 60 fps")
            time.sleep(0.5)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    engine.run_pipelined(sink, "latest", on_ready=source.start)
    source.join()
    running.clear()
    video.close()
    if source.error is not None:
        raise source.error
    if engine.errors:
        raise engine.errors[0]
    t_an = time.time()
    prog.frames = len(source.late)
    prog.show("analysing", force=True)
    gt = dict(zip(video.indices, video.gt_wcs))
    s, done = summarize(engine.records, len(source.late), args.fps, "pipelined", "latest", source, gt, lpf, bio)
    stages = stage_errors(engine.records, rec)
    s["_t0"] = source.t0
    arrays = frame_arrays(engine.records, s)
    s.pop("_t0")
    # The same joint angles with the static trial taken from Vicon (frame 0) instead of the pipeline's own 3D:
    # the biomechanics stage replayed on the 3D stream it received live (see stages.replay_angles).
    static_gt = gt[min(gt)]
    arrays["static_xyz_pipeline"] = np.asarray(engine._warm_xyz[0], np.float32)
    arrays["static_xyz_vicon"] = np.asarray(static_gt, np.float32)
    if bio is not None and len(arrays["out_xyz"]):
        bio_gt = BiomechStage(args.lpf_repo, smoothing=args.angle_smoothing)
        arrays["out_angles_gtstatic"], arrays["out_angles_gtstatic_unsmoothed"] = replay_angles(
            bio_gt, arrays["out_xyz"], arrays["out_index"], static_gt)

    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{MODELS[model][1]} {args.precision}{' compiled' if args.compile else ''} | {trial} | {info.get('gpu', 'CPU')}"
    (out_dir / "report.txt").write_text(text_report(s, title, info) + "\n")
    dropped = sorted(getattr(sink, "dropped_seqs", []) + list(engine.handoff_drops))
    try:
        plot(s, done, dropped, source, out_dir / "timing.png", title)
        filled = sorted((r for r in engine.records.values() if r.get("filled") and "xyz_lpf" in r), key=lambda r: r["seq"])
        if "accuracy" in s.get("lpf", {}):
            plot_lpf_accuracy(s, done, source, out_dir / "lpf_accuracy.png", title, filled)
        plot_biomech(s, engine.records, source, out_dir / "joint_angles.png", title)
    except Exception as e:                                        # a plot must never lose a run
        print(f"\n  (plot failed: {type(e).__name__}: {e})")
    np.savez_compressed(out_dir / "frames.npz", **arrays)
    summary = {
        "model": model, "model_name": MODELS[model][1], "trial": trial, "subject": rec.subject, "action": rec.action,
        "take": rec.take, "start_index": args.start, "n_frames": len(video), "duration_s": len(video) / args.fps,
        "config": {"fps": args.fps, "precision": args.precision, "compile": args.compile, "detector": args.detector,
                   "lpf_checkpoint": (Path(args.lpf_checkpoint).name if args.lpf_checkpoint.lower() != "none"
                                      else "random weights (test only)"),
                   "fill_dropped": args.fill and lpf is not None, "angle_smoothing": args.angle_smoothing,
                   "policy": "latest", "cams": args.cams, "prefetch": args.prefetch, "reader_threads": args.reader_threads},
        "system": info, "summary": s, "stages": stages,
        "dropped_detail": {"source": len(getattr(sink, "dropped_seqs", [])), "handoff": len(engine.handoff_drops)},
        "wall_time_s": {"setup_and_warmup": overhead, "analysis": time.time() - t_an},
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    return overhead + (time.time() - t_an)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", help="dataset root with frames/, camera_data/ and wcs_data/")
    p.add_argument("--recordings", nargs="*", default=[], help="trials to evaluate, e.g. S7A1D1 S8A2D1")
    p.add_argument("--auto", type=int, default=0, help="pick N recordings spread over subjects and actions")
    p.add_argument("--all", action="store_true", help="evaluate every recording found in --data-root")
    p.add_argument("--list", action="store_true", help="list the recordings available and exit")
    p.add_argument("--models", nargs="+", default=["vitpose", "hrnet"], choices=list(MODELS))
    p.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--no-compile", dest="compile", action="store_false", help="disable torch.compile")
    p.add_argument("--detector", default="yolo26s", help="YOLO model (yolo26n/s/m) or 'json' for the JSON boxes")
    p.add_argument("--lpf-repo", help=HELP + "; LivePoseFormer stage and joint-angle model")
    p.add_argument("--lpf-checkpoint", help="LivePoseFormer 59-keypoint checkpoint ('none' = random weights, test only)")
    p.add_argument("--no-fill", dest="fill", action="store_false", help="do not fill dropped frames with LivePoseFormer")
    p.add_argument("--angle-smoothing", default="light", choices=["off", "light", "standard"],
                   help="One Euro smoothing of the joint angles (light: ~17 ms lag; LivePoseFormer already smooths the 3D)")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--start", type=int, default=0, help="first frame of each recording (also the static trial)")
    p.add_argument("--max-seconds", type=float, default=None, help="evaluate at most this many seconds per recording")
    p.add_argument("--cams", default="C1,C2,C3")
    p.add_argument("--prefetch", type=int, default=120)
    p.add_argument("--reader-threads", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", help="results folder (default realtime_eval/results/<date>)")
    p.add_argument("--force", action="store_true", help="redo runs that are already finished")
    p.add_argument("--report-only", action="store_true", help="only (re)build the report from finished runs")
    args = p.parse_args()

    if args.report_only:
        if not args.out:
            p.error("--report-only needs --out <results folder>")
        import report
        report.build(Path(args.out))
        return
    if not args.data_root:
        p.error("--data-root is required")
    available = discover(args.data_root, args.cams.split(","))
    if args.list:
        print(f"{len(available)} recordings in {args.data_root}:")
        total = 0
        for t in available:
            n = len(json.load(open(Path(args.data_root) / "camera_data" / "C1" / f"{t}.json")))
            total += n
            print(f"  {t:<10} {n:>6} frames  {n / args.fps:6.1f} s")
        print(f"total {total / args.fps / 60:.1f} min of video; --all with {len(args.models)} models takes about "
              f"{fmt_time(len(args.models) * (total / args.fps + 45 * len(available)))} (stream + ~45 s set-up per run)")
        return
    trials = available if args.all else (list(args.recordings) or pick_spread(available, args.auto))
    missing = [t for t in trials if t not in available]
    if missing:
        p.error(f"not found (or incomplete) in {args.data_root}: {', '.join(missing)}")
    if not trials:
        p.error("choose recordings with --all, --recordings or --auto N (see --list)")
    args.lpf_repo = find_lpf_repo(args.lpf_repo, required=True)
    if not args.lpf_checkpoint:
        p.error("--lpf-checkpoint is required (use 'none' only for a test run with random weights)")
    ckpt = None if args.lpf_checkpoint.lower() == "none" else args.lpf_checkpoint

    out = Path(args.out) if args.out else HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    info = system_info()
    info["cpu_count"] = __import__("os").cpu_count()
    info["machine"] = platform.node()
    log_file = open(out / "eval.log", "a")

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print("\n" + line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    plan = []
    for m in args.models:
        for t in trials:
            n = len(json.load(open(Path(args.data_root) / "camera_data" / "C1" / f"{t}.json"))) - args.start
            if args.max_seconds:
                n = min(n, int(round(args.max_seconds * args.fps)))
            plan.append((m, t, n, (out / m / t / "summary.json").exists() and not args.force))
    (out / "plan.json").write_text(json.dumps({"models": args.models, "recordings": trials, "args": vars(args),
                                               "system": info}, indent=1, default=str))
    todo = [x for x in plan if not x[3]]
    stream_s = sum(n for _, _, n, d in plan if not d) / args.fps
    log(f"{len(plan)} runs ({len(plan) - len(todo)} already done), {len(todo)} to do: "
        f"{fmt_time(stream_s)} of 60 fps stream + set-up. Results in {out}")
    log(f"GPU {info.get('gpu', 'none')} | torch {info['torch']} | models {args.models} | {args.precision}"
        f"{' + compile' if args.compile else ''} | detector {args.detector} | LivePoseFormer {Path(ckpt).name if ckpt else 'random'}"
        f" | fill {args.fill} | angle smoothing {args.angle_smoothing}")

    prog = Progress(plan, args.fps, out)
    from stream_sim.detector import PersonDetector
    from stream_sim.lpf_stage import LpfStage
    lpf = LpfStage(args.lpf_repo, ckpt, args.device)
    bio = BiomechStage(args.lpf_repo, smoothing=args.angle_smoothing)
    bio.smoothing_name = args.angle_smoothing
    for m in args.models:
        runs = [x for x in plan if x[0] == m and not x[3]]
        if not runs:
            continue
        prog.i, prog.frames = plan.index(runs[0]), 0
        prog.show(f"loading {MODELS[m][1]}", force=True)
        pipe = build_pipeline(m, FullDataRecording(args.data_root, runs[0][1], args.cams.split(",")).subject, args, log)
        detector = None if args.detector == "json" else PersonDetector(str(ROOT / f"{args.detector}.pt"), args.device,
                                                                         precision=args.precision)
        for x in runs:
            prog.i, prog.frames = plan.index(x), 0
            try:
                oh = run_one(m, x[1], pipe, detector, lpf, bio, args, prog, out / m / x[1], info)
                prog.overheads.append(oh)
                s = json.loads((out / m / x[1] / "summary.json").read_text())["summary"]
                log(f"done {MODELS[m][1]} {x[1]}: pose {s['output_rate_fps']:.1f} fps, dropped {s['dropped_pct']:.0f}%, "
                    f"latency to angles {s.get('biomech', {}).get('delay_ms_mean', float('nan')):.0f} ms, "
                    f"MPJPE {s.get('mpjpe_world_mm', float('nan')):.1f} mm, angles "
                    f"{s.get('biomech', {}).get('mae_deg', {}).get('all output', {}).get('mean', float('nan')):.2f} deg")
            except KeyboardInterrupt:
                log("interrupted; finished runs are kept, run the same command again to continue")
                return
            except Exception as e:
                log(f"FAILED {MODELS[m][1]} {x[1]}: {type(e).__name__}: {e}")
            plan[plan.index(x)] = (x[0], x[1], x[2], True)
            prog.plan = plan
            prog.done_count += 1
        del pipe, detector
        torch.cuda.empty_cache()
    log(f"all runs finished in {fmt_time(time.time() - prog.t_start)}; building the report")
    import report
    report.build(out)
    log(f"report: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
