"""Run the live pipeline on recorded frames, with visual checks and a timing report.

    python run_pipeline.py                                 # HRNet, frame 300, all plots
    python run_pipeline.py --model vitpose --frames 300
    python run_pipeline.py --frames 0:1205:5 --viz-every 50   # many frames, plots every 50th
    python run_pipeline.py --frames 0:200 --no-viz            # speed run

Frame indices are positions in the camera JSON files (the same instant in all three cameras).
Outputs go to outputs/<model>_<frames>/:
  step1_inputs ... step6_world   one PNG per pipeline step (per visualised frame)
  summary.png                    errors over frames and time per step
  timing.txt / timing.csv        timing report: setup, every per-frame step with its share of the
                                 frame time, stage subtotals, plots and background work
  timing_breakdown.png           chart of the per-step and per-stage times (always written)
  predictions.npz                2D keypoints, 3D camera-frame and world-frame keypoints
  metrics.csv                    per-frame errors against the Vicon ground truth
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pipeline import FRAME_STEPS, STAGES, LivePipeline, Recording
from pipeline import viz

MODELS = {"hrnet": "w48_768x576_all_63", "vitpose": "ViT_B_simple_all_63"}


def parse_frames(spec, n):
    if ":" not in spec:
        return [int(spec)]
    parts = [int(p) if p else None for p in spec.split(":")]
    return list(range(n))[slice(*parts)]


def visualise(rec, pipe, fs, res, out_dir, units):
    """Save the six step plots; each is timed separately (outside the pipeline's frame time)."""
    d, cams, s, t = res.debug, rec.cams, pipe.units_to_mm, pipe.timer
    with t.step("viz: step 1 inputs"):
        viz.plot_inputs(d["images"], fs.bboxes, d["boxes"], cams, out_dir / "step1_inputs.png")
    with t.step("viz: step 2 crops + heatmaps"):
        viz.plot_crops_heatmaps(d["crops"], d["heatmaps"], d["kp_crop"], cams, out_dir / "step2_crops_heatmaps.png")
    with t.step("viz: step 3 2D pose"):
        viz.plot_pose_2d(d["images"], res.kp2d, fs.gt_xy, res.conf, cams, out_dir / "step3_pose2d.png", d["boxes"])
    with t.step("viz: step 4 triangulation"):
        viz.plot_triangulation(d["images"], res.xyz_cam * s, fs.gt_xyz_cam * s, d["reproj"], res.kp2d, cams,
                               out_dir / "step4_triangulation.png", units, res.reproj_px)
    with t.step("viz: step 5 wand"):
        wand_json = np.asarray(rec.frames[0][fs.index]["wand_xyz"], float) * s
        viz.plot_wand(d["images"], fs.wand_xy, pipe.wand_reproj, pipe.wand_cam * s, cams,
                      out_dir / "step5_wand.png", wand_json, units)
    with t.step("viz: step 6 world frame"):
        wand_wcs_pred = pipe.world.apply(pipe.wand_cam)[[1, 0, 2, 3, 4]]
        mpjpe = np.nanmean(np.linalg.norm(res.xyz_wcs - fs.gt_xyz_wcs, axis=-1))
        viz.plot_world(res.xyz_wcs, fs.gt_xyz_wcs, wand_wcs_pred, rec.wand_wcs(), out_dir / "step6_world.png", mpjpe)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="hrnet", choices=list(MODELS), help="hrnet (w48, 768x576) or vitpose (B, 256x192)")
    p.add_argument("--checkpoint", default="best_nmae_model.pth")
    p.add_argument("--trial", default="S1A3D1")
    p.add_argument("--cams", default="C1,C2,C3", help="main camera first")
    p.add_argument("--frames", default="300", help="one index, or start:stop[:step]")
    p.add_argument("--viz-every", type=int, default=1, help="save the step plots every N processed frames")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--no-prefetch", action="store_true", help="read images in the main loop instead of in the background")
    p.add_argument("--device", default="cuda")
    p.add_argument("--units-to-mm", type=float, default=60.0, help="calibration units -> mm (SUMediPose: 60)")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    rec = Recording(root / "data", args.trial, args.cams.split(","))
    frames = parse_frames(args.frames, len(rec))
    out = root / args.out / f"{args.model}_{args.frames.replace(':', '-')}"
    out.mkdir(parents=True, exist_ok=True)

    pipe = LivePipeline(root, MODELS[args.model], rec.subject, args.checkpoint, args.device, args.units_to_mm)
    print(f"model {pipe.pose.name} on {pipe.pose.device}, "
          f"cameras {rec.cams}, triangulating with views {[rec.cams[i] for i in pipe.rig.use_cams]}")
    pipe.setup_world(rec[frames[0]].wand_xy, rec.wand_wcs())
    print(f"world transform: det R = {np.linalg.det(pipe.world.R):+.0f}, wand fit residual "
          f"{pipe.world.residual_mm.mean():.1f} mm (per marker {np.round(pipe.world.residual_mm, 1)})")

    s, rows, store = args.units_to_mm, [], {"index": [], "kp2d": [], "conf": [], "xyz_cam_mm": [], "xyz_wcs_mm": []}
    prefetch = not args.no_prefetch
    pending = pipe.prefetch(rec[frames[0]].image_paths) if prefetch else None
    for n, i in enumerate(tqdm(frames, desc="frames")):
        fs = rec[i]
        do_viz = not args.no_viz and n % args.viz_every == 0
        images = pending if prefetch else fs.image_paths
        if prefetch and n + 1 < len(frames):          # start decoding the next frame set now
            pending = pipe.prefetch(rec[frames[n + 1]].image_paths)
        res = pipe.process(images, fs.bboxes, keep_debug=do_viz)
        row = {
            "index": i,
            "err2d": np.nanmean(np.linalg.norm(res.kp2d - fs.gt_xy, axis=-1), axis=1),
            "reproj": res.reproj_px,
            "mpjpe_cam": np.nanmean(np.linalg.norm((res.xyz_cam - fs.gt_xyz_cam) * s, axis=-1)),
            "mpjpe_wcs": np.nanmean(np.linalg.norm(res.xyz_wcs - fs.gt_xyz_wcs, axis=-1)),
        }
        rows.append(row)
        for k, v in (("index", i), ("kp2d", res.kp2d), ("conf", res.conf), ("xyz_cam_mm", res.xyz_cam * s),
                     ("xyz_wcs_mm", res.xyz_wcs)):
            store[k].append(v)
        if do_viz:
            visualise(rec, pipe, fs, res, out / f"frame_{i:04d}", "mm")

    np.savez(out / "predictions.npz", **{k: np.array(v) for k, v in store.items()})
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index"] + [f"err2d_{c}_px" for c in rec.cams] + [f"reproj_{c}_px" for c in rec.cams]
                   + ["mpjpe_cam_mm", "mpjpe_wcs_mm"])
        for r in rows:
            w.writerow([r["index"], *np.round(r["err2d"], 2), *np.round(r["reproj"], 2),
                        round(r["mpjpe_cam"], 2), round(r["mpjpe_wcs"], 2)])

    report = pipe.timer.report(f"Timing report - {pipe.pose.name}, {len(frames)} frame set(s), device "
                               f"{pipe.pose.device}{', prefetch on' if prefetch else ''}", FRAME_STEPS, STAGES)
    accuracy = (f"\nAccuracy vs Vicon GT over {len(rows)} frame set(s):\n"
                f"  2D error per view (px): {np.round(np.mean([r['err2d'] for r in rows], 0), 2)}\n"
                f"  reprojection (px):      {np.round(np.mean([r['reproj'] for r in rows], 0), 2)}\n"
                f"  MPJPE camera frame:     {np.mean([r['mpjpe_cam'] for r in rows]):.1f} mm\n"
                f"  MPJPE world frame:      {np.mean([r['mpjpe_wcs'] for r in rows]):.1f} mm")
    (out / "timing.txt").write_text(report + "\n" + accuracy + "\n")
    timing_rows = pipe.timer.rows(FRAME_STEPS, STAGES)
    with open(out / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing_rows[0]))
        w.writeheader()
        w.writerows(timing_rows)
    viz.plot_timing(timing_rows, STAGES, out / "timing_breakdown.png",
                    f"Timing breakdown - {pipe.pose.name}, {len(frames)} frame set(s), {pipe.pose.device}")
    if not args.no_viz:
        viz.plot_summary(rows, timing_rows, out / "summary.png")
    print("\n" + report + "\n" + accuracy + f"\n\nOutputs in {out}")


if __name__ == "__main__":
    main()
