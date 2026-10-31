"""Add the joint angles with a Vicon static trial to finished runs, without running them again.

Live, the static trial (lab alignment, ankle offsets) is the pipeline's own 3D of the first frame. This
script replays the real-time biomechanics stage on the exact 3D stream it received in each run (``out_xyz``
and the frame gaps in ``frames.npz``; same continuity, glitch hold and smoothing), calibrated with the
Vicon markers of the first frame instead, and stores the result in the run's ``frames.npz``:

    out_angles_gtstatic, out_angles_gtstatic_unsmoothed, static_xyz_vicon

New runs of run_eval.py store these themselves. Afterwards rebuild the report:

    python realtime_eval/add_gt_static.py --data-root /path/to/data --out realtime_eval/results/full_eval \\
        --lpf-repo ../LivePoseFormer
    python realtime_eval/run_eval.py --out realtime_eval/results/full_eval --report-only

Check: the same replay with the pipeline's first processed frame as the static trial should give back the
live angles (it prints the difference; small differences come from float32 storage and bf16).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from dataset import FullDataRecording  # noqa: E402
from stages import replay_angles  # noqa: E402
from stream_sim.biomech_stage import BiomechStage  # noqa: E402
from stream_sim.lpf_repo import HELP, find_lpf_repo  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True, help="evaluation results folder")
    p.add_argument("--lpf-repo", help=HELP + "; joint-angle model")
    p.add_argument("--force", action="store_true", help="redo runs that already have the Vicon-static angles")
    args = p.parse_args()
    args.lpf_repo = find_lpf_repo(args.lpf_repo, required=True)

    runs = sorted(Path(args.out).glob("*/*/summary.json"))
    bios = {}
    for k, f in enumerate(runs, 1):
        npz = f.parent / "frames.npz"
        if not npz.exists():
            continue
        fr = dict(np.load(npz))
        name = f"{f.parent.parent.name}/{f.parent.name}"
        if "out_angles_gtstatic" in fr and not args.force:
            print(f"[{k}/{len(runs)}] {name}: done already")
            continue
        s = json.loads(f.read_text())
        smoothing = s["config"]["angle_smoothing"]
        if smoothing not in bios:
            bios[smoothing] = BiomechStage(args.lpf_repo, smoothing=smoothing)
        bio = bios[smoothing]
        rec = FullDataRecording(args.data_root, s["trial"], s["config"]["cams"].split(","))
        first = rec[s.get("start_index", 0)]
        if len(fr["out_index"]) and int(fr["out_index"].min()) != first.index:
            print(f"  warning: first output frame {int(fr['out_index'].min())} is not the static frame {first.index}")
        fr["static_xyz_vicon"] = np.asarray(first.gt_xyz_wcs, np.float32)
        fr["out_angles_gtstatic"], fr["out_angles_gtstatic_unsmoothed"] = replay_angles(
            bio, fr["out_xyz"], fr["out_index"], first.gt_xyz_wcs)

        # check: the pipeline's static trial (its 3D of the first frame) must give back the live angles
        check = ""
        if len(fr["proc_index"]) and int(fr["proc_index"][0]) == first.index:
            again, _ = replay_angles(bio, fr["out_xyz"], fr["out_index"], fr["proc_xyz_raw"][0])
            d = np.abs((again - fr["out_angles"] + 180) % 360 - 180)
            check = f", replay check vs live angles: median {np.nanmedian(d):.3f}, p99 {np.nanpercentile(d, 99):.2f} deg"
        np.savez_compressed(npz, **fr)
        gt = fr["out_angles_gt"]
        mae = lambda a: float(np.nanmean(np.abs((a - gt + 180) % 360 - 180)))   # noqa: E731
        print(f"[{k}/{len(runs)}] {name}: MAE all channels {mae(fr['out_angles']):.2f} (pipeline static) -> "
              f"{mae(fr['out_angles_gtstatic']):.2f} deg (Vicon static){check}")


if __name__ == "__main__":
    main()
