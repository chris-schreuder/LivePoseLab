"""Thigh and shank rotation from anatomical landmarks instead of the wand markers, on finished runs.

Plug-in Gait sets the rotation of the thigh about its long axis with the THI wand marker and that of the
shank with TIB (the frames lie in the plane of hip centre - knee centre - THI and knee centre - ankle
centre - TIB). Wand markers sit wherever they were stuck on, so the markerless model has to guess a
subject-specific position, which shows up as a constant, per-subject rotation offset. PiG's intent is that
THI lies in the plane of the hip centre, knee centre and knee flexion axis, and TIB in the plane of the
knee centre, ankle centre and ankle axis. This script replaces them by virtual markers on exactly those
axes, from landmarks the 2D model sees:

    THI* = KJC + 100 mm x unit(KNE - MKNE)      (lateral along the epicondyle axis)
    TIB* = AJC + 100 mm x unit(ANK - MMED)      (lateral along the malleolus axis)

applied the same way to the live 3D and to the Vicon markers (fair comparison), in the static trial and in
every frame. Everything else in the model is unchanged.

For every run it replays the real-time biomechanics stage on the exact 3D stream it received live
(``out_xyz``, frame gaps, continuity, smoothing; see stages.replay_angles) and recomputes the Vicon
reference angles, with both definitions and with the static trial from the pipeline (as live) or from
Vicon. Checks that the original definition gives back the stored live and Vicon angles. Writes
``<run>/angles_definitions.npz`` and, in the results folder, ``THIGH_SHANK_DEFINITION.md``,
``thigh_shank_definition.csv`` (per channel) and ``figures/transverse/5_offsets_by_definition.png``.

    python realtime_eval/thigh_shank_definition.py --data-root data --out realtime_eval/results/full_eval \\
        --lpf-repo ../LivePoseFormer

Only ``<data_root>/wcs_data/<trial>.json`` is needed (no images or camera files).
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from stream_sim.lpf_repo import HELP, find_lpf_repo  # noqa: E402

DEFS = {"pig": "wand markers (THI, TIB)", "anat": "epicondyles + malleoli"}
STAT = {"pipeline": "pipeline (as live)", "vicon": "Vicon"}
SIDES = {"R": ("RKNE", "MKNER", "RANK", "RMMED", "RKJC", "RAJC", "RTHI", "RTIB"),
         "L": ("LKNE", "MKNEL", "LANK", "LMMED", "LKJC", "LAJC", "LTHI", "LTIB")}


def anatomical(xyz, idx):
    """(..., 59, 3) -> copy with THI / TIB replaced by virtual markers on the knee and ankle axes."""
    x = np.array(xyz, float, copy=True)
    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)   # noqa: E731
    for kne, mkne, ank, mmed, kjc, ajc, thi, tib in SIDES.values():
        x[..., idx[thi], :] = x[..., idx[kjc], :] + 100.0 * unit(x[..., idx[kne], :] - x[..., idx[mkne], :])
        x[..., idx[tib], :] = x[..., idx[ajc], :] + 100.0 * unit(x[..., idx[ank], :] - x[..., idx[mmed], :])
    return x


def process_trial(job):
    """All runs (models) of one recording."""
    trial, run_dirs, data_root, lpf_repo = job
    from pipeline.skeleton import SOURCE_TO_59
    from stages import replay_angles
    from stream_sim.biomech_stage import BiomechStage
    wcs = json.load(open(Path(data_root) / "wcs_data" / f"{trial}.json"))
    gt_all = np.array([np.asarray(w["xyz"], float)[SOURCE_TO_59] for w in wcs])
    logs, bios = [], {}
    for d in run_dirs:
        d = Path(d)
        s = json.loads((d / "summary.json").read_text())
        fr = dict(np.load(d / "frames.npz"))
        sm = s["config"]["angle_smoothing"]
        if sm not in bios:
            bios[sm] = BiomechStage(lpf_repo, smoothing=sm)
        bio = bios[sm]
        idx = fr["out_index"].astype(int)
        steps = [1] + [max(1, b - a) for a, b in zip(idx, idx[1:])]
        start = int(s.get("start_index", 0))
        static_pipe = fr["proc_xyz_raw"][0]            # the pipeline's 3D of the first frame (the live static trial)
        if int(fr["proc_index"][0]) != start:
            logs.append(f"{d}: first processed frame {int(fr['proc_index'][0])} != static frame {start}; skipped")
            continue
        res = {}
        for dn in DEFS:
            tf = (lambda a: anatomical(a, bio.idx)) if dn == "anat" else (lambda a: np.asarray(a, float))
            gt_static = tf(gt_all[start])
            res[f"gt_{dn}"] = bio.angles_batch(tf(gt_all[idx]), static=bio.static_reference(gt_static), steps=steps)
            stream = tf(fr["out_xyz"])
            res[f"live_{dn}_pipeline"], _ = replay_angles(bio, stream, idx, tf(static_pipe))
            res[f"live_{dn}_vicon"], _ = replay_angles(bio, stream, idx, gt_static)
        w = lambda a: np.abs((a + 180) % 360 - 180)   # noqa: E731
        chk_gt = np.nanpercentile(w(res["gt_pig"] - fr["out_angles_gt"]), 99)
        dl = w(res["live_pig_pipeline"] - fr["out_angles"])
        np.savez_compressed(d / "angles_definitions.npz", **{k: v.astype(np.float32) for k, v in res.items()})
        logs.append(f"{s['model_name']:8s} {trial:9s} check vs stored angles: Vicon p99 {chk_gt:.3f} deg, "
                    f"live median {np.nanmedian(dl):.3f} / mean {np.nanmean(dl):.3f} / p99 {np.nanpercentile(dl, 99):.3f} deg")
    return logs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lpf-repo", help=HELP + "; joint-angle model")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.lpf_repo = find_lpf_repo(args.lpf_repo, required=True)
    out = Path(args.out)

    trials = {}
    for f in sorted(out.glob("*/*/summary.json")):
        if args.force or not (f.parent / "angles_definitions.npz").exists():
            trials.setdefault(f.parent.name, []).append(str(f.parent))
    jobs = [(t, ds, args.data_root, args.lpf_repo) for t, ds in trials.items()]
    print(f"{sum(len(j[1]) for j in jobs)} runs to process")
    with ProcessPoolExecutor(args.workers) as ex:
        for k, logs in enumerate(ex.map(process_trial, jobs), 1):
            for line in logs:
                print(f"[{k}/{len(jobs)}] {line}", flush=True)
    summarise(out)


# ---------------------------------------------------------------------- results
def summarise(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from report import JOINT_LABEL, MIN_ROM_DEG, PLANES, ch_mean, channel_rows, r_mean, rom_median

    rows = []
    for f in sorted(out.glob("*/*/summary.json")):
        npz = f.parent / "angles_definitions.npz"
        if not npz.exists():
            continue
        s = json.loads(f.read_text())
        a = np.load(npz)
        for dn in DEFS:
            for st in STAT:
                r = dict(s, _frames={"out_angles": a[f"live_{dn}_{st}"], "out_angles_gt": a[f"gt_{dn}"]})
                for x in channel_rows(r):
                    x.update(static=st, definition=dn)
                    rows.append(x)
    if not rows:
        print(f"no runs with angles_definitions.npz in {out}")
        return
    models = list(dict.fromkeys(x["model"] for x in rows))
    with open(out / "thigh_shank_definition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in x.items()} for x in rows])

    f = lambda v, d=1: "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"   # noqa: E731
    L = ["# Thigh and shank rotation: wand markers vs anatomical landmarks\n",
         "Plug-in Gait sets the long-axis rotation of the thigh with the THI wand marker and of the shank with TIB. "
         "Here both are replaced, in the live 3D and in the Vicon markers alike, by virtual markers on the knee "
         "flexion axis (lateral along KNE − MKNE from the knee centre) and on the ankle axis (lateral along "
         "ANK − MMED from the ankle centre), which is where PiG intends THI and TIB to lie. Everything else is "
         "unchanged. Live angles are the real-time biomechanics stage replayed on the exact 3D stream of each run "
         "(same continuity and smoothing); the original definition reproduces the stored live and Vicon angles. "
         f"MAE / aligned MAE / |offset| in degrees, mean over channels and recordings (sides pooled); r: Pearson, "
         f"Fisher-z mean over channels with a Vicon ROM ≥ {MIN_ROM_DEG:g}°. Aligned = after removing the mean offset.\n"]
    affected = ["hip", "knee", "ankle", "ankle_abs", "foot"]
    for st in STAT:
        L.append(f"\n## Static trial: {STAT[st]}\n")
        for m in models:
            L.append(f"\n### {m}\n")
            L.append("| Joint | Plane | MAE wand → anat. | aligned MAE wand → anat. | mean abs. offset wand → anat. | r wand → anat. |")
            L.append("|---|---|---:|---:|---:|---:|")
            for j in affected:
                for p, plab in PLANES:
                    sel = dict(model=m, joint=j, plane=p, static=st)
                    a_, b_ = [x for x in rows if x["definition"] == "pig"], [x for x in rows if x["definition"] == "anat"]
                    L.append(f"| {JOINT_LABEL[j]} | {plab.split(' ')[0]} | "
                             f"{f(ch_mean(a_, 'mae_deg', **sel))} → **{f(ch_mean(b_, 'mae_deg', **sel))}** | "
                             f"{f(ch_mean(a_, 'mae_aligned_deg', **sel))} → {f(ch_mean(b_, 'mae_aligned_deg', **sel))} | "
                             f"{f(ch_mean(a_, 'abs_offset_deg', **sel))} → {f(ch_mean(b_, 'abs_offset_deg', **sel))} | "
                             f"{f(r_mean(a_, **sel), 3)} → {f(r_mean(b_, **sel), 3)} |")
            for lab, sel in (("All lower-limb channels above", dict(model=m, static=st)),
                             ("All channels", dict(model=m, static=st))):
                a_ = [x for x in rows if x["definition"] == "pig" and (lab == "All channels" or x["joint"] in affected)]
                b_ = [x for x in rows if x["definition"] == "anat" and (lab == "All channels" or x["joint"] in affected)]
                L.append(f"| **{lab}** | | {f(ch_mean(a_, 'mae_deg', **sel))} → **{f(ch_mean(b_, 'mae_deg', **sel))}** | "
                         f"{f(ch_mean(a_, 'mae_aligned_deg', **sel))} → {f(ch_mean(b_, 'mae_aligned_deg', **sel))} | "
                         f"{f(ch_mean(a_, 'abs_offset_deg', **sel))} → {f(ch_mean(b_, 'abs_offset_deg', **sel))} | "
                         f"{f(r_mean(a_, **sel), 3)} → {f(r_mean(b_, **sel), 3)} |")

    # per subject: is the offset still subject-specific?
    L.append("\n## Rotation offset per subject (live − Vicon, mean over the 8 actions, deg; static as live)\n")
    subjects = sorted(dict.fromkeys(x["subject"] for x in rows), key=lambda v: (len(v), v))
    for m in models:
        L.append(f"\n### {m}\n")
        L.append("| Channel | " + " | ".join(f"{s_} wand → anat." for s_ in subjects) + " |\n|---|" + "---:|" * len(subjects))
        for c in ("hip_rotation_r", "hip_rotation_l", "knee_rotation_r", "knee_rotation_l", "ankle_rotation_r", "ankle_rotation_l"):
            cells = []
            for s_ in subjects:
                v = [np.mean([x["offset_deg"] for x in rows if x["model"] == m and x["channel"] == c and x["subject"] == s_
                              and x["static"] == "pipeline" and x["definition"] == dn]) for dn in DEFS]
                cells.append(f"{v[0]:+.0f} → {v[1]:+.0f}")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
    (out / "THIGH_SHANK_DEFINITION.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # figure: offsets per recording, both definitions
    cols = {"S7": "#2563eb", "S8": "#16a34a", "S23": "#ea580c", "S25": "#9333ea"}
    chans = ["hip_rotation_r", "hip_rotation_l", "knee_rotation_r", "knee_rotation_l", "ankle_rotation_r", "ankle_rotation_l"]
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.6 * len(models)), sharey=True, squeeze=False)
    for i, m in enumerate(models):
        for jx, dn in enumerate(DEFS):
            ax = axes[i, jx]
            for x0, c in enumerate(chans):
                for x in rows:
                    if x["model"] == m and x["channel"] == c and x["static"] == "pipeline" and x["definition"] == dn:
                        ax.scatter(x0 + np.random.uniform(-0.18, 0.18), x["offset_deg"], s=12, color=cols.get(x["subject"], "k"), alpha=0.8)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(len(chans)), [c.replace("_rotation_", " rot ").upper() for c in chans], fontsize=7)
            mae = ch_mean([x for x in rows if x["definition"] == dn], "mae_deg", model=m, plane="rot", static="pipeline")
            ax.set_title(f"{m}, {DEFS[dn]}: rotation offset per recording", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        axes[i, 0].set_ylabel("offset (deg)")
    for s_, col in cols.items():
        axes[0, 1].scatter([], [], color=col, label=s_)
    axes[0, 1].legend(fontsize=7, title="subject", title_fontsize=7)
    fig.tight_layout()
    (out / "figures" / "transverse").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "transverse" / "5_offsets_by_definition.png", dpi=150)
    print(f"written: {out / 'THIGH_SHANK_DEFINITION.md'}")


if __name__ == "__main__":
    main()
