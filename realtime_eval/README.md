# 4. Evaluation: the live pipeline over a whole dataset

Runs the **full live pipeline** on every recording of the dataset, as simulated synchronised 60 fps
feeds from three cameras, and writes a paper-ready report:

```
YOLO26 person detector → 2D pose (ViTPose-B or HRNet-W48, bf16 + torch.compile) → DLT triangulation
→ world frame → LivePoseFormer (fills dropped frames) → real-time joint angles (Plug-in Gait,
continuity, One Euro smoothing)
```

It measures everything the [stream simulator](../stream_sim/README.md) reports (time per stage,
latency per frame, frame rates, dropped and filled frames) plus the error at every stage against
Vicon, and writes Markdown, LaTeX tables, figures and CSVs. It uses the same engine as the stream
simulator and the live demo, so it evaluates exactly the pipeline that runs live.

## Quick start

From the repository folder (installation in the [main README](../README.md#installation)), with the
dataset in the [layout below](#data-layout):

```bash
# 1. which recordings are there, and how long will it take?
python realtime_eval/run_eval.py --data-root /path/to/dataset --list

# 2. evaluate every recording with both models
python realtime_eval/run_eval.py --data-root /path/to/dataset --all \
    --lpf-repo ../LivePoseFormer --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59_stress.pth \
    --out realtime_eval/results/full_eval

# 3. read the report
#    realtime_eval/results/full_eval/REPORT.md
```

A quicker check: `--auto 6` picks 6 recordings spread over subjects and actions, or
`--recordings S7A1D1 S8A3D1` names them.

**Progress.** One line shows the run, the recording, frames streamed, the phase, elapsed time, time
left and the expected finish time:

```
[run 3/12] HRNet    S7A1D1    [#########...........]  1650/3600 fr  streaming 60 fps       elapsed 0:21:40  left ~1:02:10  (done ~15:42)
```

The same is written to `<out>/progress.json` (to check from another terminal), and every finished run
is logged to `<out>/eval.log`.

**Resume.** Finished runs are kept. If the evaluation stops (Ctrl-C, crash, reboot), run the same
command with the same `--out` and it continues with the remaining runs. `--force` redoes them.

**Duration.** Each run takes the recording's length in real time (it is a 60 fps simulation) plus
about 30–60 s for set-up, static trial and analysis; the first run of each model also compiles the
network. The full dataset used here (32 recordings × 2 models, 13 min of video per model) took 49 min
on an RTX 3090.

## Data layout

```
<data_root>/frames/S7/C1/C1S7A1D1/C1S7A1D1_0307.jpg
<data_root>/camera_data/C1/S7A1D1.json
<data_root>/wcs_data/S7A1D1.json
```

A recording such as `S7A1D1` (subject 7, action 1, take 1) is used when all three cameras have a JSON
file and a frames folder, and the world-frame file exists. Calibration and models come from the
repository (`calibration/`, `ViT_B_simple_all_63/`, `w48_768x576_all_63/`); the calibration session
is picked per subject.

## Options

Defaults are the full pipeline: both models, bf16, compile on, YOLO26s, LivePoseFormer filling
dropped frames, light angle smoothing, 60 fps, policy *latest*.

| Option | Default | Meaning |
|---|---|---|
| `--data-root` | required | dataset folder |
| `--all` / `--recordings` / `--auto N` | – | every recording found, the ones named, or N spread over subjects and actions |
| `--list` | – | list the recordings and the estimated time, then stop |
| `--models` | `vitpose hrnet` | pose models |
| `--precision` | `bf16` | `fp32`, `fp16` or `bf16` |
| `--no-compile` | compile on | disable `torch.compile` |
| `--detector` | `yolo26s` | `yolo26n` / `yolo26s` / `yolo26m`, or `json` for the dataset's boxes |
| `--lpf-repo`, `--lpf-checkpoint` | required | LivePoseFormer repository and checkpoint (`none` = random weights, only to test the script) |
| `--no-fill` | fill on | do not fill dropped frames with LivePoseFormer |
| `--angle-smoothing` | `light` | One Euro preset: `off`, `light` (~17 ms lag) or `standard` (~33 ms) |
| `--start` | 0 | first frame of each recording; also the static trial (choose a frame where the subject stands still) |
| `--max-seconds` | whole recording | evaluate at most this long per recording |
| `--reader-threads` | 4 | threads that decode frames ahead of the clock (raise if the report shows reader underruns) |
| `--out` | `realtime_eval/results/<date-time>` | results folder |
| `--force` | off | redo finished runs |
| `--report-only` | – | rebuild the report from finished runs (with `--out`) |
| `--fps`, `--cams`, `--prefetch`, `--device` | 60, `C1,C2,C3`, 120, `cuda` | camera rate, cameras, reader buffer, device |

## What is measured

**Real-time performance:** time per frame of every stage (detector, crop/resize, pose network,
triangulation, LivePoseFormer, biomechanics; mean and p95); pose network rate; dropped and filled
frames; output rate of the joint angles; latency from capture to the 3D, to the smoothed 3D and to the
joint angles (mean and percentiles); delay trend; whether LivePoseFormer and the biomechanics keep up;
camera clock lateness; disk reader underruns.

**Error at every stage, against Vicon:**

| Stage | Output | Metrics |
|---|---|---|
| Person detector | box per view | IoU with the dataset's box, misses, confidence |
| 2D pose | 59 keypoints × 3 views | error in px and in % of person height, PCK@5/10/20 px, per camera |
| Triangulation | 3D markers (processed frames) | MPJPE (mean, median, p95), per body region |
| LivePoseFormer | 3D at 60 fps | MPJPE on processed, filled and all output frames; 3D jitter before / after |
| Biomechanics | 72 joint-angle channels | see below |

**Joint angles,** each per plane, per joint, per action and per subject:

* MAE for the angles from the raw 3D, after LivePoseFormer and final (smoothed); measured vs filled
  frames; angle jitter; continuity corrections;
* error as % of the range of motion, and range-of-motion error (channels with a Vicon ROM under 10°
  left out);
* tracking after offset alignment (the live mean shifted to the Vicon mean per channel and recording)
  and Pearson r;
* all of these again with the static trial taken from Vicon instead of the pipeline's own 3D.

The Vicon reference angles are the same model applied to the Vicon markers (with its own static trial
from the same frame), so the errors measure the markerless pipeline, not the model.

## Outputs

Per run, `<out>/<model>/<recording>/`:

| File | Content |
|---|---|
| `summary.json` | every number: configuration, system, the simulator's full summary and the stage errors |
| `report.txt` | the simulator's text report for this run |
| `timing.png`, `lpf_accuracy.png`, `joint_angles.png` | delay, timeline and stage times; error before / after LivePoseFormer; joint angles vs Vicon |
| `frames.npz` | per-frame arrays: latencies, errors, 3D (raw and output), joint angles (final, unsmoothed, Vicon, with a Vicon static trial), filled flags, both static frames |

For the whole evaluation, `<out>/`:

| File | Content |
|---|---|
| `REPORT.md` | the report (sections below) |
| `tables.tex` | booktabs tables: real-time performance, error along the pipeline, joint-angle errors |
| `results_per_recording.csv` | every metric per model × recording |
| `results_summary.csv` | per model: mean, SD, min, max across recordings |
| `angles_subject_action_joint_plane.csv` | every model × static trial × subject × action × joint × plane: MAE, RMSE, aligned errors, offset, % ROM, Pearson r |
| `angles_per_channel.csv` | the same per channel and side |
| `pooled.json` | frame-pooled values (exact latency percentiles) |
| `figures/` | latency CDF, time per stage, 3D error by stage, joint-angle error by stage, per-recording error, knee-flexion time series (PNG and PDF) |

`REPORT.md` sections:

1. **Setup:** pipeline, simulation, hardware and software, configuration, recordings.
2. **Real-time performance:** rates, drops and fills, latency (mean, pooled p50 / p95 / p99, % of
   frames within 100 / 150 / 200 ms), time per stage.
3. **Accuracy along the pipeline:** detector → 2D → 3D → LivePoseFormer → joint angles; 3D error by
   body region; 2D error per camera.
4. **Joint angles:** per plane and stage, measured vs filled, per joint, % of ROM, offset alignment
   and Pearson r, Vicon static trial, continuity.
5. **Per-recording results.**
6. **Summary for real-time feedback.**
7. **Figures.**

Statistics: "mean ± SD" is across recordings (one value per recording); pooled values weight every
frame equally; latency percentiles are exact, from all frames of all recordings.

Everything is recomputed from the saved runs, so the report can be regrouped or extended without
running the pipeline again (`--report-only`).

## Extra analyses on finished runs

Both scripts replay the real-time biomechanics stage on the exact 3D stream each run produced (same
frames, frame gaps, continuity and smoothing; see `stages.replay_angles`), so no GPU or re-run is
needed.

**Vicon static trial** (`add_gt_static.py`). New runs store the joint angles with the static trial
taken from the Vicon markers as well. For runs made before that, this adds them (needs the dataset's
`camera_data/` and `wcs_data/`):

```bash
python realtime_eval/add_gt_static.py --data-root /path/to/dataset --out realtime_eval/results/full_eval \
    --lpf-repo ../LivePoseFormer
python realtime_eval/run_eval.py --out realtime_eval/results/full_eval --report-only
```

**Thigh and shank rotation definition** (`thigh_shank_definition.py`). Plug-in Gait sets the
long-axis rotation of the thigh with the THI wand marker and of the shank with TIB. Wands sit wherever
they were stuck on, so cameras cannot reproduce them, which shows up as a large, subject-specific
offset in hip, knee and ankle rotation. This script replaces them, in the live 3D and the Vicon
markers alike, by virtual markers on the knee axis (epicondyles KNE − MKNE) and the ankle axis
(malleoli ANK − MMED), where Plug-in Gait intends them, and compares both definitions (needs only
`wcs_data/`):

```bash
python realtime_eval/thigh_shank_definition.py --data-root /path/to/dataset \
    --out realtime_eval/results/full_eval --lpf-repo ../LivePoseFormer
```

It writes `THIGH_SHANK_DEFINITION.md`, `thigh_shank_definition.csv` and
`figures/transverse/5_offsets_by_definition.png`. On the full dataset, knee rotation error dropped
from 17° to 7.5° and ankle rotation from 14° to 7° (ViTPose; all channels 6.1° → 5.2°). The reference
is then no longer standard Plug-in Gait, so report it as a separate, landmark-based comparison.

## Files

| File | What it does |
|---|---|
| `run_eval.py` | selects recordings, runs every model × recording, shows progress and time left, resumes |
| `dataset.py` | reads the dataset layout (`frames/`, `camera_data/`, `wcs_data/`) and lists recordings |
| `stages.py` | per-stage errors not in the simulator report (2D per camera, 3D by body region), per-frame arrays, replay of the biomechanics stage |
| `report.py` | the report from all finished runs |
| `add_gt_static.py` | adds the joint angles with a Vicon static trial to older runs |
| `thigh_shank_definition.py` | compares the wand-marker and anatomical thigh/shank definitions |

## Notes

* The cameras are simulated from recorded frames on a fixed 60 Hz clock. Real camera capture,
  transfer and decoding are not part of the latency.
* The static trial is the first evaluated frame of each recording. Use `--start` to place it on a
  standing frame.
* The pipeline was tested end to end on a GTX 1660 (Windows) with fp32, no compile and random
  LivePoseFormer weights; the results in the main README come from the RTX 3090 (Linux).
