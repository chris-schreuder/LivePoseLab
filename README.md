# LivePoseLab

**Real-time markerless biomechanics from three cameras: 2D pose → 3D → LivePoseFormer → Plug-in Gait
joint angles at 60 fps, with a live feedback app and an end-to-end evaluation against Vicon.**

LivePoseLab takes three synchronised camera feeds and turns them, frame by frame and in real time, into
the 59-marker SUMediPose skeleton and 72 Plug-in Gait joint-angle channels. It covers the whole path
from a single offline frame to a live demo and a paper evaluation:

```
 3 synchronised cameras (60 fps)
        │
        ▼
 YOLO26 person detector ──► 2D pose (HRNet-W48 or ViTPose-B, 59 keypoints per view)
        │
        ▼
 DLT triangulation ──► world frame (Vicon WCS, fitted with the calibration wand)
        │
        ▼
 LivePoseFormer: streaming transformer that denoises the 3D and fills frames the pose network skipped
        │
        ▼
 Real-time biomechanics: Plug-in Gait joint angles (72 channels), continuity, One Euro smoothing
        │
        ▼
 live feedback (browser app) · OpenSim / C3D export · evaluation against Vicon
```

## Headline results

Full pipeline on an RTX 3090 (bf16 + `torch.compile`, YOLO26s, LivePoseFormer, light angle smoothing),
32 recordings (4 subjects × 8 actions, 13 min of 60 fps video per model), against Vicon:

| | ViTPose-B | HRNet-W48 |
|---|---:|---:|
| Joint angles out per second | **60** | **60** (pose network at 28 fps, LivePoseFormer fills the rest) |
| Latency, camera frame → joint angles (median / p95) | **69 / 76 ms** | 134 / 150 ms |
| 3D marker error (MPJPE) | 31.5 mm | **22.9 mm** |
| Joint-angle error, sagittal / frontal / transverse | 5.7° / 4.5° / 8.8° | 5.4° / 4.9° / 8.0° |

ViTPose runs every frame with spare capacity (about 105 fps possible). HRNet is more accurate in 3D but
runs at about half the camera rate; LivePoseFormer fills the dropped frames almost as accurately as
the measured ones.

## What is in the repository

The repository has four parts. Each builds on the one before and has its own README:

| Part | Question it answers | Run it with | Needs |
|---|---|---|---|
| [**1. Pipeline**](pipeline/README.md) | What does each step do to one frame, and how long does it take? | `run_pipeline.py` | a GPU (any) |
| [**2. Stream simulator**](stream_sim/README.md) | Does the full chain keep up with a live 60 fps stream, and how old is each result? | `stream_sim/run_stream_sim.py` | a GPU (fast for real time) |
| [**3. Live demo**](live_demo/README.md) | What does real-time feedback look like for a user? | `live_demo/server.py` | nothing for replay; a GPU for live mode |
| [**4. Evaluation**](realtime_eval/README.md) | How fast and how accurate is the live pipeline over a whole dataset? | `realtime_eval/run_eval.py` | a fast GPU and the full dataset |

* **Pipeline** (`pipeline/`, `run_pipeline.py`): the core package. Three images in, 2D keypoints, 3D
  in the camera frame and 3D in the world frame out, with a plot and a timing for every step. Use it
  to understand the method or as a library (`LivePipeline`).
* **Stream simulator** (`stream_sim/`): plays a recording as a live 60 fps 3-camera stream into a
  multi-threaded engine (detector, pose, triangulation, LivePoseFormer, biomechanics) and reports
  frame rate, dropped and filled frames, latency per stage and accuracy. The live demo and the
  evaluation use the same engine.
* **Live demo** (`live_demo/`): a browser app (FastAPI + React + three.js) for a live session:
  subject details, system check, static trial, live cameras, 3D skeleton, joint-angle numbers and
  graphs, cues and rep counting, review, and export to OpenSim / C3D / PDF. It replays precomputed
  sessions on any laptop, or computes live on a GPU.
* **Evaluation** (`realtime_eval/`): runs the full live pipeline on every recording of the dataset
  and writes a paper-ready report (Markdown, LaTeX tables, figures, CSVs) with every timing and the
  error at every stage.

## Installation

### 1. Code

```bash
git clone https://github.com/chris-schreuder/LivePoseLab.git
git clone https://github.com/chris-schreuder/LivePoseFormer.git   # temporal model + joint-angle model
cd LivePoseLab

python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install -r requirements.txt
pip install -r live_demo/requirements.txt     # only for the live demo
```

LivePoseFormer is a separate repository that provides the temporal model and the Plug-in Gait
joint-angle model (`livepose/biomech`), so it is needed for the LivePoseFormer stage and for joint
angles. The scripts find it in this order: `--lpf-repo PATH`, the `LIVEPOSEFORMER` environment
variable, or a `LivePoseFormer` folder next to this repository (as cloned above). No paths are
hard-coded; the examples below pass `--lpf-repo ../LivePoseFormer` only to be explicit.

`torch.compile` (`--compile`) needs Triton, which comes with PyTorch on Linux. Everything else runs on
Windows and Linux.

### 2. Model weights and calibration (not in git)

```
LivePoseLab/
  ViT_B_simple_all_63/  config.yaml + checkpoints/best_nmae_model.pth     ViTPose-B, fine-tuned on SUMediPose
  w48_768x576_all_63/   config.yaml + checkpoints/best_nmae_model.pth     HRNet-W48, fine-tuned on SUMediPose
  calibration/          external_calibration.csv + external/matrix/*.npz  camera calibration per session
../LivePoseFormer/checkpoints/lpf_stream_59_stress.pth                    LivePoseFormer checkpoint
```

The pose models come from
[Markerless-Biomechanical-Analysis](https://github.com/chris-schreuder/Markerless-Biomechanical-Analysis).
The first run of each pose model writes a slim `*_weights_only.pt` next to the checkpoint for faster
start-up. The YOLO26 detector weights (`yolo26s.pt`) download on first use.

### 3. Data (not in git)

The pipeline, the stream simulator and the live demo use one recording in `data/`; the evaluation
uses the full dataset in any folder:

```
data/                                      one recording (pipeline, stream simulator, live demo)
  C1S1A3D1.json  C2S1A3D1.json  C3S1A3D1.json      per camera: frames, boxes, 2D/3D ground truth, wand
  C1S1A3D1/  C2S1A3D1/  C3S1A3D1/                  the camera frames (JPEG)
  S1A3D1_wcs.json                                  Vicon markers in the world frame

<data_root>/                               full dataset (evaluation)
  frames/S7/C1/C1S7A1D1/C1S7A1D1_0307.jpg
  camera_data/C1/S7A1D1.json
  wcs_data/S7A1D1.json
```

Recordings are named `S<subject>A<action>D<take>`. C1 is the main camera.

## Quick start

Run these from the repository folder, in this order. Each step is explained in its part's README.

**1. One frame through the pipeline, with a plot of every step** ([details](pipeline/README.md))

```bash
python run_pipeline.py --model vitpose --frames 300
```

Writes `outputs/vitpose_300/frame_0300/step1_inputs.png` … `step6_world.png` and a timing breakdown.

**2. A live 60 fps stream through the full chain** ([details](stream_sim/README.md))

```bash
python stream_sim/run_stream_sim.py --model vitpose --policy latest --biomech \
    --lpf-repo ../LivePoseFormer --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59_stress.pth
```

Prints a verdict (real time or not), the output rate, latency per stage and the 3D and joint-angle
errors; results go to `stream_sim/results/`.

**3. The live feedback app** ([details](live_demo/README.md))

```bash
python live_demo/server.py --open --lpf-repo ../LivePoseFormer
```

Opens `http://localhost:8000`. Pick a session and **Replay** (no GPU needed), or **Live · compute
now** to run the pipeline on this machine's GPU.

**4. The full evaluation** ([details](realtime_eval/README.md))

```bash
python realtime_eval/run_eval.py --data-root /path/to/dataset --all \
    --lpf-repo ../LivePoseFormer --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59_stress.pth \
    --out realtime_eval/results/full_eval
```

Runs both models on every recording (resumable, with progress and time left) and writes
`realtime_eval/results/full_eval/REPORT.md`.

## How it works, in short

* **2D pose.** YOLO26s finds the person in each view (fast path: one batch, no NMS). Each crop goes
  through HRNet-W48 (768×576) or ViTPose-B (256×192); heatmaps are decoded to 59 keypoints per view.
* **3D.** DLT triangulation from the three views in the main camera's frame, then into the Vicon
  world frame with a transform fitted once to the calibration wand.
* **Live streaming.** Each stage runs in its own thread. Frames the pose network cannot take in time
  are dropped (policy *latest*), as a live system must. LivePoseFormer and the biomechanics run
  decoupled on their own queues, so they never hold up the pose network.
* **LivePoseFormer** (CUDA graph, ~2–4 ms per frame) smooths the 3D with a 3-frame lookahead (50 ms)
  and fills every dropped frame, so the output stays at the camera rate even when the pose network
  is slower.
* **Joint angles.** A vectorised port of the Plug-in Gait model (identical to the reference
  implementation, ~2 ms per frame): static trial before going live, lab alignment, Euler-branch
  continuity and glitch hold (no 180°/360° jumps), and an optional One Euro filter.

[`pipeline/CODE_WALKTHROUGH.md`](pipeline/CODE_WALKTHROUGH.md) walks through the core pipeline's
code in the order it runs.

## Repository layout

```
run_pipeline.py          1. run the pipeline on chosen frames: step plots, timing, metrics
pipeline/                   the core package (LivePipeline): pose, calibration, triangulation, world frame
stream_sim/              2. the streaming engine and simulator (also used by 3 and 4)
  engine.py                 multi-threaded stages, frame policies, LivePoseFormer + biomechanics queues
  source.py                 the camera clock and the live disk reader
  detector.py               YOLO26 person detector (fast path)
  lpf_stage.py, lpf_graph.py  LivePoseFormer stage and its CUDA graph
  biomech_stage.py          real-time Plug-in Gait joint angles
  report.py                 metrics, reports and plots
live_demo/               3. the feedback app: server, live runner, session recorder, export, web app
realtime_eval/           4. the dataset evaluation, paper report and extra analyses
calibration/                camera calibration (not in git)
data/                       the sample recording (not in git)
```

## Coordinate frames

* **Camera frame:** OpenCV axes with C1 at the origin; calibration units are mm / 60.
* **World frame (WCS):** the Vicon frame, mm, Y up, floor at y = 0. It is left-handed (a mirror of
  the camera frame). The biomechanics stage turns it into the right-handed lab frame Plug-in Gait
  expects; exports to TRC / C3D flip Z so they load correctly in OpenSim.

## Limitations

* The cameras are simulated from recorded frames on an exact 60 Hz clock; real capture, transfer and
  decoding are not in the measured latency.
* The static trial is the first frame of each recording, standing in for a separate static capture.
* Lens distortion is not corrected before triangulation.
* Joint angles are compared with the same model applied to the Vicon markers, not with Vicon Nexus
  output. Transverse-plane angles depend strongly on the thigh and shank wand markers (THI, TIB),
  which cameras cannot see; see the evaluation README.

## Related repositories

* [Markerless-Biomechanical-Analysis](https://github.com/chris-schreuder/Markerless-Biomechanical-Analysis):
  training and offline evaluation of the 2D pose models; LivePoseLab reuses its crop, heatmap
  decoding, calibration and triangulation methods.
* [LivePoseFormer](https://github.com/chris-schreuder/LivePoseFormer): the streaming 3D denoising and
  imputation transformer, and the Plug-in Gait joint-angle model.

## Licence

[MIT](LICENSE), the same as LivePoseFormer.
