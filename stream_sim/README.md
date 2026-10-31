# 2. Stream simulator: does it keep up live?

Plays a recording as three hardware-synchronised cameras streaming at **60 fps** into the full live
chain, and answers: **does it keep up, how old is each result when it arrives, how accurate is it,
and how much faster would it need to be?**

```
person detector → 2D pose → triangulation → world frame → LivePoseFormer (optional) → joint angles (optional)
```

The engine here (`engine.py`) is the one the [live demo](../live_demo/README.md) and the
[evaluation](../realtime_eval/README.md) use, so what you measure here is what runs live.

## Quick start

From the repository folder (installation in the [main README](../README.md#installation)):

```bash
# pose only: does ViTPose keep up at 60 fps?
python stream_sim/run_stream_sim.py --model vitpose --policy latest

# the full live chain: detector, pose, 3D, LivePoseFormer, joint angles
python stream_sim/run_stream_sim.py --model vitpose --policy latest --biomech \
    --lpf-repo ../LivePoseFormer --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59_stress.pth

# HRNet at its own rate, LivePoseFormer filling the dropped frames up to 60 fps (RTX 30xx, Linux)
python stream_sim/run_stream_sim.py --model hrnet --precision bf16 --compile --policy latest --biomech \
    --lpf-repo ../LivePoseFormer --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59_stress.pth \
    --lpf-fill-dropped

# what-if: how fast would the network need to be?
python stream_sim/run_stream_sim.py --model vitpose --fake-network-ms 15,10,5
```

Each run prints a verdict per scenario and writes everything to
`stream_sim/results/<model>_<gpu>_<fps>fps_<precision>_<time>/`.

## How it works

```
 LiveVideo              CameraSource (clock)          StreamEngine
 the video on disk,     releases 3 synced frames  -->  DET  (GPU: YOLO26 person detector -> boxes)
 read + decoded just    every 1/60 s, stamped          PRE  (CPU: crop, resize)
 ahead of the clock     with a capture time            GPU  (upload, network, heatmap peaks)
                                                       POST (CPU: map to image, triangulate, world)
                                                       LPF  (optional LivePoseFormer, GPU, decoupled)
                                                       BIO  (optional joint angles, CPU, decoupled)
```

* **The recording plays once as a live video**, from `--start` to its last frame (or for
  `--duration` seconds). Nothing loops, so LivePoseFormer sees one continuous recording and every
  frame is scored against its own Vicon ground truth.
* **The clock** releases a frame set every 16.67 ms, whether or not the pipeline is ready. The report
  checks that the clock itself was on time.
* **Frames are read from disk just ahead of the clock**, like a camera that hands over frames already
  in memory: `--reader-threads` background threads decode into a rolling buffer `--prefetch` frame
  sets ahead (default 120, about 2 s). If the reader is ever late, the report says so (that is the
  simulator being slow, not the pipeline: raise `--reader-threads`). `--decode-jpeg` instead times
  JPEG decoding inside the pipeline, as for cameras that send MJPEG.
* **Two frame policies** (`--policy`):
  * `latest`: the pipeline always takes the **newest** frame; frames it could not take are dropped.
    This is how a live system behaves, and shows the output rate and delay you actually get.
  * `all`: every frame is queued. If the pipeline is too slow, the queue and the delay grow without
    limit; it shows *whether* it keeps up. The camera stops early once `--max-backlog` frames wait.
* **Two execution modes** (`--mode`):
  * `pipelined`: each stage in its own thread, so the CPU prepares frame t+1 while the GPU works on
    frame t. Throughput is set by the slowest stage. With `latest`, the hand-off to the GPU also keeps
    only the newest frame.
  * `sequential`: one thread does everything, for comparison.
* **LivePoseFormer and the biomechanics are decoupled**: each runs in its own thread on an unbounded,
  ordered queue, so neither ever makes the pose network wait.
* **Warm-up.** Before the clock starts, the pipeline runs a few frames end to end, and every worker
  thread that uses the GPU runs the network in its own thread (cuDNN keeps per-thread state; a cold
  thread's first call costs ~0.7 s). The static trial for the joint angles is also done here.
* **Honest timestamps without slowing down.** The engine does not synchronise the GPU between steps;
  the GPU stage ends by copying the keypoints back, which waits for the GPU anyway.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `vitpose` | `vitpose` (ViT-B, 256×192) or `hrnet` (W48, 768×576) |
| `--trial` | `S1A3D1` | recording in `data/` |
| `--fps` | 60 | camera rate to simulate |
| `--start` | 200 | first frame of the stream (also the static trial with `--biomech`) |
| `--duration` | to the end | seconds of stream per scenario |
| `--mode` | `pipelined` | `pipelined`, `sequential` or `both` |
| `--policy` | `both` | `latest`, `all` or `both` |
| `--precision` | `fp32` | `fp16` / `bf16` autocast (use bf16 on RTX 30xx and newer) |
| `--compile` | off | `torch.compile` the network (needs Triton, i.e. Linux; falls back if missing) |
| `--detector` | `yolo26s` | person detector: `yolo26n/s/m/l/x`, a `.pt` path, or `json` for the boxes in the camera JSON files |
| `--detector-conf` | 0.25 | minimum person confidence; below it the previous box is kept and a miss is counted |
| `--lpf-repo`, `--lpf-checkpoint` | none | add the LivePoseFormer stage (random weights without a checkpoint: timing only) |
| `--lpf-fill-dropped` | off | push every dropped frame to LivePoseFormer as an all-NaN frame, so it fills it |
| `--lpf-no-graph` | off | run LivePoseFormer eagerly instead of as a CUDA graph (comparison only; slower) |
| `--biomech` | off | add the real-time joint angles after the 3D (or after LivePoseFormer) |
| `--biomech-repo` | `--lpf-repo`, else found automatically | LivePoseFormer repository with the joint-angle model (see the [main README](../README.md#installation)) |
| `--angle-smoothing` | `auto` | One Euro smoothing of the joint angles: `auto`, `off`, `light`, `standard` |
| `--fake-network-ms` | none | comma list of network times for what-if scenarios |
| `--prefetch` | 120 | frame sets the reader keeps decoded ahead of the clock (~19 MB each) |
| `--reader-threads` | 3 | threads reading and decoding frames from disk |
| `--max-backlog` | 300 | policy `all`: stop once this many frames are queued |
| `--decode-jpeg` | off | cameras send JPEG; decoding is timed in the pipeline |
| `--cams`, `--checkpoint`, `--device`, `--out` | | cameras, pose checkpoint file, device, results folder |

## Outputs

`stream_sim/results/<model>_<gpu>_<fps>fps_<precision>_<time>/`:

| File | Content |
|---|---|
| `comparison.txt` / `.csv` | one line per scenario: real time yes/no, output rate, % dropped, delay, trend, max sustainable fps, speed-up needed, 3D and angle errors |
| `<scenario>/report.txt` | the full report and verdict |
| `<scenario>/report.png` | delay per frame, timeline (processed vs dropped), time per stage against the frame budget, delay histogram |
| `<scenario>/frames.csv` | every frame's timestamps per stage and its error vs Vicon, before and after LivePoseFormer |
| `<scenario>/lpf_accuracy.png` | with LivePoseFormer: error per frame before and after, filled frames in red |
| `<scenario>/angles.csv`, `biomech.png` | with `--biomech`: every output frame's 72 live and Vicon angles; hip, knee and ankle flexion over time |
| `<scenario>/summary.json` | all numbers, system info (GPU, torch, CUDA, CPU) and the arguments |

### Reading the report

| Line | Meaning |
|---|---|
| **VERDICT** | real time = (latest) under 1% of frames dropped, or (all) the delay does not grow |
| **max sustainable fps** | 1000 / time of the slowest stage (pipelined) or of all stages together (sequential) |
| **needs N x** | how much faster the bottleneck must get for 60 fps (under 1 = already fast enough) |
| **delay** | capture → 3D result: what a live application feels (plus LivePoseFormer's 3-frame lookahead if enabled) |
| **delay trend** | ms the delay grows per second; positive with `all` = falling behind |
| **service time per stage** | time each stage spends on one frame; the bottleneck is marked |

## The stages in detail

### Person detector

The pose networks need a box around the person in each view. The DET stage runs **YOLO26s**
(Ultralytics) on all 3 views of every frame set and keeps the most confident person per view. It is
written for speed (`detector.py`) rather than through the Ultralytics `predict` API, whose Python
overhead tripled the time on the GTX 1660 (82–113 ms vs 33 ms per frame set):

* the 3 views are resized on the CPU to 640 px on the long side into a pinned buffer and uploaded as
  one batch;
* the network (BatchNorm fused, channels-last, same `--precision` as the pose network) runs on its own
  CUDA stream, in its own thread in pipelined mode;
* only one person per view is needed, so the highest person score is taken on the GPU instead of
  running non-maximum suppression, and only 3 × 5 numbers are copied back.

If a view has no person above `--detector-conf`, its previous box is kept and the miss is counted.
The detected boxes match the dataset's boxes closely (IoU 0.98) and give the same 3D error. On the
RTX 3090 it takes about 9 ms next to ViTPose; next to HRNet it waits for the GPU (35 ms).

### LivePoseFormer: accuracy before and after

With `--lpf-repo`, each report scores both outputs against Vicon in the world frame, on the same
frames:

```
  error vs Vicon GT, world frame, same 1200 frames:   before LPF    after LPF
    MPJPE mean (mm)                                        33.7         ...
    MPJPE median / p95 (mm)                                 ...         ...
    acceleration error / jitter (mm/frame^2)                ...         ...
    frames where LPF lowered the error: ..%
```

* **before LPF** is the raw triangulated 3D; **after LPF** is LivePoseFormer's output for the same
  frame (released 3 frames later).
* **Jitter** is the acceleration error over three consecutive frames; it needs a run without dropped
  frames (or with filling).
* Without `--lpf-checkpoint` the weights are random and the "after" numbers are meaningless.

### Filling dropped frames (`--lpf-fill-dropped`)

When the pose network is slower than the cameras (policy `latest`), it skips frames. With this flag,
each skipped frame is still pushed to LivePoseFormer, in order, as a frame with every keypoint missing
(all NaN). LivePoseFormer imputes it from the frames around it, so **the output keeps the full camera
rate** while the pose network runs slower. HRNet in bf16 on the RTX 3090 processes about 28 of every
60 frames; with this flag the output is 60 fps, and the filled frames are almost as accurate as the
measured ones.

Three things keep the pose network at full speed:

1. **Decoupled.** POST hands each 3D result to an unbounded queue and goes straight back to work;
   LivePoseFormer reads that queue in its own thread.
2. **Filled on the camera clock.** The moment a frame is dropped (by the source or at the hand-off to
   the GPU), a "dropped" event goes into the same queue. LivePoseFormer takes events strictly in
   stream order (a small reorder buffer waits for the frame ahead).
3. **CUDA graph** (`lpf_graph.py`). A LivePoseFormer step is hundreds of tiny GPU kernels; next to a
   heavy network every launch queued behind it, and the step took 16–27 ms on the 3090 instead of
   ~4 ms. The step is recorded once as a CUDA graph and replayed with one call per frame. The cache
   lives in a fixed 128-slot buffer with a mask, which gives the same attention as the model's
   growing cache, so the graph is used from the first frame. It is checked against the normal model
   when built (`graph vs eager max diff`, about 0.003 mm).

The report's LivePoseFormer block shows whether it keeps up (its wait does not grow over the run and
it is idle part of the time), how many frames it filled, and the error of the filled frames.

Things to know:

* **A filled frame has no observation at all**, so its error grows with the length of the gap. The
  models were trained with 25–60% of the keypoints missing, never whole frames; check the `mm fill`
  numbers before relying on it.
* A stream must start on a real frame: dropped frames before the first processed frame, and after
  the last one, are not filled.
* It only matters with `--policy latest`; with `all`, nothing is dropped.

### Real-time biomechanics (`--biomech`)

The last stage turns every 3D frame into joint angles with the Plug-in Gait model of the
LivePoseFormer repository (`livepose/biomech`): 72 channels (pelvis, hip, knee, ankle, absolute ankle,
foot progression, thorax, head, neck, spine, shoulder, elbow; right and left), each in three planes:
flexion/extension (sagittal), ab/adduction (frontal) and rotation (transverse).

* **Input.** LivePoseFormer's output when it is enabled (with `--lpf-fill-dropped`, the angles come at
  the full 60 fps), otherwise the raw 3D of every processed frame.
* **Static trial before the stream.** The ankle model needs a static calibration, recorded before
  going live in a real session. Here it is done before the clock starts, with the pipeline's own 3D of
  the first frame as a stand-in. The Vicon reference angles get their own static trial from the same
  frame.
* **Lab alignment.** The static trial also makes the world frame right-handed (this capture's is
  left-handed: Y up, Z mirrored) and turns it to the subject's facing direction, as Plug-in Gait
  expects. Without it the model's cross products flipped (hip flexion about 180° standing, ±180°
  wraps). With it, angles read clinically: about 0° standing, hip about 100° and knee about 125° at
  the bottom of a deep squat.
* **Continuous angles.** Per frame and joint, of the two equivalent Euler solutions (each moved to the
  nearest 360° multiple), the one closest to the previous frame is used, so angles never jump by
  180°/360°. A change faster than 25°/frame (1500°/s) counts as a pose glitch: the last value is held
  for up to 3 frames, then followed at that rate. The report line `continuity:` counts each
  correction.
* **Low-cost smoothing (One Euro filter).** After the continuity step, each angle can go through a
  One Euro filter: a low-pass whose cutoff rises with the angle's speed, so it smooths strongly when a
  joint is still and follows closely when it moves fast. It is causal (no lookahead) and costs about
  20 µs per frame. Vicon angles are never filtered. Tuned against Vicon on S1A3D1 (raw 3D):

  | Preset | min cutoff / beta (flexion; ab/adduction + rotation) | High-frequency noise vs Vicon | Angle error | Flexion lag |
  |---|---|---:|---:|---:|
  | `off` | – | ×5.6 (HRNet) / ×8.5 (ViTPose) | 3.39 / 4.83° | 0 |
  | `light` | 2.0 Hz / 0.05; same | ×3 / ×4 | 3.24 / 4.61° | ~17 ms |
  | `standard` | 0.7 Hz / 0.01; 0.4 Hz / 0.004 | ×1.1 / ×1.6 | 3.47 / 4.69° | ~33 ms |

  `auto` (default) uses `standard` on raw 3D and `off` with LivePoseFormer, whose lookahead already
  removes most jitter. Fixed-lag Savitzky-Golay or Gaussian smoothers removed less noise for more
  delay.
* **Fast and verified.** The reference `compute_angles` takes about 5 ms per frame; `biomech_stage.py`
  computes the same 72 channels with a vectorised port (about 2 ms) that follows scipy's own
  rotation-to-Euler path. At start-up it is checked against the reference (`fast port vs reference
  model max diff`, about 1e-13°).

The report adds a biomechanics block: compute time, whether it keeps up, delay capture → joint
angles, and the error against Vicon per plane and per joint (`all output`; with filling also
`measured` and `filled`; with LivePoseFormer also `before LPF`), plus angle jitter.

* **Elbow ab/adduction and rotation are always 0** (the model builds the forearm frame on the upper
  arm's axis), so they are left out of every angle metric.
* The absolute foot channels (`ankle_abs_*`) read about ±180° with the foot flat, because the model's
  foot axis points from toe to heel.

## Reference results

### GTX 1660 (6 GB), i5-9400F, ViTPose, fp32

Saved in `results/GTX1660_reference_vitpose*`.

| Scenario | Real time | Output fps | Dropped | Delay mean / p95 (ms) | Needs |
|---|---|---:|---:|---:|---:|
| pipelined, latest | no | 17.9 | 70% | 66 / 74 | 3.35× |
| pipelined, all | no | 17.8 | 0% | grows 2.4 s per s | 3.37× |
| sequential, latest | no | 17.5 | 70% | 66 / 73 | 3.42× |
| pipelined, latest, network 10 ms (what-if) | **yes** | 60.0 | 0% | 14 / 17 | 0.73× |
| sequential, latest, network 10 ms (what-if) | no | 37.3 | 38% | 22 / 29 | 1.61× |

The network (about 59 ms for 3 views) is the bottleneck; the CPU stages take about 1.5 ms together.
With a 10 ms network only the pipelined version reaches 60 fps: overlapping the stages matters once
the network is fast.

### RTX 3090, full chain (bf16 + compile, YOLO26s, LivePoseFormer, joint angles)

| | ViTPose | HRNet + filling |
|---|---:|---:|
| Pose network rate | 60 fps (max ~105) | 28 fps |
| Joint-angle output | 60 fps | 60 fps |
| Latency capture → joint angles | 70 ms | 135 ms |
| 3D error (MPJPE) | 31.5 mm | 22.9 mm |

These come from the [evaluation](../realtime_eval/README.md) over 32 recordings.

## Running on another PC

Copy the repository with the model weights, `calibration/` and the recording in `data/`, plus the
LivePoseFormer repository next to it (see the [main README](../README.md#installation)). The
`*_weights_only.pt` files are recreated on the first run and the detector weights download on first
use (copy `yolo26s.pt` over if that PC is offline). Each results folder name includes the GPU and
precision, so results from different machines never mix: compare their `comparison.txt` files.

Things to check on a new GPU:

* **Accuracy with fp16 / bf16.** Every report prints the MPJPE. HRNet overflowed in fp16 on the
  GTX 1660; bf16 has the fp32 range and is the safer choice. Check the MPJPE stays close to fp32
  (ViTPose about 33 mm, HRNet about 21 mm in the world frame).
* **CPU stages.** If PRE or POST becomes the bottleneck once the network is fast, the report marks it.
* **LivePoseFormer next to the network.** With `--lpf-fill-dropped`: `LPF ok` = yes and `LPF fps`
  about 60 in the comparison table, and the pose network's `GPU` time close to a run without
  LivePoseFormer.
