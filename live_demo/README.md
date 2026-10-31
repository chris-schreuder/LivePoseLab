# 3. Live demo: real-time biomechanics feedback

A browser app that runs a live markerless biomechanics session from start to finish: subject details,
system check, static trial, live recording with real-time feedback, review and export. Three
synchronised cameras stream into the pipeline (person detector → 2D pose → triangulation →
LivePoseFormer → joint angles) and the app shows the cameras, the 3D skeleton, joint angles, graphs,
cues and reps as the subject moves.

It can **replay** precomputed sessions on any laptop, or compute **live** on a GPU with the same
engine as the [stream simulator](../stream_sim/README.md).

## Quick start

From the repository folder (installation in the [main README](../README.md#installation)):

```bash
pip install -r live_demo/requirements.txt
python live_demo/server.py --open                              # replay only: no GPU or models needed
python live_demo/server.py --open --lpf-repo ../LivePoseFormer  # also live mode (GPU, models, data/)
```

This opens `http://localhost:8000`. Sessions are folders in `live_demo/sessions/` (not in git; record
one with [`record_session.py`](#recording-sessions)). The built app is in `web/dist`, so Node.js is
only needed to change the app.

**Presenter shortcuts:** `http://localhost:8000/?quick` (or `?quick=<session name>`) opens the live
screen directly with the first (or named) session in replay mode; add `&live` for live mode with the
default settings (ViTPose, fp32, YOLO26s, smoothing auto).

## Replay or live

Picked on the session screen under **Processing**:

| | Replay · precomputed | Live · compute now |
|---|---|---|
| What runs | nothing: results computed by `record_session.py` are played back at 60 fps | the real streaming pipeline on the server's GPU, at the 60 fps camera clock (frames read from disk) |
| Settings | fixed, as recorded | chosen in the app: pose model, precision, detector, LivePoseFormer + checkpoint, fill dropped frames, angle smoothing, `torch.compile` |
| Frame rate shown | always 60 fps | what the GPU really manages; frames it cannot take are dropped (or filled by LivePoseFormer) |
| Latency shown | processing time recorded with the session | measured live, capture → joint angles |
| Needs | any laptop, no GPU | the pipeline's packages, model checkpoints, the raw frames in `data/`, a CUDA GPU |
| Result | the review uses the session | the take is saved as a new session (`<trial>_<task>_live_<model>_<time>`), replayable and exportable |

```
                                          ┌─ Replay: sessions/<name>/ (precomputed) ──────────────┐
 record_session.py ──► sessions/<name>/ ──┤                                                       ├──► browser app
                                          └─ Live: server.py /ws/live ──► stream_sim engine (GPU) ┘    (React + three.js)
                                                    camera clock → detector → pose → 3D → LPF → biomech
```

In both modes the session supplies the camera images on screen, the Vicon reference and the
calibration. In live mode, everything else (2D keypoints, boxes, 3D, joint angles, measures, timings)
is what the pipeline computes during the take.

### Using live mode

1. On the session screen, pick the capture, then **Processing → Live · compute now**. The panel shows
   the server's GPU and only offers what this machine can run (models with checkpoints, fp16 and bf16
   only on a GPU, LivePoseFormer only with checkpoints in `<lpf-repo>/checkpoints/`, `torch.compile`
   only on Linux). If live mode is unavailable, the panel says why.
2. Go through the system check and static trial, then press **Record**. The server loads the model
   (cached for later takes), warms up, runs the static trial on the first frame, then starts the
   camera clock. Progress is shown over the 3D view.
3. Results arrive as they are computed. The status bar shows the real output rate, the measured
   latency per frame and the time per stage, plus a red "live · computing on <GPU>" chip.
4. **Stop** (or the end of the capture) ends the stream. The server saves the take as a session and
   the review opens on it.

Between computed frames the display holds the last result, as a live viewer would. On a GPU that
cannot keep up, the graphs therefore step: on the GTX 1660 about 10 of 60 frames are computed. On the
RTX 3090 (ViTPose bf16 at 60 fps, or HRNet with LivePoseFormer filling) every frame has a result.

## The screens

1. **Session.** Subject details (ID, name, sex, age, height, mass, dominant side, leg lengths, knee and
   ankle widths), recording name, task (bodyweight squat or free movement), notes, which capture to
   stream, and replay or live.
2. **System check.** The 3 camera views with the 2D skeleton and detector boxes; the capture volume in
   3D with the real camera positions (from the calibration and the wand-fitted world frame); checks
   for cameras, calibration, person detected, pose model, temporal model, and the joint-angle model
   verified against the reference implementation.
3. **Static trial.** A countdown, then the calibration results: estimated vs entered height, feet
   flat, upright posture, standing knee angles, segment lengths left/right with asymmetry (Vicon for
   comparison). It sets the ankle model's offsets and the baselines, such as standing pelvis height.
4. **Live.**
   * **Cameras:** 3 views with a 2D keypoint overlay (colour = side, red = low confidence).
   * **3D view:** skeleton and markers; options for the translucent **Vicon ghost** skeleton, motion
     trails, pelvis axes and camera frustums; preset views (orbit, front, side, top) or free orbit.
     Legs turn red when a knee-valgus cue fires.
   * **Numbers:** knee flexion, hip flexion, ankle dorsiflexion and knee adduction (right and left),
     trunk flexion and side bend, pelvic tilt and obliquity, squat depth, and symmetry bars. Vicon
     values can be shown next to them.
   * **Feedback:** live cues (depth reached, knee caving in, chest up, uneven bending, weight shift),
     the rep counter and a table of the last reps with their peaks.
   * **Two live graphs:** any joint and plane (right and left together, Vicon dashed if enabled) or a
     distance measure, in a scrolling 6 s window.
   * **Status bar:** recording time, frame, stream rate, latency, and a per-stage timing strip.
   * **Feedback settings:** thresholds for depth, valgus, trunk lean, asymmetry, lateral shift and rep
     start.
5. **Review.** Summary (reps, average depth, rep time), the rep table (click a rep to jump to its
   bottom), cues given, range of motion per metric, a 3D replay with a scrubber, full-take graphs with
   the reps shaded, accuracy against Vicon (marker error, joint-angle error per plane, error per
   feedback metric), and **Export**.

## What the numbers mean

**Joint angles.** Every angle in the app (numbers, graphs, cues, reps, review, export) is the output
of the real-time biomechanics stage: Plug-in Gait, 72 channels, with lab alignment from the static
trial, Euler-branch continuity and glitch hold (no 180°/360° jumps), and optional One Euro smoothing.
How these work and how the smoothing presets were tuned is in the
[stream simulator README](../stream_sim/README.md#real-time-biomechanics---biomech). In the app,
**Smoothing** in the biomechanics panel switches everything between the smoothed and the unsmoothed
angles (both are recorded), and the export follows it.

Conventions (as the model gives them): knee adduction + = varus (knee out), − = valgus (knee in);
thorax flexion + = forward; pelvic tilt + = anterior. The absolute foot channels read about ±180° with
the foot flat. The elbow channels that are always 0 are left out.

**Distance measures** (`metrics.py`, the only non-angle values): pelvis drop (standing pelvis height
minus the current one: squat depth, mm) and pelvis lateral shift (mid-hip relative to mid-ankle along
the body's left–right axis, mm).

**Squat reps.** A rep starts when **both** knees flex past 30° and ends when both are back under 15°
(hysteresis, so noise or a single-leg movement does not count). Per rep: peak knee flexion R/L, peak
hip flexion, largest pelvis drop, peak valgus R/L, peak trunk flexion, depth reached (mean knee
flexion ≥ 90° by default) and duration.

**Cues:** depth reached (mean knee flexion ≥ threshold), knee caving in (knee adduction below −8°),
keep your chest up (thorax flexion > 45°), uneven bending (knee flexion R − L > 12°), weight shift
(lateral shift > 60 mm). All thresholds are editable live; they are sensible defaults, not clinical
cut-offs.

## Export

One zip, built by the server from the recorded take (`export.py`):

| File | Content |
|---|---|
| `opensim/<name>_markers.trc` | the 59 live markers (mm, Y up, right-handed) for OpenSim scaling and inverse kinematics |
| `opensim/<name>_markers_vicon.trc` | the Vicon markers (simulation only) |
| `opensim/<name>_joint_angles.mot` | Plug-in Gait joint angles (deg), OpenSim storage format |
| `opensim/<name>_feedback_metrics.mot` | the distance measures |
| `<name>_markers.c3d` | markers as C3D |
| `csv/` | markers, joint angles and metrics as tables (live and Vicon) |
| `<name>_session.json` | subject data, recording info, pipeline and timing, static trial, reps, thresholds, cue counts |
| `<name>_summary.pdf` | report: subject and session, key angles and measures (with error vs Vicon), reps, angle curves, accuracy per plane and joint |

The capture's world frame is left-handed, so the `.trc` and `.c3d` files have Z flipped to load the
right way round in OpenSim and C3D viewers (the app's 3D view does the same). Angles and metrics are
not affected.

## Recording sessions

`record_session.py` runs the pipeline on a recording in `data/` and writes a session folder for replay:

```bash
# HRNet with YOLO26s boxes
python live_demo/record_session.py --model hrnet --name S1A3D1_squat_hrnet --biomech-repo ../LivePoseFormer
# with LivePoseFormer (smoother skeleton and angles); best recorded on a fast GPU
python live_demo/record_session.py --model hrnet --precision bf16 --lpf-repo ../LivePoseFormer \
    --lpf-checkpoint ../LivePoseFormer/checkpoints/lpf_stream_59_stress.pth --name S1A3D1_squat_hrnet_lpf
# recompute a session's joint angles and measures from its stored 3D (after a model or metric change)
python live_demo/record_session.py --refresh S1A3D1_squat_hrnet
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `hrnet` | `hrnet` or `vitpose` |
| `--precision` | `fp32` | `fp16` / `bf16` on RTX 30xx and newer |
| `--detector` | `yolo26s` | YOLO model, or `json` for the boxes in the camera JSON files |
| `--lpf-repo`, `--lpf-checkpoint` | none | add LivePoseFormer |
| `--biomech-repo` | `--lpf-repo`, else found automatically | LivePoseFormer repository with the joint-angle model (see the [main README](../README.md#installation)) |
| `--trial`, `--start`, `--frames` | `S1A3D1`, 200, to the end | which part of which recording; frame `--start` is the static trial |
| `--angle-smoothing` | `auto` | `auto` (standard on raw 3D, off with LivePoseFormer), `off`, `light`, `standard` |
| `--video-height` | 600 | size of the stored camera frames |
| `--name` | from the settings | session folder name |

Copy a session folder into `live_demo/sessions/` on any PC and it appears in the app.

**Timing in replay.** Each frame keeps the stage times measured when it was recorded, and the status
bar shows them; a chip says where the session was recorded. Sessions recorded on a slow GPU therefore
show a high latency (HRNet on the GTX 1660: about 370 ms per frame; on the RTX 3090 with bf16 +
compile: about 25 ms). Playback itself always runs at 60 fps.

## Showing it

* **One screen, locally (recommended):** `python live_demo/server.py --open`, then full screen (F11).
  Designed for 1920×1080; works down to about 1600×900.
* **Another device on the same network:** `python live_demo/server.py --host 0.0.0.0`, then open
  `http://<this PC's IP>:8000`.
* **Remote viewers:** use Tailscale or a Cloudflare Tunnel to the local server rather than opening
  ports.
* **A video:** screen-record the live screen. The exported `.trc` / `.c3d` markers can be imported
  into Blender for a rendered version.

Server options: `--host` (default `127.0.0.1`), `--port` (8000), `--open` (open the browser),
`--lpf-repo` (LivePoseFormer repository, needed for live mode), `--device` (`cuda`).

## Changing the app

```bash
cd live_demo/web
npm install
npm run dev        # http://localhost:5173, hot reload; API and sessions proxied to the server on :8000
npm run build      # writes web/dist, which server.py serves
```

| Path | What it is |
|---|---|
| `server.py` | FastAPI server: app, sessions, `/api/sessions`, `/api/options`, `/api/export`, `/ws/live` |
| `live.py` | live mode: builds (and caches) the pipeline for the chosen settings, runs the `stream_sim` engine, streams results, saves the take |
| `record_session.py` | runs the pipeline on a recording and writes a session |
| `metrics.py` | the distance measures and the static-trial checks |
| `export.py` | TRC / MOT / C3D / CSV / JSON / PDF export |
| `web/src/data/session.ts` | loads a session (binary frame table, JPEG streams); flips Z for display |
| `web/src/data/live.ts` | live mode client: WebSocket, copies each result row into the frame table |
| `web/src/data/video.ts` | decodes camera frames ahead of the playhead (`createImageBitmap`, LRU cache) |
| `web/src/data/player.ts` | the 60 fps playback clock all views follow |
| `web/src/logic/feedback.ts` | thresholds, cues, rep counter |
| `web/src/logic/signals.ts` | the signals the graphs can show |
| `web/src/components/` | camera tile, 3D scene, live graph (uPlot), panels, UI parts |
| `web/src/screens/` | Setup, SystemCheck, StaticTrial, Live, Review |

Download the web folder here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23837907.svg)](https://doi.org/10.5281/zenodo.23837907)


**How live and replay share the views.** Every view reads frames through `Session` and follows
`player.frame`. In replay the player's clock advances the frame; in live mode `live.ts` writes each
result row (binary: frame number + float32 values, layout announced by the server's "hello" message)
into the session's frame table and moves the player to the newest frame. So the screens are the same
in both modes. With real cameras instead of files, only the camera source on the server would change.

## Notes and limitations

* Both modes use a recorded capture (live mode reads its frames from disk at the camera rate), and the
  app labels which mode is running. The Vicon overlay and the accuracy panel exist only because this
  is a simulation with ground truth.
* Live mode runs one stream at a time. In live mode the system check and static-trial screens still
  show the session's recorded values; the live static trial runs on the server when you press Record.
* The ankle model's static trial is the first frame of the capture, standing in for a separate static
  recording.
* Shoulder angles come from the model's XZY decomposition; with the arms raised forward, its
  "adduction" channel reads large values. That is the model's definition, shown as it is.
