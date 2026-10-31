# Code walkthrough: the core pipeline

This file explains what the code does, in the order it runs, so you can follow it with the
source open next to it. For how to *run* it, see [README.md](README.md).

---

## 1. The big picture

```
                         ONCE (setup)                                  EVERY FRAME SET (3 images)
 ┌──────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────┐
 │ load model  (pose2d.PoseEstimator)           │   │ 1  read 3 images (prefetched in a background thread)     │
 │ load calibration -> P1, P2, P3 (calibration) │   │ 2a crop each person box (grown to 3:4)                   │
 │ GPU warm-up                                  │   │ 2b resize crops to the model input                       │
 │ triangulate wand + fit camera->world (wcs)   │   │ 3a upload batch to GPU, scale to [0, 1]                  │
 └──────────────────────────────────────────────┘   │ 3b network forward  ->  59 heatmaps per view             │
                                                    │ 4a find each heatmap peak (on the GPU)                   │
                                                    │ 4b map peaks back to full-image pixels                   │
                                                    │ 5  triangulate 59 points from 3 views  -> camera frame   │
                                                    │ 6  apply camera->world transform       -> world frame    │
                                                    └──────────────────────────────────────────────────────────┘
```

Three coordinate spaces appear along the way:

| Space | Where | Units |
|---|---|---|
| **Image** | each camera's full 1150 x 1800 picture | pixels |
| **Camera frame** | 3D, with the main camera C1 at the origin (OpenCV axes: x right, y down, z forward) | calibration units = **mm / 60** |
| **World frame (WCS)** | 3D Vicon frame, floor at y = 0, y up | **mm** |

---

## 2. Files and what each one is for

```
run_pipeline.py            the script you run: loops over frames, plots, metrics, timing report
pipeline/
  data.py                  reads the 3 camera JSONs + WCS JSON; gives one FrameSet per frame index
  calibration.py           builds the 3 projection matrices from the calibration CSV / .npz files
  pose2d.py                the 2D model: load, crop, resize, forward, decode heatmaps
  triangulation.py         DLT triangulation and reprojection
  wcs.py                   camera frame -> world frame transform, fitted to the wand
  pipeline.py              LivePipeline: runs steps 1-6 and times each one
  timing.py                StepTimer: CUDA-synchronised timing and the breakdown report
  viz.py                   one plotting function per pipeline step, plus the timing chart
  skeleton.py              keypoint names and which ones to connect when drawing
  models/hrnet.py          HRNet network definition (from your repo)
  models/vitpose.py        ViTPose network definition (from your repo)
```

The dependency direction is simple: `run_pipeline.py` uses `data.py` (inputs and ground truth),
`pipeline.py` (the actual pipeline) and `viz.py` (plots). `pipeline.py` uses `pose2d.py`,
`calibration.py`, `triangulation.py`, `wcs.py` and `timing.py`. Nothing in `pipeline/` depends on
the ground truth: GT is only used by `run_pipeline.py` for plots and error numbers.

---

## 3. What happens when you run `python run_pipeline.py`

Follow `main()` in [run_pipeline.py](../run_pipeline.py) from top to bottom.

### 3.1 Parse arguments and load the recording

```python
rec = Recording(root / "data", args.trial, args.cams.split(","))
frames = parse_frames(args.frames, len(rec))
```

**`Recording`** ([pipeline/data.py](data.py)) opens `data/C1S1A3D1.json`,
`data/C2S1A3D1.json`, `data/C3S1A3D1.json` and `data/S1A3D1_wcs.json`.

* All three camera files have the same number of entries (1205). **Entry `i` is the same
  instant in all three cameras.** That is how the views are synchronised; the image file
  numbers differ between cameras (C1 starts at 0277, C2 at 0280) because of dropped and
  duplicated video frames, so never match images by file number.
* `rec[i]` returns a **`FrameSet`** with everything for that instant:

| Field | Shape | Meaning |
|---|---|---|
| `image_paths` | 3 | e.g. `data/C1S1A3D1/C1S1A3D1_0355.jpg` (from the JSON `path`) |
| `bboxes` | (3, 4) | person box per view, `x, y, w, h` (JSON `bbx`) |
| `wand_xy` | (3, 5, 2) | the 5 wand markers in each image (JSON `wand_xy`) |
| `gt_xy` | (3, 59, 2) | Vicon ground truth projected into each image (checks only) |
| `gt_xyz_cam` | (59, 3) | ground truth in the C1 camera frame, mm / 60 (checks only) |
| `gt_xyz_wcs` | (59, 3) | ground truth in the world frame, mm (checks only) |

The JSON stores 63 Vicon channels; `SOURCE_TO_59` from
[pipeline/skeleton.py](skeleton.py) picks the 59 the models predict, in the models'
order (`KEYPOINTS_59`).

### 3.2 Build the pipeline (setup)

```python
pipe = LivePipeline(root, MODELS[args.model], rec.subject, args.checkpoint, args.device, args.units_to_mm)
```

`LivePipeline.__init__` ([pipeline/pipeline.py](pipeline.py)) does three timed setup steps.

**a) Load the model**: `PoseEstimator(...)` in [pipeline/pose2d.py](pose2d.py)

1. Reads `<model_dir>/config.yaml` (written during training) to learn the model type and input
   size: HRNet uses 576 x 768 (w x h), ViTPose 192 x 256.
2. `_build()` creates the empty network: `get_pose_model(W=48, num_joints=59)` for HRNet or
   `get_vitpose_model("B", 59, "simple")` for ViTPose.
3. `_load_weights()` loads `checkpoints/best_nmae_model.pth`. The training checkpoint also
   contains the optimizer state (that is why it is ~1 GB), so the first time it saves just the
   weights to `best_nmae_model_weights_only.pt` and uses that on later runs. It also strips a
   `module.` prefix that multi-GPU training adds to the parameter names.
4. Moves the model to the GPU, switches to eval mode, turns on `cudnn.benchmark` (cuDNN picks
   the fastest convolution algorithms for this input size) and the `channels_last` memory layout.

**b) Load the calibration**: `load_rig(root, subject)` in [pipeline/calibration.py](calibration.py)

1. Reads `calibration/external_calibration.csv`. Not every subject has its own calibration;
   `session_for("S1", ...)` picks the latest session at or before the subject (S1 -> S1,
   S5 -> S1, S12 -> S11, ...), exactly as the repo's `map_matrix` did.
2. Finds the two stereo calibrations that link the main camera (1) to camera 2 (`G12` file)
   and to camera 3 (`G13` file). Each `.npz` contains `R`, `T` (how the pair camera sits
   relative to C1) and the intrinsics `mtx_A` (C1) and `mtx_B` (the pair camera).
3. Builds one projection matrix per camera. C1 is the origin, so it has no rotation or
   translation:

   ```
   P1 = K1 [ I   | 0   ]
   P2 = K2 [ R12 | T12 ]
   P3 = K3 [ R13 | T13 ]
   ```

   A projection matrix turns a 3D point in the C1 frame into pixels in that camera.
4. Decides which cameras to triangulate with (`use_cams`), using the repo's rule: if the main
   camera's calibration `method` is 3, all three are linked, so use all three; otherwise use C1
   and the one camera it was directly calibrated with. For S1 it is all three.

**c) Warm up the GPU**: `pose.warmup()` runs the network three times on a dummy batch. The
first CUDA calls are slow (memory allocation, cuDNN autotuning), and warming up keeps that out
of the per-frame timings.

### 3.3 Set up the world frame (setup)

```python
pipe.setup_world(rec[frames[0]].wand_xy, rec.wand_wcs())
```

`LivePipeline.setup_world()` triangulates the wand and fits the transform to the world frame.
The wand is static, so this happens once.

1. **Triangulate the wand.** `triangulate(wand_xy, P)` turns the 5 wand markers seen by the 3
   cameras into 5 points in the C1 frame (the same function as for the body, see 3.5 step 5).
2. **Fit camera -> world.** `fit_world_transform()` in [pipeline/wcs.py](wcs.py)
   finds a rotation `R` and translation `t` such that

   ```
   X_world = R @ (60 * X_cam) + t        # 60 converts mm/60 to mm
   ```

   lines the triangulated wand up with the wand coordinates in `S1A3D1_wcs.json`. This is a
   least-squares rigid fit (the Kabsch / Procrustes method: centre both point sets, take the
   SVD of their cross-covariance, `R = V D Uᵀ`). Three details:

   * **Marker order differs.** The camera JSON lists the wand's short-arm end first and the
     corner second; the WCS file lists the corner first. `CAM_TO_WCS_WAND_ORDER = [1, 0, 2, 3, 4]`
     reorders them before fitting.
   * **The world frame is mirrored.** The stored Vicon frame is a mirror image of the OpenCV
     camera frame, so the right answer is a reflection (`det R = -1`), not a pure rotation.
   * **The fit alone cannot tell which.** All 5 wand markers lie on the floor, in one plane,
     so a mirrored and an unmirrored solution fit them almost equally well. The code computes
     both and keeps the one where the camera's "down" axis (its +y) points down in the world
     (negative world y). With the wrong choice the skeleton would come out upside down.

   The fit residual (about 2.5 mm per marker) is printed and stored in `pipe.world.residual_mm`.

### 3.4 The frame loop

```python
pending = pipe.prefetch(rec[frames[0]].image_paths)
for n, i in enumerate(frames):
    images = pending
    pending = pipe.prefetch(rec[frames[n + 1]].image_paths)   # start reading the NEXT frame now
    res = pipe.process(images, fs.bboxes, keep_debug=do_viz)
```

**Prefetching:** `prefetch()` hands the three JPEG paths to a thread pool, which reads and
decodes them in the background while the GPU works on the current frame. When `process()`
needs the images they are usually already decoded, so step 1 only waits about 0.1 ms instead of
about 55 ms.

After `process()`, the loop compares the result with the ground truth (2D error per view,
3D error in the camera frame and in the world frame), stores it, and, every `--viz-every`
frames, calls `visualise()` to save the six step plots.

### 3.5 Inside `LivePipeline.process()`: the per-frame steps

Each step is wrapped in `with t.step(name):`, which times it (see section 4). The step
names below match the timing report.

**Step 1: read images (wait)**
Collects the three prefetched images (RGB, `uint8`, 1800 x 1150 x 3). It also records how long
the background decoding took, as the "bg: JPEG read+decode" line in the report (not counted in
the frame time because it overlaps with GPU work). If you pass paths or arrays instead of
prefetched images, `process()` reads them here.

**Step 2a: crop to person box**: `PoseEstimator.crop()`
For each view, `adjust_box()` (your repo's `adjust_aspect_ratio`) pads the JSON box by 2 px and
grows it to the 3:4 (width:height) shape the models were trained on, then clips it to the image.
The image is sliced to that box. Slicing a NumPy array creates no copy, which is why this step
takes almost no time.

**Step 2b: resize to model input**: `PoseEstimator.resize()`
`cv2.resize` scales each crop to the model input (HRNet 576 x 768, ViTPose 192 x 256) and the
three are stacked into one `(3, H, W, 3)` `uint8` batch. Only the crop is resized, not the whole
image.

**Step 3a: upload to GPU + normalise**: `PoseEstimator.upload()`
Copies the `uint8` batch to the GPU, reorders it to `(3, 3, H, W)` and divides by 255. This is
the same input the models were trained with (RGB in [0, 1], no mean/std normalisation).
Converting on the GPU moves 4x less data than converting to float on the CPU first.

**Step 3b: network forward**: `PoseEstimator.forward()`
One forward pass for **all three views at once** (batch of 3). The output is 59 heatmaps per
view, each a small image whose brightest spot marks one keypoint: 144 x 192 for HRNet, 48 x 64
for ViTPose (a quarter of the input size). This step is about 95% of the frame time.

**Step 4a: decode heatmaps (GPU)**: `decode_heatmaps()` (your repo's `get_keypoints_from_heatmaps`)
For every heatmap:
1. find the maximum (the peak) and the second-highest value;
2. move the peak a quarter of the way towards the second-highest value. This gives sub-pixel
   precision on the coarse heatmap grid;
3. scale from heatmap pixels to model-input pixels.

The peak value is kept as a confidence score (`conf`). This runs on the GPU for all 3 x 59
heatmaps at once.

**Step 4b: copy to CPU + map to image**: `PoseEstimator.to_image()`
Copies the (3, 59, 2) keypoints to the CPU and undoes steps 2a-2b: multiply by
`box size / input size` (undo the resize) and add the box's top-left corner (undo the crop).
The keypoints are now in full-image pixels, `kp2d`.

**Step 5: triangulate (DLT)**: `triangulate()` in [pipeline/triangulation.py](triangulation.py)
Turns the 2D keypoints from the three views into 3D, one point at a time conceptually:

* A camera with projection matrix `P` sees a 3D point `X` at pixel `(u, v)`. Each such
  observation gives two linear equations in `X`:

  ```
  (v · P[2] − P[1]) · X = 0
  (P[0] − u · P[2]) · X = 0
  ```

* Three cameras give six equations, stacked into a matrix `A` (6 x 4). The best `X` is the
  singular vector of `AᵀA` with the smallest singular value, divided by its 4th component.

These are the same equations as `DLT_3_cameras` in your repo. The difference is that this
version stacks all 59 keypoints into one `(59, 4, 4)` array and runs a single batched SVD
instead of a Python loop. The result `xyz_cam` is (59, 3) in the C1 frame, in mm / 60.

**Step 6: to world frame**: `WorldTransform.apply()`
`X_world = R @ (60 * X_cam) + t` with the `R`, `t` fitted during setup. The result `xyz_wcs`
is (59, 3) in mm in the Vicon world frame.

**Quality check (not timed):** `reprojection_error()` projects the 3D points back into each
camera and measures the distance to the 2D keypoints (about 4.5 px for HRNet). If the three
views disagree (a wrong detection in one view), this number jumps.

`process()` returns a **`FrameResult`** with `kp2d`, `conf`, `xyz_cam`, `xyz_wcs` and
`reproj_px`. With `keep_debug=True` it also keeps the images, crops, heatmaps and reprojected
points for the plots.

### 3.6 Writing the outputs

After the loop, `main()` writes `predictions.npz`, `metrics.csv`, the timing report
(`timing.txt`, `timing.csv`, `timing_breakdown.png`) and `summary.png`.

---

## 4. How the timing works

[pipeline/timing.py](timing.py), class `StepTimer`.

```python
with timer.step("frame: 3b network forward"):
    heatmaps = pose.forward(x)
```

* `step()` records the wall-clock time of the block. GPU calls in PyTorch return immediately
  and run in the background, so without care a GPU step would look instant and the next CPU step
  would absorb its time. When running on CUDA, `step()` therefore calls
  `torch.cuda.synchronize()` before and after the block, so each step is charged for its own GPU
  work. This costs a little throughput (about 2 ms per frame set) in exchange for honest numbers.
* `add()` records a time measured somewhere else, used for the background JPEG decoding, which is
  timed inside the worker threads.
* `report()` groups steps into **setup**, **per frame set** (with the stages from `STAGES` in
  `pipeline.py`, a subtotal per stage, and each step's share of the frame time) and **not counted**
  (background decoding and plots).
* The per-frame total is the sum of the per-frame steps' means; frame sets per second is
  `1000 / total`.

---

## 5. The visual checks

[pipeline/viz.py](viz.py) has one function per step. Each saves a PNG into
`outputs/<run>/frame_XXXX/`. Red is the prediction, green the Vicon ground truth.

| Plot | Function | What to look for |
|---|---|---|
| `step1_inputs.png` | `plot_inputs` | yellow JSON box and cyan model crop should contain the whole person |
| `step2_crops_heatmaps.png` | `plot_crops_heatmaps` | one compact blob per keypoint; decoded points on the blobs |
| `step3_pose2d.png` | `plot_pose_2d` | red skeleton on top of the green one; per-view error in the title |
| `step4_triangulation.png` | `plot_triangulation` | 3D skeleton matches GT; purple reprojected points sit on the red 2D points |
| `step5_wand.png` | `plot_wand` | reprojected wand dots inside the blue circles; triangulated wand on the JSON crosses |
| `step6_world.png` | `plot_world` | upright skeleton standing on the floor (y = 0) at the wand; red over green |
| `timing_breakdown.png` | `plot_timing` | which step dominates the frame time |
| `summary.png` | `plot_summary` | errors over the frames of a run |

Two small helpers turn 3D coordinates into plot axes so that "up" is up on screen:
`_cam_view` (camera frame: plots x, z, −y) and `_wcs_view` (world frame: plots x, z, y).

---

## 6. Checks that the geometry is right

Things that were verified while building this, and how to repeat them:

* **Calibration and DLT match the data exactly.** Triangulating the *ground-truth* 2D points
  (`gt_xy`) gives back the JSON `xyz` with 0.0 mm error and 0 px reprojection error. So the
  projection matrices and the triangulation are the ones the dataset was made with.
* **The wand fit.** The triangulated wand matches the JSON `wand_xyz` exactly, and the fit to the
  world frame leaves about 2.5 mm per marker.
* **The world transform.** Applying it to the triangulated ground truth lands about 12 mm from the
  Vicon world coordinates. That 12 mm is a floor on the world-frame error of any prediction, and
  is why the world-frame MPJPE is a few mm higher than the camera-frame one.

---

## 7. Using the pipeline from your own code

```python
from pipeline import LivePipeline

pipe = LivePipeline(root=".", model_dir="w48_768x576_all_63", subject="S1")
pipe.setup_world(wand_xy, wand_wcs)       # (3, 5, 2) wand pixels, (5, 3) wand world coords in mm

for images, bboxes in camera_stream():    # 3 RGB arrays (C1 first) and (3, 4) person boxes
    out = pipe.process(images, bboxes)
    send(out.xyz_wcs)                     # (59, 3) mm, world frame

print(pipe.timer.report("timing"))
```

What you have to provide yourself in a live setting:

* **Synchronised frames** from C1, C2 and C3 (C1 first).
* **A person box per view.** Here they come from the JSON; live you would use a person detector,
  or the box around the previous frame's keypoints.
* **The wand**: its 2D markers in each view once, and its world coordinates.

---

## 8. Where to change things

| You want to... | Change |
|---|---|
| use the other checkpoint | `--checkpoint best_model.pth` (any file in `<model>/checkpoints/`) |
| use another subject / trial | `--trial S11A2D1` (the subject number picks the calibration) |
| use cameras 6/4/5 instead of 1/2/3 | `--cams C6,C4,C5`, and in `LivePipeline.__init__` call `load_rig(root, subject, main=6, pairs=(4, 5))` |
| correct lens distortion | undistort `kp2d` with `cv2.undistortPoints` (coefficients are in the `.npz`) before step 5 |
| add a step to the timing | wrap it in `with self.timer.step("frame: ..."):` and add the name to `STAGES` |
| change what is drawn | `SKELETON_EDGES` in `skeleton.py`, or the functions in `viz.py` |
