# 1. Pipeline: one frame set, step by step

The core of LivePoseLab. It takes three synchronised images (C1 = main camera, C2, C3), predicts the
59 SUMediPose keypoints in each with the fine-tuned **HRNet-W48** or **ViTPose-B**, triangulates them
to 3D in the main-camera frame, and moves them into the **Vicon world frame (WCS)** using the
calibration wand. Every step can be plotted, and every step is timed.

The methods are the ones from
[Markerless-Biomechanical-Analysis](https://github.com/chris-schreuder/Markerless-Biomechanical-Analysis):
the same crop and resize, heatmap decoding, calibration files and DLT triangulation, rewritten as a
small package and sped up.

This part processes frames one after another, as fast as it can, with the person boxes from the
camera JSON files. For a live 60 fps stream with a person detector, LivePoseFormer and joint angles,
see the [stream simulator](../stream_sim/README.md). For the code in the order it runs, see
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

## Quick start

From the repository folder (installation in the [main README](../README.md#installation)):

```bash
python run_pipeline.py                                   # HRNet, frame 300, all step plots
python run_pipeline.py --model vitpose --frames 300      # ViTPose
python run_pipeline.py --frames 0:1205:5 --viz-every 40  # many frames, plots every 40th
python run_pipeline.py --frames 200:400 --no-viz         # timing run
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `hrnet` | `hrnet` (W48, 768×576) or `vitpose` (ViT-B, 256×192) |
| `--frames` | `300` | one index, `a:b` or `a:b:step`; index `i` is the same instant in all three camera JSON files |
| `--trial` | `S1A3D1` | recording in `data/` |
| `--viz-every` | 1 | plot every N-th processed frame |
| `--no-viz` | off | no plots (timing and metrics only) |
| `--no-prefetch` | off | read images in the main thread instead of a background thread |
| `--cams` | `C1,C2,C3` | cameras, main camera first |
| `--checkpoint` | `best_nmae_model.pth` | checkpoint file in the model's `checkpoints/` folder |
| `--units-to-mm` | 60 | calibration units to mm |
| `--out` | `outputs` | output folder |

The first run of each model writes a slim `*_weights_only.pt` next to the checkpoint (the training
checkpoint also holds optimizer state), so later start-ups are faster.

## Steps and their plots

Each processed frame gets `outputs/<model>_<frames>/frame_XXXX/`:

| Step | What happens | Plot |
|---|---|---|
| 1 | Read the 3 images; pad each JSON person box by 2 px and grow it to 3:4 | `step1_inputs.png`: images with the JSON box and the model crop |
| 2 | Crop, resize to the model input (HRNet 576×768, ViTPose 192×256), run the network on all 3 views as one batch, decode heatmaps (peak + quarter shift to the 2nd peak) | `step2_crops_heatmaps.png`: network input with decoded keypoints, heatmaps |
| 3 | Map keypoints back to full-image pixels | `step3_pose2d.png`: prediction (red) vs projected Vicon ground truth (green), with mean px error |
| 4 | DLT triangulation with `P1 = K1[I|0]`, `P2 = K2[R12|T12]`, `P3 = K3[R13|T13]` | `step4_triangulation.png`: 3D in the C1 frame vs ground truth, and the 3D points reprojected into each view |
| 5 | Triangulate the 5 wand markers from `wand_xy` | `step5_wand.png`: wand 2D and reprojection per view, 3D wand vs the JSON `wand_xyz` |
| 6 | Fit camera → WCS with the wand (once), apply to the keypoints | `step6_world.png`: prediction vs Vicon ground truth in the WCS, floor, axes and wand |

The run folder also has:

| File | Content |
|---|---|
| `timing_breakdown.png`, `timing.txt`, `timing.csv` | time per step (written on every run, even with `--no-viz`) |
| `summary.png` | errors over frames |
| `metrics.csv` | per-frame errors |
| `predictions.npz` | `kp2d`, `conf`, `xyz_cam_mm`, `xyz_wcs_mm` per frame |

## Use it in your own code

```python
from pipeline import LivePipeline

pipe = LivePipeline(root=".", model_dir="ViT_B_simple_all_63", subject="S1")
pipe.setup_world(wand_xy, wand_wcs)          # once: (3, 5, 2) wand pixels, (5, 3) wand in WCS (mm)
out = pipe.process(images, bboxes)           # 3 RGB arrays or paths (main camera first), (3, 4) boxes
out.kp2d      # (3, 59, 2) pixels
out.xyz_cam   # (59, 3) C1 frame, calibration units (x 60 = mm)
out.xyz_wcs   # (59, 3) world frame, mm
print(pipe.timer.report("timing"))
```

The keypoint order is `KEYPOINTS_59` in `pipeline/skeleton.py`.

## Results on S1A3D1 (frames 200–399, GTX 1660, 6 GB, fp32)

| | HRNet-W48 768×576 | ViTPose-B 256×192 |
|---|---|---|
| 2D error per view (px) | 8.3 / 10.1 / 9.2 | 12.1 / 14.0 / 15.5 |
| Reprojection error (px) | 4.5 | 5.5 |
| MPJPE, camera frame | **16.1 mm** | 25.9 mm |
| MPJPE, world frame | 21.2 mm | 33.1 mm |
| Network forward (3 images) | 338 ms | 61 ms |
| Whole pipeline per frame set | 346 ms (2.9 / s) | **66 ms (15.2 / s)** |

On an RTX 3090 with bf16 and `torch.compile` the pose network takes about 5 ms (ViTPose) and 32 ms
(HRNet) per frame set; see the [evaluation](../realtime_eval/README.md).

The world-frame error includes about 12 mm from the wand fit itself: triangulating the
*ground-truth* 2D points and applying the same transform lands 11.9 mm from the Vicon WCS values (the
wand markers fit with 2.5 mm residual).

### Time per step

Mean ms per frame set over 200 frame sets, and the share of the frame time:

| Stage / step | HRNet ms | HRNet % | ViTPose ms | ViTPose % |
|---|---:|---:|---:|---:|
| **1 read images** (wait for the background decode) | 0.12 | 0.0 | 0.12 | 0.2 |
| **2 preprocess** | 5.05 | 1.5 | 1.22 | 1.9 |
| &nbsp;&nbsp; 2a crop to person box | 0.05 | 0.0 | 0.06 | 0.1 |
| &nbsp;&nbsp; 2b resize to model input | 5.00 | 1.4 | 1.16 | 1.8 |
| **3 network** | 337.6 | 97.6 | 62.6 | 95.4 |
| &nbsp;&nbsp; 3a upload to GPU + normalise | 2.27 | 0.7 | 0.62 | 0.9 |
| &nbsp;&nbsp; 3b network forward | 335.4 | 97.0 | 62.0 | 94.5 |
| **4 postprocess** | 2.55 | 0.7 | 1.22 | 1.9 |
| &nbsp;&nbsp; 4a decode heatmaps (GPU) | 2.26 | 0.7 | 0.96 | 1.5 |
| &nbsp;&nbsp; 4b copy to CPU + map to image | 0.29 | 0.1 | 0.26 | 0.4 |
| **5 3D** | 0.44 | 0.1 | 0.44 | 0.7 |
| &nbsp;&nbsp; 5 triangulate (DLT) | 0.41 | 0.1 | 0.41 | 0.6 |
| &nbsp;&nbsp; 6 to world frame | 0.03 | 0.0 | 0.03 | 0.1 |
| **Total per frame set** | **345.8** | | **65.8** | |

Not counted in the frame time:

* **Background image decoding:** reading and decoding the three JPEGs takes about 55–60 ms (summed
  over the three parallel threads), but it runs while the GPU processes the previous frame, so the
  pipeline only waits about 0.1 ms for it.
* **Setup, once:** load model 1.4–4.7 s (the first run also writes the slim weights file), load
  calibration 11 ms, GPU warm-up 0.6–2.4 s, wand triangulation + world fit under 1 ms.
* **Plots:** 0.7–1.3 s each when enabled.

Each step synchronises the GPU before and after, so GPU work is charged to the step that launched it.

## Coordinate frames and units

* **Camera frame:** OpenCV convention with C1 at the origin (x right, y down, z forward). Calibration
  and JSON camera coordinates are in **mm / 60**; the pipeline reports mm (`--units-to-mm 60`).
* **World frame (WCS):** the Vicon frame from `<trial>_wcs.json`, in mm, y up, floor at y = 0.
* **Camera → WCS:** `X_wcs = R (60 X_cam) + t`, fitted to the 5 wand markers (`pipeline/wcs.py`).
  Two points matter:
  * The camera JSON lists the wand's short-arm end first and the corner second; the WCS file lists
    the corner first. The fit reorders them.
  * The stored WCS is a mirror image of the camera frame, so R is a reflection (det R = −1). The wand
    markers are coplanar, so the fit alone cannot choose between the mirrored and the proper
    solution. The rule used is physical: the camera's "down" axis must point down in the WCS.

## Notes

* **Person boxes** come from the JSON `bbx`, as in the original evaluation. The stream simulator, the
  live demo and the evaluation use the YOLO26 person detector instead.
* **Lens distortion** is not corrected before triangulation, matching the original repository. The
  calibration files contain distortion coefficients if you want to add `cv2.undistortPoints`.
* **Precision:** this runner uses fp32. On the GTX 1660 (no tensor cores) fp16 was 3–4× slower and
  HRNet overflowed; on RTX 30xx cards the stream simulator and evaluation use bf16.
* **Calibration:** subjects without their own session use the latest earlier session in
  `calibration/external_calibration.csv`. If the main camera's calibration method is not 3, only C1
  and its directly linked pair are used (as in the original repository).

## Files

```
run_pipeline.py          command-line runner (plots, timing, metrics)
pipeline/
  pipeline.py            LivePipeline: the steps, timed
  pose2d.py              model loading, crop/resize, batched inference, heatmap decoding
  calibration.py         rig / projection matrices from external_calibration.csv
  triangulation.py       batched DLT, reprojection
  wcs.py                 wand-based camera -> world transform
  data.py                reading the synchronised camera JSONs and frames
  viz.py                 the step plots
  timing.py              CUDA-synchronised step timer and report
  skeleton.py            keypoint names, skeleton edges
  models/                HRNet and ViTPose definitions (from Markerless-Biomechanical-Analysis)
```
