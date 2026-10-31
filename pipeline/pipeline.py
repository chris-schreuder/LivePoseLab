"""The full pipeline: 3 synchronised images -> 59 keypoints in 2D -> 3D (main camera) -> 3D (Vicon world).

    pipe = LivePipeline(root=".", model_dir="w48_768x576_all_63", subject="S1")
    pipe.setup_world(wand_xy, wand_wcs)          # once: wand 2D (3, 5, 2) and WCS wand (5, 3) mm
    out = pipe.process(image_paths, bboxes)      # per frame set
    out.xyz_wcs                                  # (59, 3) mm in the world frame

Every step is timed with ``pipe.timer`` (CUDA-synchronised). ``process(..., keep_debug=True)``
also returns the images, crops and heatmaps that the visual checks need.
"""
from dataclasses import dataclass, field

import numpy as np

from .calibration import load_rig
from .pose2d import PoseEstimator
from .timing import StepTimer
from .triangulation import project, reprojection_error, triangulate
from .wcs import fit_world_transform

# Per-frame steps, in order, grouped into stages for the timing breakdown.
STAGES = {
    "1 read images": ["frame: 1 read images (wait)"],
    "2 preprocess": ["frame: 2a crop to person box", "frame: 2b resize to model input"],
    "3 network": ["frame: 3a upload to GPU + normalise", "frame: 3b network forward"],
    "4 postprocess": ["frame: 4a decode heatmaps (GPU)", "frame: 4b copy to CPU + map to image"],
    "5 3D": ["frame: 5 triangulate (DLT)", "frame: 6 to world frame"],
}
FRAME_STEPS = [s for steps in STAGES.values() for s in steps]
BACKGROUND_READ = "bg: JPEG read+decode (3 imgs, overlapped)"


@dataclass
class FrameResult:
    kp2d: np.ndarray            # (3, 59, 2) pixels, full image
    conf: np.ndarray            # (3, 59) heatmap peak value
    xyz_cam: np.ndarray         # (59, 3) main-camera frame, calibration units
    xyz_wcs: np.ndarray         # (59, 3) world frame, mm (NaN until setup_world is called)
    reproj_px: np.ndarray       # (3,) mean reprojection error per view
    debug: dict = field(default_factory=dict)


class LivePipeline:
    def __init__(self, root, model_dir, subject, checkpoint="best_nmae_model.pth", device="cuda",
                 calib_units_to_mm=60.0):
        self.root = root
        self.units_to_mm = calib_units_to_mm
        self.world = None
        with_cuda = device == "cuda"
        self.timer = StepTimer(cuda=False)
        with self.timer.step("setup: load model"):
            self.pose = PoseEstimator(f"{root}/{model_dir}", checkpoint, device)
        self.timer.cuda = with_cuda and self.pose.device.type == "cuda"
        with self.timer.step("setup: load calibration"):
            self.rig = load_rig(root, subject)
        self.P = self.rig.P[self.rig.use_cams]
        with self.timer.step("setup: GPU warm-up"):
            self.pose.warmup()

    def setup_world(self, wand_xy, wand_wcs):
        """Triangulate the wand and fit the camera -> world transform. Returns the wand in camera units."""
        with self.timer.step("setup: wand triangulation + WCS fit"):
            wand_cam = triangulate(wand_xy[self.rig.use_cams], self.P)
            self.world = fit_world_transform(wand_cam, wand_wcs, self.units_to_mm)
        self.wand_cam = wand_cam
        self.wand_reproj = project(wand_cam, self.rig.P)
        return wand_cam

    def prefetch(self, image_paths):
        """Decode the next frame set's images in the background while the GPU is busy."""
        return self.pose.read_images_async(image_paths)

    def process(self, images, bboxes, keep_debug=False):
        """images: 3 image paths, 3 RGB arrays, or the result of ``prefetch``; bboxes: (3, 4) x, y, w, h."""
        t, pose, s = self.timer, self.pose, FRAME_STEPS
        with t.step(s[0]):
            if hasattr(images[0], "result"):          # prefetched futures: wait for decoding to finish
                done = [f.result() for f in images]
                images = [img for img, _ in done]
                t.add(BACKGROUND_READ, sum(sec for _, sec in done))
            elif not isinstance(images[0], np.ndarray):
                images = pose.read_images(images)
        with t.step(s[1]):
            crops, boxes = pose.crop(images, bboxes)
        with t.step(s[2]):
            batch = pose.resize(crops)
        with t.step(s[3]):
            x = pose.upload(batch)
        with t.step(s[4]):
            heatmaps = pose.forward(x)
        with t.step(s[5]):
            kp_dev, conf_dev = pose.decode_on_device(heatmaps)
        with t.step(s[6]):
            kp2d, conf = pose.to_image(kp_dev, conf_dev, boxes)
        with t.step(s[7]):
            xyz_cam = triangulate(kp2d[self.rig.use_cams], self.P)
        with t.step(s[8]):
            xyz_wcs = self.world.apply(xyz_cam) if self.world else np.full_like(xyz_cam, np.nan)
        crops = batch

        result = FrameResult(kp2d=kp2d, conf=conf, xyz_cam=xyz_cam, xyz_wcs=xyz_wcs,
                             reproj_px=reprojection_error(kp2d, xyz_cam, self.rig.P))
        if keep_debug:
            scale = boxes[:, None, 2:4] / np.array([self.pose.input_w, self.pose.input_h])
            result.debug = {
                "images": images, "crops": crops, "boxes": boxes,
                "heatmaps": heatmaps.float().cpu().numpy(),
                "kp_crop": (kp2d - boxes[:, None, 0:2]) / scale,
                "reproj": project(xyz_cam, self.rig.P),
            }
        return result
