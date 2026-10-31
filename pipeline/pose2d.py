"""2D dense keypoint prediction with the fine-tuned HRNet or ViTPose (59 keypoints).

Pre- and post-processing follow the repo (training/SUMediPose_dataset.py,
evaluation/eval_manual.py, error_3d/error_calc_3d.py):

1. Pad the person box by 2 px and grow it to a 3:4 (w:h) aspect ratio (``adjust_box``).
2. Crop the RGB image to it, resize to the model input size, scale to [0, 1].
3. Decode each heatmap at its maximum, shifted a quarter of the way towards the second
   highest value (``decode_heatmaps``), in crop-pixel units.
4. Undo the resize and add the crop origin to get full-image pixel coordinates.

Speed-ups over the repo's evaluation loop: the three views go through the network as one
batch; images are read in parallel threads; only the crop is resized; the uint8 -> float
conversion and the heatmap decoding run on the GPU; and a slim
weights-only copy of the 1 GB training checkpoint is cached for fast start-up.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def adjust_box(x, y, w, h, img_w, img_h, padding=2, aspect=0.75):
    """Repo's adjust_aspect_ratio: pad, grow to the target aspect ratio, clip to the image."""
    x, y = max(x - padding, 0), max(y - padding, 0)
    w, h = min(w + 2 * padding, img_w - x), min(h + 2 * padding, img_h - y)
    if w / h > aspect:
        new_h = w / aspect
        y, h = y - (new_h - h) / 2, new_h
    else:
        new_w = h * aspect
        x, w = x - (new_w - w) / 2, new_w
    x, y = max(x, 0), max(y, 0)
    return x, y, min(w, img_w - x), min(h, img_h - y)


def decode_heatmaps(heatmaps, out_h, out_w):
    """Repo's get_keypoints_from_heatmaps. heatmaps: (B, J, h, w) -> keypoints (B, J, 2), peak value (B, J)."""
    B, J, h, w = heatmaps.shape
    flat = heatmaps.reshape(B, J, -1).float()
    peak, idx = flat.max(dim=-1)
    second = flat.scatter(-1, idx.unsqueeze(-1), -float("inf")).argmax(dim=-1)
    x, y = (idx % w).float(), (idx // w).float()
    x2, y2 = (second % w).float(), (second // w).float()
    x = (x + 0.25 * (x2 - x)) / w * out_w
    y = (y + 0.25 * (y2 - y)) / h * out_h
    return torch.stack([x, y], dim=-1), peak


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class PoseEstimator:
    """``PoseEstimator("w48_768x576_all_63")`` or ``PoseEstimator("ViT_B_simple_all_63")``."""

    def __init__(self, model_dir, checkpoint="best_nmae_model.pth", device="cuda"):
        self.model_dir = Path(model_dir)
        cfg = yaml.safe_load(open(self.model_dir / "config.yaml"))
        self.cfg = cfg
        self.input_w, self.input_h = cfg["dataset"]["preprocess"]["input_size"]
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.name = cfg["model"]["name"]
        self.model = self._build(cfg["model"])
        self.model.load_state_dict(self._load_weights(checkpoint))
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            self.model.to(memory_format=torch.channels_last)
        self._pool = ThreadPoolExecutor(max_workers=3)

    @staticmethod
    def _build(m):
        if m["name"] == "HRNet":
            from .models.hrnet import get_pose_model
            return get_pose_model(m["W"], m["num_joints"])
        if m["name"] == "ViTPose":
            from .models.vitpose import get_vitpose_model
            return get_vitpose_model(variant=m["variant"], num_joints=m["num_joints"], decoder=m["decoder"])
        raise ValueError(f"Unknown model {m['name']}")

    def _load_weights(self, checkpoint):
        ckpt_path = self.model_dir / "checkpoints" / checkpoint
        slim_path = ckpt_path.with_name(ckpt_path.stem + "_weights_only.pt")
        if slim_path.exists() and slim_path.stat().st_mtime >= ckpt_path.stat().st_mtime:
            return torch.load(slim_path, map_location="cpu", weights_only=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
        torch.save(state, slim_path)   # the training checkpoint also stores optimizer state
        return state

    def read_images(self, paths):
        return list(self._pool.map(read_rgb, paths))

    def read_images_async(self, paths):
        """Start decoding the next frame set in the background; pass the result to ``process``.
        Each future returns (image, seconds spent reading + decoding it)."""
        def timed_read(p):
            t0 = time.perf_counter()
            img = read_rgb(p)
            return img, time.perf_counter() - t0
        return [self._pool.submit(timed_read, p) for p in paths]

    # The per-frame work is split into small steps so each one can be timed on its own.

    def crop(self, images, bboxes):
        """Adjust each person box to 3:4 and cut it out. Returns (3 crops, boxes (3, 4))."""
        crops, boxes = [], []
        for img, (x, y, w, h) in zip(images, bboxes):
            ih, iw = img.shape[:2]
            ax, ay, aw, ah = adjust_box(x, y, w, h, iw, ih)
            crops.append(img[int(ay):int(ay + ah), int(ax):int(ax + aw)])
            boxes.append((ax, ay, aw, ah))
        return crops, np.array(boxes)

    def resize(self, crops):
        """Resize the crops to the model input. Returns a (3, H, W, 3) uint8 batch."""
        return np.stack([cv2.resize(c, (self.input_w, self.input_h)) for c in crops])

    @torch.inference_mode()
    def upload(self, batch_uint8):
        """Copy the batch to the GPU and scale it to [0, 1] there (cheaper than on the CPU)."""
        x = torch.from_numpy(batch_uint8).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if self.device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    @torch.inference_mode()
    def forward(self, x):
        """Network forward pass. Returns heatmaps (3, J, h, w) on the device."""
        return self.model(x)

    @torch.inference_mode()
    def decode_on_device(self, heatmaps):
        """Heatmap peaks -> crop-pixel keypoints (3, J, 2) and peak values (3, J), still on the device."""
        return decode_heatmaps(heatmaps, self.input_h, self.input_w)

    def to_image(self, kp, conf, boxes):
        """Copy to the CPU and map crop pixels to full-image pixels."""
        kp, conf = kp.cpu().numpy(), conf.cpu().numpy()
        scale = boxes[:, None, 2:4] / np.array([self.input_w, self.input_h])
        return kp * scale + boxes[:, None, 0:2], conf

    def prepare(self, images, bboxes):
        crops, boxes = self.crop(images, bboxes)
        return self.resize(crops), boxes

    def infer(self, batch_uint8):
        return self.forward(self.upload(batch_uint8))

    def decode(self, heatmaps, boxes):
        return self.to_image(*self.decode_on_device(heatmaps), boxes)

    def warmup(self, n=3):
        dummy = np.zeros((3, self.input_h, self.input_w, 3), np.uint8)
        for _ in range(n):
            self.infer(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
