"""YOLO26 person detector: replaces the JSON bounding boxes with live detections.

For each camera view it returns the box of the most confident person as (x, y, w, h) in image
pixels, the format the pose crop expects. Built for speed rather than through the Ultralytics
``predict`` API, whose Python overhead tripled the time per frame set:

* the 3 views are resized on the CPU to 640 px on the long side (aspect kept, padded to a multiple
  of 32, e.g. 1800x1150 -> 640x416) into a pinned buffer, and uploaded as one batch;
* the network (BatchNorm fused into the convolutions, channels-last) runs on its own CUDA stream,
  with the same autocast precision as the pose network;
* only the single best person box per view is needed, so instead of non-maximum suppression the
  highest person score is taken on the GPU, and 3 x 5 numbers are copied back.

If a view has no person above ``conf``, its previous box is kept (or the whole image at the start)
and the miss is counted.

The weights (e.g. ``yolo26s.pt``) are downloaded by Ultralytics on first use if not found.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

PRECISION = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class PersonDetector:
    def __init__(self, weights="yolo26s.pt", device="cuda", imgsz=640, precision="fp32", conf=0.25):
        from ultralytics import YOLO
        self.name = Path(weights).stem.upper()
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        cuda = self.device.type == "cuda"
        net = YOLO(weights).model.fuse().to(self.device).eval()
        self.net = net.to(memory_format=torch.channels_last) if cuda else net
        self.imgsz, self.conf = imgsz, conf
        self.amp_dtype = PRECISION[precision] if cuda else None
        self.stream = torch.cuda.Stream(self.device) if cuda else None
        self._shape = None
        self.reset()

    def reset(self):
        self.prev = None          # last good box per view
        self.misses = 0           # views with no person above ``conf``

    def _setup(self, h, w, n):
        """Input geometry for images of size (h, w): resized size, padded size, pinned buffer."""
        self.scale = self.imgsz / max(h, w)
        self.nh, self.nw = round(h * self.scale), round(w * self.scale)
        ph, pw = -(-self.nh // 32) * 32, -(-self.nw // 32) * 32
        buf = torch.full((n, ph, pw, 3), 114, dtype=torch.uint8)
        self._buf = buf.pin_memory() if self.device.type == "cuda" else buf
        self._np = self._buf.numpy()
        self._shape = (h, w, n)

    @torch.inference_mode()
    def __call__(self, images):
        """images: list of RGB uint8 arrays (one per view). Returns (boxes (n, 4) x, y, w, h, conf (n,))."""
        h, w = images[0].shape[:2]
        if self._shape != (h, w, len(images)):
            self._setup(h, w, len(images))
        for k, im in enumerate(images):
            self._np[k, :self.nh, :self.nw] = cv2.resize(im, (self.nw, self.nh), interpolation=cv2.INTER_LINEAR)
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else torch.inference_mode()
        with ctx:
            x = self._buf.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
            if self.device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            if self.amp_dtype is not None:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    out = self.net(x)[0]
            else:
                out = self.net(x)[0]                       # (n, 4 + 80, anchors): cx, cy, w, h, class scores
            best = out[:, 4].argmax(1)                     # class 0 = person
            sel = out[torch.arange(len(images), device=out.device), :5, best].float().cpu().numpy()
        cx, cy, bw, bh, conf = sel.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1) / self.scale
        boxes[:, :2] = np.maximum(boxes[:, :2], 0)
        for k in range(len(images)):
            if conf[k] < self.conf:
                self.misses += 1
                boxes[k] = self.prev[k] if self.prev is not None else (0, 0, w, h)
        self.prev = boxes
        return boxes, conf


def box_iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)
