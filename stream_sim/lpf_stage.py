"""Optional last stage: LivePoseFormer smoothing / imputation of the world-frame 3D keypoints.

Uses ``StreamingRestorer`` from the LivePoseFormer repository (pass its folder with --lpf-repo).
The restorer holds each frame until 3 future frames have arrived, so frame k is released when
frame k + 3 is pushed. ``push`` returns the released frames with their stream sequence numbers,
which lets the report measure the delay of the full live chain and the error after smoothing.

On a GPU the stage runs on its own high-priority CUDA stream and, by default, replays its step as
a CUDA graph once the cache is full (see lpf_graph.py), so it costs little when it shares the GPU
with the pose network.

Without a checkpoint the model has random weights: the timing is real, the output is not.
"""
import sys
from pathlib import Path

import torch


class LpfStage:
    def __init__(self, repo, checkpoint=None, device="cuda", num_joints=59, use_graph=True):
        sys.path.insert(0, str(Path(repo).resolve()))
        from livepose import LivePoseFormer, StreamingRestorer   # noqa: E402

        self.num_joints = num_joints
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model = LivePoseFormer(num_joints)
        self.has_weights = checkpoint is not None
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("model_state_dict", state))
        self.model = model.to(self.device).eval()
        # Own high-priority CUDA stream: this stage does not wait for the pose network running in another
        # thread, and its kernels are scheduled ahead of the network's.
        self.stream = torch.cuda.Stream(self.device, priority=-1) if self.device.type == "cuda" else None
        self.use_graph = use_graph and self.device.type == "cuda"
        if self.use_graph:
            from .lpf_graph import GraphedRestorer
            self.restorer = GraphedRestorer(self.model, device=self.device, stream=self.stream)
        else:
            self.restorer = StreamingRestorer(self.model, device=self.device)
        self.reset()

    @property
    def graph_check_mm(self):
        return getattr(self.restorer, "graph_check_mm", None)

    def reset(self):
        """Start a new stream. The restorer (and its CUDA graph, if built) is reused."""
        self.restorer.reset()
        self.pushed = []          # stream seq of every frame pushed, in order
        self.released = 0

    def _pair(self, out):
        """Match released frames to the seqs they belong to: [(seq, (59, 3) smoothed mm), ...]."""
        seqs = self.pushed[self.released:self.released + len(out)]
        self.released += len(out)
        return list(zip(seqs, out))

    def push(self, seq, xyz_wcs_mm):
        """Push one frame (59, 3) mm; returns the frames released by this push as (seq, xyz) pairs."""
        self.pushed.append(seq)
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                out = self.restorer.push(xyz_wcs_mm.astype("float32"))
        else:
            out = self.restorer.push(xyz_wcs_mm.astype("float32"))
        return self._pair(out)

    def flush(self):
        """End of stream: release the last frames still waiting for their lookahead."""
        return self._pair(self.restorer.flush())
