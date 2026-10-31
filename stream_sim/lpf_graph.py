"""CUDA-graph version of LivePoseFormer's streaming step, for when it shares the GPU.

Each LivePoseFormer frame launches hundreds of small GPU kernels from Python. Next to a heavy
network on the same GPU (e.g. HRNet), every launch queues behind the network's work and the
Python thread competes with the network's thread, so a ~3 ms step stretched to ~16 ms.

A CUDA graph records the whole step once and replays it with one call. That needs fixed shapes,
but the model's anchor KV cache grows from 0 to 128 entries at the start of every stream. So the
graph keeps the cache in a fixed 128-slot buffer, filled from the right, plus a mask of the slots
that hold an entry. Masked attention over the valid slots equals the model's attention over its
shorter cache, so the graph gives the same result from the first frame of a stream on:

    model:  keys = [cache minus its last 3 entries] + [window];   cache = last 128 anchors
    graph:  keys = [slots 0..124, masked to valid ones] + [window]; shift left, append anchor

``graph_step`` is a functional copy of ``LivePoseFormer.forward`` with that buffer. The buffer and
mask are updated in place inside the graph, so a replay is one self-contained step.
``GraphedRestorer`` checks it against the eager model when the graph is built.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from livepose.inference import StreamingRestorer
from livepose.model import StreamingTemporalBlock, apply_rope, embed_tokens


def _temporal_step(blk, x, W, N, positions, k_buf, v_buf, valid, anchor):
    """One StreamingTemporalBlock step on a fixed-size cache buffer; updates k_buf / v_buf in place."""
    B, _, D = x.shape
    xj = x.view(B, W, N, D).transpose(1, 2).reshape(B * N, W, D)
    shape = (B * N, W, blk.num_heads, blk.head_dim)
    q, k, v = blk.q_proj(xj).view(shape), blk.k_proj(xj).view(shape), blk.v_proj(xj).view(shape)
    q, k = apply_rope(q, k, positions)
    # attend to [cache without the past frames already in the window] + [window]
    k_att = torch.cat([k_buf[:, :-anchor], k], dim=1)
    v_att = torch.cat([v_buf[:, :-anchor], v], dim=1)
    mask = torch.cat([valid[:-anchor], valid.new_ones(W)])            # True = attend
    new_k = torch.cat([k_buf[:, 1:], k[:, anchor:anchor + 1]], dim=1)  # drop oldest, append the anchor
    new_v = torch.cat([v_buf[:, 1:], v[:, anchor:anchor + 1]], dim=1)
    attn = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k_att.permute(0, 2, 1, 3),
                                          v_att.permute(0, 2, 1, 3), attn_mask=mask)
    attn = attn.permute(0, 2, 1, 3).reshape(B * N, W, D)
    xj = blk.norm1(xj + blk.out_proj(attn))
    xj = blk.norm2(xj + blk.ffn(xj))
    k_buf.copy_(new_k)
    v_buf.copy_(new_v)
    return xj.view(B, N, W, D).transpose(1, 2).reshape(B, W * N, D)


def graph_step(model, x, missing, positions, bufs, valid, anchor):
    """LivePoseFormer forward for one window; advances the cache buffers and the valid mask in place."""
    B, W, N, _ = x.shape
    h = embed_tokens(model, x, missing)
    t = 0
    for block in model.blocks:
        if isinstance(block, StreamingTemporalBlock):
            h = _temporal_step(block, h, W, N, positions, bufs[t][0], bufs[t][1], valid, anchor)
            t += 1
        else:
            h = block(h, W, N)
    valid.copy_(torch.cat([valid[1:], valid.new_ones(1)]))
    return model.output_head(h).view(B, W, N, 3)


class GraphedRestorer(StreamingRestorer):
    """StreamingRestorer whose model step is a CUDA graph replay (built on the first window)."""

    def __init__(self, model, past=3, future=3, device=None, stream=None):
        self.graph = None
        self.stream = stream
        self.graph_check_mm = None
        super().__init__(model, past, future, device)

    def reset(self):
        super().reset()
        if self.graph is not None:              # empty cache for the new stream; the graph is kept
            with torch.cuda.stream(self.stream):
                self._zero_bufs()

    def _zero_bufs(self):
        for k, v in self.s_bufs:
            k.zero_()
            v.zero_()
        self.s_valid.zero_()

    def _empty_bufs(self, x):
        """Zeroed (B*N, 128, heads, head_dim) K and V buffers per temporal block, and an all-False mask."""
        bufs = []
        for b in self.model.temporal_blocks():
            shape = (x.shape[2], b.max_cache_len, b.num_heads, b.head_dim)
            bufs.append((x.new_zeros(shape), x.new_zeros(shape)))
        return bufs, torch.zeros(bufs[0][0].shape[1], dtype=torch.bool, device=x.device)

    @torch.no_grad()
    def _check(self, x, m, steps=10):
        """Max difference (mm) between graph_step and the eager model after ``steps`` windows."""
        self.model.reset_stream()
        bufs, valid = self._empty_bufs(x)
        diff = 0.0
        for s in range(steps):
            xs = torch.rand_like(x)
            ref = self.model(xs, m, frame_start=s, use_cache=True, anchor_pos=self.past)
            pos = torch.arange(s, s + self.window, device=self.device)
            out = graph_step(self.model, xs, m, pos, bufs, valid, self.past)
            diff = max(diff, float((out - ref).abs().max()))
        self.model.reset_stream()
        return diff * 4000                                        # normalized units -> mm (+-2 m box)

    @torch.no_grad()
    def _build_graph(self, x, m):
        self.graph_check_mm = self._check(x, m)
        self.s_x, self.s_m = x.clone(), m.clone()
        self.s_pos = torch.arange(self.window, device=self.device)
        self.s_bufs, self.s_valid = self._empty_bufs(x)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):                             # warm-up before capture
            for _ in range(3):
                bufs, valid = self._empty_bufs(x)
                graph_step(self.model, self.s_x, self.s_m, self.s_pos, bufs, valid, self.past)
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.s_out = graph_step(self.model, self.s_x, self.s_m, self.s_pos, self.s_bufs, self.s_valid, self.past)
        self._zero_bufs()                                         # start the stream on an empty cache

    @torch.no_grad()
    def _run_anchor(self, anchor):
        start = anchor - self.past
        end = anchor + self.future + 1
        x = torch.from_numpy(self._normalized_window(start, end))[None].to(self.device, non_blocking=True)
        m = torch.from_numpy(np.stack(self.missing[start:end]))[None].to(self.device, non_blocking=True)
        if self.graph is None:
            self._build_graph(x, m)
        t0 = time.perf_counter()
        self.s_x.copy_(x)
        self.s_m.copy_(m)
        self.s_pos.copy_(torch.arange(start, start + self.window, device=self.device))
        self.graph.replay()
        pred = self.s_out.cpu()                                   # waits for this stream only
        self.frame_times.append(time.perf_counter() - t0)
        self.last_pred = self.norm.denormalize(pred[0].numpy())
        return self.last_pred
