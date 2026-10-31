"""Streaming engine: runs the pose pipeline on a live packet stream and timestamps everything.

Pipelined mode (default) runs every stage in its own thread, as a real-time system would:

    source --> [ DET*: decode*, YOLO person detector -> boxes ]  GPU, own CUDA stream
           --> [ PRE: decode*, crop, resize ]                    CPU
           --> [ GPU: upload, network, heatmap peaks, copy back ]
           --> [ POST: map to image, triangulate, world frame ]  CPU
           ==> [ LPF*: LivePoseFormer smoothing / imputation ]   GPU, own CUDA stream
           ==> [ BIO*: joint angles (Plug-in Gait model) ]       CPU
                                                                 (* optional)

Without a detector the boxes come from the camera JSON files (DET is skipped). JPEG decoding
(--decode-jpeg) happens in the first stage that needs the pixels: DET if there is one, else PRE.

DET -> PRE -> GPU -> POST hand over through one-slot queues, so a stage that is still busy makes the
previous one wait (back-pressure). Throughput is then set by the slowest stage, not by the sum of
all stages, while each frame's delay is still roughly the sum. Sequential mode runs everything in
one thread, one frame after another, for comparison.

LivePoseFormer and the biomechanics stage are decoupled (==>): each reads an unbounded queue in
its own thread, so neither holds up the pose pipeline. BIO gets LivePoseFormer's output when it is
enabled (including filled frames), otherwise the raw 3D result of every processed frame.

LivePoseFormer's queue gets every 3D result from POST and, the moment a frame is dropped (by the
source or at the hand-off to the GPU), a "dropped" event. LivePoseFormer consumes the events in
stream order; with ``lpf_fill_dropped`` a dropped frame is pushed as an all-NaN frame, so it is
imputed and the smoothed output keeps the full camera rate even when the pose network does not.

No per-step GPU synchronisation is done here (unlike the offline timing report): the GPU stage
ends with a device-to-host copy, which waits for the GPU anyway, so the timestamps are honest
without stalling the stream.
"""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import cv2
import numpy as np
import torch

from pipeline.triangulation import triangulate

PRECISION = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class StreamEngine:
    def __init__(self, pipe, precision="fp32", fake_network_ms=None, detector=None, lpf=None,
                 decode_jpeg=False, lpf_fill_dropped=False, biomech=None):
        self.pipe, self.pose = pipe, pipe.pose
        self.cuda = self.pose.device.type == "cuda"
        self.amp_dtype = PRECISION[precision] if self.cuda else None
        self.fake_network_ms = fake_network_ms
        self.detector = detector
        self.lpf = lpf
        self.bio = biomech
        self.decode_jpeg = decode_jpeg
        self.records = {}
        self._decode_pool = ThreadPoolExecutor(3) if decode_jpeg else None
        self._fake_heatmaps = None
        self.lpf_fill_dropped = lpf_fill_dropped and lpf is not None
        self.errors = []
        self.on_output = None            # optional callback(record) once a frame's joint angles are ready
        self.start_stream(None)

    def start_stream(self, seq_info):
        """Reset per-stream state. ``seq_info(seq) -> (recording index, capture time)`` describes
        frames that were dropped before they got a record."""
        self.records = {}
        self.handoff_drops = []
        self.seq_info = seq_info
        self._lpf_q = queue.Queue()      # unbounded: LivePoseFormer never blocks the pose pipeline
        self._lpf_pending = {}           # seq -> (xyz or None for a dropped frame, time it was known)
        self._lpf_next = 0               # next seq LivePoseFormer takes, in stream order
        self._lpf_last_index = None
        self._lpf_started = False        # has LivePoseFormer seen a real frame since its last reset?
        self._bio_q = queue.Queue()      # unbounded: frames waiting for the joint angles
        if self.lpf is not None:
            self.lpf.reset()
        if self.detector is not None:
            self.detector.reset()
        if self.bio is not None:
            self.bio.reset()
        self._bio_last_index = None

    # ---------------------------------------------------------------- the stages
    def _images(self, images):
        """RGB arrays as they are; JPEG buffers decoded in parallel (MJPEG camera simulation)."""
        if images[0].ndim == 3:
            return images
        return list(self._decode_pool.map(lambda b: cv2.imdecode(b, cv2.IMREAD_COLOR_RGB), images))

    def _record(self, pkt):
        return self.records.setdefault(pkt.seq, {"seq": pkt.seq, "index": pkt.index, "t_capture": pkt.t_capture})

    def _det(self, pkt):
        """Detect the person in every view; the detected boxes replace the JSON boxes in the packet.
        The JSON boxes are kept in the record, to report how well the two agree."""
        from .detector import box_iou
        r = self._record(pkt)
        r["t_det_start"] = time.perf_counter()
        pkt.images = self._images(pkt.images)
        boxes, conf = self.detector(pkt.images)
        r["t_det_end"] = time.perf_counter()
        r["det_conf"] = conf
        r["det_misses"] = int((conf < self.detector.conf).sum())
        r["det_iou_json"] = float(np.mean([box_iou(a, b) for a, b in zip(boxes, pkt.bboxes)]))
        pkt.bboxes = boxes
        return pkt

    def _pre(self, pkt):
        r = self._record(pkt)
        r["t_pre_start"] = time.perf_counter()
        images = self._images(pkt.images)
        r["box"] = np.asarray(pkt.bboxes, float)        # the person box used (detector or JSON)
        crops, boxes = self.pose.crop(images, pkt.bboxes)
        batch = self.pose.resize(crops)
        r["t_pre_end"] = time.perf_counter()
        return pkt.seq, batch, boxes

    @torch.inference_mode()
    def _gpu_compute(self, batch):
        """Upload, network, heatmap peaks, copy back. Returns CPU tensors (kp, conf)."""
        if self.fake_network_ms is not None:
            time.sleep(self.fake_network_ms / 1000)      # "what if the network took X ms"
            heatmaps = self._fake_heatmaps
        else:
            x = self.pose.upload(batch)
            amp = torch.autocast("cuda", dtype=self.amp_dtype) if self.amp_dtype else nullcontext()
            with amp:
                heatmaps = self.pose.forward(x)
        kp, conf = self.pose.decode_on_device(heatmaps)
        return kp.cpu(), conf.cpu()                      # waits for the GPU

    def _gpu(self, seq, batch, boxes):
        r = self.records[seq]
        r["t_gpu_start"] = time.perf_counter()
        kp, conf = self._gpu_compute(batch)
        r["t_gpu_end"] = time.perf_counter()
        return seq, kp, conf, boxes

    def _post(self, seq, kp, conf, boxes):
        r = self.records[seq]
        r["t_post_start"] = time.perf_counter()
        kp2d, conf2d = self.pose.to_image(kp, conf, boxes)
        r["kp2d"], r["conf2d"] = kp2d, conf2d
        xyz_cam = triangulate(kp2d[self.pipe.rig.use_cams], self.pipe.P)
        xyz_wcs = self.pipe.world.apply(xyz_cam)
        r["t_done"] = time.perf_counter()
        r["xyz_wcs"] = xyz_wcs
        if self.lpf is not None:
            self._lpf_q.put((seq, xyz_wcs, r["t_done"]))
        elif self.bio is not None:
            self._bio_q.put((seq, xyz_wcs, r["t_done"]))
        return seq, xyz_wcs

    def _pose_stages(self, pkt):
        if self.detector is not None:
            pkt = self._det(pkt)
        return self._post(*self._gpu(*self._pre(pkt)))

    # ---------------------------------------------------------------- LivePoseFormer
    def on_drop(self, seq):
        """Called by the sinks the moment frame ``seq`` is dropped."""
        if self.lpf is not None:
            self._lpf_q.put((seq, None, time.perf_counter()))

    def _lpf_event(self, item):
        """Take one event and push every frame that is now next in stream order."""
        seq, xyz, t_known = item
        self._lpf_pending[seq] = (xyz, t_known)
        while self._lpf_next in self._lpf_pending:
            self._lpf_take(self._lpf_next)

    def _lpf_take(self, seq):
        xyz, t_known = self._lpf_pending.pop(seq)
        self._lpf_next = max(self._lpf_next, seq + 1)
        if xyz is None:                                  # a dropped frame
            if not self.lpf_fill_dropped:
                return
            index, t_capture = self.seq_info(seq)
            r = self.records.setdefault(seq, {"seq": seq, "index": index, "t_capture": t_capture})
            r["filled"] = True
            xyz = np.full((self.lpf.num_joints, 3), np.nan)
        r = self.records[seq]
        r["t_lpf_known"] = t_known
        self._push_lpf(seq, r["index"], xyz)

    def _push_lpf(self, seq, index, xyz_wcs):
        # If the recording index ever jumps back (e.g. a source that loops), treat it as a new
        # recording: flush the last frames and reset LivePoseFormer's memory. Forward gaps are
        # dropped frames and stay part of the same stream.
        if self._lpf_last_index is not None and index < self._lpf_last_index:
            self._lpf_flush()
            self.lpf.reset()
            self._lpf_started = False
        self._lpf_last_index = index
        r = self.records[seq]
        if np.isnan(xyz_wcs).all() and not self._lpf_started:
            # A stream must start on a real observation: dropped frames before the first real frame
            # of a recording cannot be filled.
            r.pop("filled", None)
            return
        self._lpf_started = True
        r["t_push_start"] = time.perf_counter()
        released = self.lpf.push(seq, xyz_wcs)
        r["t_push_end"] = time.perf_counter()
        self._store_lpf(released)

    def _store_lpf(self, released):
        now = time.perf_counter()
        for out_seq, xyz in released:
            if out_seq in self.records:
                self.records[out_seq]["t_lpf_out"] = now
                self.records[out_seq]["xyz_lpf"] = xyz
                if self.bio is not None:
                    self._bio_q.put((out_seq, xyz, now))

    def _lpf_finish(self):
        """End of stream: take what is left (in order), then release the last frames without
        waiting for more lookahead."""
        while True:
            try:
                self._lpf_event(self._lpf_q.get_nowait())
            except queue.Empty:
                break
        for seq in sorted(self._lpf_pending):            # only if an event went missing
            self._lpf_take(seq)
        self._store_lpf(self.lpf.flush())

    def _lpf_drain(self):
        """Sequential mode: handle every event queued so far, inline."""
        while True:
            try:
                self._lpf_event(self._lpf_q.get_nowait())
            except queue.Empty:
                return

    def _lpf_flush(self):
        if self.lpf is not None:
            self._store_lpf(self.lpf.flush())

    # ---------------------------------------------------------------- biomechanics
    def _bio(self, item):
        """Joint angles of one frame (smoothed if LivePoseFormer is on, raw 3D otherwise)."""
        seq, xyz, t_known = item
        r = self.records[seq]
        r["t_bio_known"] = t_known
        r["t_bio_start"] = time.perf_counter()
        gap = 1 if self._bio_last_index is None else max(1, r["index"] - self._bio_last_index)
        self._bio_last_index = r["index"]
        r["angles"] = self.bio.angles(xyz, gap)
        if self.bio.smoothing:
            r["angles_unsmoothed"] = self.bio.last_unsmoothed
        r["t_bio_end"] = time.perf_counter()
        if self.on_output is not None:
            self.on_output(r)

    def _bio_drain(self):
        while True:
            try:
                self._bio(self._bio_q.get_nowait())
            except queue.Empty:
                return

    # ---------------------------------------------------------------- running
    def warmup(self, frames):
        """Run a few frames end to end before the clock starts (cuDNN autotuning, allocator, caches).
        ``frames`` = [(index, images, bboxes), ...]. LivePoseFormer's CUDA graph is captured here."""
        from .source import Packet
        crops, _ = self.pose.crop(self._images(frames[0][1]), frames[0][2])
        self._warm_batch = self.pose.resize(crops)       # reused by the per-thread warm-up
        with torch.inference_mode():   # one real forward pass: heatmaps reused by --fake-network-ms
            self._fake_heatmaps = self.pose.forward(self.pose.upload(self._warm_batch)).float()
        self.start_stream(None)
        self._warm_xyz = []
        for k, (index, images, bboxes) in enumerate(frames):
            self._warm_xyz.append(self._pose_stages(Packet(-1 - k, index, time.perf_counter(), images, bboxes))[1])
        if self.lpf is not None:
            self._warm_thread_lpf()
        if self.bio is not None:
            # Static trial, done before any live stream: the pipeline's 3D result for the first frame
            # of the video stands in for it.
            self.bio.calibrate(self._warm_xyz[0])
            for xyz in self._warm_xyz:
                self.bio.angles(xyz)
        self.start_stream(None)
        self._warm_images = frames[0][1]                 # reused by the detector's per-thread warm-up

    def _warm_thread_gpu(self, n=3):
        """Warm-up inside the worker thread itself. cuDNN and cuBLAS keep per-thread state, so the
        first network call in a new thread pays set-up and autotuning again (about 0.7 s on an
        RTX 3090). Doing it here keeps that cost out of the timed stream."""
        for _ in range(n):
            self._gpu_compute(self._warm_batch)

    def _warm_thread_det(self, n=3):
        for _ in range(n):
            self.detector(self._images(self._warm_images))
        self.detector.reset()

    def _warm_thread_lpf(self, n=20):
        for k in range(n):
            self.lpf.push(-1, self._warm_xyz[k % len(self._warm_xyz)])
        self.lpf.reset()

    def run_sequential(self, sink, on_ready=lambda: None):
        on_ready()                     # runs in the main thread, which ``warmup`` already warmed
        while True:
            pkt = sink.get()
            if pkt is None:
                if self.lpf is not None:
                    self._lpf_finish()
                if self.bio is not None:
                    self._bio_drain()
                return
            self._pose_stages(pkt)
            r = self.records[pkt.seq]
            if self.lpf is not None:
                r["t_lpf_start"] = time.perf_counter()
                self._lpf_drain()
                r["t_lpf_done"] = time.perf_counter()
            if self.bio is not None:
                r["t_bio_drain_start"] = time.perf_counter()
                self._bio_drain()
                r["t_bio_drain_done"] = time.perf_counter()

    def run_pipelined(self, sink, policy="latest", on_ready=lambda: None):
        """policy "latest": the hand-off to the GPU also keeps only the newest preprocessed frame,
        so the GPU never works on a stale frame (older ones are dropped and counted in
        ``self.handoff_drops``). policy "all": blocking one-slot queues, nothing is dropped.

        The worker threads warm themselves up first; ``on_ready`` (which starts the camera clock)
        is called only once all of them are ready."""
        from .source import LatestSink

        q_post = queue.Queue(maxsize=1)
        stop = object()
        latest = policy == "latest"
        if latest:
            slot = LatestSink(on_drop=self.on_drop)
            self.handoff_drops = slot.dropped_seqs

            class _Slot:                     # queue-like wrapper so the workers below stay the same
                def put(self, item):
                    if item is stop:
                        slot.close()
                    else:
                        slot.put(_Item(item))

                def get(self):
                    it = slot.get()
                    return stop if it is None else it.payload

                def put_nowait(self, item):
                    self.put(item)

            class _Item:                     # LatestSink records dropped items by .seq
                def __init__(self, payload):
                    self.payload, self.seq = payload, payload[0]

            q_gpu = _Slot()
        else:
            self.handoff_drops = []
            q_gpu = queue.Queue(maxsize=1)

        q_pre = queue.Queue(maxsize=1) if self.detector is not None else None
        warm = {"gpu_worker": self._warm_thread_gpu,
                "det_worker": self._warm_thread_det if self.detector is not None else None,
                "lpf_worker": self._warm_thread_lpf if self.lpf is not None else None}
        workers_on = {"det_worker": self.detector is not None, "lpf_worker": self.lpf is not None,
                      "bio_worker": self.bio is not None}
        ready = threading.Barrier(3 + sum(workers_on.values()) + 1)

        def guard(fn):
            def wrapped():
                try:
                    if warm.get(fn.__name__):
                        warm[fn.__name__]()
                    ready.wait()
                    fn()
                except Exception as e:      # surface worker errors instead of hanging
                    self.errors.append(e)
                    ready.abort()
                    for q in (q_pre, q_gpu, q_post, self._lpf_q, self._bio_q):
                        if q is None:
                            continue
                        try:
                            q.put_nowait(stop)
                        except queue.Full:
                            pass
                    raise
            return wrapped

        def det_worker():
            while True:
                pkt = sink.get()
                if pkt is None:
                    q_pre.put(None)
                    return
                q_pre.put(self._det(pkt))

        def pre_worker():
            source = q_pre if self.detector is not None else sink
            while True:
                pkt = source.get()
                if pkt is None or pkt is stop:
                    q_gpu.put(stop)
                    return
                q_gpu.put(self._pre(pkt))

        def gpu_worker():
            while (item := q_gpu.get()) is not stop:
                q_post.put(self._gpu(*item))
            q_post.put(stop)

        def post_worker():
            while (item := q_post.get()) is not stop:
                self._post(*item)            # also queues the result for LivePoseFormer / BIO
            (self._lpf_q if self.lpf is not None else self._bio_q).put(stop)

        def lpf_worker():
            while (item := self._lpf_q.get()) is not stop:
                self._lpf_event(item)
            self._lpf_finish()
            self._bio_q.put(stop)

        def bio_worker():
            while (item := self._bio_q.get()) is not stop:
                self._bio(item)

        workers = ([det_worker] if self.detector is not None else []) + [pre_worker, gpu_worker, post_worker] \
            + ([lpf_worker] if self.lpf is not None else []) + ([bio_worker] if self.bio is not None else [])
        threads = [threading.Thread(target=guard(f), daemon=True) for f in workers]
        for t in threads:
            t.start()
        try:
            ready.wait()                 # every worker has warmed up
        except threading.BrokenBarrierError:
            raise self.errors[0] if self.errors else RuntimeError("a worker failed during warm-up")
        on_ready()
        for t in threads:
            t.join()
