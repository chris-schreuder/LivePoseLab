"""Simulated 3-camera, hardware-synchronised live video source.

``LiveVideo`` treats the recording as a live video: the stream starts at ``--start`` and runs once
through the recording, frame after frame, and each frame set is read from disk and decoded just
ahead of the camera clock by a few background threads into a small rolling buffer (``prefetch``
frame sets). That stands in for the camera's own capture: a real camera hands over a frame that is
already in memory. Nothing loops, so LivePoseFormer sees one continuous recording and every frame
is scored against its own ground truth. With ``keep_jpeg=True`` the reader only reads the JPEG
bytes, to simulate cameras that send MJPEG (decoding then happens inside the timed pipeline).

If the reader ever falls behind (the disk or CPU is too slow), the frame is released late and the
report counts it as a reader underrun, so the simulator never hides its own slowness.

``CameraSource`` releases one synchronised frame set every 1/fps seconds on a fixed clock,
stamped with its capture time. Entry ``i`` of the camera JSONs is one 60 Hz tick (the
motion-capture timing); the cameras recorded fewer images, so consecutive entries sometimes share
an image file, which is decoded once.

Two delivery policies, matching the two questions the simulation answers:
* ``QueueSink`` ("all"): every frame is queued. If the pipeline is too slow the queue, and the
  delay, keep growing (the source stops early once the queue holds ``max_backlog`` frames, so a
  hopeless run does not fill the RAM).
* ``LatestSink`` ("latest"): a single slot that always holds the newest frame. A frame that is
  replaced before the pipeline takes it is dropped, as in a real live system.
"""
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Packet:
    seq: int                 # position in the stream (0, 1, 2, ...)
    index: int               # frame index in the recording (for ground truth)
    t_capture: float         # perf_counter() time the frame set was "captured"
    images: list             # 3 RGB uint8 arrays, or 3 JPEG byte buffers (keep_jpeg)
    bboxes: np.ndarray       # (3, 4)


def _lower_priority():
    """Run the calling thread below normal priority. The reader stands in for the cameras, which
    in a real system do not use this PC's CPU, so it should not take CPU time from the pipeline."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)          # THREAD_PRIORITY_BELOW_NORMAL
        else:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)   # nice +10, this thread only
    except Exception:
        pass


class LiveVideo:
    def __init__(self, recording, start=0, n_frames=None, keep_jpeg=False, prefetch=120, threads=3):
        end = len(recording) if n_frames is None else min(len(recording), start + n_frames)
        sets = []
        for i in range(start, end):           # the video ends where its image files end
            fs = recording[i]
            if not all(p.exists() for p in fs.image_paths):
                break
            sets.append(fs)
        if not sets:
            raise FileNotFoundError(f"no images on disk for frame {start}: {recording[start].image_paths}")
        self.indices = [fs.index for fs in sets]
        self.paths = [fs.image_paths for fs in sets]
        self.bboxes = [fs.bboxes for fs in sets]
        self.gt_wcs = [fs.gt_xyz_wcs for fs in sets]
        self.keep_jpeg, self.prefetch, self.threads = keep_jpeg, prefetch, threads
        self._pool = None

    def __len__(self):
        return len(self.indices)

    def _load(self, path):
        data = np.fromfile(str(path), dtype=np.uint8)
        return data if self.keep_jpeg else cv2.imdecode(data, cv2.IMREAD_COLOR_RGB)

    def open(self):
        """Start a new pass from the first frame. Returns once the buffer is full, so the camera
        clock can start with ``prefetch`` frame sets ready."""
        self.close()
        cv2.setNumThreads(1)                     # parallelism comes from the reader threads
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="reader", initializer=_lower_priority)
        self._futs, self._next = {}, 0
        self._last = [(None, None)] * len(self.paths[0])
        self.underruns = []                      # seqs whose images were not ready on time
        for _ in range(min(self.prefetch, len(self))):
            self._submit()
        for futs in list(self._futs.values()):
            for f in futs:
                f.result()

    def _submit(self):
        k = self._next
        if k >= len(self):
            return
        futs = []
        for c, path in enumerate(self.paths[k]):
            last_path, last_fut = self._last[c]
            fut = last_fut if path == last_path else self._pool.submit(self._load, path)
            self._last[c] = (path, fut)
            futs.append(fut)
        self._futs[k] = futs
        self._next += 1

    def take(self, k):
        """Hand over frame set k's futures and schedule frame k + prefetch."""
        futs = self._futs.pop(k)
        self._submit()
        return futs

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def warm_frames(self, n):
        """The first n frame sets, loaded directly: [(index, images, bboxes), ...] for warm-up."""
        return [(self.indices[k], [self._load(p) for p in self.paths[k]], self.bboxes[k])
                for k in range(min(n, len(self)))]


class QueueSink:
    """Unbounded FIFO: nothing is dropped."""
    def __init__(self):
        self.q = queue.Queue()
        self.dropped = 0

    def put(self, packet):
        self.q.put(packet)

    def close(self):
        self.q.put(None)

    def get(self):
        return self.q.get()

    def backlog(self):
        return self.q.qsize()


class LatestSink:
    """One slot holding the newest frame; replacing an untaken frame counts as a drop.

    ``on_drop(seq)`` is called for every dropped frame, the moment it is dropped."""
    def __init__(self, on_drop=None):
        self.cv = threading.Condition()
        self.item, self.closed, self.dropped = None, False, 0
        self.dropped_seqs = []
        self.on_drop = on_drop

    def put(self, packet):
        with self.cv:
            if self.item is not None:
                self.dropped += 1
                self.dropped_seqs.append(self.item.seq)
                if self.on_drop is not None:
                    self.on_drop(self.item.seq)
            self.item = packet
            self.cv.notify()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify()

    def get(self):
        with self.cv:
            while self.item is None and not self.closed:
                self.cv.wait()
            item, self.item = self.item, None
            return item

    def backlog(self):
        return int(self.item is not None)


class CameraSource(threading.Thread):
    """Releases the video's frame sets at ``fps`` into ``sink`` on a fixed clock."""

    def __init__(self, video, sink, fps, max_backlog=None):
        super().__init__(daemon=True)
        self.video, self.sink, self.fps, self.max_backlog = video, sink, fps, max_backlog
        self.t0 = None
        self.late = []           # how late each release was (s), to check the clock itself keeps up
        self.stopped_early = None
        self.error = None
        self.stop_event = threading.Event()      # set() ends the stream early (e.g. Stop in the demo)

    def run(self):
        try:
            self._release()
        except Exception as e:                       # e.g. an unreadable image: end the stream, report it
            self.error = e
        finally:
            self.sink.close()

    def _release(self):
        period = 1.0 / self.fps
        self.t0 = time.perf_counter() + 0.05
        for seq in range(len(self.video)):
            if self.stop_event.is_set():
                break
            if self.max_backlog and self.sink.backlog() > self.max_backlog:
                self.stopped_early = seq             # the queue would only keep growing
                break
            futs = self.video.take(seq)
            target = self.t0 + seq * period
            while True:                              # sleep, then spin for the last ~1.5 ms
                remaining = target - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(remaining - 0.0015 if remaining > 0.002 else 0)
            if not all(f.done() for f in futs):
                self.video.underruns.append(seq)
            images = [f.result() for f in futs]
            now = time.perf_counter()
            self.late.append(now - target)
            self.sink.put(Packet(seq, self.video.indices[seq], now, images, self.video.bboxes[seq]))
