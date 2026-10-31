"""Live mode: the streaming pipeline runs on the server and every result is sent to the browser.

The camera clock (``stream_sim.source.CameraSource``, 60 Hz) reads the capture's frames from disk
and the ``stream_sim`` engine processes them as a live system would: person detector -> 2D pose ->
triangulation -> optional LivePoseFormer -> real-time biomechanics, each stage in its own thread,
policy "latest" (frames the pipeline cannot take are dropped, or filled by LivePoseFormer). As soon
as a frame's joint angles are ready, its results are sent as one binary row (layout announced in the
"hello" message), so the browser shows what the pipeline computes at the moment it computes it.

The session the user picked supplies what is not computed live: the camera images to display (the
same frames), the Vicon reference for the overlay and accuracy, and the calibration frustums. When
the stream ends the take is saved as a new session (``<session>_live_<time>``), so it can be
reviewed, exported and replayed like a recorded one.
"""
import json
import os
import shutil
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from metrics import compute_metrics, static_check  # noqa: E402
from pipeline import LivePipeline, Recording  # noqa: E402
from stream_sim.biomech_stage import BiomechStage  # noqa: E402
from stream_sim.engine import StreamEngine  # noqa: E402
from stream_sim.source import CameraSource, LatestSink, LiveVideo  # noqa: E402

MODELS = {"hrnet": "w48_768x576_all_63", "vitpose": "ViT_B_simple_all_63"}
MODEL_NAMES = {"hrnet": "HRNet", "vitpose": "ViTPose"}
LIVE_FIELDS = ["kp2d", "conf2d", "box", "xyz", "xyz_raw", "filled", "ang", "ang_unsmoothed", "met", "t", "latency"]
FPS = 60.0


def options(lpf_repo, session=None):
    """What live mode can offer on this machine (for the settings in the app)."""
    cuda = torch.cuda.is_available()
    ckpts = sorted(p.name for p in (Path(lpf_repo) / "checkpoints").glob("*.pth")) if lpf_repo else []
    data_ok = None
    if session:
        trial = session.get("trial", "")
        data_ok = all((ROOT / "data" / f"{c}{trial}.json").exists() for c in ("C1", "C2", "C3"))
    return {
        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU (no CUDA GPU found)", "cuda": cuda,
        "models": [m for m, d in MODELS.items() if (ROOT / d / "checkpoints" / "best_nmae_model.pth").exists()],
        "precisions": ["fp32", "fp16", "bf16"] if cuda else ["fp32"],
        "detectors": ["yolo26n", "yolo26s", "json"],
        "lpf_repo": bool(lpf_repo and (Path(lpf_repo) / "livepose").exists()), "lpf_checkpoints": ckpts,
        "compile": sys.platform != "win32", "data": data_ok,
    }


class LiveRunner:
    """Builds (and caches) the pipeline for a configuration and streams one take at a time."""

    def __init__(self, lpf_repo, device="cuda"):
        self.lpf_repo, self.device = lpf_repo, device
        self._pipes, self._dets, self._lpfs = {}, {}, {}
        self._bio = None
        self.busy = threading.Lock()
        self._source = None

    def stop(self):
        if self._source is not None:
            self._source.stop_event.set()

    # ------------------------------------------------------------------ building blocks (cached)
    def _pipeline(self, cfg, rec, emit):
        key = (cfg["model"], cfg["precision"], bool(cfg.get("compile")))
        if key not in self._pipes:
            emit({"type": "status", "text": f"Loading {MODEL_NAMES[cfg['model']]}…"})
            if len(self._pipes) >= 2:                      # keep GPU memory in check: two pipelines at most
                self._pipes.pop(next(iter(self._pipes)))
                torch.cuda.empty_cache()
            pipe = LivePipeline(ROOT, MODELS[cfg["model"]], rec.subject, "best_nmae_model.pth", self.device)
            if cfg.get("compile"):
                emit({"type": "status", "text": "Compiling the network (torch.compile)…"})
                try:
                    pipe.pose.model = torch.compile(pipe.pose.model)
                    pipe.pose.warmup()
                except Exception as e:                       # e.g. no Triton on Windows
                    pipe.pose.model = getattr(pipe.pose.model, "_orig_mod", pipe.pose.model)
                    emit({"type": "status", "text": f"torch.compile not available ({type(e).__name__}); uncompiled"})
            self._pipes[key] = pipe
        return self._pipes[key]

    def _detector(self, cfg):
        name = cfg.get("detector", "yolo26s")
        if name == "json":
            return None
        key = (name, cfg["precision"])
        if key not in self._dets:
            from stream_sim.detector import PersonDetector
            self._dets[key] = PersonDetector(str(ROOT / f"{name}.pt"), self.device, precision=cfg["precision"])
        return self._dets[key]

    def _lpf(self, cfg, emit):
        if not cfg.get("lpf"):
            return None
        ckpt = cfg.get("lpf_checkpoint")
        if ckpt not in self._lpfs:
            emit({"type": "status", "text": "Loading LivePoseFormer…"})
            from stream_sim.lpf_stage import LpfStage
            self._lpfs[ckpt] = LpfStage(self.lpf_repo, str(Path(self.lpf_repo) / "checkpoints" / ckpt) if ckpt else None,
                                        self.device)
        return self._lpfs[ckpt]

    # ------------------------------------------------------------------ one take
    def run(self, cfg, session_dir, emit):
        """Stream one take; ``emit`` gets status dicts and binary rows; None at the very end."""
        if not self.busy.acquire(blocking=False):
            emit({"type": "error", "text": "A live stream is already running."})
            emit(None)
            return
        try:
            self._run(cfg, Path(session_dir), emit)
        except Exception as e:
            emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
            raise
        finally:
            self._source = None
            self.busy.release()
            emit(None)

    def _run(self, cfg, session_dir, emit):
        s = json.loads((session_dir / "session.json").read_text())
        src = np.fromfile(session_dir / "frames.bin", np.float32).reshape(-1, s["stride"])
        rec = Recording(ROOT / "data", s["trial"])
        pipe = self._pipeline(cfg, rec, emit)
        pipe.setup_world(rec[s["start_index"]].wand_xy, rec.wand_wcs())
        detector, lpf = self._detector(cfg), self._lpf(cfg, emit)
        smoothing = cfg.get("smoothing", "auto")
        smoothing = ("off" if lpf is not None else "standard") if smoothing == "auto" else smoothing
        if self._bio is None:
            self._bio = BiomechStage(self.lpf_repo)
        bio = self._bio
        bio.set_smoothing(smoothing)
        bio.smoothing_name = smoothing

        video = LiveVideo(rec, s["start_index"], s["n_frames"], prefetch=120, threads=3)
        engine = StreamEngine(pipe, cfg["precision"], None, detector, lpf, False, bool(cfg.get("fill") and lpf), bio)
        emit({"type": "status", "text": "Warming up and static trial…"})
        engine.warmup(video.warm_frames(5))                # static trial (frame 0) and GPU warm-up, before the clock
        static_xyz = engine._warm_xyz[0]

        # binary row layout: the live fields, sized as in the session (so the browser can copy them in)
        sizes = {f: s["layout"].get(f, s["layout"]["ang"])[1] for f in LIVE_FIELDS}
        offs, L = {}, 0
        for f in LIVE_FIELDS:
            offs[f] = L
            L += sizes[f]
        rows = {}
        J = len(s["markers"])

        def on_output(r):
            row = np.full(L, np.nan, np.float32)

            def put(f, v):
                if v is not None:
                    row[offs[f]:offs[f] + sizes[f]] = np.asarray(v, np.float32).ravel()[:sizes[f]]

            put("kp2d", r.get("kp2d"))
            put("conf2d", r.get("conf2d"))
            put("box", r.get("box"))
            xyz = r.get("xyz_lpf", r.get("xyz_wcs"))
            put("xyz", xyz)
            put("xyz_raw", r.get("xyz_wcs"))
            put("filled", [1.0 if r.get("filled") else 0.0])
            put("ang", r["angles"])
            put("ang_unsmoothed", r.get("angles_unsmoothed", r["angles"]))
            m = compute_metrics(np.asarray(xyz, float).reshape(1, J, 3), static_xyz)
            put("met", [m[k][0] for k in m])
            dt = lambda a, b: (r[b] - r[a]) * 1000 if a in r and b in r else 0.0   # noqa: E731
            put("t", [dt("t_det_start", "t_det_end"), dt("t_pre_start", "t_pre_end"), dt("t_gpu_start", "t_gpu_end"),
                      dt("t_post_start", "t_done"), dt("t_push_start", "t_push_end"), dt("t_bio_start", "t_bio_end")])
            put("latency", [(r["t_bio_end"] - r["t_capture"]) * 1000])
            rows[r["seq"]] = row
            emit(struct.pack("<i", r["seq"]) + row.tobytes())

        engine.on_output = on_output
        sink = LatestSink(on_drop=engine.on_drop)
        source = CameraSource(video, sink, FPS)
        self._source = source
        video.open()
        engine.start_stream(lambda q: (video.indices[q], source.t0 + q / FPS))
        pipeline = {"pose_model": MODEL_NAMES[cfg["model"]], "precision": cfg["precision"],
                    "detector": detector.name if detector is not None else "JSON boxes",
                    "lpf": (cfg.get("lpf_checkpoint") or "random weights") if lpf is not None else None,
                    "fill_dropped": bool(cfg.get("fill") and lpf), "smoothing": smoothing,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
        emit({"type": "hello", "fields": [[f, offs[f], sizes[f]] for f in LIVE_FIELDS], "row": L,
              "n_frames": s["n_frames"], "pipeline": pipeline})
        engine.run_pipelined(sink, "latest", on_ready=source.start)
        source.join()
        video.close()
        if source.error is not None:
            raise source.error
        if engine.errors:
            raise engine.errors[0]
        name = self._save(s, session_dir, src, rows, offs, sizes, len(source.late), pipeline, cfg, static_xyz, bio)
        n = len(source.late)
        emit({"type": "done", "session": name, "captured": n, "processed": len(rows),
              "dropped": int(len(getattr(sink, "dropped_seqs", [])) + len(engine.handoff_drops))})

    def _save(self, s, session_dir, src, rows, offs, sizes, n, pipeline, cfg, static_xyz, bio):
        """Write the take as a session: live columns from the stream (a frame without output holds the
        previous one, marked filled = 2), Vicon columns and camera images from the source session."""
        n = max(1, min(n, len(src)))
        frames = src[:n].copy()
        prev = None
        for k in range(n):
            row = rows.get(k)
            if row is None and prev is not None:
                row = prev.copy()
                row[offs["filled"]] = 2.0
            if row is None:
                continue
            for f in LIVE_FIELDS:
                if f in s["layout"]:
                    a, m = s["layout"][f]
                    frames[k, a:a + m] = row[offs[f]:offs[f] + sizes[f]]
            prev = rows.get(k, row)
        name = f"{s['trial']}_{s['task']}_live_{cfg['model']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        out = session_dir.parent / name
        out.mkdir()
        frames.tofile(out / "frames.bin")
        for c in (1, 2, 3):
            try:
                os.link(session_dir / f"video_c{c}.bin", out / f"video_c{c}.bin")
            except OSError:
                shutil.copy2(session_dir / f"video_c{c}.bin", out / f"video_c{c}.bin")
        got = np.array([rows[k] for k in sorted(rows)]) if rows else np.zeros((1, max(offs.values()) + 2))
        t = got[:, offs["t"]:offs["t"] + 6]
        lat = got[:, offs["latency"]]

        def stat(a):
            a = a[np.isfinite(a)]
            return {"mean": float(a.mean()) if len(a) else 0.0, "p95": float(np.percentile(a, 95)) if len(a) else 0.0}

        meta = dict(s)
        meta.update(name=name, recorded_at=datetime.now().isoformat(timespec="seconds"), n_frames=n,
                    indices=s["indices"][:n], video_offsets=[o[:n] for o in s["video_offsets"]],
                    angle_smoothing={"preset": pipeline["smoothing"], "one_euro": bio.smoothing or None})
        meta["pipeline"] = dict(pipeline, biomech=s["pipeline"].get("biomech", ""), torch=torch.__version__,
                                mode=f"live: computed in real time on {pipeline['gpu']} (60 fps camera clock, frames the "
                                     f"pipeline cannot take are dropped{' and filled by LivePoseFormer' if pipeline['fill_dropped'] else ''}); "
                                     f"latency = capture -> joint angles, measured")
        meta["timing_ms"] = {st: stat(t[:, i]) for i, st in enumerate(s["stages"])}
        meta["timing_ms"]["latency"] = stat(lat)
        work = t[:, :4].sum(1)
        meta["timing_ms"]["work_per_frame"] = stat(work)
        meta["capacity_fps"] = float(1000 / max(work.mean(), 1e-3))
        meta["live"] = {"captured": n, "processed": len(rows), "config": cfg}
        meta["static"] = dict(s["static"], live=static_check(static_xyz), continuity=dict(bio.stats))
        (out / "session.json").write_text(json.dumps(meta))
        return name
