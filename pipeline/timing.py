"""Per-step timing with CUDA synchronisation, and a breakdown report.

Each ``with timer.step(name):`` block synchronises the GPU before and after, so GPU work is
charged to the step that launched it. This costs a little throughput but makes the
per-step numbers honest.
"""
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class StepTimer:
    def __init__(self, cuda=False):
        self.cuda = cuda
        self.times = defaultdict(list)     # step -> seconds per call
        self.order = []

    def _sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    @contextmanager
    def step(self, name):
        self._sync()
        t0 = time.perf_counter()
        yield
        self._sync()
        self.add(name, time.perf_counter() - t0)

    def add(self, name, seconds):
        """Record a duration measured elsewhere (e.g. in a background thread)."""
        if name not in self.order:
            self.order.append(name)
        self.times[name].append(seconds)

    def mean_ms(self, name):
        return 1000 * float(np.mean(self.times[name])) if self.times.get(name) else 0.0

    def rows(self, frame_steps=(), stages=None):
        """One row per step; per-frame steps also get their stage and share of the frame time."""
        frame_total = sum(self.mean_ms(s) for s in frame_steps)
        stage_of = {s: stage for stage, steps in (stages or {}).items() for s in steps}
        rows = []
        for name in self.order:
            t = np.array(self.times[name]) * 1000
            rows.append({
                "step": name, "stage": stage_of.get(name, ""), "calls": len(t),
                "mean_ms": t.mean(), "median_ms": np.median(t), "p95_ms": np.percentile(t, 95),
                "max_ms": t.max(), "total_s": t.sum() / 1000,
                "pct_of_frame": 100 * t.mean() / frame_total if name in frame_steps and frame_total else np.nan,
            })
        return rows

    def report(self, title, frame_steps=(), stages=None):
        rows = {r["step"]: r for r in self.rows(frame_steps, stages)}
        frame_total = sum(self.mean_ms(s) for s in frame_steps)
        header = f"{'step':<44}{'calls':>6}{'mean ms':>9}{'median':>9}{'p95':>9}{'max':>9}{'% frame':>9}"
        line = "-" * len(header)

        def fmt(r, indent=""):
            name = indent + r["step"].replace("frame: ", "")
            pct = f"{r['pct_of_frame']:>8.1f}%" if np.isfinite(r["pct_of_frame"]) else f"{'':>9}"
            return (f"{name:<44}{r['calls']:>6}{r['mean_ms']:>9.2f}{r['median_ms']:>9.2f}"
                    f"{r['p95_ms']:>9.2f}{r['max_ms']:>9.2f}{pct}")

        setup = [n for n in self.order if n.startswith("setup:")]
        other = [n for n in self.order if n not in setup and n not in frame_steps]
        out = [title, line, "SETUP (once)", header] + [fmt(rows[n]) for n in setup]
        out += ["", "PER FRAME SET (3 images)", header]
        for stage, steps in (stages or {"": list(frame_steps)}).items():
            present = [s for s in steps if s in rows]
            if not present:
                continue
            stage_ms = sum(rows[s]["mean_ms"] for s in present)
            out.append(f"  {stage:<42}{'':>6}{stage_ms:>9.2f}{'':>27}{100 * stage_ms / frame_total:>8.1f}%")
            out += [fmt(rows[s], indent="    ") for s in present]
        out += [line, f"{'TOTAL per frame set':<44}{'':>6}{frame_total:>9.2f} ms  ->  "
                      f"{1000 / frame_total:.1f} frame sets / s" if frame_total else ""]
        if other:
            out += ["", "NOT COUNTED IN THE FRAME TIME", header] + [fmt(rows[n]) for n in other]
        return "\n".join(out)
