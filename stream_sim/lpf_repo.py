"""Where the LivePoseFormer repository is (temporal model, checkpoints and the joint-angle model).

In order: the path given on the command line, the ``LIVEPOSEFORMER`` environment variable, or a
``LivePoseFormer`` folder next to this repository (``git clone`` both into the same folder).
"""
import os
from pathlib import Path

ENV = "LIVEPOSEFORMER"
SIBLING = Path(__file__).resolve().parents[2] / "LivePoseFormer"
HELP = (f"LivePoseFormer repository (default: ${ENV}, else ../LivePoseFormer next to this repository)")


def find_lpf_repo(path=None, required=False):
    """The repository folder as an absolute path, or None (``required``: exit with a clear message)."""
    for p in (path, os.environ.get(ENV), SIBLING if (SIBLING / "livepose").exists() else None):
        if p:
            return str(Path(p).expanduser().resolve())
    if required:
        raise SystemExit(f"LivePoseFormer repository not found: pass --lpf-repo PATH, set {ENV}=PATH, or clone "
                         f"https://github.com/chris-schreuder/LivePoseFormer next to this repository ({SIBLING}).")
    return None
