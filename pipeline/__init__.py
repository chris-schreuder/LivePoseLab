"""Live 3-camera markerless pipeline: dense 2D keypoints -> 3D triangulation -> Vicon world frame."""
from .data import Recording
from .pipeline import FRAME_STEPS, STAGES, LivePipeline

__all__ = ["LivePipeline", "Recording", "FRAME_STEPS", "STAGES"]
