"""Low-VRAM inference path for StereoWorld (spatial tiling + temporal chunking)."""
from .sampler import LowVRAMConfig, long_video_lowvram, tiled_velocity  # noqa: F401
