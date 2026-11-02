"""Side-by-side output with matching left/right geometry.

The right eye is generated from frames that were scaled to cover (H, W) and
center-cropped (util.ImageCropAndResize). Upstream builds the SBS left eye by
squeezing the *uncropped* source frame into (W, H), so whenever the aspect
ratio differs (always slightly, e.g. 1920x1080 -> 832x480) the two eyes no
longer show the same image region. Here the left eye goes through the exact
same crop as the frames the model saw.
"""
from __future__ import annotations

import imageio
import numpy as np
from PIL import Image


def cover_crop(image: Image.Image, H: int, W: int) -> Image.Image:
    """Same geometry as util.ImageCropAndResize.crop_and_resize."""
    import torchvision

    w, h = image.size
    scale = max(W / w, H / h)
    nh, nw = round(h * scale), round(w * scale)
    image = torchvision.transforms.functional.resize(
        image, (nh, nw), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True
    )
    return torchvision.transforms.functional.center_crop(image, (H, W))


def crop_note(src_w: int, src_h: int, W: int, H: int) -> str:
    """Human-readable description of how much of the source is cut off."""
    scale = max(W / src_w, H / src_h)
    keep_w = min(1.0, W / (src_w * scale))
    keep_h = min(1.0, H / (src_h * scale))
    if keep_w > 0.985 and keep_h > 0.985:  # rounding to multiples of 16 alone crops ~1%
        return ""
    if keep_w < keep_h:
        return f"{(1 - keep_w) * 100:.0f}% of the width is cropped (left and right edges)"
    return f"{(1 - keep_h) * 100:.0f}% of the height is cropped (top and bottom edges)"


def write_sbs(right_segments, segment_info, video_path, output_path, fps, overlap_frames, H, W, quality=8):
    """right_segments: list of lists of PIL frames (one list per generated segment)."""
    right = []
    for i, seg in enumerate(right_segments):
        right.extend(seg if i == 0 else seg[overlap_frames:])

    reader = imageio.get_reader(video_path, format="ffmpeg")
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, macro_block_size=8)
    n = 0
    try:
        for idx, frame in enumerate(reader):
            if idx >= len(right):
                break
            left = cover_crop(Image.fromarray(frame), H, W)
            r = right[idx].convert("RGB")
            if r.size != (W, H):
                r = r.resize((W, H), Image.LANCZOS)
            sbs = np.concatenate([np.asarray(left), np.asarray(r)], axis=1)
            writer.append_data(sbs)
            n += 1
    finally:
        writer.close()
        reader.close()
    return n
