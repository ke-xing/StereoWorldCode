"""Captioning with less VRAM.

VideoLLaMA3-7B in bf16 needs ~16 GB on its own, which is more than the
whole generation stage after tiling. Options, cheapest first:
  --prompt "..."          skip the VLM entirely (one prompt for all segments)
  --caption-quant 4bit    bitsandbytes NF4, ~5-6 GB
  --caption-quant 8bit    bitsandbytes int8, ~9 GB
  --caption-device cpu    no VRAM at all, slow, needs ~16 GB system RAM
  --caption-max-frames N  fewer visual tokens per segment (default 180 upstream)
"""
from __future__ import annotations

import functools
import gc
import inspect
import os
import tempfile

import torch

CAPTION_MAX_SIDE = 448  # px; frames are downscaled before captioning


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _blockwise_forward(orig_forward, hidden_states, *args, cu_seqlens=None, rotary_pos_emb=None, **kwargs):
    """Run a varlen (cu_seqlens) vision attention one image/frame at a time.

    Without flash-attn, VideoLLaMA3's SDPA path builds a dense
    [seq x seq] mask over *all* frames of the video, which is O(seq^2) and
    blows past any consumer GPU. Attention is block-diagonal per frame and
    everything else in the layer is per-token, so this is exact.
    """
    if cu_seqlens is None or cu_seqlens.numel() <= 2:
        return orig_forward(hidden_states, *args, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, **kwargs)
    bounds = cu_seqlens.tolist()
    total = bounds[-1]
    dim = next((d for d, n in enumerate(hidden_states.shape) if n == total), 0)
    rdim = None
    if rotary_pos_emb is not None:
        rdim = next((d for d, n in enumerate(rotary_pos_emb.shape) if n == total), 0)
    outs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        cs = torch.tensor([0, b - a], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        rp = rotary_pos_emb.narrow(rdim, a, b - a) if rotary_pos_emb is not None else None
        outs.append(orig_forward(hidden_states.narrow(dim, a, b - a), *args,
                                 cu_seqlens=cs, rotary_pos_emb=rp, **kwargs))
    return torch.cat(outs, dim=dim)


def patch_vision_attention(model) -> int:
    """Wrap every attention module that takes `cu_seqlens` (the vision encoder)."""
    n = 0
    for mod in model.modules():
        if "attention" not in type(mod).__name__.lower() or getattr(mod, "_lowvram_blockwise", False):
            continue
        try:
            params = inspect.signature(mod.forward).parameters
        except (TypeError, ValueError):
            continue
        if "cu_seqlens" not in params:
            continue
        mod.forward = functools.partial(_blockwise_forward, mod.forward)
        mod._lowvram_blockwise = True
        n += 1
    return n


def _downscale(frames, max_side=CAPTION_MAX_SIDE):
    out = []
    for im in frames:
        w, h = im.size
        s = max_side / max(w, h)
        out.append(im.resize((max(2, int(w * s)) // 2 * 2, max(2, int(h * s)) // 2 * 2)) if s < 1 else im)
    return out


def _patch_bitsandbytes_for_old_transformers():
    """transformers 4.46 (pinned by StereoWorld) calls .discard() on
    bitsandbytes.supported_torch_devices, which bitsandbytes >= 0.46 made a
    frozenset -> AttributeError. Give it a mutable copy."""
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise SystemExit("--caption-quant needs bitsandbytes: pip install \"bitsandbytes==0.45.5\"") from e
    devs = getattr(bnb, "supported_torch_devices", None)
    if isinstance(devs, frozenset):
        bnb.supported_torch_devices = set(devs)


def load_caption_model(model_dir: str, device: str, quant: str = "none"):
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs = dict(trust_remote_code=True)
    on_cuda = device.startswith("cuda")
    if quant in ("4bit", "8bit"):
        if not on_cuda:
            raise ValueError("--caption-quant needs a CUDA device")
        _patch_bitsandbytes_for_old_transformers()
        from transformers import BitsAndBytesConfig

        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                # keep the vision tower + projector in bf16 (names that don't
                # exist in a given checkpoint are simply ignored), quantise the LLM
                llm_int8_skip_modules=["vision_encoder", "vision_projector", "lm_head"],
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["vision_encoder", "vision_projector", "lm_head"],
            )
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if on_cuda else torch.float32
    kwargs["device_map"] = {"": device}
    if on_cuda:
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    if not _has_flash_attn():
        n = patch_vision_attention(model)
        print(f"[caption] no flash-attn: vision attention runs per frame ({n} layers patched)")
    return model, processor


@torch.no_grad()
def caption_segment(frames, model, processor, device, fps, max_frames=180, max_new_tokens=512):
    from util import save_frames_as_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        n = len(frames)
        if max_frames and n > max_frames:
            # keep the clip's duration: M evenly spaced frames at fps * M / N
            m = max(1, max_frames)
            idx = [round(i * (n - 1) / (m - 1)) for i in range(m)] if m > 1 else [n // 2]
            frames = [frames[i] for i in idx]
            fps = max(1.0, fps * m / n)
        save_frames_as_video(_downscale(frames), path, fps=fps)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": path, "fps": fps, "max_frames": len(frames)}},
                    {"type": "text", "text": "Describe the video."},
                ],
            },
        ]
        inputs = processor(conversation=conversation, add_system_prompt=True,
                           add_generation_prompt=True, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
