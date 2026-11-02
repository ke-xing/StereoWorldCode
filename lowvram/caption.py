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

import gc
import os
import tempfile

import torch


def load_caption_model(model_dir: str, device: str, quant: str = "none"):
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs = dict(trust_remote_code=True)
    on_cuda = device.startswith("cuda")
    if quant in ("4bit", "8bit"):
        if not on_cuda:
            raise ValueError("--caption-quant needs a CUDA device")
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
    return model, processor


@torch.no_grad()
def caption_segment(frames, model, processor, device, fps, max_frames=180, max_new_tokens=512):
    from util import save_frames_as_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        save_frames_as_video(frames, path, fps=fps)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": path, "fps": fps, "max_frames": max_frames}},
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
