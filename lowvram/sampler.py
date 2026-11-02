"""Low-VRAM replacement for WanVideoPipeline.long_video.

Same algorithm as upstream (segment-by-segment right-eye generation
conditioned on the left-eye latents, 3-latent-frame re-noised overlap
between segments, CFG, flow-matching), restructured so the DiT never
sees more tokens than one tile:

  for each segment:
    for each timestep:
      v = 0, wsum = 0
      for each (temporal window x spatial tile):
          x_tile = concat(right[t, h, w], left[t, h, w])   # along time
          v_tile = CFG(DiT(x_tile))[:, :, :len(t)]
          v[t, h, w] += v_tile * feather;  wsum += feather
      latents = scheduler.step(v / wsum, latents)

The flow-match step is linear in the velocity (x + v * dsigma), so
blending velocities and stepping once is mathematically the same as
stepping every tile and blending the results like upstream does, but it
is done in float32 and with proper 2-D feathering.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from .patches import frame_positions
from .tiling import describe, plan_tiles


@dataclass
class LowVRAMConfig:
    tile_hw: Tuple[int, int] = (60, 108)  # latent units (x8 = pixels)
    tile_overlap_hw: Tuple[int, int] = (12, 20)
    t_chunk: Optional[int] = None  # latent frames per temporal window; None = whole segment
    t_overlap: int = 4
    vae_tile: Tuple[int, int] = (30, 52)
    vae_stride: Tuple[int, int] = (15, 26)
    overlap_latents: int = 3  # = (overlap_frames - 1) // 4 + 1 with overlap_frames = 9
    offload_finished_latents: bool = True
    keep_dit_resident: bool = False  # don't move the DiT to CPU around VAE work
    teacache: float = 0.0  # TeaCache rel-L1 threshold; 0 = off (0.05 balanced, 0.08 fast)
    verbose: bool = True


_PROGRESS = os.environ.get("STEREOWORLD_PROGRESS") == "1"


def emit(kind: str, *vals) -> None:
    """Machine-readable lines for the GUI (only when STEREOWORLD_PROGRESS=1)."""
    if _PROGRESS:
        print("@@" + kind + " " + " ".join(str(v) for v in vals), flush=True)


def vram_report(device, label: str = "") -> str:
    """One-line VRAM summary. `free` is device-wide (includes other apps);
    near 0 on Windows means the driver is spilling into system RAM."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return ""
    free, total = torch.cuda.mem_get_info(device)
    alloc = torch.cuda.memory_allocated(device) / 2**30
    res = torch.cuda.memory_reserved(device) / 2**30
    msg = (f"[vram] {label}: torch allocated {alloc:.2f} GiB, reserved {res:.2f} GiB, "
           f"GPU free {free / 2**30:.2f} of {total / 2**30:.1f} GiB")
    if free / 2**30 < 0.35:
        msg += ("\n[vram] (!) GPU memory is full. On Windows the driver then moves data to system RAM "
                "instead of failing, which makes every step many times slower. Lower the resolution or "
                "VRAM preset, turn off TeaCache, close GPU-heavy apps, or set NVIDIA Control Panel > "
                "Manage 3D settings > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback "
                "to get a clear out-of-memory error instead.")
    return msg


def _gpu_stats(device) -> str:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return ""
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    res = torch.cuda.max_memory_reserved(device) / 2**30
    return f"peak alloc {peak:.2f} GiB / reserved {res:.2f} GiB"


class _TileTeaCaches:
    """One TeaCache per (tile, CFG branch).

    TeaCache caches the DiT's residual (output - input hidden states), which
    has the shape of the tile it was computed on, so tiles must not share one.
    Its skip decision depends only on the timestep embedding, so all tiles
    skip the same steps and stay consistent with each other.
    """

    def __init__(self, steps: int, thresh: float):
        from diffsynthfinal.pipelines.wan_video_new import TeaCache

        self._cls, self.steps, self.thresh = TeaCache, steps, thresh
        self.caches = {}
        self.calls = self.skips = 0

    def get(self, key):
        c = self.caches.get(key)
        if c is None:
            c = self._cls(self.steps, self.thresh, "Wan2.1-T2V-1.3B")
            orig = c.check

            def check(dit, x, t_mod, _orig=orig):
                skip = _orig(dit, x, t_mod)
                self.calls += 1
                self.skips += int(bool(skip))
                return skip

            c.check = check
            self.caches[key] = c
        return c


def tiled_velocity(call, x: torch.Tensor, T: int, tiles) -> torch.Tensor:
    """x: (1, C, 2T, H, W) = [right-eye latents | left-eye latents] along time.

    `call(sub, positions)` runs the (CFG-combined) DiT on a sub-volume shaped
    (1, C, 2n, h, w) and returns its velocity; only the first n frames (the
    right eye) are used. Returns the blended velocity for the right eye,
    shape (1, C, T, H, W).
    """
    _, C, _, H, W = x.shape
    acc = torch.zeros((1, C, T, H, W), dtype=torch.float32, device=x.device)
    wsum = torch.zeros((1, 1, T, H, W), dtype=torch.float32, device=x.device)
    for i_tile, tile in enumerate(tiles):
        ts = slice(tile.t.start, tile.t.end)
        hs = slice(tile.h.start, tile.h.end)
        ws = slice(tile.w.start, tile.w.end)
        n = tile.t.size
        sub = torch.cat(
            [x[:, :, ts, hs, ws], x[:, :, T + tile.t.start : T + tile.t.end, hs, ws]], dim=2
        )
        pos = None
        if n != T:
            # keep the right<->left RoPE offset of T frames the LoRA was trained with
            pos = torch.cat([
                torch.arange(tile.t.start, tile.t.end),
                torch.arange(T + tile.t.start, T + tile.t.end),
            ])
        v = call(sub, pos, i_tile) if call_takes_index(call) else call(sub, pos)
        del sub
        w = tile.weight.to(x.device)
        acc[:, :, ts, hs, ws] += v[:, :, :n].float() * w
        wsum[:, :, ts, hs, ws] += w
        del v
    return (acc / wsum).to(x.dtype)


def call_takes_index(call) -> bool:
    return getattr(call, "_takes_tile_index", False)


@torch.no_grad()
def long_video_lowvram(
    pipe,
    prompt: List[str],
    input_video: List[torch.Tensor],
    cfg: LowVRAMConfig,
    negative_prompt: str = "",
    seed: Optional[int] = 0,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    cfg_scale: float = 5.0,
    num_inference_steps: int = 50,
    sigma_shift: float = 5.0,
    denoising_strength: float = 1.0,
):
    device = pipe.device
    pipe.scheduler.set_timesteps(
        num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift
    )

    inputs_posi = {
        "prompt": prompt,
        "long": True,
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_shared = dict(
        input_image=None, end_image=None, input_video=input_video, input_video_r=None,
        input_video_depth=None, denoising_strength=denoising_strength, control_video=None,
        reference_image=None, camera_control_direction=None, camera_control_speed=1 / 54,
        camera_control_origin=None, vace_video=None, vace_video_mask=None,
        vace_reference_image=None, vace_scale=1.0, seed=seed, rand_device="cpu",
        height=height, width=width, num_frames=num_frames, cfg_scale=cfg_scale,
        cfg_merge=False, sigma_shift=sigma_shift, motion_bucket_id=None,
        tiled=True, tile_size=cfg.vae_tile, tile_stride=cfg.vae_stride,
        sliding_window_size=None, sliding_window_stride=None, input_audio=None,
        audio_sample_rate=16000, s2v_pose_video=None, audio_embeds=None,
        s2v_pose_latents=None, motion_video=None,
    )
    emit("stage", "encode")
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    # Everything the units produced that we feed in per-tile ourselves.
    noise = inputs_shared.pop("noise")
    seg_latents = inputs_shared.pop("input_latents")
    for k in ("latents", "latents_depth", "input_video"):
        inputs_shared.pop(k, None)
    if cfg.offload_finished_latents:
        seg_latents = [l.to("cpu") for l in seg_latents]
    prompts = inputs_posi.pop("context")
    neg_context = inputs_nega["context"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    dit = models["dit"]

    _, C, T, H, W = noise.shape
    tiles = plan_tiles(T, H, W, cfg.tile_hw, cfg.tile_overlap_hw, cfg.t_chunk, cfg.t_overlap)
    if cfg.verbose:
        tok_full = 2 * T * (H // 2) * (W // 2)
        biggest = max(t.t.size * t.h.size * t.w.size for t in tiles) * 2 // 4
        print(f"[lowvram] latent {T}x{H}x{W}; {describe(tiles)}")
        print(f"[lowvram] DiT tokens per call: {biggest:,} (untiled would be {tok_full:,})")

    tea = None  # per-segment _TileTeaCaches

    def velocity(x: torch.Tensor, timestep: torch.Tensor, ctx_posi) -> torch.Tensor:
        def call(sub, pos, idx=None):
            with frame_positions(dit, pos):
                tc_p = tea.get((idx, "pos")) if tea else None
                v_pos = pipe.model_fn(**models, **inputs_shared, context=ctx_posi,
                                      latents=sub, timestep=timestep, mode="rgb", tea_cache=tc_p)
                if cfg_scale == 1.0:
                    return v_pos
                tc_n = tea.get((idx, "neg")) if tea else None
                v_neg = pipe.model_fn(**models, **inputs_shared, context=neg_context,
                                      latents=sub, timestep=timestep, mode="rgb", tea_cache=tc_n)
                return v_neg + cfg_scale * (v_pos - v_neg)

        call._takes_tile_index = True
        return tiled_velocity(call, x, T, tiles)

    latents_all = []
    overlap_clean = None
    ov = cfg.overlap_latents
    emit("stage", "denoise")
    for i_seg, left_latents in enumerate(tqdm(seg_latents, desc="segments")):
        t0 = time.time()
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        pipe.load_models_to_device(pipe.in_iteration_models)
        left_latents = left_latents.to(device)
        latents = noise.clone()
        ctx = prompts[i_seg]
        tea = _TileTeaCaches(len(pipe.scheduler.timesteps), cfg.teacache) if cfg.teacache > 0 else None
        t_steps = time.time()
        t_last = t_steps
        if cfg.verbose and i_seg == 0:
            print(vram_report(device, "before first step"), flush=True)

        for step_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc=f"seg {i_seg+1}", leave=False)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)
            if overlap_clean is not None:
                latents[:, :, :ov] = pipe.scheduler.add_noise(
                    overlap_clean.to(device), noise[:, :, :ov], timestep
                )
            x = torch.cat([latents, left_latents], dim=2)
            v = velocity(x, timestep, ctx)
            del x
            latents = pipe.scheduler.step(v, pipe.scheduler.timesteps[step_id], latents)
            del v
            if cfg.verbose:
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                now = time.time()
                line = f"[lowvram] seg {i_seg+1} step {step_id+1}/{len(pipe.scheduler.timesteps)}: {now - t_last:.1f} s"
                if step_id == 0 or step_id == len(pipe.scheduler.timesteps) - 1:
                    line += "\n" + vram_report(device, "during denoising")
                print(line, flush=True)
                t_last = now
            emit("progress", i_seg + 1, len(seg_latents), step_id + 1, len(pipe.scheduler.timesteps))

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        step_time = (time.time() - t_steps) / max(1, len(pipe.scheduler.timesteps))
        tea_note = f", TeaCache skipped {tea.skips}/{tea.calls} DiT passes" if tea else ""
        del left_latents
        # Overlap for the next segment: decode, keep last 9 frames, re-encode (as upstream).
        vae_models = ["vae", *pipe.in_iteration_models] if cfg.keep_dit_resident else ["vae"]
        pipe.load_models_to_device(vae_models)
        if i_seg < len(seg_latents) - 1:
            tail = pipe.vae.decode(latents, device=device, tiled=True,
                                   tile_size=cfg.vae_tile, tile_stride=cfg.vae_stride)[:, :, -(4 * (ov - 1) + 1):]
            overlap_clean = pipe.vae.encode(tail, device=device, tiled=True,
                                            tile_size=cfg.vae_tile, tile_stride=cfg.vae_stride).to("cpu", pipe.torch_dtype)
            del tail
        latents_all.append(latents.to("cpu") if cfg.offload_finished_latents else latents)
        del latents
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cfg.verbose:
            print(f"[lowvram] segment {i_seg+1}/{len(seg_latents)} in {time.time()-t0:.0f}s "
                  f"({step_time:.2f} s/step{tea_note})  {_gpu_stats(device)}")

    emit("stage", "decode")
    pipe.load_models_to_device(["vae"])
    videos = []
    for lat in latents_all:
        video = pipe.vae.decode(lat, device=device, tiled=True,
                                tile_size=cfg.vae_tile, tile_stride=cfg.vae_stride)
        videos.append(pipe.vae_output_to_video(video))
        del video
    pipe.load_models_to_device([])
    return videos
