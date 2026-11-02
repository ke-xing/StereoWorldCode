"""StereoWorld inference for low-VRAM GPUs.

Same pipeline as main.py (VideoLLaMA3 captions -> Wan2.1-1.3B + stereo LoRA,
segment-by-segment right-eye generation, SBS output), but:

  * the DiT runs on spatial tiles and, if needed, temporal windows whose
    feathered velocities are blended before each scheduler step
  * exact token-chunking of attention projections / RoPE / FFN inside the DiT
  * VAE tile size follows the VRAM budget
  * input pixels stay in system RAM, finished latents are parked on CPU
  * captioning can be skipped, quantised (4/8-bit) or run on CPU

Examples
  python main_lowvram.py --video_path in.mp4 --output_path out/sbs.mp4            # auto
  python main_lowvram.py --video_path in.mp4 --preset 8gb --caption-quant 4bit
  python main_lowvram.py --video_path in.mp4 --preset 6gb --prompt "a city street at night"
  python main_lowvram.py --video_path in.mp4 --vram-gb 8 --dry-run                 # print plan only
"""
import argparse
import os
import sys

MODEL_PATH = "./models"
NEGATIVE = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
SEGMENT_FRAMES = 81
OVERLAP_FRAMES = 9


def parse_hw(s):
    s = s.lower().replace(",", "x")
    h, w = s.split("x")
    return int(h), int(w)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video_path", "--video-path", default="test/1.mp4")
    p.add_argument("--output_path", "--output-path", default="output/1.mp4")

    g = p.add_argument_group("VRAM planning")
    g.add_argument("--preset", default="auto", choices=["auto", "24gb", "16gb", "12gb", "10gb", "8gb", "6gb"],
                   help="auto = use the detected VRAM of --device")
    g.add_argument("--vram-gb", type=float, help="pretend the GPU has this much VRAM (overrides detection)")
    g.add_argument("--max-tokens", type=int, help="DiT tokens per call; overrides the estimated budget")
    g.add_argument("--dry-run", action="store_true", help="print the tiling plan and exit")

    g = p.add_argument_group("manual overrides (pixels)")
    g.add_argument("--tile-size", type=parse_hw, help="DiT tile HxW in pixels, multiple of 16, e.g. 384x864")
    g.add_argument("--tile-overlap", type=parse_hw, help="tile overlap HxW in pixels, multiple of 16, e.g. 96x160")
    g.add_argument("--temporal-chunk", type=int,
                   help="video frames per temporal window inside an 81-frame segment (4k+1, e.g. 49, 33). "
                        "0 = whole segment")
    g.add_argument("--temporal-overlap", type=int, help="frames of overlap between temporal windows (multiple of 4)")
    g.add_argument("--vae-tile", type=parse_hw, help="VAE tile HxW in pixels, e.g. 160x272")
    g.add_argument("--chunk-tokens", type=int, help="token chunk for attention-proj/RoPE/FFN; 0 disables the patches")
    g.add_argument("--vram-limit", type=float, help="GiB the weight-offloader may fill before it starts streaming")
    g.add_argument("--resolution", type=parse_hw,
                   help="process at HxW (multiple of 16) instead of the source size; biggest single VRAM lever")

    g = p.add_argument_group("captioning")
    g.add_argument("--prompt", help="use this prompt for every segment and skip VideoLLaMA3")
    g.add_argument("--caption-quant", default="none", choices=["none", "8bit", "4bit"])
    g.add_argument("--caption-device", default=None, help="e.g. cpu; default = --device")
    g.add_argument("--caption-max-frames", type=int, default=None, help="default 180 (upstream), 64 on <=8 GB")

    g = p.add_argument_group("sampling")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--cfg", type=float, default=5.0)
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def working_size(video_path, resolution):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"cannot open {video_path}")
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if resolution:
        H, W = resolution
    else:
        H, W = int(h), int(w)
        if H * W > 1920 * 1080:  # same cap as upstream
            s = (H * W / (1920 * 1080)) ** 0.5
            H, W = int(H / s), int(W / s)
    return H // 16 * 16, W // 16 * 16, fps, n


def detect_vram(device):
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30


def build_config(args, H, W, vram_gib):
    from lowvram.presets import PRESETS, auto_config
    from dataclasses import replace

    if args.preset != "auto":
        vram_gib = PRESETS[args.preset]
    T_lat = (SEGMENT_FRAMES - 1) // 4 + 1
    cfg, info = auto_config(vram_gib, H // 8, W // 8, T_lat, max_tokens=args.max_tokens)

    if args.tile_size:
        cfg = replace(cfg, tile_hw=(args.tile_size[0] // 8 // 2 * 2, args.tile_size[1] // 8 // 2 * 2))
    if args.tile_overlap:
        cfg = replace(cfg, tile_overlap_hw=(args.tile_overlap[0] // 8 // 2 * 2, args.tile_overlap[1] // 8 // 2 * 2))
    if args.temporal_chunk is not None:
        tc = None if args.temporal_chunk in (0, SEGMENT_FRAMES) else (args.temporal_chunk - 1) // 4 + 1
        cfg = replace(cfg, t_chunk=tc)
    if args.temporal_overlap is not None:
        cfg = replace(cfg, t_overlap=max(1, args.temporal_overlap // 4))
    if args.vae_tile:
        th, tw = args.vae_tile[0] // 8, args.vae_tile[1] // 8
        cfg = replace(cfg, vae_tile=(th, tw), vae_stride=(max(1, th // 2), max(1, tw // 2)))
    if args.chunk_tokens is not None:
        info["ffn_chunk_tokens"] = args.chunk_tokens
    if args.caption_max_frames is None:
        args.caption_max_frames = 64 if vram_gib <= 8 else 180
    return cfg, info, vram_gib


def main():
    args = parse_args()
    H, W, fps, n_frames = working_size(args.video_path, args.resolution)

    vram = args.vram_gb or (None if args.preset != "auto" else detect_vram(args.device))
    if vram is None and args.preset == "auto":
        sys.exit("no CUDA device found; pass --vram-gb or --preset to plan anyway")
    cfg, info, vram = build_config(args, H, W, vram or 0)

    from lowvram.tiling import describe, plan_tiles
    from lowvram.presets import tokens_for

    T_lat = (SEGMENT_FRAMES - 1) // 4 + 1
    tiles = plan_tiles(T_lat, H // 8, W // 8, cfg.tile_hw, cfg.tile_overlap_hw, cfg.t_chunk, cfg.t_overlap)
    print(f"[plan] {W}x{H} @ {fps:.3f} fps, {n_frames} frames, VRAM budget {vram:.1f} GiB")
    print(f"[plan] DiT tile {cfg.tile_hw[1]*8}x{cfg.tile_hw[0]*8}px (overlap {cfg.tile_overlap_hw[1]*8}x{cfg.tile_overlap_hw[0]*8}px), "
          f"temporal window {'whole segment' if not cfg.t_chunk else f'{(cfg.t_chunk-1)*4+1} frames'}")
    print(f"[plan] {describe(tiles)}")
    print(f"[plan] tokens/call {info['tokens_per_call']:,} vs budget {info['budget_tokens']:,}"
          f"{'' if info['fits'] else '  (!) over budget even at minimum tile; expect weight streaming / OOM'}")
    print(f"[plan] VAE tile {cfg.vae_tile[1]*8}x{cfg.vae_tile[0]*8}px, chunk tokens {info['ffn_chunk_tokens']}, "
          f"DiT resident: {info['dit_resident']}, caption: "
          f"{'fixed prompt' if args.prompt else f'{args.caption_quant} on {args.caption_device or args.device}, max_frames {args.caption_max_frames}'}")
    if args.dry_run:
        return

    import gc
    import torch
    from diffsynthfinal.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
    from util import default_video_operator, preprocess_video, concat_videos_with_blending
    from lowvram.patches import apply_dit_patches
    from lowvram.sampler import long_video_lowvram
    from lowvram import caption as cap

    device = args.device
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    loader = default_video_operator(enable_long_video=True, force_resolution=True, height=H, width=W,
                                    num_frames=SEGMENT_FRAMES, overlap_frames=OVERLAP_FRAMES)
    segments = loader(args.video_path)

    # ---- captions ---------------------------------------------------------
    if args.prompt:
        captions = [args.prompt] * len(segments)
    else:
        cdev = args.caption_device or device
        model, processor = cap.load_caption_model(f"{MODEL_PATH}/VideoLLaMA3-7B", cdev, args.caption_quant)
        captions = []
        for i, seg in enumerate(segments):
            try:
                c = cap.caption_segment(seg["frames"], model, processor, cdev, fps, args.caption_max_frames)
            except torch.cuda.OutOfMemoryError:
                sys.exit("caption model OOM: use --caption-quant 4bit, --caption-max-frames 32, "
                         "--caption-device cpu or --prompt")
            print(f"[caption {i+1}/{len(segments)}] {c[:160]}{'...' if len(c) > 160 else ''}")
            captions.append(c)
        del model, processor
        cap.free_cuda()

    # pixels stay on the CPU; the tiled VAE moves one tile at a time
    pixel_segments = [preprocess_video(s["frames"], device="cpu") for s in segments]
    for s in segments:
        s["frames"] = None  # drop PIL copies, keep start/end indices
    gc.collect()

    # ---- generation -------------------------------------------------------
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=f"{MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                        offload_device="cpu"),
            ModelConfig(path=f"{MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                        offload_device="cpu"),
            ModelConfig(path=f"{MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.load_lora(pipe.dit, f"{MODEL_PATH}/stereo.safetensors", alpha=1)
    n = apply_dit_patches(pipe.dit, info["ffn_chunk_tokens"])
    if n:
        print(f"[lowvram] chunked attention/RoPE/FFN in {n} DiT blocks ({info['ffn_chunk_tokens']} tokens)")

    from lowvram.presets import BYTES_PER_TOKEN
    if args.vram_limit is not None:
        limit = args.vram_limit
    else:
        act = info["tokens_per_call"] * BYTES_PER_TOKEN / 2**30
        limit = max(1.0, vram - act - 0.75)
    print(f"[lowvram] weight offloader limit {limit:.1f} GiB")
    pipe.enable_vram_management(vram_limit=limit, vram_buffer=0.0)

    videos = long_video_lowvram(
        pipe,
        prompt=captions,
        input_video=pixel_segments,
        cfg=cfg,
        negative_prompt=NEGATIVE,
        seed=args.seed,
        height=H,
        width=W,
        num_frames=SEGMENT_FRAMES,
        cfg_scale=args.cfg,
        num_inference_steps=args.steps,
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    concat_videos_with_blending(videos, segments, args.video_path, args.output_path,
                                fps=fps, overlap_frames=OVERLAP_FRAMES)
    print(f"[done] {args.output_path}")


if __name__ == "__main__":
    main()
