# StereoWorld on low-VRAM GPUs

## GUI: StereoWorld Studio

Double-click `StereoWorld_Studio.bat` (Windows; installs PySide6 into the
`stereoworld` env on first start) or run `python gui.py`.

* **Queue**: drag videos or whole folders in; per-video output path and prompt.
* **Show plan**: dry-run of the tiling plan for the selected video, no models loaded.
* **Start queue / Stop**: every video runs as its own process, so an OOM or a
  stop never takes the GUI down and VRAM is fully released between videos.
* Live progress (stage, segment, denoising step, ETA), colour-coded log,
  last-segment peak VRAM in the status bar, *Copy command* for the exact CLI call.
* **Models** panel checks `models/` and downloads whatever is missing.
* **Environment** tab: pick the Python of your conda env, check torch/CUDA,
  packages and ffmpeg.
* Settings and the unfinished queue are saved to `gui_settings.json`.

Every GUI option maps 1:1 to a `main_lowvram.py` flag below.

`main_lowvram.py` is a drop-in alternative to `main.py`. Same models, same
LoRA, same segment/overlap scheme, same SBS output, but the memory peaks are
restructured so it can run on 6-12 GB cards.

```bash
# plan only (no models loaded) – see what it will do on an 8 GB card
python main_lowvram.py --video_path test/1.mp4 --vram-gb 8 --dry-run

# auto-detect VRAM, 4-bit captioner
python main_lowvram.py --video_path test/1.mp4 --output_path output/1.mp4 --caption-quant 4bit

# 6 GB card, skip the 7B captioner entirely
python main_lowvram.py --video_path test/1.mp4 --preset 6gb --prompt "a crowded market street, handheld"
```

Extra dependency only for `--caption-quant`: `pip install bitsandbytes`.

Windows: skip `xformers` from `requirements.txt` (it pins a torch version and
pip then installs the CPU-only torch from PyPI). If `torch.cuda.is_available()`
is False: `pip uninstall -y xformers` and
`pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

## Speed

At 832x480 one DiT call has ~65k tokens, and self-attention is ~80% of its
compute. In order of impact:

1. **Attention kernel** (`--attention auto`, GUI: *Attention kernel*). `auto`
   times every installed kernel on your GPU at startup, rejects any whose output
   differs from PyTorch's, and uses the fastest. The log shows e.g.
   `[attention] sage 9.1 ms, sdpa 24.3 ms` and `[attention] using sage`.
   Without extra installs the DiT uses PyTorch SDPA. On Windows:
   * **SageAttention** (usually fastest on RTX 30/40): needs Triton.
     `pip install triton-windows` (pick the version the triton-windows README
     lists for your torch), then either `pip install sageattention==1.0.6`
     (Triton-only, simplest) or a SageAttention 2 wheel from
     woct0rdho/SageAttention releases whose tags match your setup
     (`+cu126torch<version>-cp311`).
   * **flash-attn 2**: community Windows wheels must match Python, torch and
     CUDA exactly (e.g. mjun0812/flash-attention-prebuild-wheels). Also used by
     the captioner when present.
   Check what's installed with *Environment -> Benchmark attention* or
   `python -m lowvram.attention`. A kernel that fails its self-test is skipped,
   never used.
2. **TeaCache** (`--teacache 0.05`, GUI: *TeaCache*): reuses the DiT result on
   steps where the timestep embedding barely changes. One cache per tile and
   CFG branch, so it is correct with tiling. ~1.5x at 0.05, ~2x at 0.08; higher
   values soften fine detail. The log reports `TeaCache skipped N/M DiT passes`.
3. **Steps** (`--steps 30`): ~40% less time than 50, usually hard to tell apart.
4. **CFG 1.0** halves the DiT passes (no negative prompt pass), at some quality cost.
5. **Resolution**: tokens grow with pixels and attention with tokens squared.

The DiT now stays on the GPU for the whole run when it fits (`DiT kept
resident` in the log). Previously the offloader decided per layer at runtime
and could end up streaming layers over PCIe on every step when other apps used
VRAM. Each segment line now reports `s/step`.

## Where upstream spends VRAM, and what changed

| Stage | Upstream `main.py` | `main_lowvram.py` |
|---|---|---|
| Captioning | VideoLLaMA3-7B bf16 on GPU (~16 GB), 180 frames | `--prompt` (0 GB), `--caption-quant 4bit` (~5-6 GB), or `--caption-device cpu`; `--caption-max-frames` |
| Model load | `from_pretrained(device="cuda")` puts **all** weights on the GPU first, incl. the 11 GB umt5-xxl text encoder | weights load to CPU; the offloader streams T5 layer by layer and keeps the 1.3B DiT resident when it fits |
| Input pixels | every segment's frames converted to a bf16 tensor **on the GPU** before anything runs (~200 MB per 81-frame 480p segment, all segments at once) | stay in system RAM; the tiled VAE moves one tile at a time |
| DiT forward | full frame if latent is exactly 60x108, else fixed 60x108 tiles; whole 81-frame segment; right+left concatenated → ~68k tokens/call at 480p | configurable spatial tile **and** temporal window; tokens/call sized to the card |
| Inside each DiT block | RoPE casts q and k to **float64 complex** for the whole sequence; FFN materialises N x 8960 | exact token-chunked q/k/v/RoPE/o-proj and FFN (bit-for-bit same maths) |
| Tile blending | step each tile in bf16, hand-written linear blends with special cases | blend the *velocity* with 2-D/3-D feather weights in fp32, step once (identical for the linear flow-match step, fewer seams) |
| VAE | fixed 240x416 px tiles | 240x416 / 160x272 / 96x160 px by budget |
| Finished segments | latents kept on GPU | parked on CPU |

The first three rows alone are why upstream does not start on a 10-12 GB card
even though the DiT is only 1.3B.

## Spatial tiling

The DiT sees `[right-eye latents | left-eye latents]` concatenated along time.
A tile cuts the same `h x w` window out of both halves, so the model always
sees the left view it has to re-project. Tiles are kept **wide** (min 512 px)
and shrink in height first, because stereo disparity is horizontal: a narrow
tile can cut off the content that should slide into view.

## Temporal chunking

Used only when spatial shrinking is not enough (≈ 6 GB at 1080p-ish inputs).
An 81-frame segment (21 latent frames) is split into overlapping windows, e.g.
13+13 latent frames with 4 of overlap. The important detail: each window is
fed with **explicit RoPE frame indices** so a right-eye frame at position *t*
still sits exactly 21 positions before its left-eye partner, as in training.
Without that, a 13-frame window would put the pair 13 positions apart and the
LoRA's left→right correspondence degrades.

Trade-off: frames in different windows no longer attend to each other, so
long-range temporal consistency inside a segment relies on the overlap blend.
Segment-to-segment continuity (the re-noised 3-latent overlap) is unchanged.

## Tuning

The sampler prints `peak alloc` per segment. Start from the preset, then:

* room to spare → `--max-tokens` higher, or a bigger `--tile-size`
* OOM in the DiT → `--max-tokens` lower, or `--temporal-chunk 49`
* OOM in VAE → `--vae-tile 160x272`
* OOM in captioning → `--caption-quant 4bit --caption-max-frames 32` or `--prompt`
* still tight → `--resolution 432x768` (the biggest lever: tokens scale with pixels)

Manual overrides take pixels: `--tile-size 384x864 --tile-overlap 96x160`.

## Tests

```bash
python -m pytest tests -q   # CPU only, no weights needed
```

They check tile coverage and alignment, that tiled + windowed velocity equals
the untiled one for any spatially-local model (catches left/right pairing
bugs), that RoPE offsets are preserved, and that the chunked DiT block is
numerically identical to the original.
