# StereoWorld on low-VRAM GPUs

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
