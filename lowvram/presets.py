"""Pick tile / chunk sizes for a given amount of VRAM.

The numbers are estimates for Wan2.1-T2V-1.3B (dim 1536, 30 blocks) in
bf16 with the exact chunking patches applied. They are deliberately
conservative; use --max-tokens to push harder once you have seen the
"peak alloc" lines the sampler prints for your card.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .sampler import LowVRAMConfig

GiB = 2**30

DIT_WEIGHTS_GIB = 2.9      # 1.3B params bf16 + LoRA fused
CONTEXT_GIB = 1.0          # CUDA context, allocator slack, VAE/T5 streaming buffers
BYTES_PER_TOKEN = 40_000   # live activations per DiT token with patches (+ safety)
BYTES_PER_TOKEN_UNPATCHED = 95_000

NATIVE_TILE = (60, 108)    # upstream's fixed latent tile (480x864 px)
MIN_TILE_H = 24
MIN_TILE_W = 64            # keep tiles wide: disparity is horizontal


def tokens_for(T: int, th: int, tw: int) -> int:
    return 2 * T * (th // 2) * (tw // 2)


def token_budget(vram_gib: float, dit_resident: bool = True, patched: bool = True) -> int:
    free = vram_gib - CONTEXT_GIB - (DIT_WEIGHTS_GIB if dit_resident else 0.6)
    per = BYTES_PER_TOKEN if patched else BYTES_PER_TOKEN_UNPATCHED
    return max(0, int(free * GiB / per))


def _even(x: int) -> int:
    return max(2, (x // 2) * 2)


def auto_config(
    vram_gib: float,
    H_lat: int,
    W_lat: int,
    T_lat: int = 21,
    max_tokens: Optional[int] = None,
    base: Optional[LowVRAMConfig] = None,
) -> tuple[LowVRAMConfig, dict]:
    """Return (config, info) fitting the DiT into `vram_gib`."""
    cfg = base or LowVRAMConfig()
    dit_resident = vram_gib >= DIT_WEIGHTS_GIB + CONTEXT_GIB + 1.5
    budget = max_tokens or token_budget(vram_gib, dit_resident)

    th, tw = min(NATIVE_TILE[0], H_lat), min(NATIVE_TILE[1], W_lat)
    t_chunk: Optional[int] = None

    # 1) shrink tile height (cheapest in quality for stereo)
    while tokens_for(T_lat, th, tw) > budget and th > MIN_TILE_H:
        th = _even(th - 4)
    # 2) temporal windows
    for tc in (13, 9, 7, 5):
        if tokens_for(t_chunk or T_lat, th, tw) <= budget:
            break
        t_chunk = min(tc, T_lat)
    # 3) last resort: narrow the tile
    while tokens_for(t_chunk or T_lat, th, tw) > budget and tw > MIN_TILE_W:
        tw = _even(tw - 8)

    ov_h = min(12, th // 4) // 2 * 2
    ov_w = min(20, tw // 4) // 2 * 2
    t_ov = 4 if t_chunk and t_chunk >= 9 else 2

    # VAE tiles (latent units); its decoder is the second-biggest consumer
    if vram_gib >= 10:
        vae_tile, vae_stride = (30, 52), (15, 26)
    elif vram_gib >= 6:
        vae_tile, vae_stride = (20, 34), (10, 17)
    else:
        vae_tile, vae_stride = (12, 20), (6, 10)

    cfg = replace(
        cfg,
        tile_hw=(th, tw),
        tile_overlap_hw=(ov_h, ov_w),
        t_chunk=t_chunk,
        t_overlap=t_ov,
        vae_tile=vae_tile,
        vae_stride=vae_stride,
    )
    chunk_tokens = 8192 if vram_gib < 8 else 16384
    info = dict(
        budget_tokens=budget,
        tokens_per_call=tokens_for(t_chunk or T_lat, th, tw),
        dit_resident=dit_resident,
        ffn_chunk_tokens=chunk_tokens,
        fits=tokens_for(t_chunk or T_lat, th, tw) <= budget,
    )
    return cfg, info


PRESETS = {
    # name: VRAM in GiB fed to auto_config
    "24gb": 24, "16gb": 16, "12gb": 12, "10gb": 10, "8gb": 8, "6gb": 6,
}
