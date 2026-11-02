"""CPU tests for the low-VRAM path. Run: python -m pytest tests -q
Needs only torch + einops (no models, no GPU)."""
import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lowvram.tiling import plan_spans, plan_tiles  # noqa: E402
from lowvram.sampler import tiled_velocity  # noqa: E402
from lowvram.presets import auto_config, tokens_for  # noqa: E402


# --------------------------------------------------------------------- tiling
@pytest.mark.parametrize("length", [2, 30, 34, 60, 68, 90, 108, 120, 135 * 2 // 2 * 2, 240])
@pytest.mark.parametrize("tile,ov", [(60, 12), (48, 8), (24, 6), (108, 20), (64, 16)])
def test_spans_cover_and_align(length, tile, ov):
    spans = plan_spans(length, tile, ov, align=2)
    assert spans[0].start == 0 and spans[-1].end == length
    for s in spans:
        assert s.start % 2 == 0 and s.size % 2 == 0
        assert s.size <= max(tile, 2) or len(spans) == 1
    for a, b in zip(spans, spans[1:]):
        assert b.start <= a.end, "gap between tiles"
        assert b.start > a.start


@pytest.mark.parametrize("T,tc,tov", [(21, None, 4), (21, 13, 4), (21, 9, 4), (21, 5, 2)])
def test_weights_partition_of_unity(T, tc, tov):
    H, W = 34, 90
    tiles = plan_tiles(T, H, W, (16, 40), (6, 10), tc, tov)
    wsum = torch.zeros(T, H, W)
    for t in tiles:
        assert t.weight.min() > 0
        wsum[t.t.start:t.t.end, t.h.start:t.h.end, t.w.start:t.w.end] += t.weight
    assert wsum.min() > 0


# ------------------------------------------------------ tiled velocity maths
def _frame_mixing_model(sub, pos):
    """Spatially local, mixes each right frame only with *its* left frame.
    Any off-by-one in pairing right/left windows breaks equality."""
    n = sub.shape[2] // 2
    r, l = sub[:, :, :n], sub[:, :, n:]
    v_r = torch.tanh(r) * 0.7 + 0.3 * l ** 2
    return torch.cat([v_r, torch.zeros_like(l)], dim=2)


@pytest.mark.parametrize("tc", [None, 13, 9, 5])
def test_tiled_equals_full_for_local_model(tc):
    torch.manual_seed(0)
    T, C, H, W = 21, 16, 30, 54
    x = torch.randn(1, C, 2 * T, H, W)
    full = _frame_mixing_model(x, None)[:, :, :T]
    tiles = plan_tiles(T, H, W, (12, 20), (4, 8), tc, 4 if tc and tc > 5 else 2)
    out = tiled_velocity(_frame_mixing_model, x, T, tiles)
    torch.testing.assert_close(out, full, rtol=1e-5, atol=1e-5)


def test_positions_keep_training_offset():
    T = 21
    seen = []

    def spy(sub, pos):
        seen.append(pos)
        return torch.zeros_like(sub)

    tiles = plan_tiles(T, 8, 8, (8, 8), (0, 0), 9, 3)
    tiled_velocity(spy, torch.zeros(1, 1, 2 * T, 8, 8), T, tiles)
    for p in seen:
        n = len(p) // 2
        assert torch.equal(p[n:] - p[:n], torch.full((n,), T))
    assert seen[0][0] == 0 and seen[-1][len(seen[-1]) // 2 - 1] == T - 1


# ------------------------------------------------------------------- presets
@pytest.mark.parametrize("vram", [4, 6, 8, 10, 12, 16, 24])
@pytest.mark.parametrize("hw", [(60, 108), (60, 104), (68, 120), (134, 240)])
def test_auto_config_monotone_and_valid(vram, hw):
    cfg, info = auto_config(vram, *hw)
    th, tw = cfg.tile_hw
    assert th % 2 == 0 and tw % 2 == 0 and th <= hw[0] and tw <= hw[1]
    assert info["tokens_per_call"] == tokens_for(cfg.t_chunk or 21, th, tw)


def test_bigger_card_never_gets_smaller_tiles():
    # Only among cards where the DiT stays resident: below ~5.4 GB the weights
    # stream, which frees room for activations (bigger tiles, much slower).
    prev = 0
    for v in (6, 8, 10, 12, 16, 24):
        cfg, info = auto_config(v, 60, 108)
        assert info["tokens_per_call"] >= prev
        prev = info["tokens_per_call"]


# -------------------------------------------- DiT patches are exact (needs repo)
def _load_dit_module():
    import types

    # Load the real wan_video_dit.py without importing the whole diffsynthfinal
    # package (which pulls modelscope, transformers, ...). Its two relative
    # imports are only used for checkpoint detection / camera control.
    for name in ("diffsynthfinal", "diffsynthfinal.models"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [os.path.join(ROOT, *name.split("."))]
            sys.modules[name] = pkg
    if "diffsynthfinal.models.utils" not in sys.modules:
        u = types.ModuleType("diffsynthfinal.models.utils")
        u.hash_state_dict_keys = lambda *a, **k: ""
        sys.modules[u.__name__] = u
    if "diffsynthfinal.models.wan_video_camera_controller" not in sys.modules:
        c = types.ModuleType("diffsynthfinal.models.wan_video_camera_controller")
        c.SimpleAdapter = type("SimpleAdapter", (torch.nn.Module,), {})
        sys.modules[c.__name__] = c

    path = os.path.join(ROOT, "diffsynthfinal", "models", "wan_video_dit.py")
    spec = importlib.util.spec_from_file_location("diffsynthfinal.models.wan_video_dit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot import wan_video_dit standalone: {e}")
    return mod


def test_chunked_block_matches_original():
    mod = _load_dit_module()
    from lowvram.patches import apply_dit_patches

    torch.manual_seed(0)
    dim, heads, ffn = 128, 4, 320
    block = mod.DiTBlock(False, dim, heads, ffn).eval()
    ref = mod.DiTBlock(False, dim, heads, ffn).eval()
    ref.load_state_dict(block.state_dict())

    holder = torch.nn.Module()
    holder.blocks = torch.nn.ModuleList([block])
    assert apply_dit_patches(holder, chunk_tokens=37) == 1  # odd chunk on purpose

    f, h, w = 3, 6, 10
    head_dim = dim // heads
    freqs3 = mod.precompute_freqs_cis_3d(head_dim)
    freqs = torch.cat([
        freqs3[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs3[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs3[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)
    x = torch.randn(1, f * h * w, dim)
    ctx = torch.randn(1, 7, dim)
    t_mod = torch.randn(1, 6, dim)
    with torch.no_grad():
        a = ref(x, ctx, t_mod, freqs)
        b = block(x, ctx, t_mod, freqs)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_frame_positions_context_manager():
    from lowvram.patches import frame_positions

    class D:
        freqs = (torch.arange(100), torch.arange(5), torch.arange(5))

    d = D()
    pos = torch.tensor([3, 4, 5, 24, 25, 26])
    with frame_positions(d, pos):
        assert torch.equal(d.freqs[0][:6], pos)
    assert torch.equal(d.freqs[0], torch.arange(100))
