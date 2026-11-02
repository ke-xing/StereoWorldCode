"""CPU tests for the low-VRAM path. Run: python -m pytest tests -q
Needs only torch + einops (no models, no GPU)."""
import importlib.util
import os
import sys
import time as time_mod

import numpy as np
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


# ------------------------------------------- captioner vision attention patch
class _DenseMaskVisionSdpaAttention(torch.nn.Module):
    """Same structure as VideoLLaMA3/Qwen2-VL vision SDPA attention: tokens are
    [seq, dim], a dense [1, seq, seq] block mask is built from cu_seqlens."""

    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        seq = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq, 3, self.heads, -1).permute(1, 0, 2, 3)
        if rotary_pos_emb is not None:  # simple per-token rotation stand-in
            q = q * torch.cos(rotary_pos_emb)[:, None, :] + torch.sin(rotary_pos_emb)[:, None, :]
            k = k * torch.cos(rotary_pos_emb)[:, None, :] - torch.sin(rotary_pos_emb)[:, None, :]
        mask = torch.zeros([1, seq, seq], dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            a, b = cu_seqlens[i - 1], cu_seqlens[i]
            mask[..., a:b, a:b] = True
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
        return self.proj(o.transpose(0, 1).reshape(seq, -1))


def test_vision_attention_blockwise_is_exact():
    from lowvram.caption import patch_vision_attention

    torch.manual_seed(0)
    attn = _DenseMaskVisionSdpaAttention()
    model = torch.nn.Sequential(attn)
    x = torch.randn(7 + 12 + 5 + 9, 32)
    cu = torch.tensor([0, 7, 19, 24, 33], dtype=torch.int32)
    rope = torch.randn(33, 8)
    with torch.no_grad():
        ref = attn(x, cu_seqlens=cu, rotary_pos_emb=rope)
        assert patch_vision_attention(model) == 1
        assert patch_vision_attention(model) == 0  # idempotent
        out = attn(x, cu_seqlens=cu, rotary_pos_emb=rope)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# --------------------------------------- end-to-end sampler with a fake pipe
def _load_file_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeTeaCache:
    """Same interface/semantics as diffsynthfinal TeaCache (skip every other step)."""

    def __init__(self, steps, thresh, model_id):
        self.step, self.steps, self.res, self.prev = 0, steps, None, None

    def check(self, dit, x, t_mod):
        skip = self.step not in (0, self.steps - 1) and self.step % 2 == 1 and self.res is not None
        self.step = (self.step + 1) % self.steps
        if not skip:
            self.prev = x.clone()
        return skip

    def store(self, h):
        self.res = h - self.prev

    def update(self, h):
        return h + self.res


def _fake_model_fn(dit=None, latents=None, timestep=None, context=None, tea_cache=None, **kw):
    """Tiny 'DiT': local in space, pairs right frame t with left frame t+n."""
    n = latents.shape[2] // 2
    x = latents
    if tea_cache is not None and tea_cache.check(dit, x, None):
        out = tea_cache.update(x)
    else:
        out = torch.tanh(x) * 0.5 + context.mean() * 0.01
        if tea_cache is not None:
            tea_cache.store(out)
    out = out.clone()
    out[:, :, :n] = out[:, :, :n] + 0.1 * latents[:, :, n:]
    return out


class _FakeVAE:
    def decode(self, z, device=None, **kw):
        b, c, t, h, w = z.shape
        return torch.zeros(b, 3, 4 * (t - 1) + 1, h * 8, w * 8)

    def encode(self, v, device=None, **kw):
        b, c, t, h, w = v.shape
        return torch.zeros(b, 16, (t + 3) // 4, h // 8, w // 8)


class _FakePipe:
    in_iteration_models = ("dit",)

    def __init__(self, T, H, W, segs):
        sched = _load_file_module("fm_sched", "diffsynthfinal/schedulers/flow_match.py")
        self.scheduler = sched.FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.device, self.torch_dtype = "cpu", torch.float32
        self.dit = torch.nn.Module()
        self.dit.freqs = (torch.zeros(1024), torch.zeros(1024), torch.zeros(1024))
        self.vae = _FakeVAE()
        self.model_fn = _fake_model_fn
        self.loaded = []
        g = torch.Generator().manual_seed(0)
        noise = torch.randn(1, 16, T, H, W, generator=g)
        left = [torch.randn(1, 16, T, H, W, generator=g) for _ in range(segs)]

        def unit(pipe, shared, posi, nega):
            shared.update(noise=noise, input_latents=left, latents=noise)
            posi["context"] = [torch.full((1, 4, 8), float(i)) for i in range(segs)]
            nega["context"] = torch.zeros(1, 4, 8)
            return shared, posi, nega

        self.units = [unit]

    def unit_runner(self, unit, pipe, shared, posi, nega):
        return unit(pipe, shared, posi, nega)

    def load_models_to_device(self, names):
        self.loaded.append(tuple(names))

    def vae_output_to_video(self, v):
        return [None] * v.shape[2]


@pytest.mark.parametrize("teacache,resident", [(0.0, False), (0.05, True)])
def test_long_video_lowvram_runs_end_to_end(monkeypatch, teacache, resident):
    import lowvram.sampler as smp

    fake_mod = type(sys)("diffsynthfinal.pipelines.wan_video_new")
    fake_mod.TeaCache = _FakeTeaCache
    monkeypatch.setitem(sys.modules, "diffsynthfinal.pipelines.wan_video_new", fake_mod)

    T, H, W = 21, 12, 20
    pipe = _FakePipe(T, H, W, segs=2)
    cfg = smp.LowVRAMConfig(tile_hw=(8, 12), tile_overlap_hw=(4, 4), teacache=teacache,
                            keep_dit_resident=resident, verbose=True)
    videos = smp.long_video_lowvram(pipe, prompt=["a", "b"], input_video=[None, None], cfg=cfg,
                                    num_inference_steps=6, height=H * 8, width=W * 8)
    assert len(videos) == 2 and len(videos[0]) == 81
    vae_loads = [n for n in pipe.loaded if "vae" in n]
    # between segments the DiT stays on the GPU when resident; the final decode frees it
    assert len(vae_loads) == 3
    assert all(("dit" in n) == resident for n in vae_loads[:-1])
    assert "dit" not in vae_loads[-1]


def test_teacache_tiled_equals_untiled_for_local_model(monkeypatch):
    """With per-tile caches, tiled + TeaCache == untiled + TeaCache for a local model."""
    import lowvram.sampler as smp

    fake_mod = type(sys)("diffsynthfinal.pipelines.wan_video_new")
    fake_mod.TeaCache = _FakeTeaCache
    monkeypatch.setitem(sys.modules, "diffsynthfinal.pipelines.wan_video_new", fake_mod)

    outs = []
    for tile in ((12, 20), (8, 12)):
        pipe = _FakePipe(21, 12, 20, segs=1)
        captured = {}
        orig_step = pipe.scheduler.step

        def step(v, t, x, _o=orig_step, _c=captured):
            _c.setdefault("v", []).append(v.clone())
            return _o(v, t, x)

        pipe.scheduler.step = step
        cfg = smp.LowVRAMConfig(tile_hw=tile, tile_overlap_hw=(4, 4), teacache=0.05, verbose=False)
        smp.long_video_lowvram(pipe, prompt=["a"], input_video=[None], cfg=cfg,
                               num_inference_steps=6, height=96, width=160)
        outs.append(torch.stack(captured["v"]))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-5)


# ----------------------------------------------------- attention selection
def test_attention_selector_picks_fastest_correct_kernel(monkeypatch, tmp_path):
    import time as _time
    import lowvram.attention as att

    dit_mod = _load_dit_module()  # stubbed import of the real wan_video_dit
    monkeypatch.setitem(sys.modules, "diffsynthfinal.models.wan_video_dit", dit_mod)
    pkg = sys.modules["diffsynthfinal.models"]
    monkeypatch.setattr(pkg, "wan_video_dit", dit_mod, raising=False)

    def slow_ok(q, k, v):
        _time.sleep(0.02)
        return att._sdpa(q, k, v)

    def fast_wrong(q, k, v):
        return torch.zeros_like(q)

    def fast_ok(q, k, v):
        return att._sdpa(q, k, v)

    # run the "GPU" code path on CPU
    monkeypatch.setattr(att.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(att, "_sync", lambda: None)
    _cat = att.check_and_time
    monkeypatch.setattr(att, "check_and_time", lambda fn, **kw: _cat(fn, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(att, "CACHE_FILE", tmp_path / "attention_cache.json")

    def _inproc_probe(n, tokens, device, timeout=None, _cmd=None):
        fn = att.available_kernels().get(n)
        if fn is None:
            return {"ok": False, "why": "not installed"}
        sec, err = att.check_and_time(fn, tokens=tokens, device=device)
        return {"ok": err <= att.MAX_ERR, "sec": sec, "err": err, "why": f"err {err:.3f}"}

    monkeypatch.setattr(att, "probe", _inproc_probe)
    sage_mod = type(sys)("sageattention")
    sage_mod.sageattn = lambda q, k, v, **kw: fast_ok(q, k, v)
    monkeypatch.setitem(sys.modules, "sageattention", sage_mod)

    monkeypatch.setattr(att, "available_kernels",
                        lambda: {"sdpa": slow_ok, "flash": fast_wrong, "sage": fast_ok})
    chosen = att.select("auto", device="cuda:0", tokens=256, verbose=False)
    assert chosen == "sage"
    assert dit_mod.SAGE_ATTN_AVAILABLE and not dit_mod.FLASH_ATTN_2_AVAILABLE and not dit_mod.FLASH_ATTN_3_AVAILABLE

    # explicit request for a broken kernel falls back to sdpa
    assert att.select("flash", device="cuda:0", tokens=256, verbose=False) == "sdpa"
    assert not dit_mod.SAGE_ATTN_AVAILABLE and not dit_mod.FLASH_ATTN_2_AVAILABLE
    # missing kernel falls back too
    assert att.select("flash3", device="cuda:0", tokens=256, verbose=False) == "sdpa"


def test_dit_flash_attention_uses_selected_backend(monkeypatch, tmp_path):
    import lowvram.attention as att

    dit_mod = _load_dit_module()
    pkg = sys.modules["diffsynthfinal.models"]
    monkeypatch.setattr(pkg, "wan_video_dit", dit_mod, raising=False)
    calls = []
    sage_mod = type(sys)("sageattention")

    def fake_sage(q, k, v, **kw):
        calls.append(q.shape)
        return att._sdpa(q, k, v)

    sage_mod.sageattn = fake_sage
    monkeypatch.setitem(sys.modules, "sageattention", sage_mod)
    monkeypatch.setattr(att, "available_kernels", lambda: {"sdpa": att._sdpa, "sage": fake_sage})
    monkeypatch.setattr(att.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(att, "_sync", lambda: None)
    _cat = att.check_and_time
    monkeypatch.setattr(att, "check_and_time", lambda fn, **kw: _cat(fn, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(att, "CACHE_FILE", tmp_path / "attention_cache.json")

    def _inproc_probe(n, tokens, device, timeout=None, _cmd=None):
        fn = att.available_kernels().get(n)
        if fn is None:
            return {"ok": False, "why": "not installed"}
        sec, err = att.check_and_time(fn, tokens=tokens, device=device)
        return {"ok": err <= att.MAX_ERR, "sec": sec, "err": err, "why": f"err {err:.3f}"}

    monkeypatch.setattr(att, "probe", _inproc_probe)
    assert att.select("sage", device="cuda:0", tokens=64, verbose=False) == "sage"
    calls.clear()
    q = torch.randn(1, 50, 4 * 16)
    out = dit_mod.flash_attention(q, q, q, num_heads=4)
    assert calls and calls[0] == (1, 4, 50, 16) and out.shape == q.shape


def test_attention_probe_survives_hang_and_crash(tmp_path):
    import lowvram.attention as att

    hang = tmp_path / "hang.py"
    hang.write_text("import time; time.sleep(60)")
    t = time_mod.time()
    r = att.probe("sage", 64, "cuda", timeout=2, _cmd=[sys.executable, str(hang)])
    assert not r["ok"] and "no result" in r["why"] and time_mod.time() - t < 30

    crash = tmp_path / "crash.py"
    crash.write_text("import sys; sys.stderr.write('Segfault in triton\\n'); sys.exit(139)")
    r = att.probe("sage", 64, "cuda", timeout=20, _cmd=[sys.executable, str(crash)])
    assert not r["ok"] and "crashed" in r["why"]

    ok = tmp_path / "ok.py"
    ok.write_text('print("noise"); print(\'@@probe {"ok": true, "sec": 0.01, "err": 0.002}\')')
    r = att.probe("sage", 64, "cuda", timeout=20, _cmd=[sys.executable, str(ok)])
    assert r == {"ok": True, "sec": 0.01, "err": 0.002}


def test_attention_cache_skips_reprobe(monkeypatch, tmp_path):
    import lowvram.attention as att

    dit_mod = _load_dit_module()
    pkg = sys.modules["diffsynthfinal.models"]
    monkeypatch.setattr(pkg, "wan_video_dit", dit_mod, raising=False)
    monkeypatch.setattr(att.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(att, "_sync", lambda: None)
    monkeypatch.setattr(att, "CACHE_FILE", tmp_path / "c.json")
    sage_mod = type(sys)("sageattention")
    sage_mod.sageattn = lambda q, k, v, **kw: att._sdpa(q, k, v)
    monkeypatch.setitem(sys.modules, "sageattention", sage_mod)
    monkeypatch.setattr(att, "available_kernels", lambda: {"sdpa": att._sdpa, "sage": sage_mod.sageattn})
    monkeypatch.setattr(att, "check_and_time", lambda fn, **kw: (0.5, 0.0))
    probes = []

    def fake_probe(n, tokens, device, timeout=None, _cmd=None):
        probes.append(n)
        return {"ok": True, "sec": 0.1, "err": 0.01}

    monkeypatch.setattr(att, "probe", fake_probe)
    assert att.select("auto", device="cuda:0", tokens=64, verbose=False) == "sage"
    assert att.select("auto", device="cuda:0", tokens=64, verbose=False) == "sage"
    assert probes == ["sage"]  # second run came from the cache


# ------------------------------------------------- SBS output geometry
def _util_crop_and_resize(image, H, W):
    """Copy of util.ImageCropAndResize.crop_and_resize (util imports transformers)."""
    import torchvision

    ow, oh = image.size
    scale = max(W / ow, H / oh)
    image = torchvision.transforms.functional.resize(
        image, (round(oh * scale), round(ow * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
    return torchvision.transforms.functional.center_crop(image, (H, W))


@pytest.mark.parametrize("src,HW", [((1920, 1080), (480, 832)), ((1920, 1080), (864, 480)),
                                    ((1080, 1920), (848, 480)), ((842, 474), (480, 848))])
def test_left_eye_matches_model_input_crop(src, HW):
    pytest.importorskip("torchvision")
    from PIL import Image
    from lowvram.output import cover_crop

    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (src[1], src[0], 3), dtype=np.uint8))
    a = np.asarray(cover_crop(im, *HW))
    b = np.asarray(_util_crop_and_resize(im, *HW))
    assert a.shape == (HW[0], HW[1], 3)
    assert np.array_equal(a, b)


def test_write_sbs_end_to_end(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("torchvision")
    import imageio
    from PIL import Image
    from lowvram.output import write_sbs

    # source: 320x180 with a white vertical bar at x=10..20 (falls in the cropped-away
    # band when processing at 112x112 -> must NOT appear on the left eye)
    src = tmp_path / "src.mp4"
    w = imageio.get_writer(str(src), fps=8, quality=9, macro_block_size=8)
    frame = np.zeros((176, 320, 3), np.uint8)
    frame[:, 10:20] = 255
    for _ in range(12):
        w.append_data(frame)
    w.close()

    H, W = 112, 112
    right = [[Image.new("RGB", (W, H), (0, 0, 255))] * 9, [Image.new("RGB", (W, H), (0, 255, 0))] * 9]
    out = tmp_path / "sbs.mp4"
    n = write_sbs(right, None, str(src), str(out), fps=8, overlap_frames=4, H=H, W=W)
    assert n == 12
    frames = [f for f in imageio.get_reader(str(out), format="ffmpeg")]  # list() trips on len()=inf
    assert frames[0].shape == (H, 2 * W, 3)
    left = frames[0][:, :W].astype(int)
    assert left.max() < 60, "uncropped edge content leaked into the left eye"
    assert frames[0][:, W:, 2].mean() > 180  # first segment (blue) on the right
    assert frames[-1][:, W:, 1].mean() > 180  # after overlap: second segment (green)
