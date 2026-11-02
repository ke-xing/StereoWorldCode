"""Attention backend selection for the Wan DiT.

wan_video_dit.flash_attention() picks a kernel from module-level flags
(FA3 > FA2 > SageAttention > PyTorch SDPA) that are set once at import time.
This module lets us choose explicitly, verifies the chosen kernel on the GPU
against SDPA, and can benchmark all installed kernels at the real shape.

Self-attention is ~80% of the DiT's FLOPs at 60k+ tokens, so the kernel is
the single biggest speed lever.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

BACKENDS = ("flash3", "flash", "sage", "sdpa")


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def available_kernels() -> Dict[str, Callable]:
    """name -> fn(q, k, v) with q/k/v in (b, heads, seq, dim) layout."""
    out: Dict[str, Callable] = {"sdpa": _sdpa}
    try:
        import flash_attn_interface

        def fa3(q, k, v):
            o = flash_attn_interface.flash_attn_func(*(t.transpose(1, 2) for t in (q, k, v)))
            o = o[0] if isinstance(o, tuple) else o
            return o.transpose(1, 2)

        out["flash3"] = fa3
    except Exception:
        pass
    try:
        import flash_attn

        def fa2(q, k, v):
            return flash_attn.flash_attn_func(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)

        out["flash"] = fa2
    except Exception:
        pass
    try:
        from sageattention import sageattn

        out["sage"] = lambda q, k, v: sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    except Exception:
        pass
    return out


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def check_and_time(fn, tokens=8192, heads=12, dim=128, device="cuda", dtype=torch.bfloat16, reps=3):
    """Returns (seconds per call, max abs error vs SDPA) or raises."""
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(1, heads, tokens, dim, generator=g).to(device, dtype) for _ in range(3))
    ref = _sdpa(q, k, v).float()
    out = fn(q, k, v)
    err = (out.float() - ref).abs().max().item()
    fn(q, k, v)  # warm-up (Triton compiles on first calls)
    _sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn(q, k, v)
    _sync()
    return (time.perf_counter() - t) / reps, err


MAX_ERR = 0.08  # bf16 / int8-QK kernels land around 1e-2; garbage is >> 0.1


def select(name: str = "auto", device: str = "cuda", tokens: int = 16384, verbose: bool = True) -> str:
    """Point wan_video_dit at one backend. Returns the backend actually used."""
    from diffsynthfinal.models import wan_video_dit as dit_mod

    kernels = available_kernels()
    on_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    def usable(n):
        if n not in kernels:
            return None
        if n == "sdpa" or not on_cuda:
            return (0.0, 0.0) if n == "sdpa" else None
        try:
            sec, err = check_and_time(kernels[n], tokens=tokens, device=device)
        except Exception as e:  # missing CUDA arch, triton not found, ...
            if verbose:
                print(f"[attention] {n}: not usable ({type(e).__name__}: {str(e)[:120]})")
            return None
        if err > MAX_ERR:
            if verbose:
                print(f"[attention] {n}: rejected, max error {err:.3f} vs SDPA")
            return None
        return sec, err

    chosen = "sdpa"
    if name == "auto":
        timings = {}
        for n in ("flash3", "flash", "sage", "sdpa"):
            if n not in kernels:
                continue
            if n == "sdpa" and on_cuda:
                try:
                    timings[n] = check_and_time(kernels[n], tokens=tokens, device=device)
                except Exception:
                    timings[n] = (float("inf"), 0.0)
                continue
            r = usable(n)
            if r is not None:
                timings[n] = r
        if verbose and on_cuda:
            print("[attention] " + ", ".join(f"{n} {t[0]*1000:.1f} ms" for n, t in timings.items())
                  + f"  (one call, {tokens} tokens)")
        if timings:
            chosen = min(timings, key=lambda n: timings[n][0])
    else:
        if name not in kernels:
            if verbose:
                print(f"[attention] {name} is not installed - using sdpa")
        elif usable(name) is None:
            if verbose:
                print(f"[attention] {name} failed its self-test - using sdpa")
        else:
            chosen = name

    # flash_attention() checks these flags in order FA3 > FA2 > Sage > SDPA
    dit_mod.FLASH_ATTN_3_AVAILABLE = chosen == "flash3"
    dit_mod.FLASH_ATTN_2_AVAILABLE = chosen == "flash"
    dit_mod.SAGE_ATTN_AVAILABLE = chosen == "sage"
    if chosen == "flash3":
        import flash_attn_interface
        dit_mod.flash_attn_interface = flash_attn_interface
    elif chosen == "flash":
        import flash_attn
        dit_mod.flash_attn = flash_attn
    elif chosen == "sage":
        from sageattention import sageattn
        dit_mod.sageattn = sageattn

    if verbose:
        extra = ""
        if chosen == "sdpa" and on_cuda:
            try:
                fa = torch.backends.cuda.is_flash_attention_available()
                extra = f" (PyTorch built-in flash kernel: {'yes' if fa else 'no'})"
            except AttributeError:
                pass
        print(f"[attention] using {chosen}{extra}")
    return chosen


if __name__ == "__main__":  # python -m lowvram.attention  -> benchmark at real DiT sizes
    import sys

    sys.path.insert(0, ".")
    ks = available_kernels()
    print("installed:", ", ".join(ks))
    for tok in (16384, 65520):
        for n, fn in ks.items():
            try:
                sec, err = check_and_time(fn, tokens=tok)
                print(f"{tok:6d} tokens  {n:7s} {sec*1000:8.1f} ms   max err {err:.4f}")
            except Exception as e:
                print(f"{tok:6d} tokens  {n:7s} failed: {type(e).__name__}: {str(e)[:100]}")
