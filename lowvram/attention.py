"""Attention backend selection for the Wan DiT.

wan_video_dit.flash_attention() picks a kernel from module-level flags
(FA3 > FA2 > SageAttention > PyTorch SDPA) that are set once at import time.
This module lets us choose explicitly, verifies the chosen kernel on the GPU
against SDPA, and can benchmark all installed kernels at the real shape.

Self-attention is ~80% of the DiT's FLOPs at 60k+ tokens, so the kernel is
the single biggest speed lever.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
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
PROBE_TIMEOUT = int(os.environ.get("STEREOWORLD_ATTN_PROBE_TIMEOUT", "240"))
ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "attention_cache.json"


def _versions() -> str:
    """Key for the probe cache: anything whose change can change the result."""
    parts = [sys.version.split()[0], torch.__version__, str(torch.version.cuda)]
    for m in ("triton", "sageattention", "flash_attn", "flash_attn_interface"):
        try:
            from importlib.metadata import version
            parts.append(f"{m}={version(m.replace('_', '-'))}")
        except Exception:
            parts.append(f"{m}=-")
    try:
        parts.append(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(data: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def probe(name: str, tokens: int, device: str, timeout: int = PROBE_TIMEOUT, _cmd=None) -> dict:
    """Test one kernel in a child process so a Triton compile that hangs or a
    native crash can't take the generation run down with it."""
    cmd = _cmd or [sys.executable, str(Path(__file__).resolve()), "--probe", name,
                   "--tokens", str(tokens), "--device", str(device)]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": f"no result after {timeout} s (Triton compile hung?)"}
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("@@probe "):
            return json.loads(line[8:])
    tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [f"exit code {r.returncode}"]
    return {"ok": False, "why": f"crashed: {tail[0][:160]}"}


def select(name: str = "auto", device: str = "cuda", tokens: int = 16384, verbose: bool = True) -> str:
    """Point wan_video_dit at one backend. Returns the backend actually used."""
    from diffsynthfinal.models import wan_video_dit as dit_mod

    kernels = available_kernels()
    on_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    cache = _load_cache()
    key = _versions()
    results = cache.get(key, {}) if on_cuda else {}
    fresh = False

    def usable(n):
        nonlocal fresh
        if n not in kernels:
            return None
        if n == "sdpa":
            return (results.get("sdpa", {}).get("sec", 0.0), 0.0)
        if not on_cuda:
            return None
        r = results.get(n)
        if r is None:
            if verbose:
                print(f"[attention] testing {n} in a separate process "
                      f"(first use may compile Triton kernels, up to {PROBE_TIMEOUT} s)...", flush=True)
            r = probe(n, tokens, device)
            results[n] = r
            fresh = True
        if not r.get("ok"):
            if verbose:
                print(f"[attention] {n}: not usable - {r.get('why', '?')}", flush=True)
            return None
        return r["sec"], r["err"]

    chosen = "sdpa"
    if name == "auto":
        timings = {}
        for n in ("flash3", "flash", "sage", "sdpa"):
            if n not in kernels:
                continue
            if n == "sdpa" and on_cuda:
                if "sdpa" not in results:
                    try:
                        sec, _ = check_and_time(kernels[n], tokens=tokens, device=device)
                        results["sdpa"] = {"ok": True, "sec": sec, "err": 0.0}
                    except Exception as e:
                        results["sdpa"] = {"ok": True, "sec": float("inf"), "err": 0.0, "why": str(e)[:100]}
                    fresh = True
                timings[n] = (results["sdpa"]["sec"], 0.0)
                continue
            r = usable(n)
            if r is not None:
                timings[n] = r
        if verbose and on_cuda:
            print("[attention] " + ", ".join(f"{n} {t[0]*1000:.1f} ms" for n, t in timings.items())
                  + f"  (one call, {tokens} tokens)")
        if timings:
            chosen = min(timings, key=lambda n: timings[n][0])
        if fresh and on_cuda:
            cache[key] = results
            _save_cache(cache)
    else:
        if name not in kernels:
            if verbose:
                print(f"[attention] {name} is not installed - using sdpa")
        elif usable(name) is None:
            if verbose:
                print(f"[attention] {name} failed its self-test - using sdpa")
        else:
            chosen = name
        if fresh and on_cuda:
            cache[key] = results
            _save_cache(cache)

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


def _probe_main(name: str, tokens: int, device: str) -> None:
    fn = available_kernels().get(name)
    if fn is None:
        res = {"ok": False, "why": "not installed"}
    else:
        try:
            sec, err = check_and_time(fn, tokens=tokens, device=device)
            res = {"ok": err <= MAX_ERR, "sec": sec, "err": err,
                   **({} if err <= MAX_ERR else {"why": f"wrong output (max error {err:.3f})"})}
        except Exception as e:
            res = {"ok": False, "why": f"{type(e).__name__}: {str(e)[:150]}"}
    print("@@probe " + json.dumps(res), flush=True)


if __name__ == "__main__":
    # python lowvram/attention.py            -> benchmark installed kernels at real DiT sizes
    # python lowvram/attention.py --probe X  -> internal: test one kernel, print JSON
    import argparse

    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.probe:
        _probe_main(a.probe, a.tokens, a.device)
        sys.exit(0)
    ks = available_kernels()
    print("installed:", ", ".join(ks), flush=True)
    for tok in (8192, 65520):
        for n in ks:
            r = probe(n, tok, a.device) if n != "sdpa" else None
            if r is None:
                try:
                    sec, err = check_and_time(ks[n], tokens=tok, device=a.device)
                    r = {"ok": True, "sec": sec, "err": err}
                except Exception as e:
                    r = {"ok": False, "why": f"{type(e).__name__}: {str(e)[:100]}"}
            if r.get("ok"):
                print(f"{tok:6d} tokens  {n:7s} {r['sec']*1000:8.1f} ms   max err {r['err']:.4f}", flush=True)
            else:
                print(f"{tok:6d} tokens  {n:7s} not usable: {r.get('why')}", flush=True)
