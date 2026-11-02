"""Exact, memory-saving patches for the Wan DiT.

None of these change the maths: every op being chunked is per-token
(Linear, RMSNorm, RoPE, GELU), so the outputs are identical to the
original up to floating-point summation order of the attention kernel,
which is untouched.

What they save, per DiT forward with N tokens (dim=1536, ffn=8960):
  * RoPE: upstream casts q and k to float64 complex for the whole sequence
    (~2 x N x 1536 x 16 B). Chunked, that transient shrinks to one chunk.
  * FFN: the N x 8960 hidden activation (+ its GELU copy) shrinks to one chunk.
"""
from __future__ import annotations

import contextlib
import types

import torch
import torch.nn as nn


def _chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


class ChunkedFFN(nn.Module):
    def __init__(self, ffn: nn.Module, chunk_tokens: int):
        super().__init__()
        self.inner = ffn
        self.chunk_tokens = chunk_tokens

    def forward(self, x):
        n = x.shape[1]
        if n <= self.chunk_tokens:
            return self.inner(x)
        out = None
        for s, e in _chunks(n, self.chunk_tokens):
            y = self.inner(x[:, s:e])
            if out is None:
                out = torch.empty(x.shape[0], n, y.shape[-1], dtype=y.dtype, device=y.device)
            out[:, s:e] = y
        return out


def _chunked_self_attn_forward(self, x, freqs):
    from diffsynthfinal.models.wan_video_dit import rope_apply

    n = x.shape[1]
    size = self._lowvram_chunk
    q = k = v = None
    for s, e in _chunks(n, size):
        xs = x[:, s:e]
        qs = rope_apply(self.norm_q(self.q(xs)), freqs[s:e], self.num_heads)
        ks = rope_apply(self.norm_k(self.k(xs)), freqs[s:e], self.num_heads)
        vs = self.v(xs)
        if q is None:
            q = torch.empty(x.shape[0], n, qs.shape[-1], dtype=qs.dtype, device=qs.device)
            k = torch.empty_like(q)
            v = torch.empty(x.shape[0], n, vs.shape[-1], dtype=vs.dtype, device=vs.device)
        q[:, s:e], k[:, s:e], v[:, s:e] = qs, ks, vs
        del qs, ks, vs
    o = self.attn(q, k, v)
    del q, k, v
    out = None
    for s, e in _chunks(n, size):
        ys = self.o(o[:, s:e])
        if out is None:
            out = torch.empty(x.shape[0], n, ys.shape[-1], dtype=ys.dtype, device=ys.device)
        out[:, s:e] = ys
    return out


def apply_dit_patches(dit: nn.Module, chunk_tokens: int | None) -> int:
    """Patch every DiTBlock of `dit` in place. Returns number of blocks patched."""
    if not chunk_tokens:
        return 0
    patched = 0
    for block in dit.modules():
        if hasattr(block, "self_attn") and hasattr(block, "ffn"):
            if not isinstance(block.ffn, ChunkedFFN):
                block.ffn = ChunkedFFN(block.ffn, chunk_tokens)
            else:
                block.ffn.chunk_tokens = chunk_tokens
            sa = block.self_attn
            sa._lowvram_chunk = chunk_tokens
            sa.forward = types.MethodType(_chunked_self_attn_forward, sa)
            patched += 1
    return patched


@contextlib.contextmanager
def frame_positions(dit: nn.Module, positions: torch.Tensor | None):
    """Make the DiT use explicit temporal RoPE indices for the next forward.

    model_fn_wan_video builds frame frequencies as `dit.freqs[0][:f]`. We
    temporarily replace freqs[0] with `freqs[0][positions]` so that slice
    picks exactly the indices we want. This lets a temporal chunk of the
    right-eye latents keep the same RoPE offset to its matching left-eye
    latents that the LoRA saw in training (offset = full segment length).
    """
    if positions is None:
        yield
        return
    orig = dit.freqs
    f0 = orig[0]
    dit.freqs = (f0[positions.to(f0.device)], orig[1], orig[2])
    try:
        yield
    finally:
        dit.freqs = orig
