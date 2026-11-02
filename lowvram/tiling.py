"""Tile / window planning and feathered blending weights.

All sizes here are in *latent* units (Wan VAE: 8x spatial, 4x temporal).
The Wan DiT patchifies with (1, 2, 2), so spatial tile sizes and starts
must be even.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class Span:
    start: int
    end: int  # exclusive

    @property
    def size(self) -> int:
        return self.end - self.start


def plan_spans(length: int, tile: int, overlap: int, align: int = 1) -> List[Span]:
    """Cover [0, length) with windows of size `tile` overlapping by >= `overlap`.

    Windows are spread evenly so every overlap is at least the requested one,
    the first window starts at 0 and the last ends at `length`. Starts are
    snapped to multiples of `align`.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    tile = max(align, (tile // align) * align)
    if length <= tile:
        return [Span(0, length)]
    overlap = max(0, min(overlap, tile - align))
    stride = tile - overlap
    n = math.ceil((length - tile) / stride) + 1
    last_start = length - tile
    if length % align:
        raise ValueError(f"length {length} is not a multiple of {align}")
    spans = []
    for i in range(n):
        s = round(i * last_start / (n - 1))
        s = (s // align) * align
        spans.append(Span(s, s + tile))
    spans[-1] = Span(last_start, length)  # always flush with the far border
    # de-duplicate
    out: List[Span] = []
    for sp in spans:
        if not out or sp != out[-1]:
            out.append(sp)
    return out


def ramp_1d(span: Span, all_spans: List[Span], length: int) -> torch.Tensor:
    """Linear feather weights for one window.

    Inside an overlap with a neighbour the weight ramps from ~0 to 1; at the
    outer borders of the whole frame it stays 1. Weights are strictly > 0 so
    the normalised sum is always defined.
    """
    w = torch.ones(span.size, dtype=torch.float32)
    idx = all_spans.index(span)
    if idx > 0:
        prev = all_spans[idx - 1]
        ov = prev.end - span.start
        if ov > 0:
            w[:ov] = torch.minimum(w[:ov], torch.arange(1, ov + 1, dtype=torch.float32) / (ov + 1))
    if idx < len(all_spans) - 1:
        nxt = all_spans[idx + 1]
        ov = span.end - nxt.start
        if ov > 0:
            w[-ov:] = torch.minimum(w[-ov:], torch.arange(ov, 0, -1, dtype=torch.float32) / (ov + 1))
    return w


@dataclass(frozen=True)
class Tile:
    t: Span  # temporal window over the *target* (right-eye) latent frames
    h: Span
    w: Span
    weight: torch.Tensor  # (t, h, w) float32 on CPU


def plan_tiles(
    T: int,
    H: int,
    W: int,
    tile_hw: Tuple[int, int],
    overlap_hw: Tuple[int, int],
    t_chunk: int | None,
    t_overlap: int,
) -> List[Tile]:
    t_spans = plan_spans(T, t_chunk or T, t_overlap, align=1)
    h_spans = plan_spans(H, tile_hw[0], overlap_hw[0], align=2)
    w_spans = plan_spans(W, tile_hw[1], overlap_hw[1], align=2)
    tiles = []
    for ts in t_spans:
        wt = ramp_1d(ts, t_spans, T)
        for hs in h_spans:
            wh = ramp_1d(hs, h_spans, H)
            for ws in w_spans:
                ww = ramp_1d(ws, w_spans, W)
                weight = wt[:, None, None] * wh[None, :, None] * ww[None, None, :]
                tiles.append(Tile(ts, hs, ws, weight))
    return tiles


def describe(tiles: List[Tile]) -> str:
    ts = sorted({(t.t.start, t.t.end) for t in tiles})
    hs = sorted({(t.h.start, t.h.end) for t in tiles})
    ws = sorted({(t.w.start, t.w.end) for t in tiles})
    return (
        f"{len(tiles)} DiT calls/step/CFG-branch: "
        f"{len(ts)} temporal x {len(hs)} rows x {len(ws)} cols "
        f"(t={ts}, h={hs}, w={ws})"
    )
