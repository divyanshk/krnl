"""Fused gated depthwise causal conv kernel — krnl optimization target.

Pipeline (mirrors the original benchmark `run()`):
  1. in_proj:  BCx = F.linear(x, in_proj_w, in_proj_b)        -> (B, S, 3H)
  2. split:    B_, C_, x_proj = chunk(BCx.T, 3, dim=1)          each (B, H, S)
  3. fused:    Bx = B_ * x_proj
               conv_out = depthwise_causal_conv1d(Bx, W, bias)   (kernel_size=K, groups=H)
               y = C_ * conv_out
  4. out_proj: out = F.linear(y.T, out_proj_w, out_proj_b)     -> (B, S, H)

The CuTe kernel below fuses step 3 (B*x_proj, depthwise causal conv, multiply by C).
Steps 1 & 4 stay as torch.F.linear in the `launch` wrapper. The naive version below
has every thread re-read B/x_proj/W for each conv tap — clear opportunities for
shared-memory tiling along the seq dim, register reuse across the conv window,
and vectorised loads along H.

Structure:
  - @cute.kernel  — device-side fused conv + gating (Claude optimizes this)
  - @cute.jit     — host-side JIT launcher (receives Constexpr B/H/S/K)
  - launch — Python wrapper; runs in_proj, splits, calls host, runs out_proj
"""

import math

import torch

import torch.nn.functional as F

import cutlass

import cutlass.cute as cute

BLOCK_H = 8

BLOCK_S = 32
from cutlass.utils import SmemAllocator
from cutlass.cute.typing import Float32

@cute.kernel
def kernel(
    b_ptr, c_ptr, x_ptr, w_ptr, bias_ptr, y_ptr,
    BATCH: cutlass.Constexpr[int],
    HIDDEN: cutlass.Constexpr[int],
    SEQ: cutlass.Constexpr[int],
    CONV_K: cutlass.Constexpr[int],
):
    bx, by, bz = cute.arch.block_idx()
    tx, _, _ = cute.arch.thread_idx()

    batch_idx = bz
    h = by
    tile_start = bx * BLOCK_S
    halo = CONV_K - 1

    smem_alloc = SmemAllocator()
    # SMEM holds Bx for [tile_start - halo ... tile_start + BLOCK_S - 1]
    # slot index s = (global pos) - (tile_start - halo)
    smem_bx = smem_alloc.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(SMEM_LEN, stride=1),
        16,
    )

    # ---- Aligned main load ----
    # smem slot (halo + i) <- Bx[tile_start + i]
    # tile_start is a multiple of BLOCK_S (=128) -> aligned, fully coalesced.
    i = tx
    while i < BLOCK_S:
        gpos = tile_start + i
        val = 0.0
        if h < HIDDEN and gpos >= 0 and gpos < SEQ:
            val = b_ptr[batch_idx, h, gpos] * x_ptr[batch_idx, h, gpos]
        smem_bx[halo + i] = val
        i = i + BLOCK_S

    # ---- Halo load (small, handled by first `halo` threads) ----
    # smem slot j (0..halo-1) <- Bx[tile_start - halo + j]
    if tx < halo:
        gpos = tile_start - halo + tx
        hval = 0.0
        if h < HIDDEN and gpos >= 0 and gpos < SEQ:
            hval = b_ptr[batch_idx, h, gpos] * x_ptr[batch_idx, h, gpos]
        smem_bx[tx] = hval

    cute.arch.sync_threads()

    s = tile_start + tx
    acc = 0.0
    if h < HIDDEN and s < SEQ:
        acc = bias_ptr[h]
        # output pos s maps to smem slot (halo + tx); tap k uses pos = s - k
        # -> smem slot = halo + tx - k.
        for k in cutlass.range(CONV_K):
            pos = s - k
            if pos >= 0:
                slot = tx + halo - k
                acc += smem_bx[slot] * w_ptr[h, CONV_K - 1 - k]
        y_ptr[batch_idx, h, s] = c_ptr[batch_idx, h, s] * acc

@cute.jit
def host(
    b_t, c_t, x_t, w_t, bias_t, y_t,
    BATCH: cutlass.Constexpr[int],
    HIDDEN: cutlass.Constexpr[int],
    SEQ: cutlass.Constexpr[int],
    CONV_K: cutlass.Constexpr[int],
):
    grid = (math.ceil(SEQ / BLOCK_S), HIDDEN, BATCH)
    kernel(
        b_t, c_t, x_t, w_t, bias_t, y_t,
        BATCH, HIDDEN, SEQ, CONV_K,
    ).launch(grid=grid, block=(BLOCK_S, 1, 1))

def launch(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
) -> torch.Tensor:
    """Public entry point: in_proj -> split -> fused CuTe kernel -> out_proj."""
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]

    BCx = F.linear(x, in_proj_weight, in_proj_bias)              # (B, S, 3H)
    BCx = BCx.transpose(-1, -2).contiguous()                      # (B, 3H, S)
    B_, C_, x_proj = BCx.chunk(3, dim=1)                          # each (B, H, S)
    B_ = B_.contiguous()
    C_ = C_.contiguous()
    x_proj = x_proj.contiguous()

    w = conv_weight.squeeze(1).contiguous()                       # (H, K)
    bias = conv_bias.contiguous()

    y = torch.empty(batch_size, hidden_size, seq_len, device=x.device, dtype=x.dtype)
    host(B_, C_, x_proj, w, bias, y, batch_size, hidden_size, seq_len, conv_kernel_size)

    y = y.transpose(-1, -2).contiguous()                          # (B, S, H)
    return F.linear(y, out_proj_weight, out_proj_bias)

def conv_gated_ref(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference — exact mirror of the original benchmark `run()`."""
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]

    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    BCx = BCx.transpose(-1, -2)
    B_, C_, x_proj = BCx.chunk(3, dim=1)
    Bx = B_ * x_proj
    Bx_padded = F.pad(Bx, (conv_kernel_size - 1, 0))
    conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=hidden_size)
    y = C_ * conv_out
    y = y.transpose(-1, -2).contiguous()
    return F.linear(y, out_proj_weight, out_proj_bias)

def get_test_inputs():
    """Deterministic inputs — config #4 from the bench table (B=2, S=2048, H=2048, K=4).

    Weights use small scaling (0.02) so accumulated values stay within fp32 precision
    for the 1e-2 atol/rtol validator (the H=2048 reductions would otherwise blow up).
    """
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 2048
    hidden_size = 2048
    conv_kernel_size = 4
    triple_hidden = 3 * hidden_size

    device = "cuda"
    dtype = torch.float32

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    in_proj_weight = torch.randn(triple_hidden, hidden_size, device=device, dtype=dtype) * 0.02
    in_proj_bias = torch.randn(triple_hidden, device=device, dtype=dtype) * 0.02
    conv_weight = torch.randn(hidden_size, 1, conv_kernel_size, device=device, dtype=dtype) * 0.1
    conv_bias = torch.randn(hidden_size, device=device, dtype=dtype) * 0.02
    out_proj_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype) * 0.02
    out_proj_bias = torch.randn(hidden_size, device=device, dtype=dtype) * 0.02

    return (x, in_proj_weight, in_proj_bias, conv_weight, conv_bias,
            out_proj_weight, out_proj_bias)
