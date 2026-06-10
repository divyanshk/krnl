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
    b_ptr,
    c_ptr,
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    BATCH: cutlass.Constexpr[int],
    HIDDEN: cutlass.Constexpr[int],
    SEQ: cutlass.Constexpr[int],
    CONV_K: cutlass.Constexpr[int],
):
    bx, by, bz = cute.arch.block_idx()
    tx, ty, _ = cute.arch.thread_idx()

    batch_idx = bz
    h = by * BLOCK_H + ty
    s = bx * BLOCK_S + tx

    HALO = CONV_K - 1
    SMEM_S = BLOCK_S + HALO

    smem_alloc = SmemAllocator()
    smem_bx = smem_alloc.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BLOCK_H, SMEM_S), stride=(SMEM_S, 1)),
        16,
    )

    # base seq index of the shared tile (left halo region starts before tile)
    tile_s0 = bx * BLOCK_S - HALO

    # Cooperatively load Bx = B * x into shared memory (including left halo).
    # Each thread loads SMEM_S/BLOCK_S columns; SMEM_S = BLOCK_S + HALO.
    # Loop over the smem columns this thread is responsible for.
    if h < HIDDEN:
        col = tx
        while col < SMEM_S:
            gpos = tile_s0 + col
            val = 0.0
            if gpos >= 0 and gpos < SEQ:
                val = b_ptr[batch_idx, h, gpos] * x_ptr[batch_idx, h, gpos]
            smem_bx[ty, col] = val
            col += BLOCK_S

    cute.arch.sync_threads()

    acc = 0.0
    if h < HIDDEN and s < SEQ:
        acc = bias_ptr[h]
        # this thread's output position s maps to smem column index (s - tile_s0)
        # = HALO + tx. tap k uses pos = s - k -> smem col = HALO + tx - k.
        base_col = HALO + tx
        for k in cutlass.range(CONV_K):
            scol = base_col - k
            gpos = s - k
            if gpos >= 0:
                acc += smem_bx[ty, scol] * w_ptr[h, CONV_K - 1 - k]
        y_ptr[batch_idx, h, s] = c_ptr[batch_idx, h, s] * acc

@cute.jit
def host(
    b_t: torch.Tensor,
    c_t: torch.Tensor,
    x_t: torch.Tensor,
    w_t: torch.Tensor,
    bias_t: torch.Tensor,
    y_t: torch.Tensor,
    BATCH: cutlass.Constexpr[int],
    HIDDEN: cutlass.Constexpr[int],
    SEQ: cutlass.Constexpr[int],
    CONV_K: cutlass.Constexpr[int],
):
    grid = (math.ceil(SEQ / BLOCK_S), math.ceil(HIDDEN / BLOCK_H), BATCH)
    kernel(
        b_t, c_t, x_t, w_t, bias_t, y_t,
        BATCH, HIDDEN, SEQ, CONV_K,
    ).launch(grid=grid, block=(BLOCK_S, BLOCK_H, 1))

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
