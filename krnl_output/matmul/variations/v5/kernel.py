"""Example: Naive matrix multiplication kernel — a basic optimization target.

Computes C = A @ B for row-major float32 matrices. Each thread computes one
output element by iterating over the K (reduction) dimension. The naive version
has poor L1/L2 reuse and no shared memory tiling — plenty of room for krnl to
optimize (shared memory tiles, register blocking, vectorized loads, etc.).

Structure:
  - @cute.kernel  — device-side GPU code
  - @cute.jit     — host-side JIT launcher (receives Constexpr M/N/K)
  - launch — regular Python wrapper; validates shapes and allocates output
"""

import math

import torch

import cutlass

import cutlass.cute as cute

BLOCK_M = 16

BLOCK_N = 16
from cutlass.utils import SmemAllocator
BLK_M_V5 = 32
BLK_N_V5 = 32
BLK_K_V5 = 32
BLK_THREADS_Y_V5 = 8

@cute.kernel
def kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    """Device-side: tiled GEMM with 32x8 block for coalesced memory access."""
    from cutlass.utils import SmemAllocator

    bx, by, _ = cute.arch.block_idx()
    tx, ty, _ = cute.arch.thread_idx()

    # BLK_M_V5=32, BLK_N_V5=32, BLK_K_V5=32
    # Block is (32, 8): tx in [0,32), ty in [0,8)
    # Thread linear ID = ty*32 + tx (0..255)
    # Warp lane = tx (since blockDim.x=32), warp = ty // (32//32) = ty
    # Each warp: same ty, tx=0..31 → accesses 32 consecutive columns → COALESCED

    smem_alloc = SmemAllocator()
    # smem_a: (BLK_M_V5 x BLK_K_V5) = (32 x 32) row-major
    smem_a = smem_alloc.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BLK_M_V5, BLK_K_V5), stride=(BLK_K_V5, 1)),
        16,
    )
    # smem_b: (BLK_K_V5 x BLK_N_V5) = (32 x 32) row-major
    smem_b = smem_alloc.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BLK_K_V5, BLK_N_V5), stride=(BLK_N_V5, 1)),
        16,
    )

    # Global row/col for this thread's output
    row = by * BLK_M_V5 + ty * (BLK_M_V5 // BLK_THREADS_Y_V5) + 0
    col = bx * BLK_N_V5 + tx

    # Accumulator for this thread's output element(s)
    # Each thread computes (BLK_M_V5/BLK_THREADS_Y_V5) x 1 output elements
    # With BLK_M_V5=32, BLK_THREADS_Y_V5=8: 4 rows per thread
    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    acc2 = cutlass.Float32(0.0)
    acc3 = cutlass.Float32(0.0)

    num_k_tiles = (K + BLK_K_V5 - 1) // BLK_K_V5

    for k_tile in cutlass.range(num_k_tiles):
        k_base = k_tile * BLK_K_V5

        # Load A tile: (32 x 32) with 256 threads = 4 elements per thread
        # Thread (tx, ty) loads rows [ty*4, ty*4+1, ty*4+2, ty*4+3], col=tx
        # tx=0..31 → consecutive columns → COALESCED
        a_row0 = by * BLK_M_V5 + ty * 4 + 0
        a_row1 = by * BLK_M_V5 + ty * 4 + 1
        a_row2 = by * BLK_M_V5 + ty * 4 + 2
        a_row3 = by * BLK_M_V5 + ty * 4 + 3
        a_col = k_base + tx

        if a_row0 < M and a_col < K:
            smem_a[ty * 4 + 0, tx] = a_ptr[a_row0, a_col]
        else:
            smem_a[ty * 4 + 0, tx] = cutlass.Float32(0.0)

        if a_row1 < M and a_col < K:
            smem_a[ty * 4 + 1, tx] = a_ptr[a_row1, a_col]
        else:
            smem_a[ty * 4 + 1, tx] = cutlass.Float32(0.0)

        if a_row2 < M and a_col < K:
            smem_a[ty * 4 + 2, tx] = a_ptr[a_row2, a_col]
        else:
            smem_a[ty * 4 + 2, tx] = cutlass.Float32(0.0)

        if a_row3 < M and a_col < K:
            smem_a[ty * 4 + 3, tx] = a_ptr[a_row3, a_col]
        else:
            smem_a[ty * 4 + 3, tx] = cutlass.Float32(0.0)

        # Load B tile: (32 x 32) with 256 threads = 4 elements per thread
        # Thread (tx, ty) loads row=(ty*4+i), col=tx for i=0..3
        # But B is (K x N): row=k, col=n
        # tx=0..31 → consecutive columns → COALESCED
        b_row0 = k_base + ty * 4 + 0
        b_row1 = k_base + ty * 4 + 1
        b_row2 = k_base + ty * 4 + 2
        b_row3 = k_base + ty * 4 + 3
        b_col = bx * BLK_N_V5 + tx

        if b_row0 < K and b_col < N:
            smem_b[ty * 4 + 0, tx] = b_ptr[b_row0, b_col]
        else:
            smem_b[ty * 4 + 0, tx] = cutlass.Float32(0.0)

        if b_row1 < K and b_col < N:
            smem_b[ty * 4 + 1, tx] = b_ptr[b_row1, b_col]
        else:
            smem_b[ty * 4 + 1, tx] = cutlass.Float32(0.0)

        if b_row2 < K and b_col < N:
            smem_b[ty * 4 + 2, tx] = b_ptr[b_row2, b_col]
        else:
            smem_b[ty * 4 + 2, tx] = cutlass.Float32(0.0)

        if b_row3 < K and b_col < N:
            smem_b[ty * 4 + 3, tx] = b_ptr[b_row3, b_col]
        else:
            smem_b[ty * 4 + 3, tx] = cutlass.Float32(0.0)

        cute.arch.sync_threads()

        # Compute: each thread computes 4 output elements (4 rows, 1 col)
        # Thread output rows: ty*4+0, ty*4+1, ty*4+2, ty*4+3
        # Thread output col: tx (= bx*BLK_N_V5 + tx)
        for k in cutlass.range(BLK_K_V5):
            b_val = smem_b[k, tx]
            acc0 += smem_a[ty * 4 + 0, k] * b_val
            acc1 += smem_a[ty * 4 + 1, k] * b_val
            acc2 += smem_a[ty * 4 + 2, k] * b_val
            acc3 += smem_a[ty * 4 + 3, k] * b_val

        cute.arch.sync_threads()

    # Write outputs
    out_col = bx * BLK_N_V5 + tx
    out_row0 = by * BLK_M_V5 + ty * 4 + 0
    out_row1 = by * BLK_M_V5 + ty * 4 + 1
    out_row2 = by * BLK_M_V5 + ty * 4 + 2
    out_row3 = by * BLK_M_V5 + ty * 4 + 3

    if out_row0 < M and out_col < N:
        c_ptr[out_row0, out_col] = acc0
    if out_row1 < M and out_col < N:
        c_ptr[out_row1, out_col] = acc1
    if out_row2 < M and out_col < N:
        c_ptr[out_row2, out_col] = acc2
    if out_row3 < M and out_col < N:
        c_ptr[out_row3, out_col] = acc3

@cute.jit
def host(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    """Host-side JIT launcher with 32x8 block for coalesced access."""
    grid = (math.ceil(N / BLK_N_V5), math.ceil(M / BLK_M_V5))
    kernel(a, b, c, M, N, K).launch(grid=grid, block=(BLK_N_V5, BLK_THREADS_Y_V5))

def launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Public entry point: validates shapes, allocates C, then calls the JIT launcher."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    host(a, b, c, M, N, K)
    return c

def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation."""
    return torch.matmul(a, b)

def get_test_inputs():
    """Generate test inputs for the kernel."""
    M, K, N = 512, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    return (a, b)
