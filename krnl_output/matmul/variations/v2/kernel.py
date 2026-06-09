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
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

@cute.kernel
def kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    """Device-side: shared-memory tiled matrix multiply C = A @ B (row-major float32)."""
    from cutlass.utils import SmemAllocator

    bx, by, _ = cute.arch.block_idx()
    tx, ty, _ = cute.arch.thread_idx()

    # Global row/col for this thread
    row = by * BLOCK_M + ty
    col = bx * BLOCK_N + tx

    # Allocate shared memory tiles for A and B
    smem_alloc = SmemAllocator()
    smem_A = smem_alloc.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BLOCK_M, BLOCK_K), stride=(BLOCK_K, 1)),
        16,
    )
    smem_B = smem_alloc.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BLOCK_K, BLOCK_N), stride=(BLOCK_N, 1)),
        16,
    )

    acc = 0.0

    num_tiles = (K + BLOCK_K - 1) // BLOCK_K

    for tile in cutlass.range(num_tiles):
        k_base = tile * BLOCK_K

        # Cooperatively load tile of A: thread (ty, tx) loads A[row, k_base + tx]
        a_row = by * BLOCK_M + ty
        a_col = k_base + tx
        if a_row < M and a_col < K:
            smem_A[ty, tx] = a_ptr[a_row, a_col]
        else:
            smem_A[ty, tx] = 0.0

        # Cooperatively load tile of B: thread (ty, tx) loads B[k_base + ty, col]
        b_row = k_base + ty
        b_col = bx * BLOCK_N + tx
        if b_row < K and b_col < N:
            smem_B[ty, tx] = b_ptr[b_row, b_col]
        else:
            smem_B[ty, tx] = 0.0

        cute.arch.sync_threads()

        # Compute partial dot product from shared memory tile
        for k in cutlass.range(BLOCK_K):
            acc = acc + smem_A[ty, k] * smem_B[k, tx]

        cute.arch.sync_threads()

    if row < M and col < N:
        c_ptr[row, col] = acc

@cute.jit
def host(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    """Host-side JIT launcher with shared-memory tiled kernel."""
    grid = (math.ceil(N / BLOCK_N), math.ceil(M / BLOCK_M))
    kernel(a, b, c, M, N, K).launch(grid=grid, block=(BLOCK_N, BLOCK_M))

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
