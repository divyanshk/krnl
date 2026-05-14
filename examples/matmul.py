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


BLOCK_M = 16  # tile rows per block
BLOCK_N = 16  # tile cols per block


@cute.kernel
def kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    """Device-side: compute C[row, col] = sum_k A[row, k] * B[k, col]."""
    bx, by, _ = cute.arch.block_idx()
    tx, ty, _ = cute.arch.thread_idx()
    dimx, dimy, _ = cute.arch.block_dim()
    row = by * dimy + ty
    col = bx * dimx + tx

    if row < M and col < N:
        acc = 0.0
        for k in cutlass.range(K):
            acc += a_ptr[row, k] * b_ptr[k, col]
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
    """Host-side JIT launcher — M/N/K are constexpr so math.ceil and grid are Python values."""
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
