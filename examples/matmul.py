"""Example: Naive matrix multiplication kernel — a basic optimization target.

Computes C = A @ B for row-major float32 matrices. Each thread computes one
output element by iterating over the K (reduction) dimension. The naive version
has poor L1/L2 reuse and no shared memory tiling — plenty of room for krnl to
optimize (shared memory tiles, register blocking, vectorized loads, etc.).

Structure:
  - @cute.kernel  — device-side GPU code
  - @cute.jit     — host-side launcher
"""

import math
import torch
import cutlass
import cutlass.cute as cute


BLOCK_M = 16  # tile rows per block
BLOCK_N = 16  # tile cols per block


@cute.kernel
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
    a_row_stride: cute.Int32,
    b_row_stride: cute.Int32,
    c_row_stride: cute.Int32,
):
    """Device-side: compute C[row, col] = sum_k A[row, k] * B[k, col]."""
    row = cute.blockIdx.y * cute.blockDim.y + cute.threadIdx.y
    col = cute.blockIdx.x * cute.blockDim.x + cute.threadIdx.x

    if row >= M or col >= N:
        return

    acc = 0.0
    k = 0
    while k < K:
        acc += a_ptr[row * a_row_stride + k] * b_ptr[k * b_row_stride + col]
        k += 1

    c_ptr[row * c_row_stride + col] = acc


@cute.jit
def matmul_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Host-side: compile and launch matmul_kernel."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (math.ceil(N / BLOCK_N), math.ceil(M / BLOCK_M))
    cute.launch(
        matmul_kernel,
        grid=grid,
        block=(BLOCK_N, BLOCK_M),
    )(
        a, b, c,
        M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
    )
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
