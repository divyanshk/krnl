"""Example: Vector addition kernel for testing krnl.

This file demonstrates the expected structure for krnl input files:
  - @cute.kernel  — device-side GPU code (runs on GPU)
  - @cute.jit     — host-side launcher (handles compilation + launch)
  - *_ref         — PyTorch reference for correctness validation
  - get_test_inputs — sample inputs for the kernel
"""

import math
import torch
import cutlass
import cutlass.cute as cute


@cute.kernel
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n: cute.Int32,
):
    """Device-side: element-wise out = x + y."""
    tid = cute.threadIdx.x + cute.blockIdx.x * cute.blockDim.x
    if tid < n:
        out_ptr[tid] = x_ptr[tid] + y_ptr[tid]


@cute.jit
def vector_add_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Host-side: compile and launch vector_add_kernel."""
    n = x.numel()
    out = torch.empty_like(x)
    block = 256
    grid = math.ceil(n / block)
    vector_add_kernel[grid, block](x, y, out, n)
    return out


def vector_add_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation."""
    return x + y


def get_test_inputs():
    """Generate test inputs for the kernel."""
    n = 1024 * 1024  # 1M elements
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, y)
