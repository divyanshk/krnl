"""Example: Fused softmax kernel — a more realistic optimization target.

Row-wise softmax over a 2D matrix. The naive version loads each row from global
memory multiple times (max pass, exp pass, sum pass, divide pass). An optimized
version fuses all passes into a single loop using warp-level reductions.

Structure:
  - @cute.kernel  — device-side GPU code
  - @cute.jit     — host-side launcher
"""

import torch
from cuda import cute


BLOCK_COLS = 256  # threads per row; tune this as an optimization parameter


@cute.kernel
def softmax_kernel(
    out_ptr,
    inp_ptr,
    n_rows: cute.Int32,
    n_cols: cute.Int32,
    inp_row_stride: cute.Int32,
    out_row_stride: cute.Int32,
    BLOCK: cute.Int32,
):
    """Device-side: compute row-wise softmax."""
    row = cute.blockIdx.x
    if row >= n_rows:
        return

    tid = cute.threadIdx.x

    # Pass 1: find row max (for numerical stability)
    row_max = -3.4028235e+38  # -inf
    col = tid
    while col < n_cols:
        val = inp_ptr[row * inp_row_stride + col]
        if val > row_max:
            row_max = val
        col += BLOCK
    row_max = cute.warp_reduce_max(row_max)

    # Pass 2: accumulate shifted-exp sum
    exp_sum = 0.0
    col = tid
    while col < n_cols:
        exp_sum += cute.exp(inp_ptr[row * inp_row_stride + col] - row_max)
        col += BLOCK
    exp_sum = cute.warp_reduce_sum(exp_sum)

    # Pass 3: write normalized output
    col = tid
    while col < n_cols:
        val = cute.exp(inp_ptr[row * inp_row_stride + col] - row_max) / exp_sum
        out_ptr[row * out_row_stride + col] = val
        col += BLOCK


@cute.jit
def softmax_launch(x: torch.Tensor) -> torch.Tensor:
    """Host-side: compile and launch softmax_kernel."""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    cute.launch(
        softmax_kernel,
        grid=(n_rows,),
        block=(BLOCK_COLS,),
    )(
        out, x,
        n_rows, n_cols,
        x.stride(0), out.stride(0),
        BLOCK_COLS,
    )
    return out


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation."""
    return torch.softmax(x, dim=-1)


def get_test_inputs():
    """Generate test inputs for the kernel."""
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    return (x,)
