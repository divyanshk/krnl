"""Mock cuteDSL vector-add kernel for demo/testing without a GPU.

Defines cute stubs so the parser can find @cute.kernel / @cute.jit via AST,
and the executor can import and run this file using plain CPU tensors.
"""

import torch

class _CuteStub:
    """Minimal stub so @cute.kernel and @cute.jit are valid Python decorators."""
    @staticmethod
    def kernel(fn):
        return fn

    @staticmethod
    def jit(fn):
        return fn

    Int32 = int

cute = _CuteStub()

N = 1024

@cute.kernel
def vector_add_kernel(x, y, out, n):
    tile = 128
    for base in range(0, n, tile):
        out[base:base + tile] = x[base:base + tile] + y[base:base + tile]

def vector_add_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Host-side: allocate output and call the kernel."""
    out = torch.empty_like(x)
    vector_add_kernel(x, y, out, x.numel())
    return out

def vector_add_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

def get_test_inputs():
    torch.manual_seed(42)
    x = torch.randn(N)
    y = torch.randn(N)
    return (x, y)
