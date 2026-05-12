# krnl

CLI tool to help kernel authors write efficient GPU kernels using AI-driven iterative optimization.

## How it works

1. Write a cuteDSL kernel with a PyTorch reference implementation
2. Run `krnl my_kernel.py`
3. krnl profiles with NCU, generates optimized variations using Claude, validates correctness, and iterates

## Install

```bash
pip install -e .
```

## Usage

```bash
# Basic usage — generate 5 variations
krnl examples/softmax_kernel.py

# Custom settings
krnl my_kernel.py -n 10 -p my_principles.md --beam-width 3

# All options
krnl my_kernel.py \
  -n 10 \
  -p PRINCIPLES.md \
  -o krnl_output \
  --beam-width 2 \
  --atol 1e-3 \
  --rtol 1e-3 \
  --model claude-sonnet-4-20250514 \
  -v
```

## Input file format

Your kernel file must define four things:

| Role | Decorator | Naming convention | Description |
|------|-----------|-------------------|-------------|
| GPU kernel | `@cute.kernel` | `*_kernel` | Device-side code that runs on the GPU |
| Host launcher | `@cute.jit` | `*_launch` / `*_wrapper` | Host-side function that handles compilation and launches the kernel |
| Reference | *(none)* | `*_ref` / `*_torch` | PyTorch reference for correctness validation |
| Test inputs | *(none)* | `get_test_inputs` | Returns sample inputs for the kernel |

### cuteDSL decorator semantics

- **`@cute.kernel`** — defines the actual GPU device function. This is the code krnl
  optimizes: tile sizes, memory access patterns, warp-level reductions, etc.
- **`@cute.jit`** — defines the host-side function. It handles JIT compilation,
  sets grid/block dims, and dispatches the `@cute.kernel` to the GPU.

### Minimal example

```python
import cutlass
import cutlass.cute as cute
import torch

@cute.kernel
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: cute.Int32):
    tid = cute.threadIdx.x + cute.blockIdx.x * cute.blockDim.x
    if tid < n:
        out_ptr[tid] = x_ptr[tid] + y_ptr[tid]

@cute.jit
def vector_add_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    out = torch.empty_like(x)
    vector_add_kernel[math.ceil(n / 256), 256](x, y, out, n)
    return out

def vector_add_ref(x, y):
    return x + y

def get_test_inputs():
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn(1 << 20, device="cuda")
    return (x, y)
```

See `examples/` for fuller examples.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA
- Nsight Compute (`ncu`) installed and on PATH
- `ANTHROPIC_API_KEY` environment variable set
