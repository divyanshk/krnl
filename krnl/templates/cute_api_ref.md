# CuTe Python DSL API Reference — Ampere (sm80) Only

**Target architecture: Ampere (sm80 / A100, A10, A6000, etc.)**
Copy atoms (`CopyG2SOp`, `cp_async_*`) and MMA atoms (`MmaF16BF16Op`, `MmaUniversalOp`)
are Ampere-specific. Do NOT apply these to Hopper (sm90) or Blackwell (sm100) kernels.

Every API listed here comes directly from working CUTLASS example kernels. Do not
invent any API not shown here.

---

## Decorators

```python
@cute.kernel          # device-side GPU function
@cute.jit             # host-side JIT launcher (handles tiling, compilation, launch)
@cute.struct          # shared memory layout struct
```

---

## Shared Memory Allocation

### Pattern 1 — Simple tensor (one buffer)
```python
from cutlass.utils import SmemAllocator

smem_alloc = SmemAllocator()
smem_tensor = smem_alloc.allocate_tensor(dtype, layout, alignment_in_bytes)
# dtype: e.g. Float16, Float32, BFloat16
# layout: any CuTe layout
# returns: a CuTe tensor backed by shared memory
```

### Pattern 2 — Struct with named fields (multiple buffers)
```python
@cute.struct
class SmemStorage:
    smem_A: cute.make_tensor_view_type(dtype_A, smem_layout_A)
    smem_B: cute.make_tensor_view_type(dtype_B, smem_layout_B)

# Inside @cute.kernel:
smem_alloc = SmemAllocator()
storage = smem_alloc.allocate(SmemStorage)
smem_A = storage.get_tensor(storage.smem_A, dtype_A, smem_layout_A)
smem_B = storage.get_tensor(storage.smem_B, dtype_B, smem_layout_B)
```

### DO NOT USE — these do not exist:
- `cute.arch.shared_memory(...)` — hallucinated, does not exist
- `cute.smem_alloc(...)` — hallucinated, does not exist
- `cute.allocate_shared(...)` — hallucinated, does not exist
- `smem_alloc.alloc(...)` — wrong method name; use `allocate` or `allocate_tensor`

---

## Layouts and Tensors

```python
# Create a layout
layout = cute.make_layout((M, N), stride=(N, 1))           # row-major
layout = cute.make_layout((M, N), stride=(1, M))           # col-major

# Tiling a tensor for a thread block
tile = cute.local_tile(tensor, tiler, coord, proj=...)
# proj: which modes to project out; e.g. proj=(1,) keeps only mode 1 free

# Partition a tiled tensor among threads
partitioned = cute.local_partition(tensor, thread_layout, thread_idx)

# Divide a layout into tiles
outer, inner = cute.zipped_divide(layout, tile_shape)

# Make a fragment (register tile) matching a tensor's layout
frag = cute.make_fragment_like(tensor)

# Convert a layout to ordered form (for copy compatibility)
ordered = cute.make_ordered_layout(layout, order)

# Flatten a layout to 1D
flat = cute.flatten(layout)

# Get size of a mode
sz = cute.size(tensor)               # total elements
sz = cute.size(tensor, mode=0)       # size of mode 0
```

### DO NOT USE:
- `cute.tile(tensor, ...)` — wrong; use `cute.local_tile(...)`
- `cute.partition(...)` — wrong; use `cute.local_partition(...)`

---

## Swizzle (bank conflict avoidance)

```python
# Create a swizzle functor
swizzle = cute.make_swizzle(bits, base, shift)
# bits: number of swizzle bits (e.g. 3)
# base: base offset in bits (e.g. 3)
# shift: shift amount (e.g. 3)

# Compose swizzle with a layout atom
composed = cute.make_composed_layout(swizzle, 0, atom_layout)

# Tile up to full shape
full_layout = cute.tile_to_shape(composed, (rows, cols))
```

### DO NOT USE:
- `cute.swizzled_layout(...)` — hallucinated
- `cute.bank_conflict_free_layout(...)` — hallucinated

---

## Copy Atoms and G2S / S2R Copy

### Global-to-Shared (async copy, Ampere)
```python
copy_atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(),
    dtype,
    num_bits_per_copy=128,   # 128-bit = ldg.128 = 16 bytes per thread
)
tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)  # or make_tiled_copy_B
```

### Shared-to-Register (ldmatrix)
```python
copy_atom_s2r = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False),
    dtype,
)
tiled_copy_s2r = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
```

### Executing a copy
```python
src_slice = tiled_copy.get_slice(thread_idx)
# partition the src (global) and dst (shared/register) tensors:
thr_src = src_slice.partition_S(gmem_tensor)
thr_dst = src_slice.partition_D(smem_tensor)
cute.copy(tiled_copy, thr_src, thr_dst)
```

### DO NOT USE:
- `cute.async_copy(...)` — hallucinated
- `cute.gmem_to_smem(...)` — hallucinated
- `cute.copy_async(...)` — hallucinated

---

## Async Copy Pipeline (mandatory for G2S)

All Global-to-Shared async copies MUST be fenced with:

```python
# After issuing G2S copies for a stage:
cute.arch.cp_async_commit_group()

# Before reading from shared memory (wait for N groups still in flight):
cute.arch.cp_async_wait_group(N)
# N=0: wait for ALL outstanding groups to finish
# N=1: allow 1 group to still be in-flight (used in double-buffering)
```

Full double-buffering pattern (2-stage pipeline):
```python
# Prologue: fill stage 0
cute.copy(tiled_copy, gmem_tile_0, smem_stage_0)
cute.arch.cp_async_commit_group()

for k in range(K_tiles):
    # Issue next stage's copy before computing current stage
    cute.copy(tiled_copy, gmem_tile_next, smem_stage_next)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(1)   # wait for all but the just-issued group
    cute.arch.syncthreads()

    # Compute on current stage
    cute.gemm(tiled_mma, acc, smem_current, smem_current_B)

cute.arch.cp_async_wait_group(0)   # drain everything before epilogue
cute.arch.syncthreads()
```

### DO NOT USE:
- `cute.arch.async_pipeline(...)` — hallucinated
- Omitting `cp_async_commit_group` after every G2S copy — required fence

---

## MMA (Matrix Multiply-Accumulate) Atoms

### SIMT (FP32, no TensorCore)
Use tensorcore if possible.
```python
tiled_mma = cute.make_tiled_mma(
    cute.nvgpu.warp.MmaUniversalOp(dtype_AB, dtype_C, (1, 2, 1)),
    layout_M=cute.make_layout(4),
    layout_N=cute.make_layout(4),
)
```

### TensorCore (FP16/BF16 → FP32, Ampere)
```python
tiled_mma = cute.make_tiled_mma(
    cute.nvgpu.warp.MmaF16BF16Op(dtype_AB, dtype_C, (16, 8, 16)),
    layout_M=cute.make_layout(2),   # 2 warps in M
    layout_N=cute.make_layout(2),   # 2 warps in N
)
```

### Running MMA
```python
acc = cute.make_fragment_like(tiled_mma.partition_fragment_C(C_tile))
cute.clear(acc)
cute.gemm(tiled_mma, acc, reg_A, reg_B)  # acc += reg_A @ reg_B
```

### DO NOT USE:
- `cute.nvgpu.warp.MmaTensorCoreOp(...)` — hallucinated name
- `cute.mma_sync(...)` — hallucinated
- `cute.wmma(...)` — hallucinated

---

## Thread / Block Indexing

```python
tidx, tidy, tidz = cute.arch.thread_idx()   # threadIdx.x/y/z
bidx, bidy, bidz = cute.arch.block_idx()    # blockIdx.x/y/z
bdimx, bdimy, bdimz = cute.arch.block_dim() # blockDim.x/y/z
cute.arch.sync_threads()                    # __syncthreads() — NOTE the underscore
cute.arch.sync_warp()                       # __syncwarp()
```

### DO NOT USE:
- `cute.arch.syncthreads()` — wrong name; use `cute.arch.sync_threads()` (underscore)

---

## Predication (bounds checking)

```python
# Identity tensor for coordinate tracking
cta_coord = cute.make_identity_tensor(cta_tile_shape)
thr_coord  = thr_copy.partition_S(cta_coord)

# Boolean register tensor for predicates
pred = cute.make_rmem_tensor(Boolean, layout)
for i in cute.range(cute.size(pred)):
    coord = thr_coord[i]
    pred[i] = cute.elem_less(coord, residue_coord)

cute.copy_if(tiled_copy, pred, src, dst)   # masked copy using predicate
```

---

## Epilogue (writing C back to global memory)

```python
# After gemm, store accumulator to global C
thr_mma = tiled_mma.get_slice(thread_idx)
thr_C_gmem = thr_mma.partition_C(C_tensor)
cute.copy(acc, thr_C_gmem)   # or element-wise store with alpha/beta scaling
```

---

## Named Barriers (for fine-grained sync, advanced)

```python
from cute.arch import pipeline
bar = pipeline.NamedBarrier(num_threads, bar_id)
bar.arrive()
bar.wait()
```

---

## Variable Initialization (CuTe DSL constraint)

CuTe DSL traces through conditional branches at compile time. Any variable used
after an `if` block MUST be initialized before that block — first assignment inside
a conditional is not visible outside it.

```python
# WRONG — acc first assigned inside `if`, runtime will error
if row < M and col < N:
    acc = 0.0
    for k in cutlass.range(K):
        acc += a_ptr[row, k] * b_ptr[k, col]
    c_ptr[row, col] = acc

# CORRECT — initialize before any conditional
acc = 0.0
if row < M and col < N:
    for k in cutlass.range(K):
        acc += a_ptr[row, k] * b_ptr[k, col]
    c_ptr[row, col] = acc
```

This applies to ALL variables: accumulators, temporaries, pointers. If the DSL
errors with "cannot access local variable X where it is not associated with a value"
or "Using variables defined in dynamic control flow is not supported", initialize X
before the branch.

---

## Module-Level Constants (tuning parameters)

Any integer constant your kernel uses (tile sizes, unroll factors, etc.) MUST be
defined at module level — not computed or assigned inside `@cute.kernel` or `@cute.jit`.
The CuTe JIT captures module globals as compile-time `Constexpr` values.

```python
# CORRECT — define at module level
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16   # <-- must exist here if used anywhere in the kernel

@cute.kernel
def kernel(...):
    smem_a = smem_alloc.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BLOCK_M, BLOCK_K), stride=(BLOCK_K, 1)),
        16,
    )
```

### DO NOT USE:
- Defining a constant only inside `@cute.kernel` or `@cute.jit` then using it in layouts — it won't be in scope
- `from ... import ...` inside a `@cute.kernel` body — imports must be at file top level

## Common Import Pattern

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
from cutlass.cute.typing import Float16, Float32, BFloat16, Boolean, Int32
```
