# Kernel Optimization Principles

## Memory Coalescing
Ensure that threads within a warp access consecutive memory addresses. Uncoalesced
accesses lead to multiple memory transactions and wasted bandwidth.

**Applies when**: NCU shows low memory throughput or high L2 cache miss rate
**Action**: Restructure data layout and access patterns so that thread i in a warp accesses element i (or i + constant stride of 1)

## Shared Memory Tiling
Load tiles of data from global memory (HBM) into shared memory (SRAM), compute on
the tile, then store results back. This reduces redundant global memory accesses when
multiple threads need the same data.

**Applies when**: Repeated reads from global memory for overlapping data regions
**Action**: Partition the computation into tiles, load each tile into shared memory with cooperative loads, synchronize, then compute from shared memory

## Reduce Register Pressure
High register usage per thread limits occupancy — fewer warps can run concurrently,
reducing the GPU's ability to hide memory latency.

**Applies when**: NCU shows low occupancy due to high register count per thread
**Action**: Reduce the number of live variables, recompute values instead of storing them, use smaller data types (fp16/bf16) where precision allows

## Minimize Global Memory Traffic
Every byte transferred between HBM and the SMs costs energy and time. Fuse operations
to avoid writing intermediate results to global memory.

**Applies when**: Kernel pipeline involves writing intermediate tensors to HBM that are immediately read by the next operation
**Action**: Fuse sequential operations into a single kernel, compute intermediate values in registers or shared memory

## Vectorized Loads and Stores
Use wider memory transactions (e.g., load 4 floats at once with 128-bit loads) to
improve memory bandwidth utilization.

**Applies when**: Memory throughput is below peak and access patterns are aligned and contiguous
**Action**: Use tl.load with appropriate block sizes that are multiples of the warp size, ensure pointer alignment

## Loop Unrolling
Unroll inner loops to reduce loop overhead and enable instruction-level parallelism.
The compiler can interleave independent operations from multiple unrolled iterations.

**Applies when**: Kernel has tight inner loops with independent iterations
**Action**: Increase BLOCK_SIZE or explicitly unroll loops to expose more ILP, but balance against register pressure

## Optimal Block Size Selection
The BLOCK_SIZE (tile size) controls the granularity of work per thread block. Too small
wastes parallelism overhead; too large causes register spilling and reduces occupancy.

**Applies when**: Kernel performance is sensitive to block/tile size configuration
**Action**: Try multiple block sizes (64, 128, 256, 512, 1024) and profile each; consider using autotune to search automatically

## Reduce Synchronization Overhead
Barrier synchronizations (tl.debug_barrier, shared memory fences) stall all threads
in a block. Minimize their frequency.

**Applies when**: Profiling shows high warp stall due to barrier waits
**Action**: Restructure computation to reduce the number of synchronization points, prefer independent per-thread computation where possible

## Three-Generation Feature Table

| Feature | SM80 Ampere | SM90 Hopper | SM100 Blackwell |
|---|---|---|---|
| Load GMEM→SMEM | `cp.async` (all threads) | TMA G2S (warp 0) | TMA G2S (warp 0) |
| MMA unit | `mma.sync` (1 warp, 32 threads) | WGMMA (1 warpgroup, 128 threads) | `tcgen05` (1–2 warpgroups) |
| Accumulator lives | Registers | Registers | TMEM (new 256 KB on-chip) |
| Epilogue path | `cute.copy(rC → gC)` | stmatrix → SMEM → TMA S2G → GMEM | tcgen05.ld → regs → GMEM |
| Sync primitive | `cpasync.fence/wait/syncthreads` | `PipelineTmaAsync` | `PipelineTmaUmma` + `UmmaAsync` |
| Thread Block Cluster | None | Up to 8 CTAs, TMA multicast | Up to 16 CTAs, TMA multicast |
| SMEM swizzle | Manual `cute.Swizzle(3,3,3)` | `sm90_utils` (auto) | `sm100_utils` (auto) |
| SMEM re-use | N/A | sA reused as sC in epilogue | N/A (TMEM holds accumulators) |
