# Kernel Optimization Principles

## Memory Coalescing
Ensure that threads within a warp access consecutive memory addresses. Uncoalesced
accesses cause multiple memory transactions, wasting bandwidth and inflating latency.

**Triggers**: sectors_per_request_ld > 2.0; sectors_per_request_st > 2.0
**Applies when**: NCU shows sectors_per_request > 2 (ideal = 1.0 for fully coalesced)
**Action**: Restructure data layout so thread i in a warp accesses element i (stride-1). Transpose data if needed. Pad rows to 128-byte alignment.
**Contraindicated when**: sectors_per_request_ld <= 1.5
**Expected delta**: sectors_per_request_ld drops toward 1.0; memory_throughput_pct rises 20–50%
**Failure signal**: sectors_per_request unchanged after restructure → access pattern is indirect (gather/scatter), not a simple stride issue

## Shared Memory Tiling
Load tiles of data from global memory (HBM) into shared memory (SRAM), compute on
the tile, then store results back. Reduces redundant HBM round-trips when multiple
threads in a block need the same data.

**Triggers**: stall_long_scoreboard_pct > 30; l2_hit_rate_pct < 50
**Applies when**: Long-scoreboard stalls are high and L2 hit rate is low, indicating repeated HBM fetches
**Action**: Partition the computation into tiles; load each tile into shared memory with cooperative loads, synchronize with __syncthreads(), then compute from SRAM.
**Contraindicated when**: Each element is accessed exactly once (streaming access — tiling adds overhead with no reuse benefit)
**Expected delta**: stall_long_scoreboard_pct drops; l2_hit_rate_pct or l1_hit_rate_pct rises; duration_ns decreases
**Failure signal**: Shared memory bank conflicts appear (check smsp__sass_l1tex_data_bank_conflicts); stalls persist despite tiling

## Reduce Register Pressure
High register usage per thread limits occupancy — fewer warps can run concurrently
per SM, reducing the GPU's ability to hide memory latency with other work.

**Triggers**: registers_per_thread > 64; occupancy < 0.5
**Applies when**: NCU shows registers_per_thread > 64 and achieved occupancy is below 0.5
**Action**: Reduce live variables at any one time. Recompute values instead of storing them. Use fp16/bf16 where precision allows. Add __launch_bounds__ to cap register use.
**Contraindicated when**: occupancy >= 0.75 (register count is not the bottleneck)
**Expected delta**: registers_per_thread drops; occupancy rises toward theoretical_occupancy_pct; stall_long_scoreboard_pct decreases
**Failure signal**: Compiler ignores __launch_bounds__ or spills to local memory (check lmem_transactions); occupancy doesn't improve

## Minimize Global Memory Traffic
Every HBM round-trip costs latency and bandwidth. Fuse operations so intermediate
results stay in registers or shared memory instead of being written to HBM.

**Triggers**: memory_throughput_pct > 70; stall_long_scoreboard_pct > 25
**Applies when**: Memory throughput is near peak and long-scoreboard stalls are high, suggesting sequential kernel launches that each touch HBM
**Action**: Fuse sequential element-wise operations into a single kernel. Accumulate intermediate values in registers. Use epilogue fusion for matrix ops.
**Contraindicated when**: Operations have data dependencies that require barrier synchronization between them (fusion would cause correctness issues)
**Expected delta**: duration_ns decreases proportionally to number of fused passes; memory_throughput_pct may rise briefly then fall as fewer passes hit HBM
**Failure signal**: Fused kernel exceeds register budget and spills; duration_ns increases instead of decreasing

## Vectorized Loads and Stores
Wider memory transactions (128-bit loads = 4 × fp32) amortize the per-transaction
overhead and fill the memory pipeline more efficiently.

**Triggers**: memory_throughput_pct < 60; sectors_per_request_ld <= 2.0
**Applies when**: Memory throughput is well below peak yet access patterns are already coalesced
**Action**: Use 128-bit (float4) or 64-bit (float2) load/store intrinsics. Ensure pointers are 16-byte aligned. Set block sizes to multiples of 32 (warp size).
**Contraindicated when**: sectors_per_request_ld > 2.0 (fix coalescing first — vectorization on uncoalesced access makes things worse)
**Expected delta**: memory_throughput_pct rises 10–30%; duration_ns decreases proportionally
**Failure signal**: No throughput change → bottleneck is compute or occupancy, not bandwidth

## Loop Unrolling and ILP
Unrolling inner loops exposes instruction-level parallelism: the compiler can
interleave independent operations from multiple unrolled iterations, hiding
compute pipeline latency.

**Triggers**: stall_short_scoreboard_pct > 20; compute_throughput_pct < 60
**Applies when**: Short-scoreboard (compute pipeline) stalls are high, meaning the compute pipeline has low ILP
**Action**: Increase BLOCK_SIZE or add #pragma unroll N. Reorder instructions so independent ops are adjacent. Break long dependency chains by accumulating into multiple partial sums.
**Contraindicated when**: registers_per_thread > 64 — unrolling worsens register pressure; fix register pressure first
**Expected delta**: stall_short_scoreboard_pct drops; compute_throughput_pct rises; active_warps_pct may increase
**Failure signal**: Duration unchanged or worse due to increased register spills → reduce unroll factor

## Optimal Block Size Selection
Block size controls work granularity per SM. Too small wastes SM capacity (too few
warps to hide latency); too large causes register spilling and cuts occupancy.

**Triggers**: occupancy < 0.5; active_warps_pct < 50
**Applies when**: Achieved occupancy and active warp % are both low, suggesting the launch config is underutilizing the SM
**Action**: Profile block sizes 64, 128, 256, 512 — the sweet spot is where occupancy × compute_throughput_pct is maximized. On Hopper+ use cluster_size for multi-CTA tiles.
**Contraindicated when**: theoretical_occupancy_pct is also low (block size isn't the issue — shared memory or register budget is the limiter)
**Expected delta**: occupancy rises; stall_long_scoreboard_pct drops (more warps to overlap latency); duration_ns decreases
**Failure signal**: Occupancy plateau: increasing block size no longer improves occupancy (hit shared memory or register wall)

## Reduce Synchronization Overhead
Barrier synchronizations (__syncthreads, shared memory fences) stall all threads in
a block until the slowest thread reaches the barrier. Excess barriers serialize warps.

**Triggers**: stall_barrier_pct > 15
**Applies when**: stall_barrier_pct > 15% of active cycles are spent waiting at barriers
**Action**: Reduce __syncthreads() calls: merge producer/consumer stages, use fine-grained async copies (cp.async / TMA), or restructure to eliminate unnecessary shared-memory reuse across stages.
**Contraindicated when**: stall_barrier_pct < 10 (barriers are not the bottleneck)
**Expected delta**: stall_barrier_pct drops; overall throughput improves; duration_ns decreases
**Failure signal**: Correctness issues appear (race conditions) → barriers were load-bearing; use async pipeline primitives instead of removing them

---

## Architecture Feature Reference

| Feature | SM80 Ampere | SM90 Hopper | SM100 Blackwell |
|---|---|---|---|
| Load GMEM→SMEM | `cp.async` (all threads) | TMA G2S (warp 0) | TMA G2S (warp 0) |
| MMA unit | `mma.sync` (1 warp, 32 threads) | WGMMA (1 warpgroup, 128 threads) | `tcgen05` (1–2 warpgroups) |
| Accumulator lives | Registers | Registers | TMEM (256 KB on-chip) |
| Epilogue path | `cute.copy(rC → gC)` | stmatrix → SMEM → TMA S2G → GMEM | tcgen05.ld → regs → GMEM |
| Sync primitive | `cpasync.fence/wait/syncthreads` | `PipelineTmaAsync` | `PipelineTmaUmma` + `UmmaAsync` |
| Thread Block Cluster | None | Up to 8 CTAs, TMA multicast | Up to 16 CTAs, TMA multicast |
| SMEM swizzle | Manual `cute.Swizzle(3,3,3)` | `sm90_utils` (auto) | `sm100_utils` (auto) |
