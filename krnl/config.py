"""Configuration and defaults for krnl."""

from dataclasses import dataclass, field
from pathlib import Path


# ── NCU metrics ───────────────────────────────────────────────────────────────
# Tier 1: roofline position (always collected)
_TIER1 = [
    "gpu__time_duration.sum",                                  # kernel wall time (ns)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",        # compute throughput %
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",      # HBM bandwidth %
    "sm__warps_active.avg.pct_of_peak_sustained_active",       # achieved occupancy %
    "sm__ctas_active.avg.pct_of_peak_sustained_elapsed",       # CTA utilization over elapsed time
]

# Tier 2: causal diagnostics — explains WHY the kernel is at its roofline position
_TIER2 = [
    # Occupancy limiters
    "launch__registers_per_thread",               # registers allocated per thread
    "sm__maximum_warps_per_active_cycle_pct",     # theoretical max occupancy %

    # Memory access quality
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",  # coalescing: global loads  (ideal=1.0)
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",  # coalescing: global stores (ideal=1.0)
    "l1tex__t_sector_hit_rate.pct",               # L1 cache hit rate %
    "lts__t_sector_hit_rate.pct",                 # L2 cache hit rate %

    # Warp stall reasons (% of active warp cycles spent stalling on each cause)
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",   # memory latency (HBM/L2)
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",  # compute pipeline latency
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",           # __syncthreads / barriers
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",      # warp-scheduler pressure
]

DEFAULT_NCU_METRICS = _TIER1 + _TIER2

DEFAULT_NCU_SECTIONS = [
    "SpeedOfLight_RooflineChart",
]


@dataclass
class KrnlConfig:
    """Runtime configuration for a krnl optimization session."""

    input_file: Path
    principles_file: Path = Path("PRINCIPLES.md")
    output_dir: Path = Path("krnl_output")
    num_variations: int = 5
    model: str = "claude-sonnet-4-6"
    ncu_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_NCU_METRICS))
    atol: float = 1e-2
    rtol: float = 1e-2
    variations_per_iteration: int = 2
    verbose: bool = False
