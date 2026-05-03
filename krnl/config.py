"""Configuration and defaults for krnl."""

from dataclasses import dataclass, field
from pathlib import Path


# NCU metrics we collect (basic set — user can expand later)
DEFAULT_NCU_METRICS = [
    "gpu__time_duration.sum",                    # kernel duration (ns)
    "sm__throughput.avg.pct_of_peak_sustained",  # compute throughput %
    "dram__throughput.avg.pct_of_peak_sustained", # memory throughput %
    "launch__occupancy",                          # achieved occupancy
    "sm__warps_active.avg.pct_of_peak_sustained", # active warps %
]

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
    model: str = "claude-sonnet-4-20250514"
    ncu_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_NCU_METRICS))
    atol: float = 1e-2
    rtol: float = 1e-2
    variations_per_iteration: int = 2  # beam width: how many variations per round
    verbose: bool = False
