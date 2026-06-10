"""Interface to NVIDIA Nsight Compute (NCU) for kernel profiling."""

import csv
import io
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from krnl.config import DEFAULT_NCU_METRICS


@dataclass
class NCUMetrics:
    """Parsed NCU profiling metrics for a kernel."""

    kernel_name: str

    # ── Tier 1: roofline position ─────────────────────────────────────────
    duration_ns: float = 0.0
    compute_throughput_pct: float = 0.0   # sm__throughput.avg.pct_of_peak_sustained_elapsed
    memory_throughput_pct: float = 0.0    # dram__throughput.avg.pct_of_peak_sustained_elapsed
    occupancy: float = 0.0                # achieved occupancy (0–1) = sm__warps_active.../100
    active_warps_pct: float = 0.0         # sm__warps_active.avg.pct_of_peak_sustained_active
    cta_utilization_pct: float = 0.0      # sm__ctas_active.avg.pct_of_peak_sustained_elapsed

    # ── Tier 1.5: Speed-of-Light section (from --section SpeedOfLight) ────
    sol_combined_pct: float = 0.0         # max(SM SOL, Memory SOL) — single roofline %
    l1_throughput_pct: float = 0.0        # L1/TEX cache pipeline SOL %
    l2_throughput_pct: float = 0.0        # L2 cache pipeline SOL %
    sol_bottleneck_text: str = ""         # NCU's own verdict (SOLBottleneck rule)

    # ── Tier 2: causal diagnostics ────────────────────────────────────────

    # Occupancy limiters
    registers_per_thread: float = 0.0        # launch__registers_per_thread
    theoretical_occupancy_pct: float = 0.0   # sm__maximum_warps_per_active_cycle_pct

    # Memory access quality
    sectors_per_request_ld: float = 0.0   # l1tex__average_t_sectors_per_request...ld (ideal=1.0)
    sectors_per_request_st: float = 0.0   # l1tex__average_t_sectors_per_request...st
    l1_hit_rate_pct: float = 0.0          # l1tex__t_sector_hit_rate.pct
    l2_hit_rate_pct: float = 0.0          # lts__t_sector_hit_rate.pct

    # Warp stall reasons (% of active cycles)
    stall_long_scoreboard_pct: float = 0.0    # memory latency (HBM/L2)
    stall_short_scoreboard_pct: float = 0.0   # compute pipeline latency
    stall_barrier_pct: float = 0.0            # __syncthreads / barriers
    stall_not_selected_pct: float = 0.0       # warp-scheduler pressure

    # Raw dict — everything NCU returned, unparsed
    raw_metrics: dict[str, str] = field(default_factory=dict)

    def as_metric_dict(self) -> dict[str, float]:
        """Return all metrics as {ncu_metric_name: float} for principles matching."""
        structured = {
            "gpu__time_duration.sum": self.duration_ns,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": self.compute_throughput_pct,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": self.memory_throughput_pct,
            "sm__warps_active.avg.pct_of_peak_sustained_active": self.active_warps_pct,
            "sm__ctas_active.avg.pct_of_peak_sustained_elapsed": self.cta_utilization_pct,
            "sol_combined_pct": self.sol_combined_pct,
            "L1/TEX Cache Throughput": self.l1_throughput_pct,
            "L2 Cache Throughput": self.l2_throughput_pct,
            "launch__registers_per_thread": self.registers_per_thread,
            "sm__maximum_warps_per_active_cycle_pct": self.theoretical_occupancy_pct,
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": self.sectors_per_request_ld,
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio": self.sectors_per_request_st,
            "l1tex__t_sector_hit_rate.pct": self.l1_hit_rate_pct,
            "lts__t_sector_hit_rate.pct": self.l2_hit_rate_pct,
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": self.stall_long_scoreboard_pct,
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": self.stall_short_scoreboard_pct,
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct": self.stall_barrier_pct,
            "smsp__warp_issue_stalled_not_selected_per_warp_active.pct": self.stall_not_selected_pct,
        }
        # Merge raw_metrics so any extra collected metrics are also available
        raw_floats = {k: _safe_float(v) for k, v in self.raw_metrics.items()}
        return {**raw_floats, **structured}  # structured values win on conflict

    def summary(self) -> str:
        lines = [
            f"Kernel: {self.kernel_name}",
            f"  Duration:                {self.duration_ns:.0f} ns",
            f"  Compute Throughput:      {self.compute_throughput_pct:.1f}%",
            f"  Memory Throughput:       {self.memory_throughput_pct:.1f}%",
            f"  Occupancy:               {self.occupancy:.2f}  (theoretical max: {self.theoretical_occupancy_pct:.1f}%)",
            f"  Active Warps:            {self.active_warps_pct:.1f}%",
            f"  CTA Utilization:         {self.cta_utilization_pct:.1f}%  (over elapsed time)",
            f"  Combined SOL:            {self.sol_combined_pct:.1f}%  (max of compute/memory)",
            f"  L1/TEX Throughput:       {self.l1_throughput_pct:.1f}%",
            f"  L2 Throughput:           {self.l2_throughput_pct:.1f}%",
            f"  Registers/thread:        {self.registers_per_thread:.0f}",
            f"  Sectors/request (load):  {self.sectors_per_request_ld:.2f}  (ideal=1.0)",
            f"  Sectors/request (store): {self.sectors_per_request_st:.2f}",
            f"  L1 hit rate:             {self.l1_hit_rate_pct:.1f}%",
            f"  L2 hit rate:             {self.l2_hit_rate_pct:.1f}%",
            f"  Stall — mem latency:     {self.stall_long_scoreboard_pct:.1f}%",
            f"  Stall — compute:         {self.stall_short_scoreboard_pct:.1f}%",
            f"  Stall — barrier:         {self.stall_barrier_pct:.1f}%",
        ]
        return "\n".join(lines)

    def bottleneck_summary(self) -> str:
        """Diagnose the primary bottleneck using causal metrics where available."""
        issues = []

        # NCU's own SOL verdict — surface verbatim if present (NVIDIA-authored)
        if self.sol_bottleneck_text:
            issues.append(f"NCU SOL VERDICT: {self.sol_bottleneck_text.strip()}")

        # Coalescing check (most specific — overrides generic memory diagnosis)
        if self.sectors_per_request_ld > 2.0:
            issues.append(
                f"UNCOALESCED LOADS (sectors_per_request_ld={self.sectors_per_request_ld:.1f}, ideal=1.0) — "
                "threads in a warp are accessing non-contiguous addresses"
            )

        # Barrier stalls
        if self.stall_barrier_pct > 15:
            issues.append(
                f"BARRIER STALLS ({self.stall_barrier_pct:.1f}% of cycles) — "
                "too many __syncthreads() or shared memory fences"
            )

        # Memory latency stalls → occupancy too low to hide it
        if self.stall_long_scoreboard_pct > 30:
            if self.occupancy < 0.5:
                issues.append(
                    f"MEMORY LATENCY NOT HIDDEN (long_scoreboard={self.stall_long_scoreboard_pct:.1f}%, "
                    f"occupancy={self.occupancy:.2f}) — need more active warps to overlap HBM latency"
                )
            else:
                issues.append(
                    f"MEMORY LATENCY BOUND (long_scoreboard={self.stall_long_scoreboard_pct:.1f}%) — "
                    "kernel is blocked on HBM/L2 round-trips"
                )

        # Compute pipeline stalls
        if self.stall_short_scoreboard_pct > 20:
            issues.append(
                f"COMPUTE LATENCY STALL ({self.stall_short_scoreboard_pct:.1f}%) — "
                "instruction-level parallelism is low; try loop unrolling"
            )

        # Register pressure limiting occupancy
        if self.registers_per_thread > 64 and self.occupancy < 0.5:
            issues.append(
                f"REGISTER PRESSURE (registers_per_thread={self.registers_per_thread:.0f}) — "
                "high register count limits active warps per SM"
            )

        # Occupancy/theoretical gap (config mismatch)
        if (self.theoretical_occupancy_pct > 0
                and self.occupancy * 100 < self.theoretical_occupancy_pct * 0.6):
            issues.append(
                f"OCCUPANCY GAP (achieved={self.occupancy:.2f}, "
                f"theoretical_max={self.theoretical_occupancy_pct:.1f}%) — "
                "block/grid config is leaving SM capacity unused"
            )

        # Fall back to generic roofline if no causal signal
        if not issues:
            if self.compute_throughput_pct > 70 and self.memory_throughput_pct > 70:
                issues.append("ROOFLINE PEAK — kernel is near hardware limits; gains require algorithmic changes")
            elif self.compute_throughput_pct > self.memory_throughput_pct:
                issues.append(
                    f"COMPUTE BOUND (compute={self.compute_throughput_pct:.1f}%, "
                    f"memory={self.memory_throughput_pct:.1f}%)"
                )
            else:
                issues.append(
                    f"MEMORY BOUND (memory={self.memory_throughput_pct:.1f}%, "
                    f"compute={self.compute_throughput_pct:.1f}%)"
                )

        return "\n".join(issues)


def profile_kernel(
    script_path: Path,
    launcher_fn_name: str,
    test_inputs_fn_name: str,
    metrics: list[str] | None = None,
    log_dir: Path | None = None,
    kernel_fn_names: list[str] | None = None,
) -> list[NCUMetrics]:
    """Run NCU on a kernel script and return parsed metrics.

    kernel_fn_names: Python function names of the @cute.kernel(s) to target.
    When provided, ncu is told to profile only kernels whose names contain any
    of those strings — filters out torch.randn, CUB, and other setup noise.

    If log_dir is provided, saves ncu_raw.csv and ncu_summary.md there.
    """
    if metrics is None:
        metrics = list(DEFAULT_NCU_METRICS)

    wrapper = _create_ncu_wrapper(script_path, launcher_fn_name, test_inputs_fn_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        wrapper_path = Path(f.name)

    try:
        parsed, raw_csv = _run_ncu(wrapper_path, metrics, kernel_fn_names)
    finally:
        wrapper_path.unlink(missing_ok=True)

    if log_dir is not None and raw_csv:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "ncu_raw.csv").write_text(raw_csv)
        summary_lines = [m.summary() for m in parsed]
        (log_dir / "ncu_summary.md").write_text("\n\n".join(summary_lines))

    return parsed


def _create_ncu_wrapper(
    script_path: Path, launcher_fn_name: str, test_inputs_fn_name: str
) -> str:
    return f"""
import sys
sys.path.insert(0, "{script_path.parent}")
import importlib.util

spec = importlib.util.spec_from_file_location("user_kernel", "{script_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

inputs = mod.{test_inputs_fn_name}()
if isinstance(inputs, dict):
    mod.{launcher_fn_name}(**inputs)
elif isinstance(inputs, (list, tuple)):
    mod.{launcher_fn_name}(*inputs)
else:
    mod.{launcher_fn_name}(inputs)
"""


def _run_ncu(
    script_path: Path,
    metrics: list[str],
    kernel_fn_names: list[str] | None = None,
) -> tuple[list[NCUMetrics], str]:
    """Execute NCU and return (parsed metrics, raw CSV string)."""
    metrics_arg = ",".join(metrics)

    ncu_bin = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    cmd = [
        ncu_bin,
        "--csv",
        "--metrics", metrics_arg,
        "--section", "SpeedOfLight",   # adds L1/L2 SOL + NCU's own SOLBottleneck verdict
        "--target-processes", "all",
    ]

    if kernel_fn_names:
        # CUTLASS mangles device kernel symbols as kernel_cutlass_<fn_name>_<params>.
        # Anchoring on this prefix precisely excludes torch and CUB kernels
        # (e.g. at::vectorized_elementwise_kernel, DeviceSelectSweepKernel).
        pattern = "|".join(kernel_fn_names)
        cmd += ["--kernel-name", f"regex:kernel_cutlass_({pattern})_.*"]

    cmd += [sys.executable, str(script_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        if "ERR_NVGPUCTRPERM" in result.stdout or "ERR_NVGPUCTRPERM" in result.stderr:
            raise RuntimeError(
                "NCU failed: GPU performance counter access denied.\n"
                "Fix (no reboot needed):\n"
                "  sudo sh -c 'echo 0 > /proc/driver/nvidia/params/RestrictProfilingToAdminUsers'\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        raise RuntimeError(f"NCU failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    raw_csv = result.stdout
    return _parse_ncu_csv(raw_csv), raw_csv


def _parse_ncu_csv(csv_output: str) -> list[NCUMetrics]:
    lines = [line for line in csv_output.strip().split("\n") if not line.startswith("==")]
    if not lines:
        return []

    csv_text = "\n".join(lines)
    reader = csv.DictReader(io.StringIO(csv_text))

    kernels: dict[str, dict[str, str]] = {}
    rules: dict[str, dict[str, str]] = {}   # kernel -> {rule_name: rule_description}
    for row in reader:
        kernel_name = row.get("Kernel Name", "unknown")
        if kernel_name not in kernels:
            kernels[kernel_name] = {}
            rules[kernel_name] = {}

        # Section rules (e.g. SOLBottleneck) carry their verdict in Rule Description,
        # not Metric Value — Metric Name is empty for these rows.
        rule_name = row.get("Rule Name", "") or ""
        if rule_name:
            rules[kernel_name][rule_name] = row.get("Rule Description", "") or ""
            continue

        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "0")
        kernels[kernel_name][metric_name] = metric_value

    results = []
    for kernel_name, raw in kernels.items():
        kernel_rules = rules.get(kernel_name, {})
        active_warps_pct = _safe_float(raw.get("sm__warps_active.avg.pct_of_peak_sustained_active", "0"))
        compute_pct = _safe_float(raw.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "0"))
        memory_pct = _safe_float(raw.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", "0"))
        m = NCUMetrics(
            kernel_name=kernel_name,
            # Tier 1
            duration_ns=_safe_float(raw.get("gpu__time_duration.sum", "0")),
            compute_throughput_pct=compute_pct,
            memory_throughput_pct=memory_pct,
            occupancy=active_warps_pct / 100.0,   # achieved occupancy as 0–1
            active_warps_pct=active_warps_pct,
            cta_utilization_pct=_safe_float(raw.get("sm__ctas_active.avg.pct_of_peak_sustained_elapsed", "0")),
            # Tier 1.5 — SOL section
            sol_combined_pct=max(compute_pct, memory_pct),
            l1_throughput_pct=_safe_float(raw.get("L1/TEX Cache Throughput", "0")),
            l2_throughput_pct=_safe_float(raw.get("L2 Cache Throughput", "0")),
            sol_bottleneck_text=kernel_rules.get("SOLBottleneck", ""),
            # Tier 2 — occupancy
            registers_per_thread=_safe_float(raw.get("launch__registers_per_thread", "0")),
            theoretical_occupancy_pct=_safe_float(raw.get("sm__maximum_warps_per_active_cycle_pct", "0")),
            # Tier 2 — memory access quality
            sectors_per_request_ld=_safe_float(raw.get(
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", "0")),
            sectors_per_request_st=_safe_float(raw.get(
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio", "0")),
            l1_hit_rate_pct=_safe_float(raw.get("l1tex__t_sector_hit_rate.pct", "0")),
            l2_hit_rate_pct=_safe_float(raw.get("lts__t_sector_hit_rate.pct", "0")),
            # Tier 2 — stall reasons
            stall_long_scoreboard_pct=_safe_float(raw.get(
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "0")),
            stall_short_scoreboard_pct=_safe_float(raw.get(
                "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct", "0")),
            stall_barrier_pct=_safe_float(raw.get(
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct", "0")),
            stall_not_selected_pct=_safe_float(raw.get(
                "smsp__warp_issue_stalled_not_selected_per_warp_active.pct", "0")),
            raw_metrics=raw,
        )
        results.append(m)

    return results


def _safe_float(val: str) -> float:
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0
