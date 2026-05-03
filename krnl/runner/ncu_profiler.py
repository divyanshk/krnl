"""Interface to NVIDIA Nsight Compute (NCU) for kernel profiling."""

import csv
import io
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from krnl.config import DEFAULT_NCU_METRICS


@dataclass
class NCUMetrics:
    """Parsed NCU profiling metrics for a kernel."""

    kernel_name: str
    duration_ns: float = 0.0
    compute_throughput_pct: float = 0.0
    memory_throughput_pct: float = 0.0
    occupancy: float = 0.0
    active_warps_pct: float = 0.0
    raw_metrics: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Kernel: {self.kernel_name}",
            f"  Duration:            {self.duration_ns:.0f} ns",
            f"  Compute Throughput:  {self.compute_throughput_pct:.1f}%",
            f"  Memory Throughput:   {self.memory_throughput_pct:.1f}%",
            f"  Occupancy:           {self.occupancy:.2f}",
            f"  Active Warps:        {self.active_warps_pct:.1f}%",
        ]
        return "\n".join(lines)

    def bottleneck_summary(self) -> str:
        issues = []
        if self.compute_throughput_pct < 30:
            issues.append(
                f"LOW COMPUTE THROUGHPUT ({self.compute_throughput_pct:.1f}%) — "
                "kernel is not utilizing compute units effectively"
            )
        if self.memory_throughput_pct < 30:
            issues.append(
                f"LOW MEMORY THROUGHPUT ({self.memory_throughput_pct:.1f}%) — "
                "kernel is not saturating memory bandwidth"
            )
        if self.occupancy < 0.5:
            issues.append(
                f"LOW OCCUPANCY ({self.occupancy:.2f}) — "
                "not enough warps to hide latency, check register/shared memory pressure"
            )
        if self.compute_throughput_pct > 70 and self.memory_throughput_pct > 70:
            issues.append(
                "BOTH compute and memory near peak — kernel is well-balanced, "
                "further optimization may require algorithmic changes"
            )

        if not issues:
            if self.compute_throughput_pct > self.memory_throughput_pct:
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
) -> list[NCUMetrics]:
    """Run NCU on a kernel script and return parsed metrics.

    If log_dir is provided, saves ncu_raw.csv and ncu_summary.md there.
    """
    if metrics is None:
        metrics = list(DEFAULT_NCU_METRICS)

    wrapper = _create_ncu_wrapper(script_path, launcher_fn_name, test_inputs_fn_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        wrapper_path = Path(f.name)

    try:
        parsed, raw_csv = _run_ncu(wrapper_path, metrics)
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


def _run_ncu(script_path: Path, metrics: list[str]) -> tuple[list[NCUMetrics], str]:
    """Execute NCU and return (parsed metrics, raw CSV string)."""
    metrics_arg = ",".join(metrics)

    cmd = [
        "ncu",
        "--csv",
        "--metrics", metrics_arg,
        "--target-processes", "all",
        "python", str(script_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
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
    for row in reader:
        kernel_name = row.get("Kernel Name", "unknown")
        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "0")

        if kernel_name not in kernels:
            kernels[kernel_name] = {}
        kernels[kernel_name][metric_name] = metric_value

    results = []
    for kernel_name, raw in kernels.items():
        m = NCUMetrics(
            kernel_name=kernel_name,
            duration_ns=_safe_float(raw.get("gpu__time_duration.sum", "0")),
            compute_throughput_pct=_safe_float(
                raw.get("sm__throughput.avg.pct_of_peak_sustained", "0")
            ),
            memory_throughput_pct=_safe_float(
                raw.get("dram__throughput.avg.pct_of_peak_sustained", "0")
            ),
            occupancy=_safe_float(raw.get("launch__occupancy", "0")),
            active_warps_pct=_safe_float(
                raw.get("sm__warps_active.avg.pct_of_peak_sustained", "0")
            ),
            raw_metrics=raw,
        )
        results.append(m)

    return results


def _safe_float(val: str) -> float:
    try:
        return float(val.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0
