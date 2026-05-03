"""Track kernel variations, metrics, and optimization history."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class VariationRecord:
    """Record of a single kernel variation."""

    variation_id: int
    file_path: str
    parent_id: int  # which variation this was derived from (-1 for baseline)
    principles_cited: list[str]
    predicted_effect: str
    is_correct: bool
    duration_ns: float = 0.0
    compute_throughput_pct: float = 0.0
    memory_throughput_pct: float = 0.0
    occupancy: float = 0.0
    speedup_vs_baseline: float = 1.0
    speedup_vs_pytorch: float = 1.0
    error: str | None = None
    notes: str = ""

    def summary_line(self) -> str:
        """One-line summary for logging."""
        status = "PASS" if self.is_correct else "FAIL"
        return (
            f"v{self.variation_id} [{status}] "
            f"duration={self.duration_ns:.0f}ns "
            f"speedup_vs_pytorch={self.speedup_vs_pytorch:.2f}x "
            f"compute={self.compute_throughput_pct:.1f}% "
            f"memory={self.memory_throughput_pct:.1f}% "
            f"principles={self.principles_cited}"
        )


@dataclass
class OptimizationTracker:
    """Tracks the full optimization history."""

    variations: list[VariationRecord] = field(default_factory=list)
    pytorch_baseline_ns: float = 0.0
    kernel_baseline_ns: float = 0.0
    best_variation_id: int = 0

    def add_variation(self, record: VariationRecord) -> None:
        """Add a new variation record."""
        self.variations.append(record)

        # Update best if this is correct and faster
        if record.is_correct and record.duration_ns > 0:
            best = self.get_best()
            if best is None or record.duration_ns < best.duration_ns:
                self.best_variation_id = record.variation_id

    def get_best(self) -> VariationRecord | None:
        """Get the best (fastest correct) variation so far."""
        correct = [v for v in self.variations if v.is_correct and v.duration_ns > 0]
        if not correct:
            return None
        return min(correct, key=lambda v: v.duration_ns)

    def get_variation(self, variation_id: int) -> VariationRecord | None:
        """Get a variation by ID."""
        for v in self.variations:
            if v.variation_id == variation_id:
                return v
        return None

    def history_for_llm(self) -> str:
        """Format optimization history for LLM context."""
        if not self.variations:
            return "No variations tried yet."

        lines = ["## Optimization History", ""]

        for v in self.variations:
            status = "CORRECT" if v.is_correct else "INCORRECT"
            lines.append(f"### Variation {v.variation_id} ({status})")
            if v.is_correct:
                lines.append(f"- Duration: {v.duration_ns:.0f} ns")
                lines.append(f"- Speedup vs PyTorch: {v.speedup_vs_pytorch:.2f}x")
                lines.append(f"- Compute throughput: {v.compute_throughput_pct:.1f}%")
                lines.append(f"- Memory throughput: {v.memory_throughput_pct:.1f}%")
            lines.append(f"- Principles applied: {', '.join(v.principles_cited)}")
            lines.append(f"- Predicted effect: {v.predicted_effect}")
            if v.error:
                lines.append(f"- Error: {v.error}")
            lines.append("")

        best = self.get_best()
        if best:
            lines.append(
                f"**Current best: Variation {best.variation_id}** "
                f"({best.duration_ns:.0f} ns, {best.speedup_vs_pytorch:.2f}x vs PyTorch)"
            )

        return "\n".join(lines)

    def save_report(self, output_dir: Path) -> Path:
        """Save a JSON report of the optimization history."""
        report_path = output_dir / "optimization_report.json"
        report = {
            "pytorch_baseline_ns": self.pytorch_baseline_ns,
            "kernel_baseline_ns": self.kernel_baseline_ns,
            "best_variation_id": self.best_variation_id,
            "variations": [asdict(v) for v in self.variations],
        }
        report_path.write_text(json.dumps(report, indent=2))
        return report_path

    def print_leaderboard(self) -> str:
        """Print a ranked leaderboard of all correct variations."""
        correct = [v for v in self.variations if v.is_correct and v.duration_ns > 0]
        correct.sort(key=lambda v: v.duration_ns)

        lines = [
            "┌──────────┬──────────────┬────────────────┬─────────────┬──────────────┐",
            "│ Rank     │ Variation    │ Duration (ns)  │ vs PyTorch  │ Principles   │",
            "├──────────┼──────────────┼────────────────┼─────────────┼──────────────┤",
        ]
        for i, v in enumerate(correct):
            principles = ", ".join(v.principles_cited[:2])
            if len(v.principles_cited) > 2:
                principles += "..."
            lines.append(
                f"│ {i + 1:<8} │ v{v.variation_id:<11} │ {v.duration_ns:<14.0f} │ "
                f"{v.speedup_vs_pytorch:<11.2f}x │ {principles:<12} │"
            )
        lines.append(
            "└──────────┴──────────────┴────────────────┴─────────────┴──────────────┘"
        )
        return "\n".join(lines)
