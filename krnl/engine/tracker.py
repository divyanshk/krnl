"""Track kernel variations, metrics, and optimization history."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rich.tree import Tree


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
    diff_vs_parent: str = ""   # unified diff of @cute.kernel source vs parent
    bottleneck: str = ""       # bottleneck at the time this variation was generated

    def summary_line(self) -> str:
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
        self.variations.append(record)
        if record.is_correct and record.duration_ns > 0:
            best = self.get_best()
            if best is None or record.duration_ns < best.duration_ns:
                self.best_variation_id = record.variation_id

    def get_best(self) -> VariationRecord | None:
        correct = [v for v in self.variations if v.is_correct and v.duration_ns > 0]
        if not correct:
            return None
        return min(correct, key=lambda v: v.duration_ns)

    def get_variation(self, variation_id: int) -> VariationRecord | None:
        for v in self.variations:
            if v.variation_id == variation_id:
                return v
        return None

    def get_frontier(self, n: int = 3) -> list[VariationRecord]:
        """Return up to n correct variations with distinct bottleneck profiles.

        Always includes the global best. Fills remaining slots with variations
        whose bottleneck type differs from already-chosen ones, then by speed.
        This gives the LLM diverse branches to explore from rather than always
        deriving from the single fastest result.
        """
        correct = [v for v in self.variations if v.is_correct and v.duration_ns > 0]
        if not correct:
            return []

        best = min(correct, key=lambda v: v.duration_ns)
        frontier = [best]
        seen_types = {_bottleneck_type(best.bottleneck)}

        for v in sorted(correct, key=lambda v: v.duration_ns):
            if len(frontier) >= n:
                break
            bt = _bottleneck_type(v.bottleneck)
            if bt not in seen_types:
                frontier.append(v)
                seen_types.add(bt)

        # Fill remaining slots with next-fastest not already in frontier
        for v in sorted(correct, key=lambda v: v.duration_ns):
            if len(frontier) >= n:
                break
            if v not in frontier:
                frontier.append(v)

        return frontier

    def history_for_llm(self) -> str:
        """Rich optimization history: NCU deltas, prediction accuracy, diff excerpts."""
        if not self.variations:
            return "No variations tried yet."

        lines = []
        for v in self.variations:
            if v.variation_id == 0:
                continue  # skip baseline in history body
            status = "CORRECT" if v.is_correct else "INCORRECT"
            lines.append(f"### v{v.variation_id} (parent: v{v.parent_id}) — {status}")

            lines.append(f"- Principles: {', '.join(v.principles_cited)}")
            lines.append(f"- Predicted: {v.predicted_effect[:200]}")

            if v.is_correct:
                parent = self.get_variation(v.parent_id)
                parent_dur = parent.duration_ns if parent and parent.duration_ns > 0 else self.kernel_baseline_ns
                delta_pct = (
                    (parent_dur - v.duration_ns) / parent_dur * 100
                    if parent_dur > 0 else 0
                )
                direction = "faster" if delta_pct > 0 else "slower"
                lines.append(
                    f"- Actual: {v.duration_ns:.0f} ns "
                    f"({abs(delta_pct):.1f}% {direction} than parent v{v.parent_id})"
                )
                lines.append(f"- Speedup vs baseline: {v.speedup_vs_baseline:.2f}x")
                lines.append(
                    f"- NCU: compute={v.compute_throughput_pct:.1f}% "
                    f"memory={v.memory_throughput_pct:.1f}% "
                    f"occupancy={v.occupancy:.2f}"
                )
                if parent and parent.is_correct:
                    c_delta = v.compute_throughput_pct - parent.compute_throughput_pct
                    m_delta = v.memory_throughput_pct - parent.memory_throughput_pct
                    o_delta = v.occupancy - parent.occupancy
                    lines.append(
                        f"- NCU delta vs parent: "
                        f"compute{c_delta:+.1f}% "
                        f"memory{m_delta:+.1f}% "
                        f"occupancy{o_delta:+.2f}"
                    )
                # Prediction accuracy signal
                predicted_improvement = any(
                    w in v.predicted_effect.lower()
                    for w in ("faster", "improv", "reduc", "increase", "better")
                )
                if predicted_improvement and delta_pct < 0:
                    lines.append("- Prediction: WRONG DIRECTION — predicted improvement but regressed")
                elif predicted_improvement and delta_pct > 0:
                    lines.append("- Prediction: correct direction")
            else:
                lines.append(f"- Error: {v.error}")

            # Include a compact diff excerpt (first 25 lines)
            if v.diff_vs_parent:
                diff_lines = v.diff_vs_parent.strip().splitlines()
                excerpt = diff_lines[:25]
                if len(diff_lines) > 25:
                    excerpt.append(f"... ({len(diff_lines) - 25} more diff lines)")
                lines.append("- Kernel diff vs parent:")
                lines.append("  ```diff")
                for dl in excerpt:
                    lines.append(f"  {dl}")
                lines.append("  ```")

            lines.append("")

        best = self.get_best()
        if best:
            lines.append(
                f"**Current best: v{best.variation_id}** "
                f"({best.duration_ns:.0f} ns, "
                f"{best.speedup_vs_baseline:.2f}x vs baseline)"
            )

        return "\n".join(lines)

    def dead_ends_for_llm(self) -> str:
        """Explicitly enumerate approaches that failed or regressed so Claude avoids repeating them."""
        dead = []

        for v in self.variations:
            if v.variation_id == 0:
                continue

            parent = self.get_variation(v.parent_id)
            parent_dur = (
                parent.duration_ns
                if parent and parent.is_correct and parent.duration_ns > 0
                else self.kernel_baseline_ns
            )

            if not v.is_correct:
                dead.append(
                    f"- v{v.variation_id}: FAILED ({v.error[:120] if v.error else 'unknown error'}) "
                    f"— tried: {', '.join(v.principles_cited[:2])}"
                )
            elif v.duration_ns > 0 and parent_dur > 0 and v.duration_ns > parent_dur * 1.02:
                regress_pct = (v.duration_ns - parent_dur) / parent_dur * 100
                dead.append(
                    f"- v{v.variation_id}: REGRESSED {regress_pct:.1f}% slower than parent v{v.parent_id} "
                    f"— tried: {', '.join(v.principles_cited[:2])} | "
                    f"NCU after: compute={v.compute_throughput_pct:.1f}% "
                    f"memory={v.memory_throughput_pct:.1f}% "
                    f"occupancy={v.occupancy:.2f}"
                )

        if not dead:
            return "No dead ends yet — all correct variations showed improvement or neutral results."

        lines = ["The following approaches did not help. Do not repeat them:"]
        lines.extend(dead)
        return "\n".join(lines)

    def headroom_for_llm(self) -> str:
        """Estimate remaining performance headroom vs theoretical roofline peak."""
        best = self.get_best()
        if not best or best.duration_ns <= 0:
            return "No profiled variations yet — headroom unknown."

        lines = []

        baseline_dur = self.kernel_baseline_ns
        best_dur = best.duration_ns
        improvement_so_far = (
            (baseline_dur - best_dur) / baseline_dur * 100
            if baseline_dur > 0 else 0
        )
        lines.append(f"Current best: v{best.variation_id} at {best_dur:.0f} ns")
        if baseline_dur > 0:
            lines.append(
                f"Progress so far: {improvement_so_far:.1f}% faster than original baseline"
            )

        # Compute peak vs current utilization gaps
        compute_gap = 100.0 - best.compute_throughput_pct
        memory_gap = 100.0 - best.memory_throughput_pct
        occupancy_gap = 1.0 - best.occupancy

        lines.append(f"Compute utilization: {best.compute_throughput_pct:.1f}% of peak ({compute_gap:.1f}% headroom)")
        lines.append(f"Memory utilization:  {best.memory_throughput_pct:.1f}% of peak ({memory_gap:.1f}% headroom)")
        lines.append(f"Occupancy:           {best.occupancy:.2f} ({occupancy_gap:.2f} headroom to 1.0)")

        # Identify primary opportunity
        if compute_gap > memory_gap and compute_gap > 30:
            lines.append(
                f"→ Primary opportunity: COMPUTE ({compute_gap:.0f}% unused). "
                "Focus on instruction-level parallelism, vectorization, or reducing wasted work."
            )
        elif memory_gap > compute_gap and memory_gap > 30:
            lines.append(
                f"→ Primary opportunity: MEMORY BANDWIDTH ({memory_gap:.0f}% unused). "
                "Focus on coalescing, shared memory, or reducing global memory traffic."
            )
        elif occupancy_gap > 0.3:
            lines.append(
                f"→ Primary opportunity: OCCUPANCY ({occupancy_gap:.2f} gap). "
                "Reduce register pressure or shared memory usage to increase active warps."
            )
        else:
            lines.append(
                "→ Kernel is approaching roofline limits. "
                "Further gains require algorithmic changes (e.g., fusing ops, reducing passes)."
            )

        # Theoretical remaining multiplier from roofline
        bottleneck_utilization = max(
            best.compute_throughput_pct, best.memory_throughput_pct, 0.01
        )
        theoretical_remaining = 100.0 / bottleneck_utilization
        if theoretical_remaining > 1.05:
            lines.append(
                f"Theoretical remaining multiplier (roofline): up to {theoretical_remaining:.1f}x "
                f"if bottleneck utilization reaches 100%"
            )

        return "\n".join(lines)

    def format_tree(self) -> Tree:
        """Render the parent-child variation graph as a Rich Tree.

        Shows each variation's id, correctness, duration, speedup, and
        bottleneck so the user can see exactly which branch led where.
        """
        best = self.get_best()
        root = Tree("[bold]Optimization Tree[/bold]")

        # Build a dict: parent_id → list of child VariationRecords
        children: dict[int, list[VariationRecord]] = {}
        for v in self.variations:
            children.setdefault(v.parent_id, []).append(v)

        def _node_label(v: VariationRecord) -> str:
            is_best = best and v.variation_id == best.variation_id
            star = " [bold yellow]★ BEST[/]" if is_best else ""

            if v.variation_id == 0:
                label = f"[bold]v0[/] [dim]baseline[/]{star}"
                if v.duration_ns > 0:
                    label += f"  {v.duration_ns:.0f} ns"
                label += f"  compute={v.compute_throughput_pct:.0f}%  memory={v.memory_throughput_pct:.0f}%  occ={v.occupancy:.2f}"
                return label

            if not v.is_correct:
                error_snippet = (v.error or "")[:60].replace("\n", " ")
                return (
                    f"[red]v{v.variation_id}[/] [red]FAIL[/]  "
                    f"parent=v{v.parent_id}  "
                    f"[dim]{error_snippet}[/]"
                )

            label = f"[green]v{v.variation_id}[/]{star}  "
            label += f"{v.duration_ns:.0f} ns  "
            label += f"[cyan]{v.speedup_vs_baseline:.2f}x[/] vs baseline  "
            label += f"compute={v.compute_throughput_pct:.0f}%  memory={v.memory_throughput_pct:.0f}%  occ={v.occupancy:.2f}"
            if v.principles_cited:
                principles = ", ".join(v.principles_cited[:2])
                if len(v.principles_cited) > 2:
                    principles += "…"
                label += f"  [dim][{principles}][/]"
            return label

        def _attach(parent_rich_node: Tree, parent_vid: int) -> None:
            for v in children.get(parent_vid, []):
                child_node = parent_rich_node.add(_node_label(v))
                _attach(child_node, v.variation_id)

        # Root of the variation graph is v0 (parent_id == -1)
        for v in self.variations:
            if v.parent_id == -1:
                v0_node = root.add(_node_label(v))
                _attach(v0_node, v.variation_id)

        return root

    def save_report(self, output_dir: Path) -> Path:
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
        correct = [v for v in self.variations if v.is_correct and v.duration_ns > 0]
        correct.sort(key=lambda v: v.duration_ns)

        lines = [
            "┌──────────┬──────────────┬────────────────┬─────────────┬──────────────┐",
            "│ Rank     │ Variation    │ Duration (ns)  │ vs baseline │ Principles   │",
            "├──────────┼──────────────┼────────────────┼─────────────┼──────────────┤",
        ]
        for i, v in enumerate(correct):
            principles = ", ".join(v.principles_cited[:2])
            if len(v.principles_cited) > 2:
                principles += "..."
            lines.append(
                f"│ {i + 1:<8} │ v{v.variation_id:<11} │ {v.duration_ns:<14.0f} │ "
                f"{v.speedup_vs_baseline:<11.2f}x │ {principles:<12} │"
            )
        lines.append(
            "└──────────┴──────────────┴────────────────┴─────────────┴──────────────┘"
        )
        return "\n".join(lines)


def _bottleneck_type(bottleneck: str) -> str:
    """Classify a bottleneck string into a canonical type for frontier diversity."""
    b = bottleneck.upper()
    if "COMPUTE BOUND" in b or "LOW COMPUTE" in b:
        return "compute"
    if "MEMORY BOUND" in b or "LOW MEMORY" in b:
        return "memory"
    if "LOW OCCUPANCY" in b:
        return "occupancy"
    if "BALANCED" in b or "BOTH" in b:
        return "balanced"
    return "unknown"
