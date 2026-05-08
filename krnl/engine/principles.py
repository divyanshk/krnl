"""Load and index PRINCIPLES.md for structured retrieval."""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TriggerCondition:
    """A single numeric threshold condition on an NCU metric."""

    metric: str   # e.g. "sectors_per_request_ld"
    op: str       # ">", "<", ">=", "<="
    threshold: float

    def evaluate(self, metric_values: dict[str, float]) -> bool:
        val = metric_values.get(self.metric)
        if val is None:
            return False
        if self.op == ">":
            return val > self.threshold
        if self.op == ">=":
            return val >= self.threshold
        if self.op == "<":
            return val < self.threshold
        if self.op == "<=":
            return val <= self.threshold
        return False


@dataclass
class Principle:
    """A single optimization principle."""

    title: str
    body: str
    applies_when: str
    action: str
    triggers: list[TriggerCondition] = field(default_factory=list)
    contraindicated: str = ""
    expected_delta: str = ""
    failure_signal: str = ""

    def matches(self, metric_values: dict[str, float]) -> bool:
        """True if ANY trigger condition fires."""
        return any(t.evaluate(metric_values) for t in self.triggers)

    def to_prompt_str(self) -> str:
        lines = [f"### {self.title}", self.body]
        if self.applies_when:
            lines.append(f"- Applies when: {self.applies_when}")
        if self.action:
            lines.append(f"- Action: {self.action}")
        if self.expected_delta:
            lines.append(f"- Expected delta: {self.expected_delta}")
        if self.failure_signal:
            lines.append(f"- Failure signal: {self.failure_signal}")
        if self.contraindicated:
            lines.append(f"- Contraindicated when: {self.contraindicated}")
        return "\n".join(lines)


# ── NCU metric name aliases ───────────────────────────────────────────────────
# Map short names used in **Triggers** blocks to full NCU metric names.
_METRIC_ALIASES: dict[str, str] = {
    "sectors_per_request_ld":       "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "sectors_per_request_st":       "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
    "l1_hit_rate_pct":              "l1tex__t_hit_rate.pct",
    "l2_hit_rate_pct":              "l2__t_hit_rate.pct",
    "occupancy":                    "launch__occupancy",
    "compute_throughput_pct":       "sm__throughput.avg.pct_of_peak_sustained",
    "memory_throughput_pct":        "dram__throughput.avg.pct_of_peak_sustained",
    "active_warps_pct":             "sm__warps_active.avg.pct_of_peak_sustained",
    "registers_per_thread":         "launch__registers_per_thread",
    "theoretical_occupancy_pct":    "sm__maximum_warps_per_active_cycle_pct",
    "stall_long_scoreboard_pct":    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_short_scoreboard_pct":   "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "stall_barrier_pct":            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "stall_not_selected_pct":       "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "duration_ns":                  "gpu__time_duration.sum",
}


def load_principles(filepath: Path) -> list[Principle]:
    """Parse a PRINCIPLES.md file into structured Principle objects.

    Expected section format:
        ## Title
        Description...
        **Triggers**: metric_alias > threshold [; metric_alias < threshold ...]
        **Applies when**: human-readable condition
        **Action**: what to do
        **Contraindicated when**: condition
        **Expected delta**: what NCU numbers should move
        **Failure signal**: what to look for if it doesn't help
    """
    if not filepath.exists():
        return []

    content = filepath.read_text()
    sections = re.split(r"^## ", content, flags=re.MULTILINE)

    principles = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        # Skip non-principle sections (e.g. tables with no tags)
        if not re.search(r"\*\*(Applies when|Action|Triggers)\*\*", body):
            continue

        applies_when = _extract_tag(body, "Applies when")
        action = _extract_tag(body, "Action")
        contraindicated = _extract_tag(body, "Contraindicated when")
        expected_delta = _extract_tag(body, "Expected delta")
        failure_signal = _extract_tag(body, "Failure signal")
        triggers = _extract_triggers(body)

        clean_body = re.sub(
            r"\*\*(Applies when|Action|Triggers|Contraindicated when|Expected delta|Failure signal)\*\*\s*:.*$",
            "",
            body,
            flags=re.MULTILINE,
        ).strip()

        principles.append(
            Principle(
                title=title,
                body=clean_body,
                applies_when=applies_when,
                action=action,
                triggers=triggers,
                contraindicated=contraindicated,
                expected_delta=expected_delta,
                failure_signal=failure_signal,
            )
        )

    return principles


def find_relevant_principles(
    principles: list[Principle],
    bottleneck_summary: str,
    metric_values: dict[str, float] | None = None,
) -> list[Principle]:
    """Return principles whose triggers fire against metric_values.

    If metric_values is None or no trigger-based matches, falls back to
    keyword overlap against bottleneck_summary.
    """
    if metric_values:
        matched = [p for p in principles if p.matches(metric_values)]
        if matched:
            return matched

    # Keyword fallback
    bottleneck_lower = bottleneck_summary.lower()
    scored = []
    for p in principles:
        applies_lower = p.applies_when.lower()
        keywords = set(applies_lower.split())
        bottleneck_words = set(bottleneck_lower.split())
        overlap = len(keywords & bottleneck_words)
        if overlap > 0 or not p.applies_when:
            scored.append((overlap, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored and scored[0][0] > 0:
        return [p for score, p in scored if score > 0]

    return principles


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_tag(text: str, tag_name: str) -> str:
    pattern = rf"\*\*{re.escape(tag_name)}\*\*\s*:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _extract_triggers(text: str) -> list[TriggerCondition]:
    """Parse **Triggers**: lines into TriggerCondition objects.

    Supports semicolon-separated conditions on the same line, e.g.:
        **Triggers**: sectors_per_request_ld > 2.0; occupancy < 0.5
    Also handles multi-line blocks where each subsequent line starts with
    whitespace or a bullet.
    """
    pattern = r"\*\*Triggers\*\*\s*:\s*(.+?)(?=\n\*\*|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []

    raw = match.group(1).strip()
    # Flatten multi-line into a single string, split on semicolons and newlines
    parts = re.split(r"[;\n]", raw)

    conditions = []
    for part in parts:
        part = part.strip().lstrip("-").strip()
        cond = _parse_condition(part)
        if cond:
            conditions.append(cond)

    return conditions


def _parse_condition(expr: str) -> TriggerCondition | None:
    """Parse 'metric_alias op threshold' into a TriggerCondition."""
    m = re.match(r"^([\w.]+)\s*(>=|<=|>|<)\s*([\d.]+)$", expr.strip())
    if not m:
        return None
    metric_raw, op, threshold_str = m.group(1), m.group(2), m.group(3)
    # Resolve alias to full NCU name (or use as-is if it's already a full name)
    metric = _METRIC_ALIASES.get(metric_raw, metric_raw)
    return TriggerCondition(metric=metric, op=op, threshold=float(threshold_str))
