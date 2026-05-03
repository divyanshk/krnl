"""Mock NCU profiler and Claude client for demo runs without GPU or API key."""

import re
from pathlib import Path
from krnl.runner.ncu_profiler import NCUMetrics


# ── Mock NCU ──────────────────────────────────────────────────────────────────

def _variation_number(script_path: Path) -> int:
    """Extract the variation index from a path like .../v3/kernel.py."""
    match = re.search(r"/v(\d+)/", str(script_path))
    return int(match.group(1)) if match else 0


def mock_profile_kernel(script_path, launcher_fn_name, test_inputs_fn_name,
                        metrics=None, log_dir=None):
    """Return realistic NCUMetrics that improve as variation number grows."""
    vid = _variation_number(script_path)

    # Simulate a memory-bound kernel that gets progressively better
    duration_ns  = max(120_000 - vid * 9_000, 45_000)
    compute_pct  = min(18.0 + vid * 7.0,  88.0)
    memory_pct   = min(22.0 + vid * 11.0, 91.0)
    occupancy    = round(min(0.82 + vid * 0.025, 0.97), 3)

    m = NCUMetrics(
        kernel_name="mock_vector_add_kernel",
        duration_ns=duration_ns,
        compute_throughput_pct=compute_pct,
        memory_throughput_pct=memory_pct,
        occupancy=occupancy,
        active_warps_pct=occupancy * 100,
    )

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "ncu_raw.csv").write_text(
            "\"Kernel Name\",\"Metric Name\",\"Metric Value\"\n"
            f'"mock_vector_add_kernel","gpu__time_duration.sum","{duration_ns}"\n'
        )
        (log_dir / "ncu_summary.md").write_text(m.summary())

    return [m]


# ── Mock Claude ───────────────────────────────────────────────────────────────

# Four pre-baked kernel variations — each a valid, correct vector-add.
# The mock client cycles through them so each Claude call returns a new one.
_VARIATIONS = [
    (
        "tiling",
        "Tiling the inner loop improves cache locality and should reduce memory latency.",
        """\
@cute.kernel
def vector_add_kernel(x, y, out, n):
    tile = 128
    for base in range(0, n, tile):
        out[base:base + tile] = x[base:base + tile] + y[base:base + tile]""",
    ),
    (
        "vectorised slice",
        "A single tensor slice op reduces Python loop overhead, boosting throughput.",
        """\
@cute.kernel
def vector_add_kernel(x, y, out, n):
    out[:n] = x[:n] + y[:n]""",
    ),
    (
        "two-pass split",
        "Splitting the range in two independent halves can expose pipeline parallelism.",
        """\
@cute.kernel
def vector_add_kernel(x, y, out, n):
    mid = n // 2
    out[:mid] = x[:mid] + y[:mid]
    out[mid:n] = x[mid:n] + y[mid:n]""",
    ),
    (
        "torch.add fused",
        "Using torch.add with an out= argument avoids an intermediate allocation.",
        """\
@cute.kernel
def vector_add_kernel(x, y, out, n):
    import torch
    torch.add(x[:n], y[:n], out=out[:n])""",
    ),
]


def _make_response_text(principles: str, predicted: str, kernel_code: str) -> str:
    return (
        f"**Principles Applied**: {principles}\n\n"
        f"**Predicted Effect**: {predicted}\n\n"
        "**Changes Made**: Updated the @cute.kernel body.\n\n"
        f"KERNEL_CODE\n```python\n{kernel_code}\n```\n\n"
        "LAUNCHER_CODE\n```python\nNO CHANGE\n```\n"
    )


class _MockMessages:
    def __init__(self):
        self._call_count = 0

    def create(self, **kwargs):
        idx = self._call_count % len(_VARIATIONS)
        self._call_count += 1
        principles, predicted, kernel_code = _VARIATIONS[idx]
        text = _make_response_text(principles, predicted, kernel_code)

        class _Content:
            pass

        content = _Content()
        content.text = text

        class _Response:
            pass

        resp = _Response()
        resp.content = [content]
        return resp


class MockAnthropic:
    """Drop-in stub for anthropic.Anthropic() with no network calls."""
    def __init__(self, **kwargs):
        self.messages = _MockMessages()
