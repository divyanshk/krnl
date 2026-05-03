"""Demo runner — exercises the full krnl loop without a GPU or API key.

Usage:
    python testing/run_demo.py
    python testing/run_demo.py --variations 6 --beam-width 2
"""

import argparse
import sys
import types
from pathlib import Path

# Ensure project root is on sys.path regardless of how this script is invoked
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Inject stubs before krnl imports them ────────────────────────────────────

# Stub anthropic if not installed (or even if it is — we always want the mock)
mock_anthropic = types.ModuleType("anthropic")
sys.modules["anthropic"] = mock_anthropic

# Now safe to import krnl
import krnl.runner.ncu_profiler as _ncu_mod
import krnl.engine.variation_gen as _vgen

from testing.mocks import mock_profile_kernel, MockAnthropic

# Patch the two external calls
_ncu_mod.profile_kernel = mock_profile_kernel
_vgen.anthropic = mock_anthropic          # module-level ref in variation_gen
mock_anthropic.Anthropic = MockAnthropic  # client = anthropic.Anthropic()

# ── Run ───────────────────────────────────────────────────────────────────────

from rich.console import Console
from krnl.config import KrnlConfig
from krnl.engine.optimizer import run_optimization

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="krnl demo (no GPU / no API key)")
    parser.add_argument("--variations", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    args = parser.parse_args()

    config = KrnlConfig(
        input_file=(HERE / "mock_kernel.py").resolve(),
        principles_file=Path("PRINCIPLES.md").resolve(),  # empty list if missing
        output_dir=args.output.resolve(),
        num_variations=args.variations,
        model="claude-sonnet-4-6",
        variations_per_iteration=args.beam_width,
    )

    console = Console()
    console.print("[bold green]krnl demo[/] — mock NCU + mock Claude\n")
    run_optimization(config, console)


if __name__ == "__main__":
    main()
