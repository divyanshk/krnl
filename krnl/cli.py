"""CLI entry point for krnl."""

import click
from pathlib import Path
from rich.console import Console

from krnl.config import KrnlConfig
from krnl.engine.optimizer import run_optimization

console = Console()


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-n", "--num-variations", default=5, show_default=True,
    help="Total number of kernel variations to generate.",
)
@click.option(
    "-p", "--principles", default="PRINCIPLES.md", show_default=True,
    type=click.Path(path_type=Path),
    help="Path to the PRINCIPLES.md file.",
)
@click.option(
    "-o", "--output-dir", default="krnl_output", show_default=True,
    type=click.Path(path_type=Path),
    help="Directory to write variations and reports.",
)
@click.option(
    "--model", default="claude-sonnet-4-20250514", show_default=True,
    help="Claude model to use for variation generation.",
)
@click.option(
    "--beam-width", default=2, show_default=True,
    help="Number of variations to generate per iteration.",
)
@click.option(
    "--atol", default=1e-2, show_default=True,
    help="Absolute tolerance for correctness validation.",
)
@click.option(
    "--rtol", default=1e-2, show_default=True,
    help="Relative tolerance for correctness validation.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def main(
    input_file: Path,
    num_variations: int,
    principles: Path,
    output_dir: Path,
    model: str,
    beam_width: int,
    atol: float,
    rtol: float,
    verbose: bool,
):
    """krnl — optimize GPU kernels with AI + profiling.

    Pass a Python file containing a cuteDSL kernel and a PyTorch reference
    implementation. krnl will iteratively generate optimized variations,
    validate correctness, and profile with NCU.
    """
    console.print("[bold green]krnl[/] — GPU kernel optimizer\n")

    config = KrnlConfig(
        input_file=input_file.resolve(),
        principles_file=principles.resolve() if principles.exists() else principles,
        output_dir=output_dir.resolve(),
        num_variations=num_variations,
        model=model,
        variations_per_iteration=beam_width,
        atol=atol,
        rtol=rtol,
        verbose=verbose,
    )

    run_optimization(config, console)


if __name__ == "__main__":
    main()
