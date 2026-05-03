"""Main optimization loop — the core of krnl."""

import math
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from krnl.config import KrnlConfig
from krnl.engine.principles import load_principles, find_relevant_principles
from krnl.engine.tracker import OptimizationTracker, VariationRecord
from krnl.engine.variation_gen import generate_variations
from krnl.parsers.cutedsl_parser import parse_kernel_file, reconstruct_kernel_file
from krnl.runner.executor import run_reference, run_kernel
from krnl.runner.ncu_profiler import profile_kernel, NCUMetrics
from krnl.runner.validator import validate_outputs


def run_optimization(config: KrnlConfig, console: Console) -> None:
    """Run the full krnl optimization loop."""

    # ── Setup ──────────────────────────────────────────────────────────────
    config.output_dir.mkdir(parents=True, exist_ok=True)
    variations_dir = config.output_dir / "variations"
    variations_dir.mkdir(exist_ok=True)

    console.print(f"Input:      {config.input_file}")
    console.print(f"Principles: {config.principles_file}")
    console.print(f"Output:     {config.output_dir}")
    console.print(f"Variations: {config.num_variations}")
    console.print(f"Model:      {config.model}")
    console.print()

    # Parse the input kernel file
    console.print("[bold]Step 0: Parsing kernel file...[/]")
    kernel_info = parse_kernel_file(config.input_file)

    if not kernel_info.kernel_fn_names:
        console.print("[red]Error: No @cute.kernel function found in input file.[/]")
        return

    console.print(f"  Found kernel(s): {kernel_info.kernel_fn_names}")
    console.print(f"  Launcher:        {kernel_info.launcher_fn_name}")
    console.print(f"  Reference:       {kernel_info.ref_fn_name}")
    console.print(f"  Test inputs:     {kernel_info.test_inputs_fn_name}")

    if not kernel_info.launcher_fn_name:
        console.print("[red]Error: No launcher function found. Need a function ending in _launch or _wrapper.[/]")
        return
    if not kernel_info.ref_fn_name:
        console.print("[red]Error: No reference function found. Need a function ending in _ref, _torch, or _pytorch.[/]")
        return
    if not kernel_info.test_inputs_fn_name:
        console.print("[red]Error: No test inputs function found. Need get_test_inputs() or *_inputs().[/]")
        return

    # Load principles
    principles = load_principles(config.principles_file)
    console.print(f"\n  Loaded {len(principles)} optimization principles")

    # ── Step 1: Baseline ──────────────────────────────────────────────────
    console.print("\n[bold]Step 1: Establishing baseline...[/]")
    tracker = OptimizationTracker()

    # Run PyTorch reference
    console.print("  Running PyTorch reference...")
    ref_result = run_reference(
        config.input_file, kernel_info.ref_fn_name, kernel_info.test_inputs_fn_name
    )
    if not ref_result.success:
        console.print(f"[red]  Reference execution failed: {ref_result.error}[/]")
        return
    console.print("  [green]Reference OK[/]")

    # Run original kernel
    console.print("  Running original kernel...")
    kernel_result = run_kernel(
        config.input_file, kernel_info.launcher_fn_name, kernel_info.test_inputs_fn_name
    )
    if not kernel_result.success:
        console.print(f"[red]  Kernel execution failed: {kernel_result.error}[/]")
        return

    # Validate original kernel against reference
    console.print("  Validating original kernel...")
    val_result = validate_outputs(ref_result, kernel_result, config.atol, config.rtol)
    if not val_result.is_correct:
        console.print(f"[red]  Original kernel does not match reference: {val_result.details}[/]")
        console.print("[yellow]  Proceeding anyway — will use reference for validation.[/]")

    # Profile PyTorch reference with NCU
    console.print("  Profiling PyTorch reference with NCU...")
    try:
        pytorch_ncu = profile_kernel(
            config.input_file,
            kernel_info.ref_fn_name,
            kernel_info.test_inputs_fn_name,
            config.ncu_metrics,
        )
        if pytorch_ncu:
            tracker.pytorch_baseline_ns = pytorch_ncu[0].duration_ns
            console.print(f"  PyTorch baseline: {tracker.pytorch_baseline_ns:.0f} ns")
    except Exception as e:
        console.print(f"[yellow]  NCU profiling of PyTorch failed: {e}[/]")
        console.print("[yellow]  Continuing without PyTorch NCU baseline.[/]")

    # Profile original kernel with NCU
    console.print("  Profiling original kernel with NCU...")
    try:
        baseline_ncu = profile_kernel(
            config.input_file,
            kernel_info.launcher_fn_name,
            kernel_info.test_inputs_fn_name,
            config.ncu_metrics,
        )
    except Exception as e:
        console.print(f"[red]  NCU profiling failed: {e}[/]")
        console.print("[yellow]  Continuing without NCU data.[/]")
        baseline_ncu = []

    if baseline_ncu:
        tracker.kernel_baseline_ns = baseline_ncu[0].duration_ns
        for m in baseline_ncu:
            console.print(f"\n{m.summary()}")

    # Record baseline
    baseline_record = VariationRecord(
        variation_id=0,
        file_path=str(config.input_file),
        parent_id=-1,
        principles_cited=["original"],
        predicted_effect="baseline",
        is_correct=val_result.is_correct,
        duration_ns=tracker.kernel_baseline_ns,
        compute_throughput_pct=baseline_ncu[0].compute_throughput_pct if baseline_ncu else 0,
        memory_throughput_pct=baseline_ncu[0].memory_throughput_pct if baseline_ncu else 0,
        occupancy=baseline_ncu[0].occupancy if baseline_ncu else 0,
        speedup_vs_baseline=1.0,
        speedup_vs_pytorch=(
            tracker.pytorch_baseline_ns / tracker.kernel_baseline_ns
            if tracker.kernel_baseline_ns > 0 and tracker.pytorch_baseline_ns > 0
            else 1.0
        ),
    )
    tracker.add_variation(baseline_record)

    # ── Step 2: Optimization loop ─────────────────────────────────────────
    console.print(f"\n[bold]Step 2: Generating {config.num_variations} variations...[/]")

    variation_id = 1
    current_best_source = kernel_info.full_source
    current_ncu = baseline_ncu

    while variation_id <= config.num_variations:
        remaining = config.num_variations - variation_id + 1
        batch_size = min(config.variations_per_iteration, remaining)

        console.print(f"\n[bold cyan]── Iteration (v{variation_id}–v{variation_id + batch_size - 1}) ──[/]")

        # Find relevant principles based on current bottleneck
        bottleneck = current_ncu[0].bottleneck_summary() if current_ncu else "Unknown"
        relevant = find_relevant_principles(principles, bottleneck)
        console.print(f"  Bottleneck: {bottleneck}")
        console.print(f"  Relevant principles: {[p.title for p in relevant]}")

        # Generate variations via LLM
        console.print(f"  Generating {batch_size} variation(s) with Claude...")
        try:
            generated = generate_variations(
                config=config,
                kernel_info=kernel_info,
                current_best_source=current_best_source,
                ncu_metrics=current_ncu,
                relevant_principles=relevant,
                tracker=tracker,
                n=batch_size,
            )
        except Exception as e:
            console.print(f"[red]  LLM generation failed: {e}[/]")
            variation_id += batch_size
            continue

        if not generated:
            console.print("[yellow]  No valid variations generated, skipping.[/]")
            variation_id += batch_size
            continue

        # Evaluate each generated variation
        for gen in generated:
            vid = variation_id
            variation_id += 1

            console.print(f"\n  [bold]Evaluating v{vid}...[/]")

            # Write variation to file
            var_source = reconstruct_kernel_file(
                kernel_info, gen.kernel_code, gen.launcher_code
            )
            var_path = variations_dir / f"v{vid}_kernel.py"
            var_path.write_text(var_source)
            console.print(f"    Written to {var_path}")

            # Run and validate
            console.print("    Running variation...")
            var_result = run_kernel(
                var_path, kernel_info.launcher_fn_name, kernel_info.test_inputs_fn_name
            )

            if not var_result.success:
                console.print(f"    [red]Execution failed: {var_result.error}[/]")
                tracker.add_variation(VariationRecord(
                    variation_id=vid,
                    file_path=str(var_path),
                    parent_id=tracker.best_variation_id,
                    principles_cited=gen.principles_cited,
                    predicted_effect=gen.predicted_effect,
                    is_correct=False,
                    error=var_result.error,
                ))
                continue

            console.print("    Validating correctness...")
            val = validate_outputs(ref_result, var_result, config.atol, config.rtol)

            if not val.is_correct:
                console.print(f"    [red]Validation failed: {val.details}[/]")
                tracker.add_variation(VariationRecord(
                    variation_id=vid,
                    file_path=str(var_path),
                    parent_id=tracker.best_variation_id,
                    principles_cited=gen.principles_cited,
                    predicted_effect=gen.predicted_effect,
                    is_correct=False,
                    error=val.details,
                ))
                continue

            console.print("    [green]Correctness OK[/]")

            # Profile with NCU
            var_ncu: list[NCUMetrics] = []
            console.print("    Profiling with NCU...")
            try:
                var_ncu = profile_kernel(
                    var_path,
                    kernel_info.launcher_fn_name,
                    kernel_info.test_inputs_fn_name,
                    config.ncu_metrics,
                )
            except Exception as e:
                console.print(f"    [yellow]NCU profiling failed: {e}[/]")

            duration = var_ncu[0].duration_ns if var_ncu else 0
            speedup_baseline = (
                tracker.kernel_baseline_ns / duration
                if duration > 0 and tracker.kernel_baseline_ns > 0
                else 0
            )
            speedup_pytorch = (
                tracker.pytorch_baseline_ns / duration
                if duration > 0 and tracker.pytorch_baseline_ns > 0
                else 0
            )

            record = VariationRecord(
                variation_id=vid,
                file_path=str(var_path),
                parent_id=tracker.best_variation_id,
                principles_cited=gen.principles_cited,
                predicted_effect=gen.predicted_effect,
                is_correct=True,
                duration_ns=duration,
                compute_throughput_pct=var_ncu[0].compute_throughput_pct if var_ncu else 0,
                memory_throughput_pct=var_ncu[0].memory_throughput_pct if var_ncu else 0,
                occupancy=var_ncu[0].occupancy if var_ncu else 0,
                speedup_vs_baseline=speedup_baseline,
                speedup_vs_pytorch=speedup_pytorch,
            )
            tracker.add_variation(record)

            console.print(f"    {record.summary_line()}")

            # If this is the new best, update the source for next iteration
            best = tracker.get_best()
            if best and best.variation_id == vid:
                console.print(f"    [bold green]New best! v{vid}[/]")
                current_best_source = var_source
                current_ncu = var_ncu

    # ── Step 3: Report ────────────────────────────────────────────────────
    console.print("\n[bold]Step 3: Final Report[/]")
    console.print(tracker.print_leaderboard())

    report_path = tracker.save_report(config.output_dir)
    console.print(f"\nReport saved to {report_path}")

    best = tracker.get_best()
    if best:
        console.print(
            f"\n[bold green]Best kernel: v{best.variation_id}[/] "
            f"({best.duration_ns:.0f} ns, "
            f"{best.speedup_vs_pytorch:.2f}x vs PyTorch, "
            f"{best.speedup_vs_baseline:.2f}x vs original kernel)"
        )
        console.print(f"  File: {best.file_path}")
    else:
        console.print("[yellow]No correct variations were generated.[/]")
