"""Main optimization loop — the core of krnl."""

import difflib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from krnl.config import KrnlConfig
from krnl.engine.principles import load_principles, find_relevant_principles
from krnl.engine.tracker import OptimizationTracker, VariationRecord
from krnl.engine.variation_gen import generate_variations, ParentCandidate
from krnl.parsers.cutedsl_parser import parse_kernel_file, reconstruct_kernel_file, KernelInfo
from krnl.runner.executor import run_reference, run_kernel
from krnl.runner.ncu_profiler import profile_kernel, NCUMetrics
from krnl.runner.validator import validate_outputs


@dataclass
class _FrontierEntry:
    """Stores the runnable state of a variation so it can serve as a future parent."""
    full_source: str
    kernel_source: str
    launcher_source: str
    ncu: list[NCUMetrics]


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

    # ── Step 0: Parse ─────────────────────────────────────────────────────
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

    principles = load_principles(config.principles_file)
    console.print(f"\n  Loaded {len(principles)} optimization principles")

    # ── Step 1: Baseline ──────────────────────────────────────────────────
    console.print("\n[bold]Step 1: Establishing baseline...[/]")
    tracker = OptimizationTracker()

    console.print("  Running PyTorch reference...")
    ref_result = run_reference(
        config.input_file, kernel_info.ref_fn_name, kernel_info.test_inputs_fn_name
    )
    if not ref_result.success:
        console.print(f"[red]  Reference execution failed: {ref_result.error}[/]")
        return
    console.print("  [green]Reference OK[/]")

    console.print("  Running original kernel...")
    kernel_result = run_kernel(
        config.input_file, kernel_info.launcher_fn_name, kernel_info.test_inputs_fn_name
    )
    if not kernel_result.success:
        console.print(f"[red]  Kernel execution failed: {kernel_result.error}[/]")
        return

    console.print("  Validating original kernel...")
    val_result = validate_outputs(ref_result, kernel_result, config.atol, config.rtol)
    if not val_result.is_correct:
        console.print(f"[red]  Original kernel does not match reference: {val_result.details}[/]")
        console.print("[yellow]  Proceeding anyway — will use reference for validation.[/]")

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

    # Profile baseline kernel and save to its own directory
    baseline_dir = variations_dir / "v0"
    baseline_dir.mkdir(exist_ok=True)

    console.print("  Profiling original kernel with NCU...")
    try:
        baseline_ncu = profile_kernel(
            config.input_file,
            kernel_info.launcher_fn_name,
            kernel_info.test_inputs_fn_name,
            config.ncu_metrics,
            log_dir=baseline_dir,
            kernel_fn_names=kernel_info.kernel_fn_names,
        )
    except Exception as e:
        console.print(f"[red]  NCU profiling failed: {e}[/]")
        console.print("[yellow]  Continuing without NCU data.[/]")
        baseline_ncu = []

    if baseline_ncu:
        tracker.kernel_baseline_ns = baseline_ncu[0].duration_ns
        for m in baseline_ncu:
            console.print(f"\n{m.summary()}")

    baseline_bottleneck = baseline_ncu[0].bottleneck_summary() if baseline_ncu else "Unknown"

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
        bottleneck=baseline_bottleneck,
    )
    tracker.add_variation(baseline_record)

    # Save baseline metadata
    _write_meta(baseline_dir, baseline_record)

    # frontier_sources maps variation_id → _FrontierEntry (source + ncu)
    # so any past variation can serve as a parent for future generations
    frontier_sources: dict[int, _FrontierEntry] = {
        0: _FrontierEntry(
            full_source=kernel_info.full_source,
            kernel_source=kernel_info.kernel_source,
            launcher_source=kernel_info.jit_launcher_source or "",
            ncu=baseline_ncu,
        )
    }

    # ── Step 2: Optimization loop ─────────────────────────────────────────
    console.print(f"\n[bold]Step 2: Generating {config.num_variations} variations...[/]")

    variation_id = 1

    while variation_id <= config.num_variations:
        remaining = config.num_variations - variation_id + 1
        batch_size = min(config.variations_per_iteration, remaining)

        console.print(f"\n[bold cyan]── Iteration (v{variation_id}–v{variation_id + batch_size - 1}) ──[/]")

        # Build parent candidates from the frontier
        frontier_records = tracker.get_frontier(n=batch_size)
        parents = _build_parent_candidates(frontier_records, frontier_sources, batch_size)

        for p in parents:
            console.print(f"  Parent: v{p.variation_id} | Bottleneck: {p.bottleneck[:60]}")

        # Find relevant principles using NCU metric thresholds when available
        primary_bottleneck = parents[0].bottleneck if parents else "Unknown"
        primary_ncu_list = parents[0].ncu_metrics if parents else []
        metric_values = primary_ncu_list[0].as_metric_dict() if primary_ncu_list else {}
        relevant = find_relevant_principles(principles, primary_bottleneck, metric_values)
        console.print(f"  Relevant principles: {[p.title for p in relevant]}")

        console.print(f"  Calling Claude for {batch_size} variation(s)...")
        try:
            generated = generate_variations(
                config=config,
                kernel_info=kernel_info,
                parents=parents,
                relevant_principles=relevant,
                tracker=tracker,
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

            console.print(f"\n  [bold]Evaluating v{vid} (from parent v{gen.parent_id})...[/]")

            # Create per-variation directory
            var_dir = variations_dir / f"v{vid}"
            var_dir.mkdir(exist_ok=True)

            # Reconstruct full file and write kernel.py
            parent_entry = frontier_sources.get(gen.parent_id)
            parent_kernel_info = _make_kernel_info_for_parent(kernel_info, parent_entry)

            var_source = reconstruct_kernel_file(
                parent_kernel_info, gen.kernel_code, gen.launcher_code
            )  # gen.launcher_code is the new @cute.jit source (or None → keep original)
            var_path = var_dir / "kernel.py"
            var_path.write_text(var_source)
            console.print(f"    Written: {var_path}")

            # Compute diff vs parent kernel source
            parent_kernel_src = parent_entry.kernel_source if parent_entry else kernel_info.kernel_source
            diff_vs_parent = _compute_diff(parent_kernel_src, gen.kernel_code)

            # Run and validate
            console.print("    Running...")
            var_result = run_kernel(
                var_path, kernel_info.launcher_fn_name, kernel_info.test_inputs_fn_name
            )

            if not var_result.success:
                console.print(f"    [red]Execution failed: {var_result.error}[/]")
                record = VariationRecord(
                    variation_id=vid,
                    file_path=str(var_path),
                    parent_id=gen.parent_id,
                    principles_cited=gen.principles_cited,
                    predicted_effect=gen.predicted_effect,
                    is_correct=False,
                    error=var_result.error,
                    diff_vs_parent=diff_vs_parent,
                    bottleneck=primary_bottleneck,
                )
                tracker.add_variation(record)
                _write_meta(var_dir, record)
                continue

            console.print("    Validating...")
            val = validate_outputs(ref_result, var_result, config.atol, config.rtol)

            if not val.is_correct:
                console.print(f"    [red]Validation failed: {val.details}[/]")
                record = VariationRecord(
                    variation_id=vid,
                    file_path=str(var_path),
                    parent_id=gen.parent_id,
                    principles_cited=gen.principles_cited,
                    predicted_effect=gen.predicted_effect,
                    is_correct=False,
                    error=val.details,
                    diff_vs_parent=diff_vs_parent,
                    bottleneck=primary_bottleneck,
                )
                tracker.add_variation(record)
                _write_meta(var_dir, record)
                continue

            console.print("    [green]Correctness OK[/]")

            # Profile with NCU, saving raw CSV + summary to var_dir
            var_ncu: list[NCUMetrics] = []
            console.print("    Profiling with NCU...")
            try:
                var_ncu = profile_kernel(
                    var_path,
                    kernel_info.launcher_fn_name,
                    kernel_info.test_inputs_fn_name,
                    config.ncu_metrics,
                    log_dir=var_dir,
                    kernel_fn_names=kernel_info.kernel_fn_names,
                )
            except Exception as e:
                console.print(f"    [yellow]NCU profiling failed: {e}[/]")

            duration = var_ncu[0].duration_ns if var_ncu else 0
            speedup_baseline = (
                tracker.kernel_baseline_ns / duration
                if duration > 0 and tracker.kernel_baseline_ns > 0 else 0
            )
            speedup_pytorch = (
                tracker.pytorch_baseline_ns / duration
                if duration > 0 and tracker.pytorch_baseline_ns > 0 else 0
            )
            var_bottleneck = var_ncu[0].bottleneck_summary() if var_ncu else "Unknown"

            record = VariationRecord(
                variation_id=vid,
                file_path=str(var_path),
                parent_id=gen.parent_id,
                principles_cited=gen.principles_cited,
                predicted_effect=gen.predicted_effect,
                is_correct=True,
                duration_ns=duration,
                compute_throughput_pct=var_ncu[0].compute_throughput_pct if var_ncu else 0,
                memory_throughput_pct=var_ncu[0].memory_throughput_pct if var_ncu else 0,
                occupancy=var_ncu[0].occupancy if var_ncu else 0,
                speedup_vs_baseline=speedup_baseline,
                speedup_vs_pytorch=speedup_pytorch,
                diff_vs_parent=diff_vs_parent,
                bottleneck=var_bottleneck,
            )
            tracker.add_variation(record)
            _write_meta(var_dir, record)

            console.print(f"    {record.summary_line()}")

            # Add to frontier so future iterations can branch from here.
            # launcher_source tracks the @cute.jit source for this variation.
            frontier_sources[vid] = _FrontierEntry(
                full_source=var_source,
                kernel_source=gen.kernel_code,
                launcher_source=gen.launcher_code or (parent_entry.launcher_source if parent_entry else kernel_info.jit_launcher_source or ""),
                ncu=var_ncu,
            )

            best = tracker.get_best()
            if best and best.variation_id == vid:
                console.print(f"    [bold green]New best! v{vid}[/]")

    # ── Step 3: Report ────────────────────────────────────────────────────
    console.print("\n[bold]Step 3: Final Report[/]")
    console.print(tracker.format_tree())
    console.print()
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_parent_candidates(
    frontier_records: list,
    frontier_sources: dict[int, _FrontierEntry],
    n: int,
) -> list[ParentCandidate]:
    """Map frontier VariationRecords to ParentCandidates, padding with best if needed."""
    candidates = []
    for fr in frontier_records[:n]:
        entry = frontier_sources.get(fr.variation_id)
        if entry is None:
            continue
        candidates.append(ParentCandidate(
            variation_id=fr.variation_id,
            full_source=entry.full_source,
            kernel_source=entry.kernel_source,
            launcher_source=entry.launcher_source,
            ncu_metrics=entry.ncu,
            bottleneck=fr.bottleneck or (
                entry.ncu[0].bottleneck_summary() if entry.ncu else "Unknown"
            ),
        ))

    # If frontier yielded nothing (e.g. all variations failed validation),
    # fall back to the baseline so the loop can still make progress.
    if not candidates:
        entry = frontier_sources.get(0)
        if entry:
            candidates.append(ParentCandidate(
                variation_id=0,
                full_source=entry.full_source,
                kernel_source=entry.kernel_source,
                launcher_source=entry.launcher_source,
                ncu_metrics=entry.ncu,
                bottleneck=entry.ncu[0].bottleneck_summary() if entry.ncu else "Unknown",
            ))

    # Pad with the first candidate repeated if frontier is smaller than batch_size
    while len(candidates) < n and candidates:
        candidates.append(candidates[0])

    return candidates


def _make_kernel_info_for_parent(original: KernelInfo, entry: "_FrontierEntry | None") -> KernelInfo:
    """Return a KernelInfo-like object whose jit_launcher_source reflects the parent."""
    if entry is None:
        return original
    from krnl.parsers.cutedsl_parser import KernelInfo as KI
    return KI(
        source_path=original.source_path,
        full_source=entry.full_source,
        kernel_fn_names=original.kernel_fn_names,
        kernel_source=entry.kernel_source,
        jit_launcher_fn_name=original.jit_launcher_fn_name,
        jit_launcher_source=entry.launcher_source or original.jit_launcher_source,
        launcher_fn_name=original.launcher_fn_name,
        launcher_source=original.launcher_source,
        ref_fn_name=original.ref_fn_name,
        ref_source=original.ref_source,
        test_inputs_fn_name=original.test_inputs_fn_name,
        test_inputs_source=original.test_inputs_source,
        preamble=original.preamble,
    )


def _compute_diff(old_source: str, new_source: str) -> str:
    """Compute a unified diff between two kernel source strings."""
    old_lines = old_source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile="parent_kernel",
        tofile="new_kernel",
        lineterm="",
    ))
    return "".join(diff)


def _write_meta(var_dir: Path, record: VariationRecord) -> None:
    """Write meta.json for a variation directory."""
    meta = {
        "variation_id": record.variation_id,
        "parent_id": record.parent_id,
        "is_correct": record.is_correct,
        "duration_ns": record.duration_ns,
        "speedup_vs_baseline": record.speedup_vs_baseline,
        "speedup_vs_pytorch": record.speedup_vs_pytorch,
        "compute_throughput_pct": record.compute_throughput_pct,
        "memory_throughput_pct": record.memory_throughput_pct,
        "occupancy": record.occupancy,
        "bottleneck": record.bottleneck,
        "principles_cited": record.principles_cited,
        "predicted_effect": record.predicted_effect,
        "error": record.error,
    }
    (var_dir / "meta.json").write_text(json.dumps(meta, indent=2))
