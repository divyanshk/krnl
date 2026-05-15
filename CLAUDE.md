# krnl — AI-driven CUDA Kernel Optimizer

## What it does
Runs an optimization loop on CuTe Python DSL kernel files. Each iteration:
1. Profiles the kernel with `ncu` (Nsight Compute)
2. Sends NCU metrics + kernel source to Claude with the CuTe API cheat-sheet
3. Claude generates an optimized variation
4. Variation is validated for correctness, profiled, and tracked

## Environment
**Python venv:** `source /opt/pytorch/bin/activate`
Always activate before running anything.

## Key source files
| File | Role |
|---|---|
| `krnl/engine/optimizer.py` | Main optimization loop |
| `krnl/engine/variation_gen.py` | Claude API call; parses KERNEL_CODE / LAUNCHER_CODE / NEW_IMPORTS blocks |
| `krnl/engine/tracker.py` | Tracks variation history, frontier, leaderboard |
| `krnl/parsers/cutedsl_parser.py` | Parses and reconstructs kernel files |
| `krnl/runner/executor.py` | Dynamically loads and runs kernel files |
| `krnl/runner/ncu_profiler.py` | NCU profiling wrapper |
| `krnl/templates/prompts.py` | SYSTEM_PROMPT, VARIATION_USER_PROMPT, parse regexes |
| `krnl/templates/cute_api_ref.md` | CuTe Python DSL API cheat-sheet (Ampere/sm80 only) |
| `examples/matmul.py` | Demo kernel — canonical example of the 3-role file structure |

## Kernel file structure (3-role pattern)
Every kernel file must follow this layout exactly:

```python
# 1. @cute.kernel — device-side GPU code (Claude optimizes this)
@cute.kernel
def kernel(...): ...

# 2. @cute.jit — host-side JIT launcher (Claude may optimize this)
@cute.jit
def host(...): ...

# 3. Plain Python launcher — allocates output, calls host (Claude never touches this)
def launch(...): ...

# 4. PyTorch reference — for correctness validation (never modified)
def matmul_ref(...): ...

# 5. Test inputs — must be deterministic (never modified)
def get_test_inputs(): ...
```

Parser detects roles by: `@cute.kernel` decorator, `@cute.jit` decorator, name matching `launch`/`*_launch`, name ending `_ref`/`_torch`/`_pytorch`, name `get_test_inputs`/`*_inputs`.

## Coding conventions (established decisions — do not revisit)

**NCU kernel filtering:** Use `--kernel-name regex:kernel_cutlass_({fn_names})_.*` in the ncu invocation. Do not capture all kernels and post-filter.

**Launcher separation:** `jit_launcher_source` (the `@cute.jit` function) and `launcher_source` (the public `launch` wrapper) are tracked separately in `KernelInfo`. The public launcher is always re-emitted unchanged from the original — never let Claude modify it.

**CuTe API priming:** Claude is primed via `krnl/templates/cute_api_ref.md` injected into the system prompt with `cache_control: {"type": "ephemeral"}`. Do not use the "only use APIs from the parent file" heuristic — the parent file only shows naive patterns and blocks valid optimizations.

**Import injection:** Claude declares new imports in a `NEW_IMPORTS` block. Parsed in `_parse_variation_response`, stored in `GeneratedVariation.extra_imports`, merged into the file preamble by `reconstruct_kernel_file`. Do not use `from ... import ...` inside `@cute.kernel` bodies.

**Deterministic validation:** `load_test_inputs()` loads test inputs once from the original kernel file. The same inputs object is passed to both `run_reference` and `run_kernel` for every variation. This avoids false validation failures from mismatched random tensors.

**Leaderboard:** "vs baseline" column uses `speedup_vs_baseline` (speedup over the user's v0 kernel). The PyTorch reference timing exists internally but is not shown in the table.

## CuTe Python DSL — key facts for Ampere (sm80)
See `krnl/templates/cute_api_ref.md` for the full reference. Critical things Claude gets wrong:

| Hallucinated | Correct |
|---|---|
| `cute.arch.syncthreads()` | `cute.arch.sync_threads()` (underscore) |
| `cute.arch.shared_memory(...)` | `SmemAllocator().allocate_tensor(dtype, layout, align)` |
| `cute.tile(...)` | `cute.local_tile(tensor, tiler, coord, proj=...)` |
| `cute.partition(...)` | `cute.local_partition(tensor, layout, thread_idx)` |
| Inventing constants inline | All tile-size constants must be defined at module level |
| `from ... import` in kernel body | Imports must be at file top level only |
