"""LLM prompt templates for kernel variation generation."""

SYSTEM_PROMPT = """\
You are an expert GPU kernel optimization engineer. You specialize in writing \
high-performance CUDA kernels using cuteDSL (NVIDIA's CuTe DSL for Python).

In cuteDSL:
- @cute.kernel defines device-side GPU code — the actual function that runs on the GPU.
- @cute.jit defines the host-side function that handles meta-programming, \
compilation, and launching of @cute.kernel functions.

Your task is to analyze a cuteDSL kernel, its NCU profiling metrics, and a set of \
optimization principles, then generate an optimized variation of the kernel.

Rules:
1. The optimized kernel MUST produce the same outputs as the original for the same inputs.
2. You MUST cite which principle(s) from the provided PRINCIPLES you are applying.
3. You MUST predict the expected performance effect of your changes.
4. Only modify the @cute.kernel function(s) and the @cute.jit host launcher. Do NOT modify \
the public Python launcher (launch / *_launch), the reference implementation, test inputs, \
or imports unless absolutely necessary.
5. Preserve the function signatures of the @cute.jit launcher and test inputs generator.
6. Be specific about what you changed and why.
7. Study the dead ends carefully — do not re-apply strategies that have already failed or regressed.
8. Use the remaining headroom analysis to target the right bottleneck.
"""

VARIATION_USER_PROMPT = """\
## Parent Kernel: Variation {parent_id} (Bottleneck: {parent_bottleneck})

```python
{kernel_source}
```

## Parent Launcher (@cute.jit)

```python
{launcher_source}
```

## Full Parent File (for context)

```python
{full_source}
```

## NCU Profiling Metrics (Parent v{parent_id})

{ncu_summary}

## Bottleneck Analysis

{bottleneck_summary}

## Remaining Performance Headroom

{headroom}

## Relevant Optimization Principles

{principles_text}

## What Has Been Tried and Why It Didn't Help

{dead_ends}

## Full Optimization History

{history}

## Instructions

Generate an optimized variation of this cuteDSL kernel. You are deriving from \
v{parent_id} (bottleneck: {parent_bottleneck}). Apply one or more of the provided \
principles to address the identified bottleneck. Study the dead ends and history \
before deciding — every new variation must be a function of what was tried, how it \
performed, and where the remaining headroom is.

You must respond with:

1. **Principles Applied**: List which principles you are applying and why they \
are relevant to the current bottleneck.

2. **Predicted Effect**: What performance improvement do you expect and why \
(be specific about which NCU metric should improve and by how much).

3. **Changes Made**: A brief description of what you changed relative to v{parent_id}.

4. **Complete Kernel Code**: The full optimized @cute.kernel function(s) (device-side). \
Include the complete function, not just the diff.

5. **Complete Launcher Code**: The full @cute.jit launcher if it needs to change, \
or "NO CHANGE" if the launcher stays the same.

6. **New Imports**: List any new import lines your kernel requires that are NOT already \
in the parent file (e.g. `from cutlass.utils import SmemAllocator`). Write "NONE" if \
no new imports are needed.

7. **New Constants**: List any new module-level constants your kernel introduces \
(e.g. `BLOCK_K = 16`). These MUST be plain `NAME = value` assignments. Write "NONE" \
if no new constants are needed. Do NOT put constants as comments — they must be \
actual assignments so the runtime can see them.

Format your response with these tagged blocks:

KERNEL_CODE
```python
@cute.kernel
def optimized_kernel(...):
    ...
```

LAUNCHER_CODE
```python
@cute.jit
def kernel_launch(...):
    ...
```

NEW_IMPORTS
```python
from cutlass.utils import SmemAllocator
```

NEW_CONSTANTS
```python
BLOCK_K = 16
TILE_SIZE = 32
```
"""

VARIATION_PARSE_REGEX_KERNEL = r"KERNEL_CODE\s*```python\s*\n(.*?)```"
VARIATION_PARSE_REGEX_LAUNCHER = r"LAUNCHER_CODE\s*```python\s*\n(.*?)```"
VARIATION_PARSE_REGEX_IMPORTS = r"NEW_IMPORTS\s*```python\s*\n(.*?)```"
VARIATION_PARSE_REGEX_CONSTANTS = r"NEW_CONSTANTS\s*```python\s*\n(.*?)```"
VARIATION_PARSE_REGEX_PRINCIPLES = r"\*\*Principles Applied\*\*:?\s*(.*?)(?:\n\n|\*\*)"
VARIATION_PARSE_REGEX_PREDICTED = r"\*\*Predicted Effect\*\*:?\s*(.*?)(?:\n\n|\*\*)"
