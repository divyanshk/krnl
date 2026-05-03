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
4. Only modify the @cute.kernel function(s) and the @cute.jit launcher. Do NOT modify \
the reference implementation, test inputs, or imports unless absolutely necessary.
5. Preserve the function signatures of the @cute.jit launcher and test inputs generator.
6. Be specific about what you changed and why.
"""

VARIATION_USER_PROMPT = """\
## Current Kernel (Variation {parent_id})

```python
{kernel_source}
```

## Launcher (@cute.jit host function)

```python
{launcher_source}
```

## Full File (for context)

```python
{full_source}
```

## NCU Profiling Metrics

{ncu_summary}

## Bottleneck Analysis

{bottleneck_summary}

## Relevant Optimization Principles

{principles_text}

## Optimization History

{history}

## Instructions

Generate an optimized variation of this cuteDSL kernel. Apply one or more of the \
provided principles to address the identified bottlenecks.

You must respond with:

1. **Principles Applied**: List which principles you are applying and why they \
are relevant to the current bottleneck.

2. **Predicted Effect**: What performance improvement do you expect and why.

3. **Changes Made**: A brief description of what you changed.

4. **Complete Kernel Code**: The full optimized @cute.kernel function(s) (device-side). \
Include the complete function, not just the diff.

5. **Complete Launcher Code**: The full @cute.jit launcher if it needs to change, \
or "NO CHANGE" if the launcher stays the same.

Format your kernel code in a ```python block tagged with KERNEL_CODE and launcher \
in a block tagged with LAUNCHER_CODE:

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
"""

VARIATION_PARSE_REGEX_KERNEL = r"KERNEL_CODE\s*```python\s*\n(.*?)```"
VARIATION_PARSE_REGEX_LAUNCHER = r"LAUNCHER_CODE\s*```python\s*\n(.*?)```"
VARIATION_PARSE_REGEX_PRINCIPLES = r"\*\*Principles Applied\*\*:?\s*(.*?)(?:\n\n|\*\*)"
VARIATION_PARSE_REGEX_PREDICTED = r"\*\*Predicted Effect\*\*:?\s*(.*?)(?:\n\n|\*\*)"
