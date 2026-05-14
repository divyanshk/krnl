"""Parse cuteDSL kernel files to extract kernel functions, launch configs, and reference implementations.

Expected file structure:
    - One or more functions decorated with @cute.kernel (device-side GPU code)
    - A @cute.jit decorated host function that compiles and launches the kernel
    - A plain Python launcher that allocates outputs and calls the host function
    - A PyTorch reference function for correctness comparison
    - Naming conventions to identify roles

Convention:
    The input file should define at minimum:
        - A function named `kernel` or `*_kernel` decorated with @cute.kernel (device code)
        - A function named `host` or `*_host` decorated with @cute.jit (JIT host launcher)
        - A function named `launch` or `*_launch` as the public Python entry point
        - A function named `*_ref` or `*_torch` as the PyTorch reference
        - A function named `get_test_inputs` that returns sample inputs
"""

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelInfo:
    """Parsed information about a cuteDSL kernel file."""

    source_path: Path
    full_source: str

    # The @cute.kernel device function(s)
    kernel_fn_names: list[str]
    kernel_source: str  # combined source of all kernel functions

    # The @cute.jit host launcher (compiles and launches the kernel)
    jit_launcher_fn_name: str | None
    jit_launcher_source: str | None

    # The public Python launcher (*_launch) — allocates outputs and calls the
    # @cute.jit function.  Never modified by Claude; always re-emitted unchanged.
    # launcher_fn_name is what the executor / profiler calls.  When there is no
    # separate public launcher, it falls back to jit_launcher_fn_name.
    launcher_fn_name: str | None
    launcher_source: str | None

    # The PyTorch reference function
    ref_fn_name: str | None
    ref_source: str | None

    # Test input generator
    test_inputs_fn_name: str | None
    test_inputs_source: str | None

    # Imports and top-level code
    preamble: str


def _get_fn_source(source: str, node: ast.FunctionDef) -> str:
    """Return the full source of a function including its decorators."""
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    lines = source.splitlines()
    return "\n".join(lines[start - 1 : node.end_lineno])


def parse_kernel_file(filepath: Path) -> KernelInfo:
    """Parse a cuteDSL kernel file and extract its components."""
    source = filepath.read_text()
    tree = ast.parse(source)

    kernel_fns: list[tuple[str, str]] = []
    jit_launcher_fn: tuple[str, str] | None = None
    public_launcher_fn: tuple[str, str] | None = None
    ref_fn: tuple[str, str] | None = None
    test_inputs_fn: tuple[str, str] | None = None
    preamble_parts: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            preamble_parts.append(ast.get_source_segment(source, node))
            continue

        if isinstance(node, ast.Assign):
            preamble_parts.append(ast.get_source_segment(source, node))
            continue

        if not isinstance(node, ast.FunctionDef):
            segment = ast.get_source_segment(source, node)
            if segment:
                preamble_parts.append(segment)
            continue

        fn_name = node.name
        fn_source = _get_fn_source(source, node)

        if _has_cute_kernel_decorator(node):
            kernel_fns.append((fn_name, fn_source))
        elif _has_cute_jit_decorator(node):
            jit_launcher_fn = (fn_name, fn_source)
        elif fn_name == "get_test_inputs" or fn_name.endswith("_inputs"):
            test_inputs_fn = (fn_name, fn_source)
        elif fn_name.endswith(("_ref", "_torch", "_pytorch")):
            ref_fn = (fn_name, fn_source)
        elif _is_public_launcher(fn_name):
            public_launcher_fn = (fn_name, fn_source)
        else:
            preamble_parts.append(fn_source)

    # launcher_fn_name is what the executor calls — prefer the public wrapper,
    # fall back to the @cute.jit function if no wrapper exists.
    if public_launcher_fn:
        launcher_fn_name = public_launcher_fn[0]
        launcher_source = public_launcher_fn[1]
    else:
        launcher_fn_name = jit_launcher_fn[0] if jit_launcher_fn else None
        launcher_source = None

    return KernelInfo(
        source_path=filepath,
        full_source=source,
        kernel_fn_names=[k[0] for k in kernel_fns],
        kernel_source="\n\n".join(k[1] for k in kernel_fns),
        jit_launcher_fn_name=jit_launcher_fn[0] if jit_launcher_fn else None,
        jit_launcher_source=jit_launcher_fn[1] if jit_launcher_fn else None,
        launcher_fn_name=launcher_fn_name,
        launcher_source=launcher_source,
        ref_fn_name=ref_fn[0] if ref_fn else None,
        ref_source=ref_fn[1] if ref_fn else None,
        test_inputs_fn_name=test_inputs_fn[0] if test_inputs_fn else None,
        test_inputs_source=test_inputs_fn[1] if test_inputs_fn else None,
        preamble="\n\n".join(p for p in preamble_parts if p),
    )


def _is_public_launcher(fn_name: str) -> bool:
    """Return True for plain Python launcher functions (launch / *_launch / *_wrapper)."""
    return fn_name in ("launch", "wrapper") or fn_name.endswith(("_launch", "_wrapper"))


def _has_cute_kernel_decorator(node: ast.FunctionDef) -> bool:
    """Return True if the function is decorated with @cute.kernel."""
    for dec in node.decorator_list:
        if (
            isinstance(dec, ast.Attribute)
            and isinstance(dec.value, ast.Name)
            and dec.value.id == "cute"
            and dec.attr == "kernel"
        ):
            return True
    return False


def _has_cute_jit_decorator(node: ast.FunctionDef) -> bool:
    """Return True if the function is decorated with @cute.jit."""
    for dec in node.decorator_list:
        if (
            isinstance(dec, ast.Attribute)
            and isinstance(dec.value, ast.Name)
            and dec.value.id == "cute"
            and dec.attr == "jit"
        ):
            return True
    return False


def reconstruct_kernel_file(
    info: KernelInfo,
    new_kernel_source: str,
    new_jit_launcher_source: str | None = None,
) -> str:
    """Reconstruct a full runnable file with a new kernel implementation.

    Keeps the preamble, public launcher, reference function, and test inputs
    from the original (info).  Replaces the @cute.kernel function(s) and
    optionally the @cute.jit host launcher.
    """
    parts = [info.preamble, "", new_kernel_source, ""]

    jit_src = new_jit_launcher_source or info.jit_launcher_source
    if jit_src:
        parts.append(jit_src)
        parts.append("")

    # Public launcher is never modified — always taken from the original file.
    if info.launcher_source:
        parts.append(info.launcher_source)
        parts.append("")

    if info.ref_source:
        parts.append(info.ref_source)
        parts.append("")

    if info.test_inputs_source:
        parts.append(info.test_inputs_source)
        parts.append("")

    return "\n".join(parts)
