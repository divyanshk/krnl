"""Execute kernel scripts and capture outputs for validation."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExecutionResult:
    """Result of running a kernel."""

    outputs: list[torch.Tensor]  # output tensors from the kernel
    success: bool
    error: str | None = None


def load_module_from_file(filepath: Path, module_name: str = "krnl_user_kernel"):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_reference(filepath: Path, ref_fn_name: str, test_inputs_fn_name: str) -> ExecutionResult:
    """Run the PyTorch reference implementation and capture outputs."""
    try:
        module = load_module_from_file(filepath)
        get_inputs = getattr(module, test_inputs_fn_name)
        ref_fn = getattr(module, ref_fn_name)

        inputs = get_inputs()
        if isinstance(inputs, dict):
            outputs = ref_fn(**inputs)
        elif isinstance(inputs, (list, tuple)):
            outputs = ref_fn(*inputs)
        else:
            outputs = ref_fn(inputs)

        outputs = _normalize_outputs(outputs)
        return ExecutionResult(outputs=outputs, success=True)

    except Exception as e:
        return ExecutionResult(outputs=[], success=False, error=str(e))


def run_kernel(filepath: Path, launcher_fn_name: str, test_inputs_fn_name: str) -> ExecutionResult:
    """Run the kernel via its launcher and capture outputs."""
    try:
        module = load_module_from_file(filepath, module_name="krnl_variant_kernel")
        get_inputs = getattr(module, test_inputs_fn_name)
        launcher = getattr(module, launcher_fn_name)

        inputs = get_inputs()
        if isinstance(inputs, dict):
            outputs = launcher(**inputs)
        elif isinstance(inputs, (list, tuple)):
            outputs = launcher(*inputs)
        else:
            outputs = launcher(inputs)

        outputs = _normalize_outputs(outputs)
        return ExecutionResult(outputs=outputs, success=True)

    except Exception as e:
        return ExecutionResult(outputs=[], success=False, error=str(e))


def _normalize_outputs(outputs: Any) -> list[torch.Tensor]:
    """Normalize function outputs to a list of tensors."""
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (list, tuple)):
        return [o for o in outputs if isinstance(o, torch.Tensor)]
    return []
