"""Validate kernel correctness by comparing outputs against PyTorch reference."""

from dataclasses import dataclass

import torch

from krnl.runner.executor import ExecutionResult


@dataclass
class ValidationResult:
    """Result of validating a kernel variant against the reference."""

    is_correct: bool
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    details: str = ""


def validate_outputs(
    reference: ExecutionResult,
    candidate: ExecutionResult,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> ValidationResult:
    """Compare candidate kernel outputs against reference outputs.

    Uses torch.allclose with the specified tolerances. Reports per-tensor
    max absolute and relative errors for debugging.
    """
    if not reference.success:
        return ValidationResult(
            is_correct=False, details=f"Reference execution failed: {reference.error}"
        )

    if not candidate.success:
        return ValidationResult(
            is_correct=False, details=f"Candidate execution failed: {candidate.error}"
        )

    if len(reference.outputs) != len(candidate.outputs):
        return ValidationResult(
            is_correct=False,
            details=(
                f"Output count mismatch: reference={len(reference.outputs)}, "
                f"candidate={len(candidate.outputs)}"
            ),
        )

    max_abs = 0.0
    max_rel = 0.0
    all_close = True
    failure_details = []

    for i, (ref_t, cand_t) in enumerate(zip(reference.outputs, candidate.outputs)):
        if ref_t.shape != cand_t.shape:
            failure_details.append(
                f"Tensor {i}: shape mismatch ref={ref_t.shape} vs cand={cand_t.shape}"
            )
            all_close = False
            continue

        abs_diff = (ref_t.float() - cand_t.float()).abs()
        tensor_max_abs = abs_diff.max().item()
        max_abs = max(max_abs, tensor_max_abs)

        # Relative error where reference is non-zero
        ref_abs = ref_t.float().abs()
        nonzero_mask = ref_abs > 0
        if nonzero_mask.any():
            rel_diff = abs_diff[nonzero_mask] / ref_abs[nonzero_mask]
            tensor_max_rel = rel_diff.max().item()
            max_rel = max(max_rel, tensor_max_rel)

        if not torch.allclose(ref_t.float(), cand_t.float(), atol=atol, rtol=rtol):
            failure_details.append(
                f"Tensor {i}: max_abs_err={tensor_max_abs:.6e}, "
                f"exceeds atol={atol} or rtol={rtol}"
            )
            all_close = False

    if all_close:
        details = f"All outputs match (max_abs_err={max_abs:.6e}, max_rel_err={max_rel:.6e})"
    else:
        details = "Validation FAILED:\n" + "\n".join(failure_details)

    return ValidationResult(
        is_correct=all_close,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        details=details,
    )
