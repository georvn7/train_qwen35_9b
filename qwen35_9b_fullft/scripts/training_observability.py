#!/usr/bin/env python3
"""Small, backend-neutral training observability helpers."""

from __future__ import annotations

import resource
import sys
from collections import Counter
from typing import Any, Iterable


def peak_process_rss_mib() -> float:
    """Return the process high-water RSS with Linux/macOS unit handling."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return value / divisor


def cuda_memory_observability() -> dict[str, float]:
    """Return allocator high-water marks when CUDA is available."""
    import torch

    if not torch.cuda.is_available():
        return {}
    scale = 1024.0**2
    return {
        "allocated_mib": float(torch.cuda.memory_allocated()) / scale,
        "reserved_mib": float(torch.cuda.memory_reserved()) / scale,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated()) / scale,
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved()) / scale,
    }


def _tensor_dtype_counts(values: Iterable[Any]) -> dict[str, dict[str, int]]:
    import torch

    tensors: Counter[str] = Counter()
    elements: Counter[str] = Counter()
    bytes_by_dtype: Counter[str] = Counter()
    for value in values:
        if not torch.is_tensor(value):
            continue
        dtype = str(value.dtype).removeprefix("torch.")
        tensors[dtype] += 1
        elements[dtype] += int(value.numel())
        bytes_by_dtype[dtype] += int(value.numel() * value.element_size())
    return {
        "tensor_counts": dict(sorted(tensors.items())),
        "element_counts": dict(sorted(elements.items())),
        "bytes": dict(sorted(bytes_by_dtype.items())),
    }


def optimizer_observability(optimizer: Any, configured_name: str) -> dict[str, Any]:
    """Describe the instantiated optimizer and its materialized tensor state."""
    state_values = []
    for state in getattr(optimizer, "state", {}).values():
        if isinstance(state, dict):
            state_values.extend(state.values())
    return {
        "configured_name": configured_name,
        "implementation": (
            f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        ),
        "state_entries": len(getattr(optimizer, "state", {})),
        "state_tensors": _tensor_dtype_counts(state_values),
    }


def model_parameter_observability(model: Any) -> dict[str, Any]:
    """Record trainable and total parameter dtypes without serializing values."""
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    return {
        "total_parameters": sum(int(parameter.numel()) for parameter in parameters),
        "trainable_parameters": sum(int(parameter.numel()) for parameter in trainable),
        "all_parameter_tensors": _tensor_dtype_counts(parameters),
        "trainable_parameter_tensors": _tensor_dtype_counts(trainable),
    }


def module_residency_observability(model: Any | None) -> dict[str, Any]:
    """Describe whether a model object and its parameter storage are resident."""
    if model is None:
        return {
            "present": False,
            "devices": [],
            "parameter_bytes": 0,
            "cuda_parameter_bytes": 0,
        }
    parameters = list(model.parameters())
    devices = sorted({str(parameter.device) for parameter in parameters})
    parameter_bytes = sum(
        int(parameter.numel() * parameter.element_size()) for parameter in parameters
    )
    cuda_parameter_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in parameters
        if parameter.device.type == "cuda"
    )
    return {
        "present": True,
        "devices": devices,
        "parameter_bytes": parameter_bytes,
        "cuda_parameter_bytes": cuda_parameter_bytes,
    }


def token_throughput(
    tokens_per_epoch: int,
    completed_epochs: float,
    runtime_seconds: float,
) -> dict[str, float | int]:
    """Report a transparent token-exposure estimate for non-packed training."""
    exposed = float(tokens_per_epoch) * float(completed_epochs)
    return {
        "tokens_per_epoch": int(tokens_per_epoch),
        "completed_epochs": float(completed_epochs),
        "estimated_tokens_processed": exposed,
        "estimated_tokens_per_second": (
            exposed / float(runtime_seconds) if runtime_seconds > 0 else 0.0
        ),
    }
