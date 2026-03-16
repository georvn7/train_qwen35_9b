#!/usr/bin/env python3
"""Run one Unsloth fine-tuning session for Qwen3.5 9B from a manifest."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5 9B with Unsloth using a session manifest."
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Session directory created by create_session.py",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.5-9B",
        help="Unsloth or HF model id.",
    )
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument(
        "--truncation-side",
        default="left",
        choices=["right", "left"],
        help="How to truncate overlength samples. 'right' keeps the start of each sample.",
    )
    parser.add_argument(
        "--truncate-overlength-samples",
        dest="truncate_overlength_samples",
        action="store_true",
        default=True,
        help="Apply deterministic text truncation before trainer tokenization.",
    )
    parser.add_argument(
        "--no-truncate-overlength-samples",
        dest="truncate_overlength_samples",
        action="store_false",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Attention backend (e.g. eager, sdpa).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="",
        help="Shared Hugging Face cache directory. Defaults to <workspace>/qwen35_9b_fullft/.cache/huggingface",
    )
    parser.add_argument(
        "--python-headers-root",
        default="",
        help="Optional local include root containing python3.12 headers (e.g. .local_py312dev/usr/include).",
    )
    parser.add_argument(
        "--triton-ptxas-path",
        default="auto",
        help="Path to ptxas used by Triton JIT. 'auto' prefers /usr/local/cuda/bin/ptxas if present.",
    )
    parser.add_argument(
        "--cuda-alloc-conf",
        default="",
        help=(
            "Optional PYTORCH_CUDA_ALLOC_CONF override, e.g. "
            "'expandable_segments:True,max_split_size_mb:128'."
        ),
    )
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="Model device placement for from_pretrained (e.g. cuda:0, auto).",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=4,
        help="Max parallel tokenization workers inside SFTTrainer dataset prep. Lower this to reduce RAM spikes.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--optim",
        default="adamw_8bit",
        choices=["adamw_8bit", "paged_adamw_8bit", "adamw_torch"],
        help="Optimizer backend used by Trainer/SFTConfig.",
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["steps", "epoch", "no"],
        help="Checkpoint save strategy for Trainer.",
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--save-only-model",
        action="store_true",
        default=False,
        help="Save only model weights at checkpoints (skip optimizer/scheduler/rng state).",
    )
    parser.add_argument(
        "--no-save-only-model",
        dest="save_only_model",
        action="store_false",
    )
    parser.add_argument(
        "--checkpoint-max-shard-size",
        default="",
        help=(
            "Optional max shard size passed to save_pretrained during checkpoint writes "
            "(for example: 512MB, 1GB, 2GB). Empty uses library default."
        ),
    )
    parser.add_argument(
        "--checkpoint-safe-serialization",
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Checkpoint serialization format override for save_pretrained. "
            "'auto' keeps library default, 'true' forces safetensors, 'false' forces torch .bin."
        ),
    )
    parser.add_argument(
        "--checkpoint-presave-gc",
        dest="checkpoint_presave_gc",
        action="store_true",
        default=True,
        help="Run gc.collect() immediately before checkpoint writes.",
    )
    parser.add_argument(
        "--no-checkpoint-presave-gc",
        dest="checkpoint_presave_gc",
        action="store_false",
    )
    parser.add_argument(
        "--checkpoint-presave-empty-cache",
        dest="checkpoint_presave_empty_cache",
        action="store_true",
        default=True,
        help="Call torch.cuda.empty_cache() immediately before checkpoint writes.",
    )
    parser.add_argument(
        "--no-checkpoint-presave-empty-cache",
        dest="checkpoint_presave_empty_cache",
        action="store_false",
    )
    parser.add_argument(
        "--checkpoint-presave-disable-cuda-history",
        dest="checkpoint_presave_disable_cuda_history",
        action="store_true",
        default=True,
        help=(
            "Disable and clear CUDA allocator history right before checkpoint write "
            "(only relevant when --debug-cuda-memory-history is enabled)."
        ),
    )
    parser.add_argument(
        "--no-checkpoint-presave-disable-cuda-history",
        dest="checkpoint_presave_disable_cuda_history",
        action="store_false",
    )
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=0,
        help="If >0, call torch.cuda.empty_cache() every N steps.",
    )
    parser.add_argument(
        "--eval-holdout-ratio",
        type=float,
        default=0.0,
        help="If >0, split this fraction from training data for evaluation.",
    )
    parser.add_argument(
        "--eval-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation scheduling when holdout eval is enabled.",
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument(
        "--load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_false",
    )
    parser.add_argument(
        "--metric-for-best-model",
        default="eval_loss",
        help="Metric name for best-checkpoint selection when enabled.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=False)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--full-finetuning", dest="full_finetuning", action="store_true", default=True)
    parser.add_argument("--no-full-finetuning", dest="full_finetuning", action="store_false")
    parser.add_argument(
        "--force-causal-lm-loader",
        dest="force_causal_lm_loader",
        action="store_true",
        default=True,
        help="Force AutoModelForCausalLM loading path to skip multimodal vision tower weights.",
    )
    parser.add_argument(
        "--no-force-causal-lm-loader",
        dest="force_causal_lm_loader",
        action="store_false",
    )
    parser.add_argument(
        "--freeze-visual-modules",
        dest="freeze_visual_modules",
        action="store_true",
        default=True,
        help="Freeze model.visual.* parameters for text-only training.",
    )
    parser.add_argument(
        "--no-freeze-visual-modules",
        dest="freeze_visual_modules",
        action="store_false",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        default="unsloth",
        help="Use 'unsloth', 'true', or 'false'.",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "bf16", "fp16", "float32"],
        help="Mixed precision mode for trainer args.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype passed to FastLanguageModel.from_pretrained.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Optional compute dtype override for BitsAndBytes 4-bit quantization.",
    )
    parser.add_argument(
        "--unsloth-mixed-precision",
        default="auto",
        choices=["auto", "float32", "bfloat16"],
        help="Optional UNSLOTH_MIXED_PRECISION override.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help="Optional: low, medium, or high for tokenizer.apply_chat_template.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--gguf-quantization", default="q4_k_m")
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        help="Skip final model/adapter save to speed up hyperparameter probes.",
    )
    parser.add_argument("--skip-merged-export", action="store_true")
    parser.add_argument("--skip-gguf-export", action="store_true")
    parser.add_argument(
        "--disable-unsloth-compile",
        dest="disable_unsloth_compile",
        action="store_true",
        default=True,
        help="Disable torch.compile path in Unsloth for compatibility.",
    )
    parser.add_argument(
        "--enable-unsloth-compile",
        dest="disable_unsloth_compile",
        action="store_false",
    )
    parser.add_argument(
        "--disable-moe-triton",
        dest="disable_moe_triton",
        action="store_true",
        default=True,
        help="Disable MoE Triton kernels in Unsloth temporary patches.",
    )
    parser.add_argument(
        "--enable-moe-triton",
        dest="disable_moe_triton",
        action="store_false",
    )
    parser.add_argument(
        "--disable-flex-attention",
        dest="disable_flex_attention",
        action="store_true",
        default=True,
        help="Set UNSLOTH_ENABLE_FLEX_ATTENTION=0 to avoid Unsloth flex-attention patch path.",
    )
    parser.add_argument(
        "--enable-flex-attention",
        dest="disable_flex_attention",
        action="store_false",
    )
    parser.add_argument(
        "--disable-cce",
        dest="disable_cce",
        action="store_true",
        default=True,
        help="Set UNSLOTH_ENABLE_CCE=0 to disable Unsloth fused/chunked cross-entropy kernels.",
    )
    parser.add_argument(
        "--enable-cce",
        dest="disable_cce",
        action="store_false",
    )
    parser.add_argument(
        "--causal-loss-mode",
        default="default",
        choices=[
            "default",
            "no_upcast",
            "chunked_fp32",
            "chunked_no_upcast",
            "active_no_upcast",
            "active_chunked_fp32",
            "active_chunked_no_upcast",
            "forward_chunked_fp32",
            "forward_chunked_no_upcast",
            "forward_active_chunked_fp32",
            "forward_active_chunked_no_upcast",
        ],
        help=(
            "Causal LM loss implementation. "
            "'default' = Transformers default (full logits upcast to fp32), "
            "'no_upcast' = avoid fp32 upcast, "
            "'chunked_fp32' = chunked CE with per-chunk fp32 upcast, "
            "'chunked_no_upcast' = chunked CE without fp32 upcast, "
            "'active_*' variants first filter to non-ignored labels then apply the selected mode, "
            "'forward_*' variants avoid materializing full-sequence logits by computing CE from "
            "hidden states in token chunks."
        ),
    )
    parser.add_argument(
        "--causal-loss-chunk-tokens",
        type=int,
        default=2048,
        help="Token chunk size used by chunked causal-loss modes.",
    )
    parser.add_argument(
        "--group-by-length",
        dest="group_by_length",
        action="store_true",
        default=True,
        help="Bucket similar sequence lengths per batch to reduce padding waste.",
    )
    parser.add_argument("--no-group-by-length", dest="group_by_length", action="store_false")
    parser.add_argument(
        "--packing",
        dest="packing",
        action="store_true",
        default=False,
        help="Pack multiple short samples into one sequence window.",
    )
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument(
        "--assistant-only-loss",
        dest="assistant_only_loss",
        action="store_true",
        default=True,
        help="Train loss only on assistant turns.",
    )
    parser.add_argument(
        "--no-assistant-only-loss",
        dest="assistant_only_loss",
        action="store_false",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest/dataset/config pipeline without loading model or training.",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        default=-1.0,
        help=(
            "Hard-stop guard for GPU memory. "
            "-1 = auto (110 GiB for Qwen3.5-9B full-finetune, disabled otherwise), "
            "0 = disabled, >0 = explicit GiB limit."
        ),
    )
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.0,
        help=(
            "Optional hard CUDA allocator cap via torch.cuda.set_per_process_memory_fraction. "
            "Use a value in (0, 1), e.g. 0.88, to force catchable CUDA OOM before system-level kill."
        ),
    )
    parser.add_argument(
        "--debug-cuda-memory-history",
        action="store_true",
        default=False,
        help="Enable CUDA allocator history recording for post-failure snapshots.",
    )
    parser.add_argument(
        "--debug-cuda-memory-history-max-entries",
        type=int,
        default=250000,
        help="Max allocator history events when --debug-cuda-memory-history is enabled.",
    )
    parser.add_argument(
        "--debug-cuda-snapshot-on-error",
        action="store_true",
        default=False,
        help="Dump CUDA allocator snapshot/summary if training raises an exception.",
    )
    parser.add_argument(
        "--resume-torch-load-mmap",
        dest="resume_torch_load_mmap",
        action="store_true",
        default=True,
        help=(
            "When resuming, force torch.load(..., mmap=True) for files under the resume checkpoint path "
            "(optimizer/scheduler/rng) to reduce host-RAM spikes."
        ),
    )
    parser.add_argument(
        "--no-resume-torch-load-mmap",
        dest="resume_torch_load_mmap",
        action="store_false",
    )
    parser.add_argument(
        "--debug-resume-memory-phases",
        action="store_true",
        default=False,
        help=(
            "Add detailed host/GPU memory probes around resume internals "
            "(_load_from_checkpoint, optimizer/scheduler restore, rng restore, torch.load)."
        ),
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive only
        return {"cmd": cmd, "error": repr(exc)}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_gradient_checkpointing(value: str) -> str | bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


def parse_device_map(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"cuda:0", "0", "single"}:
        return {"": 0}
    return value


def resolve_torch_dtype(value: str) -> Any:
    import torch

    lowered = value.strip().lower()
    if lowered == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[lowered]


def resolve_triton_ptxas_path(value: str) -> Path | None:
    lowered = value.strip().lower()
    if lowered in {"", "none", "off", "disable"}:
        return None
    if lowered == "auto":
        candidates = [
            Path("/usr/local/cuda/bin/ptxas"),
            Path("/usr/local/cuda-13.0/bin/ptxas"),
            Path("/usr/local/cuda-12.9/bin/ptxas"),
            Path("/usr/local/cuda-12.8/bin/ptxas"),
            Path(sys.prefix) / "bin" / "ptxas",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None
    candidate = Path(value).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Requested Triton ptxas path not found: {candidate}")
    return candidate


def resolve_precision_flags(model: Any, requested: str) -> tuple[bool, bool, str]:
    import torch

    lowered = requested.strip().lower()
    if lowered == "bf16":
        return True, False, "requested:bf16"
    if lowered == "fp16":
        return False, True, "requested:fp16"
    if lowered == "float32":
        return False, False, "requested:float32"

    dtype = getattr(model.config, "dtype", None) or getattr(model.config, "torch_dtype", None)
    if dtype is None:
        try:
            dtype = model.get_input_embeddings().weight.dtype
        except Exception:
            dtype = None
    dtype_name = str(dtype).lower() if dtype is not None else ""
    if "bfloat16" in dtype_name:
        return True, False, "auto:model_bfloat16"
    if "float16" in dtype_name:
        return False, True, "auto:model_float16"

    try:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return True, False, "auto:cuda_bf16_supported"
            return False, True, "auto:cuda_fp16_fallback"
    except Exception:
        pass
    return False, False, "auto:no_cuda_or_unknown"


def prepend_env_path(var_name: str, paths: list[Path]) -> None:
    clean = [str(path) for path in paths if path.exists()]
    if not clean:
        return
    current = os.environ.get(var_name, "")
    if current:
        os.environ[var_name] = ":".join(clean + [current])
    else:
        os.environ[var_name] = ":".join(clean)


def resolve_max_gpu_memory_gib(requested: float, model_name: str) -> float:
    if requested > 0:
        return requested
    if requested == 0:
        return 0.0
    lowered = model_name.lower()
    if "qwen3.5-9b" in lowered and "base" in lowered:
        return 110.0
    if "qwen3.5-9b" in lowered:
        return 110.0
    return 0.0


def current_process_max_reserved_mib() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    peak_mib = 0.0
    for idx in range(torch.cuda.device_count()):
        try:
            reserved = torch.cuda.max_memory_reserved(idx)
        except Exception:
            continue
        peak_mib = max(peak_mib, float(reserved) / (1024.0 * 1024.0))
    return peak_mib


def current_process_nvidia_used_mib() -> float:
    pid = os.getpid()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 0.0

    if result.returncode != 0:
        return 0.0

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            line_pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if line_pid == pid:
            return used_mib
    return 0.0


def enable_cuda_memory_history(max_entries: int) -> bool:
    import torch

    if not torch.cuda.is_available():
        print("CUDA memory history skipped: CUDA not available")
        return False
    if not hasattr(torch.cuda.memory, "_record_memory_history"):
        print("CUDA memory history skipped: torch.cuda.memory._record_memory_history unavailable")
        return False
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=max_entries,
        )
        print(f"CUDA memory history enabled (max_entries={max_entries})")
        return True
    except Exception as exc:
        print(f"CUDA memory history enable failed: {exc!r}")
        return False


def disable_cuda_memory_history(clear_history: bool = True) -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    if not hasattr(torch.cuda.memory, "_record_memory_history"):
        return False
    try:
        torch.cuda.memory._record_memory_history(
            enabled=None,
            clear_history=clear_history,
        )
        return True
    except Exception as exc:
        print(f"CUDA memory history disable failed: {exc!r}")
        return False


def current_process_rss_mib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def current_mem_available_mib() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def current_process_status_mib() -> dict[str, float]:
    results: dict[str, float] = {
        "rss_mib": 0.0,
        "hwm_mib": 0.0,
        "vms_mib": 0.0,
        "swap_mib": 0.0,
    }
    key_map = {
        "VmRSS:": "rss_mib",
        "VmHWM:": "hwm_mib",
        "VmSize:": "vms_mib",
        "VmSwap:": "swap_mib",
    }
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 2:
                    continue
                mapped = key_map.get(parts[0])
                if mapped is None:
                    continue
                try:
                    results[mapped] = float(parts[1]) / 1024.0
                except ValueError:
                    continue
    except Exception:
        return results
    return results


def current_process_cuda_allocated_mib() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def current_process_cuda_reserved_mib() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def log_memory_probe(prefix: str) -> None:
    status = current_process_status_mib()
    print(
        f"[memory_probe] {prefix}: "
        f"rss_mib={status['rss_mib']:.1f}, "
        f"hwm_mib={status['hwm_mib']:.1f}, "
        f"vms_mib={status['vms_mib']:.1f}, "
        f"swap_mib={status['swap_mib']:.1f}, "
        f"mem_avail_mib={current_mem_available_mib():.1f}, "
        f"cuda_allocated_mib={current_process_cuda_allocated_mib():.1f}, "
        f"cuda_reserved_mib={current_process_cuda_reserved_mib():.1f}, "
        f"torch_peak_reserved_mib={current_process_max_reserved_mib():.1f}, "
        f"nvidia_used_mib={current_process_nvidia_used_mib():.1f}"
    )


def dump_cuda_debug_artifacts(metadata_dir: Path, error_kind: str) -> dict[str, str]:
    import torch

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outputs: dict[str, str] = {}
    if not torch.cuda.is_available():
        outputs["cuda_available"] = "false"
        return outputs

    summary_path = metadata_dir / f"cuda_memory_summary_{error_kind}_{timestamp}.txt"
    snapshot_path = metadata_dir / f"cuda_memory_snapshot_{error_kind}_{timestamp}.pickle"
    stats_path = metadata_dir / f"cuda_memory_stats_{error_kind}_{timestamp}.json"

    try:
        summary = torch.cuda.memory_summary()
        summary_path.write_text(summary, encoding="utf-8")
        outputs["memory_summary"] = str(summary_path)
    except Exception as exc:
        outputs["memory_summary_error"] = repr(exc)

    try:
        stats = {
            "created_at_utc": utc_now(),
            "error_kind": error_kind,
            "torch_peak_reserved_mib": current_process_max_reserved_mib(),
            "nvidia_process_used_mib": current_process_nvidia_used_mib(),
            "device_count": int(torch.cuda.device_count()),
        }
        save_json(stats_path, stats)
        outputs["memory_stats"] = str(stats_path)
    except Exception as exc:
        outputs["memory_stats_error"] = repr(exc)

    if hasattr(torch.cuda.memory, "_dump_snapshot"):
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            outputs["memory_snapshot"] = str(snapshot_path)
        except Exception as exc:
            outputs["memory_snapshot_error"] = repr(exc)
    else:
        outputs["memory_snapshot_error"] = "_dump_snapshot_unavailable"

    return outputs


def render_messages_as_text(
    examples: dict[str, Any], tokenizer: Any, reasoning_effort: str
) -> dict[str, list[str]]:
    def should_fallback(exc: ValueError) -> bool:
        message = str(exc).lower()
        fallback_markers = (
            "chat template",
            "incorrect image source",
            "must be a valid url",
            "base64",
            "image source",
        )
        return any(marker in message for marker in fallback_markers)

    def fallback_join(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", ""))
            thinking = msg.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                lines.append(f"{role}.thinking: {thinking}")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    texts = []
    for messages in examples["messages"]:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            text = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            try:
                text = tokenizer.apply_chat_template(messages, **kwargs)
            except ValueError as exc:
                if should_fallback(exc):
                    text = fallback_join(messages)
                else:
                    raise
        except ValueError as exc:
            if should_fallback(exc):
                text = fallback_join(messages)
            else:
                raise
        texts.append(text)
    return {"text": texts}


def truncate_text_batch_to_max_tokens(
    examples: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
    truncation_side: str,
) -> dict[str, Any]:
    texts = examples["text"]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_batches = encoded["input_ids"]

    truncated_flags: list[int] = []
    original_tokens: list[int] = []
    final_tokens: list[int] = []
    final_token_batches: list[list[int]] = []

    for token_ids in token_batches:
        original_len = len(token_ids)
        original_tokens.append(original_len)

        if original_len > max_seq_length:
            if truncation_side == "left":
                clipped = token_ids[-max_seq_length:]
            else:
                clipped = token_ids[:max_seq_length]
            truncated_flags.append(1)
        else:
            clipped = token_ids
            truncated_flags.append(0)

        final_tokens.append(len(clipped))
        final_token_batches.append(clipped)

    final_texts = tokenizer.batch_decode(
        final_token_batches,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return {
        "text": final_texts,
        "__orig_tokens": original_tokens,
        "__final_tokens": final_tokens,
        "__was_truncated": truncated_flags,
    }


def configure_causal_lm_loss(
    model: Any,
    mode: str,
    chunk_tokens: int,
) -> dict[str, Any]:
    import types

    import torch
    import torch.nn.functional as F
    from transformers.modeling_outputs import CausalLMOutputWithPast

    normalized = mode.strip().lower()
    valid_modes = {
        "default",
        "no_upcast",
        "chunked_fp32",
        "chunked_no_upcast",
        "active_no_upcast",
        "active_chunked_fp32",
        "active_chunked_no_upcast",
        "forward_chunked_fp32",
        "forward_chunked_no_upcast",
        "forward_active_chunked_fp32",
        "forward_active_chunked_no_upcast",
    }
    if normalized not in valid_modes:
        raise ValueError(f"Unsupported causal loss mode: {mode}")

    forward_chunked = normalized.startswith("forward_")
    core_mode = normalized[len("forward_") :] if forward_chunked else normalized
    active_only = core_mode.startswith("active_")
    base_mode = core_mode[len("active_") :] if active_only else core_mode

    if base_mode.startswith("chunked") and chunk_tokens <= 0:
        raise ValueError("--causal-loss-chunk-tokens must be > 0 for chunked modes.")
    if forward_chunked and base_mode not in {"chunked_fp32", "chunked_no_upcast"}:
        raise ValueError("forward_* causal loss modes require a chunked_* base mode.")

    if base_mode == "default":
        return {
            "mode": normalized,
            "installed": False,
            "chunked": False,
            "forward_chunked_logits": False,
            "chunk_tokens": 0,
            "upcast_logits": True,
            "active_only": bool(active_only),
            "impl": "transformers_default",
        }

    chunked = base_mode in {"chunked_fp32", "chunked_no_upcast"}
    upcast_logits = base_mode in {"chunked_fp32"}
    if base_mode == "no_upcast":
        upcast_logits = False

    if forward_chunked:
        if not hasattr(model, "model") or not hasattr(model, "lm_head"):
            raise ValueError(
                "forward_* causal loss mode requires model.model and model.lm_head attributes."
            )
        original_forward = model.forward

        def forward_with_chunked_loss(
            self: Any,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Any | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            cache_position: torch.LongTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs: Any,
        ) -> CausalLMOutputWithPast:
            # Keep non-training behavior identical to model-native forward.
            if labels is None:
                return original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            loss_kwargs = dict(kwargs)
            num_items_in_batch = loss_kwargs.pop("num_items_in_batch", None)
            ignore_index = int(loss_kwargs.pop("ignore_index", -100))

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **loss_kwargs,
            )
            hidden_states = outputs.last_hidden_state

            # Standard causal shift: token t predicts label t+1.
            shift_hidden = hidden_states[:, :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_hidden = shift_hidden.view(-1, shift_hidden.shape[-1])
            flat_labels = shift_labels.view(-1).to(flat_hidden.device)
            supervised_mask = flat_labels.ne(ignore_index)
            supervised_count = supervised_mask.sum()
            supervised_indices = (
                supervised_mask.nonzero(as_tuple=False).squeeze(-1)
                if active_only
                else None
            )

            step = int(chunk_tokens)
            total_loss: torch.Tensor | None = None

            if active_only:
                for start in range(0, supervised_indices.shape[0], step):
                    end = min(start + step, supervised_indices.shape[0])
                    token_indices = supervised_indices[start:end]
                    hidden_chunk = flat_hidden.index_select(0, token_indices)
                    labels_chunk = flat_labels.index_select(0, token_indices).to(torch.long)
                    logits_chunk = self.lm_head(hidden_chunk)
                    if upcast_logits:
                        logits_chunk = logits_chunk.float()
                    chunk_loss = F.cross_entropy(logits_chunk, labels_chunk, reduction="sum")
                    total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
            else:
                for start in range(0, flat_labels.shape[0], step):
                    end = min(start + step, flat_labels.shape[0])
                    hidden_chunk = flat_hidden[start:end]
                    labels_chunk = flat_labels[start:end].to(torch.long)
                    logits_chunk = self.lm_head(hidden_chunk)
                    if upcast_logits:
                        logits_chunk = logits_chunk.float()
                    chunk_loss = F.cross_entropy(
                        logits_chunk,
                        labels_chunk,
                        ignore_index=ignore_index,
                        reduction="sum",
                    )
                    total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss

            if total_loss is None:  # pragma: no cover - empty/fully masked safety path.
                total_loss = flat_hidden[:1].sum() * 0.0

            if num_items_in_batch is None:
                denom_source = supervised_count
                denom = denom_source.clamp(min=1).to(total_loss.device, dtype=total_loss.dtype)
            elif torch.is_tensor(num_items_in_batch):
                denom = num_items_in_batch.to(total_loss.device, dtype=total_loss.dtype)
            else:
                denom = torch.tensor(
                    float(num_items_in_batch),
                    device=total_loss.device,
                    dtype=total_loss.dtype,
                )
            loss = total_loss / denom

            # Keep a tiny logits tensor in output to satisfy downstream consumers.
            last_logits = self.lm_head(hidden_states[:, -1:, :])
            return CausalLMOutputWithPast(
                loss=loss,
                logits=last_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        model.forward = types.MethodType(forward_with_chunked_loss, model)
        return {
            "mode": normalized,
            "installed": True,
            "chunked": True,
            "forward_chunked_logits": True,
            "chunk_tokens": int(chunk_tokens),
            "upcast_logits": bool(upcast_logits),
            "active_only": bool(active_only),
            "impl": "forward_chunked_hidden_ce",
        }

    def custom_for_causal_lm_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: torch.Tensor | int | None = None,
        ignore_index: int = -100,
        shift_labels: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if shift_labels is None:
            labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        flat_logits = logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1).to(flat_logits.device)
        active_count = flat_labels.ne(ignore_index).sum()

        if active_only:
            active_mask = flat_labels.ne(ignore_index)
            flat_logits = flat_logits[active_mask]
            flat_labels = flat_labels[active_mask]

        if not chunked:
            source = flat_logits.float() if upcast_logits else flat_logits
            if source.shape[0] == 0:
                return source.sum() * 0.0
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = F.cross_entropy(source, flat_labels, ignore_index=ignore_index, reduction=reduction)
            if reduction == "sum":
                if torch.is_tensor(num_items_in_batch):
                    denom = num_items_in_batch.to(loss.device, dtype=loss.dtype)
                else:
                    denom = torch.tensor(float(num_items_in_batch), device=loss.device, dtype=loss.dtype)
                loss = loss / denom
            return loss

        # Chunked CE path to bound temporary loss memory at long context.
        step = int(chunk_tokens)
        total_loss: torch.Tensor | None = None
        for start in range(0, flat_labels.shape[0], step):
            end = min(start + step, flat_labels.shape[0])
            logits_chunk = flat_logits[start:end]
            labels_chunk = flat_labels[start:end]
            if upcast_logits:
                logits_chunk = logits_chunk.float()
            chunk_loss = F.cross_entropy(
                logits_chunk,
                labels_chunk,
                ignore_index=ignore_index,
                reduction="sum",
            )
            total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss

        if total_loss is None:  # pragma: no cover - empty batch defensive path
            total_loss = flat_logits[:1].sum() * 0.0

        if num_items_in_batch is None:
            denom_source = active_count if active_only else flat_labels.ne(ignore_index).sum()
            denom = denom_source.clamp(min=1)
            denom = denom.to(total_loss.device, dtype=total_loss.dtype)
        elif torch.is_tensor(num_items_in_batch):
            denom = num_items_in_batch.to(total_loss.device, dtype=total_loss.dtype)
        else:
            denom = torch.tensor(float(num_items_in_batch), device=total_loss.device, dtype=total_loss.dtype)
        return total_loss / denom

    model.loss_function = custom_for_causal_lm_loss
    return {
        "mode": normalized,
        "installed": True,
        "chunked": bool(chunked),
        "forward_chunked_logits": False,
        "chunk_tokens": int(chunk_tokens) if chunked else 0,
        "upcast_logits": bool(upcast_logits),
        "active_only": bool(active_only),
        "impl": "custom_for_causal_lm_loss",
    }


def main() -> None:
    args = parse_args()
    if args.eval_holdout_ratio < 0 or args.eval_holdout_ratio >= 1:
        raise ValueError("--eval-holdout-ratio must be in [0, 1).")
    resolved_max_gpu_memory_gib = resolve_max_gpu_memory_gib(
        args.max_gpu_memory_gib, args.model_name
    )
    resolved_max_gpu_memory_mib = resolved_max_gpu_memory_gib * 1024.0

    session_dir = Path(args.session_dir).expanduser().resolve()
    metadata_dir = session_dir / "metadata"
    artifacts_dir = session_dir / "artifacts"
    checkpoints_dir = session_dir / "checkpoints"
    full_model_dir = artifacts_dir / "full_model"
    adapter_dir = artifacts_dir / "adapter"
    merged_dir = artifacts_dir / "merged_16bit"
    gguf_dir = artifacts_dir / "gguf"

    manifest_path = metadata_dir / "dataset_manifest.json"
    session_meta_path = metadata_dir / "session.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    if not session_meta_path.exists():
        raise FileNotFoundError(f"Missing session metadata: {session_meta_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace_root = Path(manifest["workspace_root"])
    dataset_files: list[Path] = []
    for item in manifest["files"]:
        if "relative_path" in item:
            dataset_files.append((workspace_root / item["relative_path"]).resolve())
            continue
        if "absolute_path" in item:
            dataset_files.append(Path(item["absolute_path"]).expanduser().resolve())
            continue
        raise ValueError(f"Manifest entry missing relative_path/absolute_path: {item}")
    missing = [str(path) for path in dataset_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Dataset files missing:\n" + "\n".join(missing))

    cache_root = (
        Path(args.hf_cache_dir).expanduser().resolve()
        if args.hf_cache_dir
        else workspace_root / "qwen35_9b_fullft" / ".cache" / "huggingface"
    )
    (cache_root / "datasets").mkdir(parents=True, exist_ok=True)
    (cache_root / "transformers").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")

    headers_root = (
        Path(args.python_headers_root).expanduser().resolve()
        if args.python_headers_root
        else (workspace_root / ".local_py312dev" / "usr" / "include").resolve()
    )
    header_paths = [
        headers_root,
        headers_root / "python3.12",
        headers_root / "aarch64-linux-gnu" / "python3.12",
    ]
    prepend_env_path("CPATH", header_paths)
    prepend_env_path("C_INCLUDE_PATH", header_paths)
    prepend_env_path("CPLUS_INCLUDE_PATH", header_paths)

    resolved_triton_ptxas = resolve_triton_ptxas_path(args.triton_ptxas_path)
    if resolved_triton_ptxas is not None:
        os.environ["TRITON_PTXAS_PATH"] = str(resolved_triton_ptxas)
        print(f"Using TRITON_PTXAS_PATH={resolved_triton_ptxas}")
    else:
        print("Using Triton default PTXAS path resolution")

    from datasets import load_dataset

    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi": run_command(["nvidia-smi"]),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }
    save_json(metadata_dir / "environment.json", environment)

    print(f"Loading {len(dataset_files)} dataset files from manifest")
    dataset = load_dataset(
        "json",
        data_files=[str(path) for path in dataset_files],
        split="train",
    )
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if "text" in dataset.column_names:
        dataset_text_field = "text"
        has_messages = "messages" in dataset.column_names
    elif "messages" in dataset.column_names:
        dataset_text_field = "text"
        has_messages = True
    else:
        raise ValueError(
            f"Dataset columns must include 'messages' or 'text'. Got: {dataset.column_names}"
        )

    target_modules = [
        module.strip()
        for module in args.target_modules.split(",")
        if module.strip()
    ]

    if args.full_finetuning and args.load_in_4bit:
        raise ValueError(
            "full-finetuning with 4-bit loading is not supported. "
            "Use --no-load-in-4bit for full-weight training."
        )

    if args.cuda_alloc_conf.strip():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf.strip()
        print(f"Using PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    if args.dry_run:
        run_config = {
            "created_at_utc": utc_now(),
            "session_dir": str(session_dir),
            "model_name": args.model_name,
            "attn_implementation": args.attn_implementation,
            "dataset_manifest": str(manifest_path),
            "dataset_num_rows": len(dataset),
            "dataset_columns": dataset.column_names,
            "dry_run": True,
            "reasoning_effort": args.reasoning_effort,
            "device_map": args.device_map,
            "hf_cache_dir": str(cache_root),
            "python_headers_root": str(headers_root),
            "disable_unsloth_compile": args.disable_unsloth_compile,
            "disable_moe_triton": args.disable_moe_triton,
            "disable_flex_attention": args.disable_flex_attention,
            "disable_cce": args.disable_cce,
            "causal_loss_mode": args.causal_loss_mode,
            "causal_loss_chunk_tokens": args.causal_loss_chunk_tokens,
            "cuda_alloc_conf": args.cuda_alloc_conf,
            "group_by_length": args.group_by_length,
            "packing": args.packing,
            "assistant_only_loss": args.assistant_only_loss,
            "precision": args.precision,
            "torch_dtype": args.torch_dtype,
            "load_in_4bit": args.load_in_4bit,
            "full_finetuning": args.full_finetuning,
            "force_causal_lm_loader": args.force_causal_lm_loader,
            "freeze_visual_modules": args.freeze_visual_modules,
            "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
            "unsloth_mixed_precision": args.unsloth_mixed_precision,
            "triton_ptxas_path": str(resolved_triton_ptxas) if resolved_triton_ptxas else "",
            "truncation_side": args.truncation_side,
            "truncate_overlength_samples": args.truncate_overlength_samples,
            "max_gpu_memory_gib_guard": resolved_max_gpu_memory_gib,
            "cuda_memory_fraction": args.cuda_memory_fraction,
            "optim": args.optim,
            "save_only_model": args.save_only_model,
            "checkpoint_max_shard_size": args.checkpoint_max_shard_size,
            "checkpoint_safe_serialization": args.checkpoint_safe_serialization,
            "checkpoint_presave_gc": args.checkpoint_presave_gc,
            "checkpoint_presave_empty_cache": args.checkpoint_presave_empty_cache,
            "checkpoint_presave_disable_cuda_history": args.checkpoint_presave_disable_cuda_history,
            "torch_empty_cache_steps": args.torch_empty_cache_steps,
            "training_mode": "full_finetuning" if args.full_finetuning else "lora",
            "lora": (
                {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": target_modules,
                    "gradient_checkpointing": args.gradient_checkpointing,
                }
                if not args.full_finetuning
                else None
            ),
            "dataset_num_proc": args.dataset_num_proc,
            "save_strategy": args.save_strategy,
            "eval_holdout_ratio": args.eval_holdout_ratio,
            "eval_strategy": args.eval_strategy,
            "eval_steps": args.eval_steps,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "eval_max_samples": args.eval_max_samples,
            "load_best_model_at_end": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
            "skip_final_save": args.skip_final_save,
        }
        save_json(metadata_dir / "run_config.json", run_config)

        session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
        session_meta["status"] = "validated"
        session_meta["last_updated_utc"] = utc_now()
        save_json(session_meta_path, session_meta)

        print("Dry run complete: manifest and dataset validated.")
        print(f"Rows considered: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")
        if has_messages:
            print("Detected messages-format samples.")
        if resolved_max_gpu_memory_gib > 0:
            print(f"GPU memory guard enabled: {resolved_max_gpu_memory_gib:.2f} GiB")
        else:
            print("GPU memory guard disabled")
        return

    if args.disable_unsloth_compile:
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    if args.disable_moe_triton:
        os.environ["UNSLOTH_DISABLE_MOE_TRITON"] = "1"
    if args.disable_flex_attention:
        os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
    if args.disable_cce:
        os.environ["UNSLOTH_ENABLE_CCE"] = "0"
    if args.unsloth_mixed_precision != "auto":
        os.environ["UNSLOTH_MIXED_PRECISION"] = args.unsloth_mixed_precision

    from unsloth import FastLanguageModel
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    cuda_memory_history_enabled = False
    if args.debug_cuda_memory_history:
        cuda_memory_history_enabled = enable_cuda_memory_history(
            max_entries=max(1, args.debug_cuda_memory_history_max_entries)
        )

    if args.cuda_memory_fraction > 0:
        if not 0.0 < args.cuda_memory_fraction < 1.0:
            raise ValueError("--cuda-memory-fraction must be in (0, 1).")
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda-memory-fraction requested but CUDA is not available.")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
        print(
            "CUDA per-process allocator cap enabled: "
            f"{args.cuda_memory_fraction:.4f} of device memory"
        )

    print(f"Loading model: {args.model_name}")
    requested_dtype = resolve_torch_dtype(args.torch_dtype)
    from_pretrained_kwargs: dict[str, Any] = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "dtype": requested_dtype,
        "load_in_4bit": args.load_in_4bit,
        "full_finetuning": args.full_finetuning,
        "device_map": parse_device_map(args.device_map),
        "attn_implementation": args.attn_implementation,
    }
    if args.force_causal_lm_loader:
        from_pretrained_kwargs["auto_model"] = AutoModelForCausalLM
        print("Forcing AutoModelForCausalLM load path (text-only; skips vision tower weights)")
    if args.load_in_4bit:
        bnb_compute_dtype = requested_dtype
        if args.bnb_4bit_compute_dtype != "auto":
            bnb_compute_dtype = resolve_torch_dtype(args.bnb_4bit_compute_dtype)
        if bnb_compute_dtype is not None:
            from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print(f"Forcing bnb_4bit_compute_dtype={bnb_compute_dtype}")
    model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
    causal_loss_config = configure_causal_lm_loss(
        model,
        mode=args.causal_loss_mode,
        chunk_tokens=args.causal_loss_chunk_tokens,
    )
    print(
        "Causal loss config: "
        f"mode={causal_loss_config['mode']}, "
        f"chunked={causal_loss_config['chunked']}, "
        f"forward_chunked_logits={causal_loss_config.get('forward_chunked_logits', False)}, "
        f"chunk_tokens={causal_loss_config['chunk_tokens']}, "
        f"upcast_logits={causal_loss_config['upcast_logits']}, "
        f"installed={causal_loss_config['installed']}"
    )
    frozen_visual_params = 0
    if args.freeze_visual_modules:
        visual_module = getattr(getattr(model, "model", None), "visual", None)
        if visual_module is None:
            print("Visual module freeze enabled: model has no model.visual module")
        else:
            for parameter in visual_module.parameters():
                if parameter.requires_grad:
                    frozen_visual_params += parameter.numel()
                    parameter.requires_grad_(False)
            print(
                "Visual module freeze enabled: "
                f"froze {frozen_visual_params:,} params ({(frozen_visual_params * 2) / (1024**3):.3f} GiB bf16 weights)"
            )
    else:
        print("Visual module freeze disabled")
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tokenizer is not tokenizer:
        print("Detected multimodal processor; using inner text tokenizer for text-only SFT.")

    tokenizer.truncation_side = args.truncation_side
    if getattr(text_tokenizer, "truncation_side", None) is not None:
        text_tokenizer.truncation_side = args.truncation_side
    print(
        f"Tokenizer truncation_side={getattr(text_tokenizer, 'truncation_side', args.truncation_side)} "
        f"(max_seq_length={args.max_seq_length})"
    )
    for cfg in (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
    ):
        if cfg is not None:
            setattr(cfg, "_attn_implementation", args.attn_implementation)
    bf16_flag, fp16_flag, precision_reason = resolve_precision_flags(model, args.precision)
    print(f"Resolved precision: bf16={bf16_flag}, fp16={fp16_flag} ({precision_reason})")

    if args.full_finetuning:
        print("Training mode: full_finetuning (no LoRA adapter wrapping)")
    else:
        print(f"Applying LoRA config to modules: {target_modules}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing=parse_gradient_checkpointing(args.gradient_checkpointing),
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

    if "messages" in dataset.column_names:
        dataset = dataset.map(
            lambda batch: render_messages_as_text(batch, tokenizer, args.reasoning_effort),
            batched=True,
            desc="Rendering chat template",
        )
        dataset_text_field = "text"

    truncation_stats = {
        "enabled": bool(args.truncate_overlength_samples),
        "max_seq_length": int(args.max_seq_length),
        "truncation_side": args.truncation_side,
        "total_rows": int(len(dataset)),
        "rows_truncated": 0,
        "pct_rows_truncated": 0.0,
        "max_original_tokens": 0,
        "max_final_tokens": 0,
    }
    if args.truncate_overlength_samples and dataset_text_field == "text":
        dataset = dataset.map(
            lambda batch: truncate_text_batch_to_max_tokens(
                batch,
                text_tokenizer,
                args.max_seq_length,
                args.truncation_side,
            ),
            batched=True,
            desc="Applying max_seq_length truncation",
        )
        orig_tokens = dataset["__orig_tokens"]
        final_tokens = dataset["__final_tokens"]
        truncated_flags = dataset["__was_truncated"]
        rows_truncated = int(sum(int(flag) for flag in truncated_flags))
        total_rows = int(len(dataset))
        truncation_stats = {
            "enabled": True,
            "max_seq_length": int(args.max_seq_length),
            "truncation_side": args.truncation_side,
            "total_rows": total_rows,
            "rows_truncated": rows_truncated,
            "pct_rows_truncated": round((rows_truncated / total_rows) * 100.0, 4)
            if total_rows > 0
            else 0.0,
            "max_original_tokens": int(max(orig_tokens)) if orig_tokens else 0,
            "max_final_tokens": int(max(final_tokens)) if final_tokens else 0,
        }
        print(
            "Pre-truncation stats: "
            f"rows_truncated={rows_truncated}/{total_rows}, "
            f"max_original_tokens={truncation_stats['max_original_tokens']}, "
            f"max_final_tokens={truncation_stats['max_final_tokens']}"
        )
        dataset = dataset.remove_columns(["__orig_tokens", "__final_tokens", "__was_truncated"])
    save_json(metadata_dir / "truncation_stats.json", truncation_stats)

    train_dataset = dataset
    eval_dataset = None
    if args.eval_holdout_ratio > 0:
        if len(dataset) < 2:
            raise ValueError("Need at least 2 rows to enable holdout eval split.")
        split = dataset.train_test_split(
            test_size=args.eval_holdout_ratio,
            seed=args.seed,
            shuffle=True,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        if args.eval_max_samples > 0:
            eval_dataset = eval_dataset.select(range(min(args.eval_max_samples, len(eval_dataset))))
        print(
            "Holdout eval enabled: "
            f"train_rows={len(train_dataset)}, eval_rows={len(eval_dataset)}, "
            f"ratio={args.eval_holdout_ratio}"
        )

    eval_enabled = eval_dataset is not None and args.eval_strategy != "no"

    checkpoint_save_overrides: dict[str, Any] = {}
    if args.checkpoint_max_shard_size.strip():
        checkpoint_save_overrides["max_shard_size"] = args.checkpoint_max_shard_size.strip()
    if args.checkpoint_safe_serialization != "auto":
        checkpoint_save_overrides["safe_serialization"] = args.checkpoint_safe_serialization == "true"
    if checkpoint_save_overrides:
        print(f"Checkpoint save_pretrained overrides: {checkpoint_save_overrides}")

    trainer_gradient_checkpointing = parse_gradient_checkpointing(args.gradient_checkpointing)
    if isinstance(trainer_gradient_checkpointing, str):
        lowered = trainer_gradient_checkpointing.strip().lower()
        trainer_gradient_checkpointing = lowered not in {"", "false", "off", "0"}

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "max_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "dataset_num_proc": args.dataset_num_proc if args.dataset_num_proc > 0 else None,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": bool(trainer_gradient_checkpointing),
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": args.save_only_model,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "report_to": args.report_to,
        "optim": args.optim,
        "bf16": bf16_flag,
        "fp16": fp16_flag,
        "group_by_length": args.group_by_length,
        "packing": args.packing,
        "assistant_only_loss": args.assistant_only_loss,
    }
    if args.torch_empty_cache_steps > 0:
        sft_kwargs["torch_empty_cache_steps"] = args.torch_empty_cache_steps
    if eval_enabled:
        greater_is_better = not args.metric_for_best_model.strip().lower().endswith("loss")
        sft_kwargs.update(
            {
                "do_eval": True,
                "eval_strategy": args.eval_strategy,
                "eval_steps": args.eval_steps,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "load_best_model_at_end": args.load_best_model_at_end,
                "metric_for_best_model": args.metric_for_best_model,
                "greater_is_better": greater_is_better,
            }
        )
    elif eval_dataset is not None:
        print("Holdout eval dataset exists but eval_strategy=no; evaluation disabled.")
    if args.max_steps > 0:
        sft_kwargs["max_steps"] = args.max_steps
    else:
        sft_kwargs["num_train_epochs"] = args.num_train_epochs

    sft_init_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    filtered_sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_init_params}
    dropped_sft_kwargs = sorted(set(sft_kwargs.keys()) - set(filtered_sft_kwargs.keys()))
    if dropped_sft_kwargs:
        print(f"Dropping unsupported SFTConfig args for current trl version: {dropped_sft_kwargs}")

    training_args = SFTConfig(**filtered_sft_kwargs)
    callbacks: list[Any] = []
    if resolved_max_gpu_memory_mib > 0:
        print(f"GPU memory guard enabled: {resolved_max_gpu_memory_gib:.2f} GiB")

        class MaxGpuMemoryGuardCallback(TrainerCallback):
            def __init__(self, max_mib: float) -> None:
                self.max_mib = max_mib

            def _check(self) -> None:
                torch_peak_mib = current_process_max_reserved_mib()
                nvidia_used_mib = current_process_nvidia_used_mib()
                measured_mib = max(torch_peak_mib, nvidia_used_mib)
                if measured_mib > self.max_mib:
                    raise RuntimeError(
                        "GPU memory guard triggered: "
                        f"measured={measured_mib:.1f} MiB "
                        f"(torch_peak_reserved={torch_peak_mib:.1f} MiB, "
                        f"nvidia_process_used={nvidia_used_mib:.1f} MiB) "
                        f"limit={self.max_mib:.1f} MiB "
                        f"(~{self.max_mib / 1024.0:.2f} GiB)"
                    )

            def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._check()
                return control

            def on_substep_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._check()
                return control

            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._check()
                return control

        callbacks.append(MaxGpuMemoryGuardCallback(resolved_max_gpu_memory_mib))
    else:
        print("GPU memory guard disabled")

    if (
        args.checkpoint_presave_gc
        or args.checkpoint_presave_empty_cache
        or args.checkpoint_presave_disable_cuda_history
    ):
        checkpoint_presave_gc = bool(args.checkpoint_presave_gc)
        checkpoint_presave_empty_cache = bool(args.checkpoint_presave_empty_cache)
        checkpoint_presave_disable_cuda_history = bool(args.checkpoint_presave_disable_cuda_history)
        print(
            "Checkpoint pre-save housekeeping enabled: "
            f"gc={checkpoint_presave_gc}, "
            f"empty_cache={checkpoint_presave_empty_cache}, "
            f"disable_cuda_history={checkpoint_presave_disable_cuda_history}"
        )

        class CheckpointPreSaveCallback(TrainerCallback):
            def __init__(self) -> None:
                self.cuda_history_disabled = False

            def _log_memory(self, prefix: str) -> None:
                print(
                    f"{prefix}: "
                    f"rss_mib={current_process_rss_mib():.1f}, "
                    f"mem_avail_mib={current_mem_available_mib():.1f}, "
                    f"torch_peak_reserved_mib={current_process_max_reserved_mib():.1f}, "
                    f"nvidia_used_mib={current_process_nvidia_used_mib():.1f}"
                )

            def on_step_end(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                if not control.should_save:
                    return control
                self._log_memory("Checkpoint pre-save memory")
                if checkpoint_presave_disable_cuda_history and cuda_memory_history_enabled:
                    if disable_cuda_memory_history(clear_history=True):
                        self.cuda_history_disabled = True
                        print("Checkpoint pre-save action: disabled CUDA memory history and cleared history")
                if checkpoint_presave_gc:
                    gc.collect()
                    print("Checkpoint pre-save action: gc.collect()")
                if checkpoint_presave_empty_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("Checkpoint pre-save action: torch.cuda.empty_cache()")
                self._log_memory("Checkpoint pre-save memory (after cleanup)")
                return control

            def on_save(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._log_memory("Checkpoint post-save memory")
                return control

        callbacks.append(CheckpointPreSaveCallback())

    if checkpoint_save_overrides:
        original_save_pretrained = model.save_pretrained

        def save_pretrained_with_overrides(
            save_directory: str, *sp_args: Any, **sp_kwargs: Any
        ) -> Any:
            for key, value in checkpoint_save_overrides.items():
                sp_kwargs.setdefault(key, value)
            return original_save_pretrained(save_directory, *sp_args, **sp_kwargs)

        model.save_pretrained = save_pretrained_with_overrides  # type: ignore[assignment]

    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_enabled else None,
        dataset_text_field=dataset_text_field,
        args=training_args,
        callbacks=callbacks,
    )

    run_config = {
        "created_at_utc": utc_now(),
        "session_dir": str(session_dir),
        "model_name": args.model_name,
        "attn_implementation": args.attn_implementation,
        "dataset_manifest": str(manifest_path),
        "dataset_num_rows": len(dataset),
        "train_num_rows": len(train_dataset),
        "eval_num_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "eval_enabled": eval_enabled,
        "train_args": sft_kwargs,
        "dry_run": False,
        "reasoning_effort": args.reasoning_effort,
        "device_map": args.device_map,
        "hf_cache_dir": str(cache_root),
        "python_headers_root": str(headers_root),
        "disable_unsloth_compile": args.disable_unsloth_compile,
        "disable_moe_triton": args.disable_moe_triton,
        "disable_flex_attention": args.disable_flex_attention,
        "disable_cce": args.disable_cce,
        "causal_loss": causal_loss_config,
        "group_by_length": args.group_by_length,
        "packing": args.packing,
        "assistant_only_loss": args.assistant_only_loss,
        "precision": args.precision,
        "torch_dtype": args.torch_dtype,
        "load_in_4bit": args.load_in_4bit,
        "full_finetuning": args.full_finetuning,
        "force_causal_lm_loader": args.force_causal_lm_loader,
        "freeze_visual_modules": args.freeze_visual_modules,
        "frozen_visual_params": frozen_visual_params,
        "training_mode": "full_finetuning" if args.full_finetuning else "lora",
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "unsloth_mixed_precision": args.unsloth_mixed_precision,
        "triton_ptxas_path": str(resolved_triton_ptxas) if resolved_triton_ptxas else "",
        "truncation_side": args.truncation_side,
        "truncate_overlength_samples": args.truncate_overlength_samples,
        "truncation_stats": truncation_stats,
        "max_gpu_memory_gib_guard": resolved_max_gpu_memory_gib,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "save_strategy": args.save_strategy,
        "checkpoint_max_shard_size": args.checkpoint_max_shard_size,
        "checkpoint_safe_serialization": args.checkpoint_safe_serialization,
        "checkpoint_presave_gc": args.checkpoint_presave_gc,
        "checkpoint_presave_empty_cache": args.checkpoint_presave_empty_cache,
        "checkpoint_presave_disable_cuda_history": args.checkpoint_presave_disable_cuda_history,
        "eval_holdout_ratio": args.eval_holdout_ratio,
        "eval_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "eval_max_samples": args.eval_max_samples,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": args.metric_for_best_model,
        "skip_final_save": args.skip_final_save,
        "resolved_precision": {
            "bf16": bf16_flag,
            "fp16": fp16_flag,
            "reason": precision_reason,
        },
        "lora": (
            {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": target_modules,
                "gradient_checkpointing": args.gradient_checkpointing,
            }
            if not args.full_finetuning
            else None
        ),
        "dataset_num_proc": args.dataset_num_proc,
        "debug_cuda_memory_history": args.debug_cuda_memory_history,
        "debug_cuda_memory_history_max_entries": args.debug_cuda_memory_history_max_entries,
        "debug_cuda_snapshot_on_error": args.debug_cuda_snapshot_on_error,
        "resume_torch_load_mmap": args.resume_torch_load_mmap,
        "debug_resume_memory_phases": args.debug_resume_memory_phases,
    }
    save_json(metadata_dir / "run_config.json", run_config)

    probe_restore_ops: list[tuple[Any, str, Any]] = []
    resume_checkpoint_dir = (
        Path(args.resume_from_checkpoint).resolve()
        if args.resume_from_checkpoint
        else None
    )
    install_resume_hooks = bool(
        args.resume_from_checkpoint and (args.debug_resume_memory_phases or args.resume_torch_load_mmap)
    )
    if install_resume_hooks:
        if args.debug_resume_memory_phases:
            log_memory_probe("resume_probe:pre_install")

        def install_method_probe(instance: Any, method_name: str) -> None:
            original = getattr(instance, method_name, None)
            if original is None:
                return

            def wrapped(*wrapped_args: Any, **wrapped_kwargs: Any) -> Any:
                log_memory_probe(f"resume_probe:{method_name}:begin")
                try:
                    result = original(*wrapped_args, **wrapped_kwargs)
                except Exception as exc:
                    log_memory_probe(f"resume_probe:{method_name}:error:{type(exc).__name__}")
                    raise
                log_memory_probe(f"resume_probe:{method_name}:end")
                return result

            setattr(instance, method_name, wrapped)
            probe_restore_ops.append((instance, method_name, original))
            print(f"resume_probe: installed method hook for {method_name}")

        if args.debug_resume_memory_phases:
            install_method_probe(trainer, "_load_from_checkpoint")
            install_method_probe(trainer, "_load_optimizer_and_scheduler")
            install_method_probe(trainer, "_load_rng_state")
            install_method_probe(trainer, "training_step")

        import torch as _torch

        original_torch_load = getattr(_torch, "load", None)
        if original_torch_load is not None:
            def torch_load_with_probe(*load_args: Any, **load_kwargs: Any) -> Any:
                target = "<unknown>"
                if load_args:
                    target = str(load_args[0])
                elif "f" in load_kwargs:
                    target = str(load_kwargs["f"])

                target_path: Path | None = None
                try:
                    target_path = Path(target).resolve()
                except Exception:
                    target_path = None

                use_mmap = False
                if (
                    args.resume_torch_load_mmap
                    and resume_checkpoint_dir is not None
                    and target_path is not None
                ):
                    use_mmap = str(target_path).startswith(str(resume_checkpoint_dir))

                effective_kwargs = dict(load_kwargs)
                if use_mmap and "mmap" not in effective_kwargs:
                    effective_kwargs["mmap"] = True

                if args.debug_resume_memory_phases:
                    log_memory_probe(f"resume_probe:torch.load:begin:{target}")
                try:
                    loaded = original_torch_load(*load_args, **effective_kwargs)
                except Exception as exc:
                    if args.resume_torch_load_mmap and use_mmap and effective_kwargs.get("mmap") is True:
                        fallback_kwargs = dict(effective_kwargs)
                        fallback_kwargs.pop("mmap", None)
                        print(
                            "resume_mmap: torch.load with mmap=True failed for "
                            f"{target}: {exc!r}; retrying with mmap disabled"
                        )
                        loaded = original_torch_load(*load_args, **fallback_kwargs)
                    else:
                        if args.debug_resume_memory_phases:
                            log_memory_probe(f"resume_probe:torch.load:error:{target}:{type(exc).__name__}")
                        raise
                if args.debug_resume_memory_phases:
                    log_memory_probe(f"resume_probe:torch.load:end:{target}")
                return loaded

            _torch.load = torch_load_with_probe  # type: ignore[assignment]
            probe_restore_ops.append((_torch, "load", original_torch_load))
            if args.debug_resume_memory_phases:
                print("resume_probe: installed torch.load hook")
            elif args.resume_torch_load_mmap:
                print("resume_mmap: installed torch.load mmap hook")

        if args.debug_resume_memory_phases:
            try:
                import safetensors.torch as _st
            except Exception:
                _st = None
            if _st is not None:
                original_st_load_file = getattr(_st, "load_file", None)
                if original_st_load_file is not None:
                    def st_load_file_with_probe(*load_args: Any, **load_kwargs: Any) -> Any:
                        target = "<unknown>"
                        if load_args:
                            target = str(load_args[0])
                        elif "filename" in load_kwargs:
                            target = str(load_kwargs["filename"])
                        log_memory_probe(f"resume_probe:safetensors.load_file:begin:{target}")
                        try:
                            loaded = original_st_load_file(*load_args, **load_kwargs)
                        except Exception as exc:
                            log_memory_probe(
                                f"resume_probe:safetensors.load_file:error:{target}:{type(exc).__name__}"
                            )
                            raise
                        log_memory_probe(f"resume_probe:safetensors.load_file:end:{target}")
                        return loaded

                    _st.load_file = st_load_file_with_probe  # type: ignore[assignment]
                    probe_restore_ops.append((_st, "load_file", original_st_load_file))
                    print("resume_probe: installed safetensors.load_file hook")

            log_memory_probe("resume_probe:post_install")

    print("Starting training")
    train_result = None
    try:
        if args.debug_resume_memory_phases and args.resume_from_checkpoint:
            log_memory_probe("resume_probe:before_trainer_train")
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception as exc:
        error_kind = "runtime_error" if isinstance(exc, RuntimeError) else "exception"
        error_payload = {
            "created_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_repr": repr(exc),
            "traceback": traceback.format_exc(),
            "cuda_memory_history_enabled": cuda_memory_history_enabled,
            "debug_artifacts": {},
        }
        if args.debug_cuda_snapshot_on_error:
            debug_artifacts = dump_cuda_debug_artifacts(metadata_dir, error_kind=error_kind)
            error_payload["debug_artifacts"] = debug_artifacts
            print(f"CUDA debug artifacts: {debug_artifacts}")
        save_json(metadata_dir / "train_error.json", error_payload)

        session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
        session_meta["status"] = "failed"
        session_meta["last_updated_utc"] = utc_now()
        session_meta["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        save_json(session_meta_path, session_meta)
        raise
    finally:
        if install_resume_hooks:
            if args.debug_resume_memory_phases and args.resume_from_checkpoint:
                log_memory_probe("resume_probe:after_trainer_train")
            for target, attr_name, original in reversed(probe_restore_ops):
                try:
                    setattr(target, attr_name, original)
                except Exception as restore_exc:
                    print(
                        "resume_probe: failed to restore hook "
                        f"{target}.{attr_name}: {restore_exc!r}"
                    )
            if args.debug_resume_memory_phases and args.resume_from_checkpoint:
                log_memory_probe("resume_probe:after_restore_hooks")

    if train_result is None:  # pragma: no cover - defensive only
        raise RuntimeError("Training returned no result and no exception")
    metrics = train_result.metrics
    save_json(metadata_dir / "train_metrics.json", metrics)
    save_json(metadata_dir / "train_log_history.json", {"log_history": trainer.state.log_history})

    if args.skip_final_save:
        print("Skipping final model/adapter save (--skip-final-save)")
    else:
        if args.full_finetuning:
            print(f"Saving full model to: {full_model_dir}")
            trainer.save_model(str(full_model_dir))
            text_tokenizer.save_pretrained(str(full_model_dir))
        else:
            print(f"Saving adapter to: {adapter_dir}")
            trainer.save_model(str(adapter_dir))
            text_tokenizer.save_pretrained(str(adapter_dir))

    skip_merged_export = args.skip_merged_export or args.skip_final_save
    skip_gguf_export = args.skip_gguf_export or args.skip_final_save
    export_report: dict[str, Any] = {"created_at_utc": utc_now()}
    if not skip_merged_export:
        try:
            print(f"Exporting merged 16-bit model to: {merged_dir}")
            if args.full_finetuning:
                model.save_pretrained(str(merged_dir))
                text_tokenizer.save_pretrained(str(merged_dir))
            else:
                model.save_pretrained_merged(str(merged_dir), text_tokenizer, save_method="merged_16bit")
            export_report["merged_16bit"] = {"status": "ok", "path": str(merged_dir)}
        except Exception as exc:  # pragma: no cover - depends on runtime env
            export_report["merged_16bit"] = {"status": "error", "error": repr(exc)}
    else:
        export_report["merged_16bit"] = {"status": "skipped"}

    if not skip_gguf_export:
        try:
            print(f"Exporting GGUF ({args.gguf_quantization}) to: {gguf_dir}")
            model.save_pretrained_gguf(
                str(gguf_dir),
                text_tokenizer,
                quantization_method=args.gguf_quantization,
            )
            export_report["gguf"] = {
                "status": "ok",
                "path": str(gguf_dir),
                "quantization": args.gguf_quantization,
            }
        except Exception as exc:  # pragma: no cover - depends on runtime env
            export_report["gguf"] = {"status": "error", "error": repr(exc)}
    else:
        export_report["gguf"] = {"status": "skipped"}

    save_json(metadata_dir / "export_report.json", export_report)

    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
    session_meta["status"] = "trained"
    session_meta["last_updated_utc"] = utc_now()
    save_json(session_meta_path, session_meta)

    print("Training session completed")
    print(f"Session dir: {session_dir}")


if __name__ == "__main__":
    main()
