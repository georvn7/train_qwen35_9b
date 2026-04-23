#!/usr/bin/env python3
"""Run one DPO training session for Qwen3.5 9B from a manifest."""

from __future__ import annotations

import argparse
import gc
import hashlib
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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "command": command,
            "error": repr(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5 9B with TRL DPOTrainer using a session manifest."
    )
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--max-prompt-length", type=int, default=14848)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--truncation-mode", default="keep_end", choices=["keep_end", "keep_start"])
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--save-only-model", action="store_true", default=False)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--seed", type=int, default=3413)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--resume-torch-load-mmap",
        dest="resume_torch_load_mmap",
        action="store_true",
        default=True,
        help=(
            "When resuming, force torch.load(..., mmap=True) for files under the resume checkpoint path "
            "and fall back automatically if mmap is unsupported."
        ),
    )
    parser.add_argument(
        "--no-resume-torch-load-mmap",
        dest="resume_torch_load_mmap",
        action="store_false",
    )
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--loss-type", default="sigmoid")
    parser.add_argument("--precompute-ref-log-probs", action="store_true", default=True)
    parser.add_argument("--no-precompute-ref-log-probs", dest="precompute_ref_log_probs", action="store_false")
    parser.add_argument("--precompute-ref-batch-size", type=int, default=1)
    parser.add_argument("--use-logits-to-keep", action="store_true", default=True)
    parser.add_argument("--no-use-logits-to-keep", dest="use_logits_to_keep", action="store_false")
    parser.add_argument("--padding-free", action="store_true", default=False)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "float32"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-gpu-memory-gib", type=float, default=110.0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.88)
    parser.add_argument("--cuda-alloc-conf", default="")
    parser.add_argument("--checkpoint-max-shard-size", default="512MB")
    parser.add_argument(
        "--checkpoint-safe-serialization",
        default="true",
        choices=["auto", "true", "false"],
    )
    parser.add_argument("--checkpoint-presave-gc", action="store_true", default=True)
    parser.add_argument("--no-checkpoint-presave-gc", dest="checkpoint_presave_gc", action="store_false")
    parser.add_argument("--checkpoint-presave-empty-cache", action="store_true", default=True)
    parser.add_argument(
        "--no-checkpoint-presave-empty-cache",
        dest="checkpoint_presave_empty_cache",
        action="store_false",
    )
    parser.add_argument(
        "--extra-save-steps",
        default="10",
        help="Comma-separated extra trainer save steps in addition to save_steps. Example: 10,25",
    )
    parser.add_argument(
        "--resume-warm-marker-path",
        default="",
        help="Optional file path to write after the first completed training step of this process.",
    )
    parser.add_argument(
        "--checkpoint-save-marker-path",
        default="",
        help="Optional file path to write immediately before trainer checkpoint save and clear after save completes.",
    )
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        help="Skip final full-model export after training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> Any | None:
    import torch

    mapping = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def resolve_precision_flags(precision: str) -> tuple[bool, bool, str]:
    import torch

    if precision == "bf16":
        return True, False, "user_forced_bf16"
    if precision == "fp16":
        return False, True, "user_forced_fp16"
    if precision == "float32":
        return False, False, "user_forced_float32"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False, "auto_bf16"
    if torch.cuda.is_available():
        return False, True, "auto_fp16"
    return False, False, "cpu_float32"


def parse_device_map(value: str) -> Any:
    value = value.strip()
    if value == "auto":
        return "auto"
    return value


def current_process_rss_mib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def current_mem_available_mib() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def current_process_max_reserved_mib() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_reserved()) / (1024.0**2)
    except Exception:
        return 0.0


def current_process_nvidia_used_mib() -> float:
    try:
        pid = str(os.getpid())
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return 0.0
        peak = 0.0
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            if parts[0] == pid:
                peak = max(peak, float(parts[1]))
        return peak
    except Exception:
        return 0.0


def latest_checkpoint(session_dir: Path) -> str | None:
    checkpoints = sorted(
        session_dir.glob("checkpoints/checkpoint-*"),
        key=lambda path: path.name,
    )
    if not checkpoints:
        return None
    return str(checkpoints[-1])


class PreparedDPOTrainerMixin:
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        column_names = set(getattr(dataset, "column_names", []))
        required = {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
        if required.issubset(column_names):
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    def get_train_dataloader(self):
        column_names = set(getattr(self.train_dataset, "column_names", []))
        cached = {"ref_chosen_logps", "ref_rejected_logps"}
        if self.precompute_ref_log_probs and cached.issubset(column_names):
            if not self._precomputed_train_ref_log_probs:
                print("Using cached train reference log probs from dataset columns")
            self._precomputed_train_ref_log_probs = True
        return super().get_train_dataloader()


def parse_extra_save_steps(value: str) -> list[int]:
    steps: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        step = int(chunk)
        if step <= 0:
            raise ValueError(f"extra save step must be positive, got {step}")
        steps.append(step)
    return sorted(set(steps))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_ref_logprob_cache_signature(
    *,
    args: argparse.Namespace,
    dataset_files: list[Path],
    train_num_rows: int,
    tokenization_stats: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for path in dataset_files:
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path),
            }
        )
    return {
        "model_name": str(Path(args.model_name).expanduser().resolve())
        if Path(args.model_name).expanduser().exists()
        else args.model_name,
        "train_num_rows": train_num_rows,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_length": args.max_length,
        "truncation_mode": args.truncation_mode,
        "files": files,
        "tokenization_stats": tokenization_stats,
    }


def load_ref_logprob_cache(
    data_path: Path,
    meta_path: Path,
    signature: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    if not data_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("signature") != signature:
        print("Ignoring stale train ref-logprob cache: signature mismatch")
        return None

    import numpy as np

    payload = np.load(data_path)
    ref_chosen = payload["ref_chosen_logps"].tolist()
    ref_rejected = payload["ref_rejected_logps"].tolist()
    expected_rows = signature["train_num_rows"]
    if len(ref_chosen) != expected_rows or len(ref_rejected) != expected_rows:
        print(
            "Ignoring stale train ref-logprob cache: "
            f"rows=({len(ref_chosen)}, {len(ref_rejected)}) expected={expected_rows}"
        )
        return None
    return ref_chosen, ref_rejected


def save_ref_logprob_cache(
    data_path: Path,
    meta_path: Path,
    *,
    ref_chosen_logps: list[float],
    ref_rejected_logps: list[float],
    signature: dict[str, Any],
) -> None:
    import numpy as np

    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_data_path = data_path.with_name(data_path.name + ".tmp")
    tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")

    with tmp_data_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            ref_chosen_logps=np.asarray(ref_chosen_logps, dtype=np.float32),
            ref_rejected_logps=np.asarray(ref_rejected_logps, dtype=np.float32),
        )
    save_json(
        tmp_meta_path,
        {
            "created_at_utc": utc_now(),
            "signature": signature,
            "rows": len(ref_chosen_logps),
            "data_path": str(data_path),
        },
    )
    tmp_data_path.replace(data_path)
    tmp_meta_path.replace(meta_path)


def build_tokenized_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int,
    max_completion_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eos_token_id = tokenizer.eos_token_id
    prompt_original_lengths: list[int] = []
    chosen_original_lengths: list[int] = []
    rejected_original_lengths: list[int] = []
    prompt_final_lengths: list[int] = []
    chosen_final_lengths: list[int] = []
    rejected_final_lengths: list[int] = []
    prompt_truncated = 0
    chosen_truncated = 0
    rejected_truncated = 0
    tokenized_rows: list[dict[str, Any]] = []

    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(row["rejected"], add_special_tokens=False)["input_ids"]

        if eos_token_id is not None:
            chosen_ids = chosen_ids + [eos_token_id]
            rejected_ids = rejected_ids + [eos_token_id]

        prompt_original_lengths.append(len(prompt_ids))
        chosen_original_lengths.append(len(chosen_ids))
        rejected_original_lengths.append(len(rejected_ids))

        final_prompt_ids = prompt_ids[-max_prompt_length:]
        final_chosen_ids = chosen_ids[:max_completion_length]
        final_rejected_ids = rejected_ids[:max_completion_length]

        if len(final_prompt_ids) != len(prompt_ids):
            prompt_truncated += 1
        if len(final_chosen_ids) != len(chosen_ids):
            chosen_truncated += 1
        if len(final_rejected_ids) != len(rejected_ids):
            rejected_truncated += 1

        prompt_final_lengths.append(len(final_prompt_ids))
        chosen_final_lengths.append(len(final_chosen_ids))
        rejected_final_lengths.append(len(final_rejected_ids))

        tokenized_row = {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "prompt_input_ids": final_prompt_ids,
            "chosen_input_ids": final_chosen_ids,
            "rejected_input_ids": final_rejected_ids,
        }
        if "row_index" in row:
            tokenized_row["row_index"] = row["row_index"]
        if "meta" in row:
            tokenized_row["meta"] = row["meta"]
        tokenized_rows.append(tokenized_row)

    return tokenized_rows, {
        "rows": len(tokenized_rows),
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "prompt_original_max": max(prompt_original_lengths) if prompt_original_lengths else 0,
        "prompt_final_max": max(prompt_final_lengths) if prompt_final_lengths else 0,
        "chosen_original_max": max(chosen_original_lengths) if chosen_original_lengths else 0,
        "chosen_final_max": max(chosen_final_lengths) if chosen_final_lengths else 0,
        "rejected_original_max": max(rejected_original_lengths) if rejected_original_lengths else 0,
        "rejected_final_max": max(rejected_final_lengths) if rejected_final_lengths else 0,
        "prompt_truncated_rows": prompt_truncated,
        "chosen_truncated_rows": chosen_truncated,
        "rejected_truncated_rows": rejected_truncated,
    }


def main() -> None:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    metadata_dir = session_dir / "metadata"
    artifacts_dir = session_dir / "artifacts"
    checkpoints_dir = session_dir / "checkpoints"
    full_model_dir = artifacts_dir / "full_model"
    manifest_path = metadata_dir / "dataset_manifest.json"
    session_meta_path = metadata_dir / "session.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    if not session_meta_path.exists():
        raise FileNotFoundError(f"missing session metadata: {session_meta_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace_root = Path(manifest["workspace_root"])
    dataset_files: list[Path] = []
    for item in manifest["files"]:
        if "relative_path" in item:
            dataset_files.append((workspace_root / item["relative_path"]).resolve())
        elif "absolute_path" in item:
            dataset_files.append(Path(item["absolute_path"]).expanduser().resolve())
        else:
            raise ValueError(f"manifest entry missing dataset path: {item}")
    missing = [str(path) for path in dataset_files if not path.exists()]
    if missing:
        raise FileNotFoundError("dataset files missing:\n" + "\n".join(missing))

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

    if args.cuda_alloc_conf.strip():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf.strip()
        print(f"Using PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi": run_command(["nvidia-smi"]),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }
    save_json(metadata_dir / "environment.json", environment)

    from datasets import Dataset, load_dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    if args.cuda_memory_fraction > 0:
        if not 0.0 < args.cuda_memory_fraction < 1.0:
            raise ValueError("--cuda-memory-fraction must be in (0, 1)")
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda-memory-fraction requested but CUDA is not available")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
        print(
            "CUDA per-process allocator cap enabled: "
            f"{args.cuda_memory_fraction:.4f} of device memory"
        )

    raw_dataset = load_dataset(
        "json",
        data_files=[str(path) for path in dataset_files],
        split="train",
    )
    if args.max_samples > 0:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))

    required_columns = {"prompt", "chosen", "rejected"}
    if not required_columns.issubset(set(raw_dataset.column_names)):
        raise ValueError(
            "Prepared DPO dataset must include prompt/chosen/rejected columns. "
            f"Got columns: {raw_dataset.column_names}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    raw_rows = [raw_dataset[i] for i in range(len(raw_dataset))]
    tokenized_rows, tokenization_stats = build_tokenized_rows(
        rows=raw_rows,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )
    ref_logprob_cache_data_path = metadata_dir / "train_ref_logprobs_cache.npz"
    ref_logprob_cache_meta_path = metadata_dir / "train_ref_logprobs_cache.meta.json"
    ref_logprob_cache_signature = build_ref_logprob_cache_signature(
        args=args,
        dataset_files=dataset_files,
        train_num_rows=len(tokenized_rows),
        tokenization_stats=tokenization_stats,
    )
    cached_ref_logprobs = load_ref_logprob_cache(
        ref_logprob_cache_data_path,
        ref_logprob_cache_meta_path,
        ref_logprob_cache_signature,
    )
    if cached_ref_logprobs is not None:
        ref_chosen_logps, ref_rejected_logps = cached_ref_logprobs
        for index, row in enumerate(tokenized_rows):
            row["ref_chosen_logps"] = ref_chosen_logps[index]
            row["ref_rejected_logps"] = ref_rejected_logps[index]
        print(
            "Loaded durable train ref-logprob cache: "
            f"rows={len(ref_chosen_logps)} path={ref_logprob_cache_data_path}"
        )
    save_json(metadata_dir / "dpo_tokenization_stats.json", tokenization_stats)
    train_dataset = Dataset.from_list(tokenized_rows)

    if args.dry_run:
        run_config = {
            "created_at_utc": utc_now(),
            "session_dir": str(session_dir),
            "model_name": args.model_name,
            "dataset_manifest": str(manifest_path),
            "dataset_num_rows": len(raw_dataset),
            "dry_run": True,
            "attn_implementation": args.attn_implementation,
            "device_map": args.device_map,
            "hf_cache_dir": str(cache_root),
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "max_length": args.max_length,
            "truncation_mode": args.truncation_mode,
            "beta": args.beta,
            "loss_type": args.loss_type,
            "precompute_ref_log_probs": args.precompute_ref_log_probs,
            "use_logits_to_keep": args.use_logits_to_keep,
            "padding_free": args.padding_free,
            "precision": args.precision,
            "torch_dtype": args.torch_dtype,
            "optim": args.optim,
            "tokenization_stats": tokenization_stats,
        }
        save_json(metadata_dir / "run_config.json", run_config)
        session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
        session_meta["status"] = "validated"
        session_meta["last_updated_utc"] = utc_now()
        save_json(session_meta_path, session_meta)
        print(f"Dry run complete: rows={len(raw_dataset)}")
        return

    requested_dtype = resolve_torch_dtype(args.torch_dtype)
    bf16_flag, fp16_flag, precision_reason = resolve_precision_flags(args.precision)
    print(f"Resolved precision: bf16={bf16_flag}, fp16={fp16_flag} ({precision_reason})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=requested_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=parse_device_map(args.device_map),
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.config._attn_implementation = args.attn_implementation
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    resolved_max_gpu_memory_mib = args.max_gpu_memory_gib * 1024.0
    callbacks: list[Any] = []
    if resolved_max_gpu_memory_mib > 0:
        print(f"GPU memory guard enabled: {args.max_gpu_memory_gib:.2f} GiB")

        class MaxGpuMemoryGuardCallback(TrainerCallback):
            def __init__(self, max_mib: float) -> None:
                self.max_mib = max_mib

            def _check(self) -> None:
                measured_mib = max(current_process_max_reserved_mib(), current_process_nvidia_used_mib())
                if measured_mib > self.max_mib:
                    raise RuntimeError(
                        "GPU memory guard triggered: "
                        f"measured={measured_mib:.1f} MiB limit={self.max_mib:.1f} MiB"
                    )

            def on_train_begin(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._check()
                return control

            def on_substep_end(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._check()
                return control

            def on_step_end(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._check()
                return control

        callbacks.append(MaxGpuMemoryGuardCallback(resolved_max_gpu_memory_mib))

    extra_save_steps = parse_extra_save_steps(args.extra_save_steps)
    checkpoint_save_marker_path = (
        Path(args.checkpoint_save_marker_path)
        if args.checkpoint_save_marker_path.strip()
        else None
    )

    if args.checkpoint_presave_gc or args.checkpoint_presave_empty_cache or checkpoint_save_marker_path is not None:
        class CheckpointPreSaveCallback(TrainerCallback):
            def __init__(self, marker_path: Path | None) -> None:
                self.marker_path = marker_path

            def _log_memory(self, prefix: str) -> None:
                print(
                    f"{prefix}: "
                    f"rss_mib={current_process_rss_mib():.1f}, "
                    f"mem_avail_mib={current_mem_available_mib():.1f}, "
                    f"torch_peak_reserved_mib={current_process_max_reserved_mib():.1f}, "
                    f"nvidia_used_mib={current_process_nvidia_used_mib():.1f}"
                )

            def _checkpoint_dir(self, train_args: Any, state: Any) -> Path:
                return Path(train_args.output_dir) / f"checkpoint-{int(state.global_step)}"

            def on_step_end(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                should_prepare_save = control.should_save or int(state.global_step) in extra_save_steps
                if not should_prepare_save:
                    return control
                if self.marker_path is not None:
                    payload = {
                        "created_at_utc": utc_now(),
                        "global_step": int(state.global_step),
                        "pid": os.getpid(),
                    }
                    self.marker_path.parent.mkdir(parents=True, exist_ok=True)
                    self.marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    print(
                        "Checkpoint save marker written: "
                        f"step={int(state.global_step)} path={self.marker_path}"
                    )
                self._log_memory("Checkpoint pre-save memory")
                if args.checkpoint_presave_gc:
                    gc.collect()
                    print("Checkpoint pre-save action: gc.collect()")
                if args.checkpoint_presave_empty_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("Checkpoint pre-save action: torch.cuda.empty_cache()")
                self._log_memory("Checkpoint pre-save memory (after cleanup)")
                return control

            def on_save(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                self._log_memory("Checkpoint post-save memory")
                checkpoint_dir = self._checkpoint_dir(train_args, state)
                if checkpoint_dir.exists():
                    marker_payload = {
                        "created_at_utc": utc_now(),
                        "global_step": int(state.global_step),
                        "pid": os.getpid(),
                    }
                    (checkpoint_dir / "checkpoint_complete.json").write_text(
                        json.dumps(marker_payload, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        "Checkpoint complete marker written: "
                        f"step={int(state.global_step)} path={checkpoint_dir / 'checkpoint_complete.json'}"
                    )
                if self.marker_path is not None and self.marker_path.exists():
                    self.marker_path.unlink()
                    print(f"Checkpoint save marker cleared: path={self.marker_path}")
                return control

        callbacks.append(CheckpointPreSaveCallback(checkpoint_save_marker_path))

    if extra_save_steps:
        class ExtraSaveStepsCallback(TrainerCallback):
            def __init__(self, steps: list[int]) -> None:
                self.steps = set(steps)

            def on_step_end(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                if int(state.global_step) in self.steps:
                    print(f"Extra checkpoint trigger at step={int(state.global_step)}")
                    control.should_save = True
                return control

        callbacks.append(ExtraSaveStepsCallback(extra_save_steps))

    if args.resume_warm_marker_path.strip():
        resume_warm_marker_path = Path(args.resume_warm_marker_path)

        class ResumeWarmMarkerCallback(TrainerCallback):
            def __init__(self, marker_path: Path) -> None:
                self.marker_path = marker_path
                self.written = False

            def on_step_end(self, train_args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                if self.written:
                    return control
                payload = {
                    "created_at_utc": utc_now(),
                    "global_step": int(state.global_step),
                    "pid": os.getpid(),
                }
                self.marker_path.parent.mkdir(parents=True, exist_ok=True)
                self.marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                self.written = True
                print(
                    "Resume warm marker written: "
                    f"step={int(state.global_step)} path={self.marker_path}"
                )
                return control

        callbacks.append(ResumeWarmMarkerCallback(resume_warm_marker_path))

    checkpoint_save_overrides: dict[str, Any] = {}
    if args.checkpoint_max_shard_size.strip():
        checkpoint_save_overrides["max_shard_size"] = args.checkpoint_max_shard_size.strip()
    if args.checkpoint_safe_serialization != "auto":
        checkpoint_save_overrides["safe_serialization"] = args.checkpoint_safe_serialization == "true"
    if checkpoint_save_overrides:
        original_save_pretrained = model.save_pretrained

        def save_pretrained_with_overrides(save_directory: str, *sp_args: Any, **sp_kwargs: Any) -> Any:
            for key, value in checkpoint_save_overrides.items():
                sp_kwargs.setdefault(key, value)
            return original_save_pretrained(save_directory, *sp_args, **sp_kwargs)

        model.save_pretrained = save_pretrained_with_overrides  # type: ignore[assignment]

    dpo_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": args.save_only_model,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "report_to": args.report_to,
        "optim": args.optim,
        "bf16": bf16_flag,
        "fp16": fp16_flag,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_length": args.max_length,
        "truncation_mode": args.truncation_mode,
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "loss_type": args.loss_type,
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
        "precompute_ref_batch_size": args.precompute_ref_batch_size,
        "use_logits_to_keep": args.use_logits_to_keep,
        "padding_free": args.padding_free,
        "dataset_num_proc": args.dataset_num_proc,
        "remove_unused_columns": False,
    }
    if args.max_steps > 0:
        dpo_kwargs["max_steps"] = args.max_steps
    else:
        dpo_kwargs["num_train_epochs"] = args.num_train_epochs

    training_args = DPOConfig(**dpo_kwargs)
    PreparedDPOTrainer = type("PreparedDPOTrainer", (PreparedDPOTrainerMixin, DPOTrainer), {})
    trainer = PreparedDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    run_config = {
        "created_at_utc": utc_now(),
        "session_dir": str(session_dir),
        "model_name": args.model_name,
        "dataset_manifest": str(manifest_path),
        "dataset_num_rows": len(raw_dataset),
        "train_num_rows": len(train_dataset),
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "hf_cache_dir": str(cache_root),
        "precision": args.precision,
        "torch_dtype": args.torch_dtype,
        "resolved_precision": {
            "bf16": bf16_flag,
            "fp16": fp16_flag,
            "reason": precision_reason,
        },
        "dpo_args": dpo_kwargs,
        "tokenization_stats": tokenization_stats,
        "max_gpu_memory_gib_guard": args.max_gpu_memory_gib,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "cuda_alloc_conf": args.cuda_alloc_conf,
        "checkpoint_max_shard_size": args.checkpoint_max_shard_size,
        "checkpoint_safe_serialization": args.checkpoint_safe_serialization,
        "checkpoint_presave_gc": args.checkpoint_presave_gc,
        "checkpoint_presave_empty_cache": args.checkpoint_presave_empty_cache,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "resume_torch_load_mmap": args.resume_torch_load_mmap,
        "latest_checkpoint_at_launch": latest_checkpoint(session_dir),
        "skip_final_save": args.skip_final_save,
        "extra_save_steps": extra_save_steps,
        "resume_warm_marker_path": args.resume_warm_marker_path,
        "checkpoint_save_marker_path": args.checkpoint_save_marker_path,
        "train_ref_logprob_cache": {
            "data_path": str(ref_logprob_cache_data_path),
            "meta_path": str(ref_logprob_cache_meta_path),
            "loaded_at_launch": cached_ref_logprobs is not None,
        },
    }
    save_json(metadata_dir / "run_config.json", run_config)

    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
    session_meta["status"] = "running"
    session_meta["last_updated_utc"] = utc_now()
    session_meta["train_ref_logprob_cache"] = {
        "data_path": str(ref_logprob_cache_data_path),
        "meta_path": str(ref_logprob_cache_meta_path),
        "loaded_at_launch": cached_ref_logprobs is not None,
    }
    save_json(session_meta_path, session_meta)

    if args.precompute_ref_log_probs and cached_ref_logprobs is None:
        print("Precomputing durable train reference log probs before trainer.train()")
        trainer.get_train_dataloader()
        ref_chosen_logps = list(trainer.train_dataset["ref_chosen_logps"])
        ref_rejected_logps = list(trainer.train_dataset["ref_rejected_logps"])
        save_ref_logprob_cache(
            ref_logprob_cache_data_path,
            ref_logprob_cache_meta_path,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            signature=ref_logprob_cache_signature,
        )
        session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
        session_meta["status"] = "ref_logprobs_precomputed"
        session_meta["last_updated_utc"] = utc_now()
        session_meta["train_ref_logprob_cache"] = {
            "data_path": str(ref_logprob_cache_data_path),
            "meta_path": str(ref_logprob_cache_meta_path),
            "loaded_at_launch": False,
            "created_at_utc": utc_now(),
            "rows": len(ref_chosen_logps),
        }
        save_json(session_meta_path, session_meta)
        print(
            "Durable train ref-logprob cache saved: "
            f"rows={len(ref_chosen_logps)} path={ref_logprob_cache_data_path}"
        )

    print("Starting DPO training")
    train_result = None
    resume_checkpoint_dir = (
        Path(args.resume_from_checkpoint).resolve()
        if args.resume_from_checkpoint
        else None
    )
    restore_torch_load: tuple[Any, str, Any] | None = None
    if args.resume_from_checkpoint and args.resume_torch_load_mmap:
        import torch as _torch

        original_torch_load = getattr(_torch, "load", None)
        if original_torch_load is not None:
            def torch_load_with_resume_mmap(*load_args: Any, **load_kwargs: Any) -> Any:
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

                use_mmap = bool(
                    resume_checkpoint_dir is not None
                    and target_path is not None
                    and str(target_path).startswith(str(resume_checkpoint_dir))
                )
                effective_kwargs = dict(load_kwargs)
                if use_mmap and "mmap" not in effective_kwargs:
                    effective_kwargs["mmap"] = True
                try:
                    return original_torch_load(*load_args, **effective_kwargs)
                except Exception as exc:
                    if use_mmap and effective_kwargs.get("mmap") is True:
                        fallback_kwargs = dict(effective_kwargs)
                        fallback_kwargs.pop("mmap", None)
                        print(
                            "resume_mmap: torch.load with mmap=True failed for "
                            f"{target}: {exc!r}; retrying with mmap disabled"
                        )
                        return original_torch_load(*load_args, **fallback_kwargs)
                    raise

            _torch.load = torch_load_with_resume_mmap  # type: ignore[assignment]
            restore_torch_load = (_torch, "load", original_torch_load)
            print("resume_mmap: installed torch.load mmap hook")
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception as exc:
        error_payload = {
            "created_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_repr": repr(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(metadata_dir / "train_error.json", error_payload)
        session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
        session_meta["status"] = "failed"
        session_meta["last_updated_utc"] = utc_now()
        session_meta["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save_json(session_meta_path, session_meta)
        raise
    finally:
        if restore_torch_load is not None:
            setattr(restore_torch_load[0], restore_torch_load[1], restore_torch_load[2])

    if train_result is None:  # pragma: no cover - defensive only
        raise RuntimeError("Training returned no result and no exception")

    save_json(metadata_dir / "train_metrics.json", train_result.metrics)
    save_json(metadata_dir / "train_log_history.json", {"log_history": trainer.state.log_history})

    if args.skip_final_save:
        print("Skipping final model save (--skip-final-save)")
    else:
        print(f"Saving full model to: {full_model_dir}")
        trainer.save_model(str(full_model_dir))
        tokenizer.save_pretrained(str(full_model_dir))

    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
    session_meta["status"] = "trained"
    session_meta["last_updated_utc"] = utc_now()
    save_json(session_meta_path, session_meta)

    print("DPO session completed")
    print(f"Session dir: {session_dir}")


if __name__ == "__main__":
    main()
