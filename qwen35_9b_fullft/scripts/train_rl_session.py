#!/usr/bin/env python3
"""Train one memory-bounded checkpointed debugger RL session.

The rollout reward is terminal, but every policy response in the rollout is
trained. Prompt tokens are masked; Qwen reasoning and final answer tokens are
both completion tokens. Old-policy log-probabilities are computed from the
exact input checkpoint before the first optimizer update.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"command": command, "error": repr(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serialized 32K checkpointed RL training")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", default="10,20,40,60")
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--optim", choices=["adamw_8bit", "adamw_torch"], default="adamw_8bit")
    parser.add_argument("--seed", type=int, default=3413)
    parser.add_argument("--clip-epsilon", type=float, default=0.20)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "float32"], default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--max-gpu-memory-gib", type=float, default=110.0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.88)
    parser.add_argument("--cuda-alloc-conf", default="")
    parser.add_argument("--checkpoint-max-shard-size", default="512MB")
    parser.add_argument("--checkpoint-safe-serialization", choices=["true", "false"], default="true")
    parser.add_argument("--frozen-eval-max-sequences", type=int, default=16)
    parser.add_argument(
        "--smoke-optimizer-steps",
        type=int,
        default=0,
        help=(
            "Run only this many real optimizer steps, record metrics, and skip "
            "model publication. Zero performs normal production training."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_positive_steps(value: str) -> list[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step <= 0:
            raise ValueError("save steps must be positive")
        result.add(step)
    return sorted(result)


def resolve_optimizer_step_limit(total_steps: int, smoke_steps: int) -> tuple[int, bool]:
    if total_steps <= 0:
        raise ValueError("RL training requires at least one optimizer step")
    if smoke_steps < 0:
        raise ValueError("smoke optimizer steps cannot be negative")
    if smoke_steps == 0:
        return total_steps, False
    return min(total_steps, smoke_steps), True


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Hen's thinking field to Qwen's reasoning_content without mutation."""
    normalized: list[dict[str, Any]] = []
    for raw in messages:
        message = dict(raw)
        if str(message.get("role", "")).lower() == "assistant":
            existing = message.get("reasoning_content")
            thinking = message.get("thinking")
            if not (isinstance(existing, str) and existing.strip()):
                if isinstance(thinking, str) and thinking.strip():
                    message["reasoning_content"] = thinking.strip()
        normalized.append(message)
    return normalized


def apply_chat_template_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(output, dict) or (hasattr(output, "keys") and "input_ids" in output):
        output = output["input_ids"]
    if output and isinstance(output[0], list):
        output = output[0]
    return [int(token) for token in output]


def common_prefix_length(left: list[int], right: list[int]) -> int:
    result = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        result += 1
    return result


def tokenize_policy_step(
    policy_step: dict[str, Any], tokenizer: Any, max_length: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = policy_step.get("prompt")
    completion = policy_step.get("completion")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("RL policy prompt must be a non-empty message array")
    if not isinstance(completion, list) or not completion:
        raise ValueError("RL policy completion must be a non-empty message array")
    if any(not isinstance(message, dict) for message in prompt + completion):
        raise ValueError("RL policy messages must be objects")
    if completion[-1].get("role") != "assistant":
        raise ValueError("RL policy completion must end with an assistant message")

    normalized_prompt = normalize_messages(prompt)
    normalized_completion = normalize_messages(completion)
    prefix_ids = apply_chat_template_ids(
        tokenizer, normalized_prompt, add_generation_prompt=True
    )
    full_ids = apply_chat_template_ids(
        tokenizer,
        normalized_prompt + normalized_completion,
        add_generation_prompt=False,
    )
    target_start = len(prefix_ids)
    boundary_adjustment = 0
    if full_ids[: len(prefix_ids)] != prefix_ids:
        common = common_prefix_length(full_ids, prefix_ids)
        boundary_adjustment = len(prefix_ids) - common
        if boundary_adjustment < 0 or boundary_adjustment > 8:
            raise ValueError(
                "RL final-assistant prefix mismatch: "
                f"prefix={len(prefix_ids)} common={common} full={len(full_ids)}"
            )
        target_start = common
    if target_start <= 0 or target_start >= len(full_ids):
        raise ValueError("RL row produced no completion tokens")

    original_length = len(full_ids)
    completion_length = original_length - target_start
    if completion_length >= max_length:
        raise ValueError(
            f"RL completion has {completion_length} tokens and cannot fit max_length={max_length}"
        )
    truncated_prompt_tokens = 0
    if original_length > max_length:
        truncated_prompt_tokens = original_length - max_length
        if truncated_prompt_tokens >= target_start:
            raise ValueError("RL truncation would remove the entire prompt")
        full_ids = full_ids[truncated_prompt_tokens:]
        target_start -= truncated_prompt_tokens

    tokenized = {
        "input_ids": full_ids,
        "target_start": target_start,
    }
    stats = {
        "original_tokens": original_length,
        "final_tokens": len(full_ids),
        "completion_tokens": len(full_ids) - target_start,
        "prompt_tokens_removed": truncated_prompt_tokens,
        "boundary_adjustment_tokens": boundary_adjustment,
        "thinking_present": bool(
            isinstance(completion[-1].get("thinking"), str)
            and completion[-1]["thinking"].strip()
        ),
    }
    return tokenized, stats


def flatten_rollouts(
    rows: Iterable[dict[str, Any]], tokenizer: Any, max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    rollout_count = 0
    groups: set[str] = set()
    for row_index, row in enumerate(rows):
        group_id = str(row["group_id"])
        rollout_id = str(row["rollout_id"])
        groups.add(group_id)
        rollout_count += 1
        steps = row["policy_steps"]
        expected_weight = 1.0 / len(steps)
        supplied_weight = float(row["policy_step_weight"])
        if abs(supplied_weight - expected_weight) > 1e-6:
            raise ValueError(
                f"RL rollout {group_id}/{rollout_id} has inconsistent policy_step_weight"
            )
        for step_index, policy_step in enumerate(steps):
            tokenized, token_stats = tokenize_policy_step(
                policy_step, tokenizer, max_length
            )
            record_id = f"{group_id}/{rollout_id}/{step_index}"
            records.append(
                {
                    **tokenized,
                    "record_id": record_id,
                    "group_id": group_id,
                    "rollout_id": rollout_id,
                    "policy_step_index": step_index,
                    "reward": float(row["reward"]),
                    "advantage": float(row["advantage"]),
                    "weight": supplied_weight,
                }
            )
            stats.append({"record_id": record_id, **token_stats})
    if not records:
        raise ValueError("RL dataset produced no policy sequences")
    return records, {
        "groups": len(groups),
        "rollouts": rollout_count,
        "policy_sequences": len(records),
        "truncated_sequences": sum(item["prompt_tokens_removed"] > 0 for item in stats),
        "thinking_sequences": sum(item["thinking_present"] for item in stats),
        "max_original_tokens": max(item["original_tokens"] for item in stats),
        "max_final_tokens": max(item["final_tokens"] for item in stats),
        "max_completion_tokens": max(item["completion_tokens"] for item in stats),
        "max_prompt_tokens_removed": max(item["prompt_tokens_removed"] for item in stats),
        "max_boundary_adjustment_tokens": max(
            item["boundary_adjustment_tokens"] for item in stats
        ),
        "records": stats,
    }


def select_frozen_subset(records: list[dict[str, Any]], maximum: int) -> list[int]:
    """Select a stable, group-diverse subset before training starts."""
    if maximum <= 0:
        return []
    buckets: dict[str, list[tuple[str, int]]] = {}
    for index, record in enumerate(records):
        digest = hashlib.sha256(record["record_id"].encode("utf-8")).hexdigest()
        buckets.setdefault(record["group_id"], []).append((digest, index))
    for bucket in buckets.values():
        bucket.sort()
    selected: list[int] = []
    depth = 0
    while len(selected) < maximum:
        added = False
        for group_id in sorted(buckets):
            bucket = buckets[group_id]
            if depth < len(bucket):
                selected.append(bucket[depth][1])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return selected


def clipped_policy_objective(
    current_logps: Any,
    old_logps: Any,
    advantage: Any,
    clip_epsilon: float,
    kl_beta: float,
) -> tuple[Any, dict[str, Any]]:
    """Token-level clipped objective sampled from the frozen old policy."""
    import torch

    if current_logps.shape != old_logps.shape or current_logps.numel() == 0:
        raise ValueError("current and old completion log-probabilities must align")
    log_ratio = (current_logps.float() - old_logps.float()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    scalar_advantage = torch.as_tensor(
        advantage, dtype=ratio.dtype, device=ratio.device
    )
    surrogate = torch.minimum(ratio * scalar_advantage, clipped_ratio * scalar_advantage)
    policy_loss = -surrogate.mean()
    approximate_kl = (ratio - 1.0 - log_ratio).mean()
    total = policy_loss + float(kl_beta) * approximate_kl
    metrics = {
        "policy_loss": policy_loss.detach(),
        "approx_kl": approximate_kl.detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_epsilon).float().mean().detach(),
        "ratio_mean": ratio.mean().detach(),
    }
    return total, metrics


def completion_logps(model: Any, input_ids: Any, target_start: int) -> Any:
    """Return one log-probability per completion token without full-sequence logits."""
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("serialized RL requires one sequence per forward")
    completion_tokens = input_ids.shape[1] - target_start
    if target_start <= 0 or completion_tokens <= 0:
        raise ValueError("invalid RL completion boundary")
    logits_to_keep = completion_tokens + 1
    outputs = model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        logits_to_keep=logits_to_keep,
    )
    logits = outputs.logits
    if logits.shape[1] < logits_to_keep:
        raise ValueError(
            f"model returned {logits.shape[1]} logits, expected at least {logits_to_keep}"
        )
    prediction_logits = logits[:, -logits_to_keep:-1, :].float()
    labels = input_ids[:, target_start:]
    if prediction_logits.shape[1] != labels.shape[1]:
        raise ValueError("RL completion logits do not align with completion labels")
    selected = prediction_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (selected - torch.logsumexp(prediction_logits, dim=-1)).squeeze(0)


def aggregate_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(set().union(*(value.keys() for value in values)))
    return {
        key: sum(float(value[key]) for value in values if key in value)
        / sum(1 for value in values if key in value)
        for key in keys
    }


def aggregate_weighted_metrics(
    values: list[dict[str, float]], weights: list[float]
) -> dict[str, float]:
    """Aggregate metrics using the rollout-normalized policy-step weights."""
    if len(values) != len(weights):
        raise ValueError("metric values and weights must align")
    if not values:
        return {}
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("metric weights must be finite and positive")
    keys = sorted(set().union(*(value.keys() for value in values)))
    result: dict[str, float] = {}
    for key in keys:
        denominator = sum(
            weight for value, weight in zip(values, weights) if key in value
        )
        result[key] = sum(
            float(value[key]) * weight
            for value, weight in zip(values, weights)
            if key in value
        ) / denominator
    result["weight_sum"] = sum(weights)
    return result


def resolve_dtype(name: str) -> Any | None:
    import torch

    return {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_autocast_dtype(precision: str) -> Any | None:
    import torch

    if precision == "float32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else None


def cuda_snapshot() -> dict[str, float]:
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


def release_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_device(model: Any) -> Any:
    return next(model.parameters()).device


def forward_context(device: Any, autocast_dtype: Any | None) -> Any:
    """Return a fresh context manager for one model forward."""
    import torch

    if autocast_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)
    return contextlib.nullcontext()


def compute_old_policy_logps(
    model: Any,
    records: list[dict[str, Any]],
    autocast_dtype: Any | None,
) -> list[Any]:
    import torch

    result: list[Any] = []
    device = model_device(model)
    model.eval()
    for index, record in enumerate(records, start=1):
        input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad(), forward_context(device, autocast_dtype):
            logps = completion_logps(model, input_ids, int(record["target_start"]))
        result.append(logps.detach().cpu().float())
        del input_ids, logps
        release_memory()
        print(f"Old-policy log-probs: {index}/{len(records)}", flush=True)
    return result


def evaluate_records(
    model: Any,
    records: list[dict[str, Any]],
    old_logps: list[Any],
    indices: list[int],
    clip_epsilon: float,
    kl_beta: float,
    autocast_dtype: Any | None,
) -> dict[str, Any]:
    import torch

    model.eval()
    device = model_device(model)
    metrics: list[dict[str, float]] = []
    weights: list[float] = []
    for index in indices:
        record = records[index]
        input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad(), forward_context(device, autocast_dtype):
            current = completion_logps(model, input_ids, int(record["target_start"]))
            objective, values = clipped_policy_objective(
                current,
                old_logps[index].to(device),
                float(record["advantage"]),
                clip_epsilon,
                kl_beta,
            )
        metrics.append(
            {
                "loss": float(objective.detach().cpu()),
                **{key: float(value.cpu()) for key, value in values.items()},
            }
        )
        weights.append(float(record["weight"]))
        del input_ids, current, objective
        release_memory()
    return {
        "sequences": len(indices),
        **aggregate_metrics(metrics),
        "policy_step_weighted": aggregate_weighted_metrics(metrics, weights),
    }


def retain_latest_checkpoints(checkpoints_dir: Path, limit: int) -> None:
    checkpoints = sorted(
        (path for path in checkpoints_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    for path in checkpoints[: max(0, len(checkpoints) - limit)]:
        shutil.rmtree(path)


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    destination: Path,
    *,
    max_shard_size: str,
    safe_serialization: bool,
    metadata: dict[str, Any],
) -> None:
    release_memory()
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(destination),
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )
    tokenizer.save_pretrained(str(destination))
    save_json(destination / "checkpoint_complete.json", metadata)


def main() -> None:
    args = parse_args()
    if args.num_train_epochs != 1:
        raise ValueError("checkpointed RL currently requires exactly one epoch")
    if not 0.0 < args.clip_epsilon < 1.0 or args.kl_beta < 0.0:
        raise ValueError("invalid RL clip/KL configuration")
    if not 0 < args.max_length <= 32768:
        raise ValueError("max_length must be in [1, 32768]")
    save_steps = parse_positive_steps(args.save_steps)

    session_dir = Path(args.session_dir).expanduser().resolve()
    metadata_dir = session_dir / "metadata"
    checkpoints_dir = session_dir / "checkpoints"
    full_model_dir = session_dir / "artifacts" / "full_model"
    manifest_path = metadata_dir / "dataset_manifest.json"
    session_meta_path = metadata_dir / "session.json"
    if not manifest_path.is_file() or not session_meta_path.is_file():
        raise FileNotFoundError("session metadata or dataset manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace_root = Path(manifest["workspace_root"])
    dataset_files = [
        (workspace_root / item["relative_path"]).resolve()
        if "relative_path" in item
        else Path(item["absolute_path"]).expanduser().resolve()
        for item in manifest["files"]
    ]
    if len(dataset_files) != 1 or not dataset_files[0].is_file():
        raise ValueError("serialized RL requires exactly one immutable JSONL input")

    cache_root = (
        Path(args.hf_cache_dir).expanduser().resolve()
        if args.hf_cache_dir
        else workspace_root / "qwen35_9b_fullft" / ".cache" / "huggingface"
    )
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi": run_command(["nvidia-smi"]),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }
    save_json(metadata_dir / "environment.json", environment)

    rows = [
        json.loads(line)
        for line in dataset_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    records, tokenization_stats = flatten_rollouts(rows, tokenizer, args.max_length)
    full_training_steps = len(records) * args.num_train_epochs
    optimizer_step_limit, smoke_mode = resolve_optimizer_step_limit(
        full_training_steps, args.smoke_optimizer_steps
    )
    save_json(metadata_dir / "rl_tokenization_stats.json", tokenization_stats)
    frozen_indices = select_frozen_subset(records, args.frozen_eval_max_sequences)
    frozen_contract = {
        "indices": frozen_indices,
        "record_ids": [records[index]["record_id"] for index in frozen_indices],
    }
    frozen_contract["sha256"] = canonical_sha256(frozen_contract)
    save_json(metadata_dir / "frozen_eval_subset.json", frozen_contract)

    run_config = {
        "created_at_utc": utc_now(),
        "model_name": args.model_name,
        "dataset_sha256": file_sha256(dataset_files[0]),
        "dataset_rows": len(rows),
        "policy_sequences": len(records),
        "max_length": args.max_length,
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "clip_epsilon": args.clip_epsilon,
        "kl_beta": args.kl_beta,
        "completion_loss": "thinking_and_final_content",
        "prompt_loss": "masked",
        "old_policy": "exact_base_checkpoint_precomputed_before_updates",
        "save_steps": save_steps,
        "frozen_eval_subset_sha256": frozen_contract["sha256"],
        "full_training_steps": full_training_steps,
        "optimizer_step_limit": optimizer_step_limit,
        "smoke_mode": smoke_mode,
        "dry_run": args.dry_run,
    }
    save_json(metadata_dir / "run_config.json", run_config)
    if args.dry_run:
        save_json(
            metadata_dir / "train_metrics.json",
            {"dry_run": True, "policy_sequences": len(records)},
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("serialized full-finetuning RL requires CUDA")
    if not 0.0 < args.cuda_memory_fraction < 1.0:
        raise ValueError("cuda_memory_fraction must be in (0, 1)")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
    requested_dtype = resolve_dtype(args.torch_dtype)
    autocast_dtype = resolve_autocast_dtype(args.precision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=requested_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.config._attn_implementation = args.attn_implementation
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("RL full fine-tuning unexpectedly found frozen model parameters")

    if args.optim == "adamw_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=optimizer_step_limit,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    old_cache_path = metadata_dir / "old_policy_logprobs.pt"
    old_cache_meta = metadata_dir / "old_policy_logprobs.meta.json"
    cache_signature = canonical_sha256(
        {
            "model_name": args.model_name,
            "dataset_sha256": run_config["dataset_sha256"],
            "max_length": args.max_length,
            "record_ids": [record["record_id"] for record in records],
            "target_starts": [record["target_start"] for record in records],
        }
    )
    if old_cache_path.is_file() and old_cache_meta.is_file():
        cache_meta = json.loads(old_cache_meta.read_text(encoding="utf-8"))
        if cache_meta.get("signature") != cache_signature:
            raise RuntimeError("existing old-policy cache does not match this immutable job")
        old_logps = torch.load(old_cache_path, map_location="cpu", weights_only=True)
    else:
        old_logps = compute_old_policy_logps(model, records, autocast_dtype)
        torch.save(old_logps, old_cache_path)
        save_json(
            old_cache_meta,
            {
                "created_at_utc": utc_now(),
                "signature": cache_signature,
                "records": len(old_logps),
                "model_name": args.model_name,
            },
        )
    if len(old_logps) != len(records):
        raise RuntimeError("old-policy cache row count mismatch")

    initial_eval = evaluate_records(
        model,
        records,
        old_logps,
        frozen_indices,
        args.clip_epsilon,
        args.kl_beta,
        autocast_dtype,
    )
    eval_path = metadata_dir / "frozen_checkpoint_metrics.jsonl"
    if eval_path.exists():
        eval_path.unlink()
    append_jsonl(eval_path, {"global_step": 0, "checkpoint": "base", **initial_eval})

    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
    session_meta["status"] = "running"
    session_meta["last_updated_utc"] = utc_now()
    save_json(session_meta_path, session_meta)

    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    train_history: list[dict[str, float]] = []
    started = time.monotonic()
    global_step = 0
    model.train()
    try:
        for _epoch in range(args.num_train_epochs):
            for record_index in order:
                if global_step >= optimizer_step_limit:
                    break
                global_step += 1
                record = records[record_index]
                device = model_device(model)
                input_ids = torch.tensor(
                    [record["input_ids"]], dtype=torch.long, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                with forward_context(device, autocast_dtype):
                    current = completion_logps(
                        model, input_ids, int(record["target_start"])
                    )
                    unweighted_loss, values = clipped_policy_objective(
                        current,
                        old_logps[record_index].to(device),
                        float(record["advantage"]),
                        args.clip_epsilon,
                        args.kl_beta,
                    )
                    loss = unweighted_loss * float(record["weight"])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                row_metrics = {
                    "global_step": float(global_step),
                    "loss": float(loss.detach().cpu()),
                    "unweighted_loss": float(unweighted_loss.detach().cpu()),
                    "policy_loss": float(values["policy_loss"].cpu()),
                    "approx_kl": float(values["approx_kl"].cpu()),
                    "clip_fraction": float(values["clip_fraction"].cpu()),
                    "ratio_mean": float(values["ratio_mean"].cpu()),
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                train_history.append(row_metrics)
                if global_step % args.logging_steps == 0:
                    print(
                        "RL step "
                        f"{global_step}/{optimizer_step_limit}: "
                        f"loss={row_metrics['loss']:.6f} "
                        f"kl={row_metrics['approx_kl']:.6f} "
                        f"clip={row_metrics['clip_fraction']:.4f}",
                        flush=True,
                    )
                del input_ids, current, unweighted_loss, loss
                release_memory()

                if args.max_gpu_memory_gib > 0:
                    reserved_gib = float(torch.cuda.max_memory_reserved()) / (1024.0**3)
                    if reserved_gib > args.max_gpu_memory_gib:
                        raise RuntimeError(
                            f"GPU memory guard triggered: {reserved_gib:.2f} GiB "
                            f"> {args.max_gpu_memory_gib:.2f} GiB"
                        )
                if not smoke_mode and global_step in save_steps:
                    evaluation = evaluate_records(
                        model,
                        records,
                        old_logps,
                        frozen_indices,
                        args.clip_epsilon,
                        args.kl_beta,
                        autocast_dtype,
                    )
                    checkpoint_dir = checkpoints_dir / f"checkpoint-{global_step}"
                    append_jsonl(
                        eval_path,
                        {
                            "global_step": global_step,
                            "checkpoint": str(checkpoint_dir),
                            **evaluation,
                        },
                    )
                    save_checkpoint(
                        model,
                        tokenizer,
                        checkpoint_dir,
                        max_shard_size=args.checkpoint_max_shard_size,
                        safe_serialization=args.checkpoint_safe_serialization == "true",
                        metadata={"global_step": global_step, "created_at_utc": utc_now()},
                    )
                    retain_latest_checkpoints(checkpoints_dir, args.save_total_limit)
                    model.train()
            if global_step >= optimizer_step_limit:
                break
    except Exception as exc:
        save_json(
            metadata_dir / "train_error.json",
            {
                "created_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        session_meta["status"] = "failed"
        session_meta["last_updated_utc"] = utc_now()
        save_json(session_meta_path, session_meta)
        raise

    final_eval = evaluate_records(
        model,
        records,
        old_logps,
        frozen_indices,
        args.clip_epsilon,
        args.kl_beta,
        autocast_dtype,
    )
    append_jsonl(
        eval_path,
        {"global_step": global_step, "checkpoint": "final", **final_eval},
    )
    if not smoke_mode:
        save_checkpoint(
            model,
            tokenizer,
            full_model_dir,
            max_shard_size=args.checkpoint_max_shard_size,
            safe_serialization=args.checkpoint_safe_serialization == "true",
            metadata={"global_step": global_step, "created_at_utc": utc_now()},
        )
    elapsed = time.monotonic() - started
    train_metrics = {
        "train_runtime_seconds": elapsed,
        "train_steps": global_step,
        "full_training_steps": full_training_steps,
        "smoke_mode": smoke_mode,
        "groups": tokenization_stats["groups"],
        "rollouts": tokenization_stats["rollouts"],
        "policy_sequences": tokenization_stats["policy_sequences"],
        "completion_loss": "thinking_and_final_content",
        "prompt_loss": "masked",
        "old_policy_cache_signature": cache_signature,
        "frozen_eval_subset_sha256": frozen_contract["sha256"],
        "initial_frozen_eval": initial_eval,
        "final_frozen_eval": final_eval,
        "training": aggregate_metrics(train_history),
        "tokenization": {
            key: value for key, value in tokenization_stats.items() if key != "records"
        },
        "memory": cuda_snapshot(),
    }
    save_json(metadata_dir / "train_metrics.json", train_metrics)
    save_json(metadata_dir / "train_log_history.json", {"log_history": train_history})
    session_meta["status"] = "smoke_passed" if smoke_mode else "trained"
    session_meta["last_updated_utc"] = utc_now()
    save_json(session_meta_path, session_meta)
    if smoke_mode:
        print("Serialized checkpointed RL smoke completed without publication", flush=True)
    else:
        print("Serialized checkpointed RL session completed", flush=True)


if __name__ == "__main__":
    main()
