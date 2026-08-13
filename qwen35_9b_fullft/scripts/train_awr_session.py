#!/usr/bin/env python3
"""Train one serialized 32K repair-distance AWR session.

Each row trains only its final assistant response, including reasoning and
visible content. Prompt tokens are masked. Deterministic repair-distance and
local-credit weights scale completion negative log-likelihood.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from train_rl_session import (
    aggregate_metrics,
    aggregate_weighted_metrics,
    canonical_sha256,
    completion_logps,
    cuda_snapshot,
    file_sha256,
    forward_context,
    model_device,
    parse_positive_steps,
    release_memory,
    retain_latest_checkpoints,
    resolve_autocast_dtype,
    resolve_dtype,
    resolve_optimizer_step_limit,
    run_command,
    save_checkpoint,
    save_json,
    select_frozen_subset,
    tokenize_policy_step,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "float32"], default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--max-gpu-memory-gib", type=float, default=110.0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.88)
    parser.add_argument("--cuda-alloc-conf", default="")
    parser.add_argument("--checkpoint-max-shard-size", default="512MB")
    parser.add_argument("--checkpoint-safe-serialization", choices=["true", "false"], default="true")
    parser.add_argument("--frozen-eval-max-sequences", type=int, default=16)
    parser.add_argument("--smoke-optimizer-steps", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def flatten_awr_rows(
    rows: Iterable[dict[str, Any]], tokenizer: Any, max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    token_stats: list[dict[str, Any]] = []
    groups: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        if row.get("objective") != "repair_distance_awr":
            raise ValueError(f"AWR row {index} has an invalid objective")
        weight = row.get("sample_weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise ValueError(f"AWR row {index} has an invalid sample_weight")
        sample_id = str(row.get("sample_id", "")).strip()
        group_id = str(row.get("group_id", "")).strip()
        if not sample_id or not group_id:
            raise ValueError(f"AWR row {index} requires sample_id and group_id")
        identity = (group_id, sample_id)
        if identity in identities:
            raise ValueError(f"AWR row {index} duplicates {group_id}/{sample_id}")
        identities.add(identity)
        completion = row.get("completion")
        if (
            not isinstance(completion, list)
            or len(completion) != 1
            or not isinstance(completion[0], dict)
            or completion[0].get("role") != "assistant"
            or not isinstance(completion[0].get("thinking"), str)
            or not completion[0]["thinking"].strip()
        ):
            raise ValueError(
                f"AWR row {index} requires one thinking-enabled assistant completion"
            )
        tokenized, stats = tokenize_policy_step(
            {"prompt": row.get("prompt"), "completion": row.get("completion")},
            tokenizer,
            max_length,
        )
        records.append(
            {
                **tokenized,
                "record_id": sample_id,
                "group_id": group_id,
                "weight": float(weight),
            }
        )
        groups.add(group_id)
        token_stats.append({"record_id": sample_id, **stats})
    if not records:
        raise ValueError("AWR dataset produced no policy sequences")
    return records, {
        "groups": len(groups),
        "policy_sequences": len(records),
        "truncated_sequences": sum(item["prompt_tokens_removed"] > 0 for item in token_stats),
        "thinking_sequences": sum(item["thinking_present"] for item in token_stats),
        "max_original_tokens": max(item["original_tokens"] for item in token_stats),
        "max_final_tokens": max(item["final_tokens"] for item in token_stats),
        "max_completion_tokens": max(item["completion_tokens"] for item in token_stats),
        "max_prompt_tokens_removed": max(item["prompt_tokens_removed"] for item in token_stats),
        "max_boundary_adjustment_tokens": max(
            item["boundary_adjustment_tokens"] for item in token_stats
        ),
        "sample_weight_min": min(record["weight"] for record in records),
        "sample_weight_max": max(record["weight"] for record in records),
        "sample_weight_mean": sum(record["weight"] for record in records) / len(records),
        "records": token_stats,
    }


def completion_nll(logps: Any) -> Any:
    if logps.numel() == 0:
        raise ValueError("AWR completion has no target tokens")
    return -logps.float().mean()


def evaluate_records(
    model: Any,
    records: list[dict[str, Any]],
    indices: list[int],
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
            logps = completion_logps(model, input_ids, int(record["target_start"]))
            nll = completion_nll(logps)
        metrics.append({"nll": float(nll.detach().cpu())})
        weights.append(float(record["weight"]))
        del input_ids, logps, nll
        release_memory()
    return {
        "sequences": len(indices),
        **aggregate_metrics(metrics),
        "sample_weighted": aggregate_weighted_metrics(metrics, weights),
    }


def main() -> None:
    args = parse_args()
    if args.num_train_epochs != 1:
        raise ValueError("repair-distance AWR currently requires exactly one epoch")
    if not 0 < args.max_length <= 32768:
        raise ValueError("max_length must be in [1, 32768]")
    if args.save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive")
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
        raise ValueError("serialized AWR requires exactly one immutable JSONL input")

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
    save_json(
        metadata_dir / "environment.json",
        {
            "created_at_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "nvidia_smi": run_command(["nvidia-smi"]),
            "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
        },
    )

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
    records, tokenization_stats = flatten_awr_rows(rows, tokenizer, args.max_length)
    full_training_steps = len(records)
    optimizer_step_limit, smoke_mode = resolve_optimizer_step_limit(
        full_training_steps, args.smoke_optimizer_steps
    )
    save_json(metadata_dir / "awr_tokenization_stats.json", tokenization_stats)
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
        "completion_loss": "thinking_and_final_content",
        "prompt_loss": "masked",
        "objective": "sample_weighted_completion_nll",
        "save_steps": save_steps,
        "save_total_limit": args.save_total_limit,
        "frozen_eval_subset_sha256": frozen_contract["sha256"],
        "optimizer_step_limit": optimizer_step_limit,
        "smoke_mode": smoke_mode,
        "dry_run": args.dry_run,
    }
    save_json(metadata_dir / "run_config.json", run_config)
    if args.dry_run:
        save_json(metadata_dir / "train_metrics.json", {"dry_run": True, **run_config})
        return

    if not torch.cuda.is_available():
        raise RuntimeError("serialized full-finetuning AWR requires CUDA")
    if not 0.0 < args.cuda_memory_fraction < 1.0:
        raise ValueError("cuda_memory_fraction must be in (0, 1)")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
    autocast_dtype = resolve_autocast_dtype(args.precision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=resolve_dtype(args.torch_dtype),
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
        raise RuntimeError("AWR full fine-tuning unexpectedly found frozen parameters")
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
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=optimizer_step_limit,
    )
    torch.cuda.reset_peak_memory_stats()
    initial_eval = evaluate_records(model, records, frozen_indices, autocast_dtype)
    eval_path = metadata_dir / "frozen_checkpoint_metrics.jsonl"
    eval_path.write_text(
        json.dumps({"global_step": 0, "checkpoint": "base", **initial_eval}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
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
        for record_index in order:
            if global_step >= optimizer_step_limit:
                break
            global_step += 1
            record = records[record_index]
            device = model_device(model)
            input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with forward_context(device, autocast_dtype):
                logps = completion_logps(model, input_ids, int(record["target_start"]))
                nll = completion_nll(logps)
                loss = nll * float(record["weight"])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            metrics = {
                "global_step": float(global_step),
                "loss": float(loss.detach().cpu()),
                "nll": float(nll.detach().cpu()),
                "sample_weight": float(record["weight"]),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            train_history.append(metrics)
            if global_step % args.logging_steps == 0:
                print(
                    f"AWR step {global_step}/{optimizer_step_limit}: "
                    f"loss={metrics['loss']:.6f} nll={metrics['nll']:.6f} "
                    f"weight={metrics['sample_weight']:.4f}",
                    flush=True,
                )
            del input_ids, logps, nll, loss
            release_memory()
            reserved_gib = float(torch.cuda.max_memory_reserved()) / (1024.0**3)
            if args.max_gpu_memory_gib > 0 and reserved_gib > args.max_gpu_memory_gib:
                raise RuntimeError(
                    f"GPU memory guard triggered: {reserved_gib:.2f} GiB "
                    f"> {args.max_gpu_memory_gib:.2f} GiB"
                )
            if not smoke_mode and global_step in save_steps:
                evaluation = evaluate_records(model, records, frozen_indices, autocast_dtype)
                checkpoint_dir = checkpoints_dir / f"checkpoint-{global_step}"
                with eval_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"global_step": global_step, "checkpoint": str(checkpoint_dir), **evaluation},
                            sort_keys=True,
                        )
                        + "\n"
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

    final_eval = evaluate_records(model, records, frozen_indices, autocast_dtype)
    with eval_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"global_step": global_step, "checkpoint": "final", **final_eval},
                sort_keys=True,
            )
            + "\n"
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
    train_metrics = {
        "train_runtime_seconds": time.monotonic() - started,
        "train_steps": global_step,
        "full_training_steps": full_training_steps,
        "smoke_mode": smoke_mode,
        "policy_sequences": len(records),
        "completion_loss": "thinking_and_final_content",
        "prompt_loss": "masked",
        "objective": "sample_weighted_completion_nll",
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
    print(
        "Serialized repair-distance AWR smoke completed without publication"
        if smoke_mode
        else "Serialized repair-distance AWR session completed",
        flush=True,
    )


if __name__ == "__main__":
    main()
