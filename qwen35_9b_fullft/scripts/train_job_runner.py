#!/usr/bin/env python3
"""Run immutable remote training jobs for the Hayabusa/Qwen stack.

This runner wraps the validated SFT, DPO, checkpointed RL, and repair-distance
AWR trainers.
It owns job claiming, validation, durable status/result files, and optional
vLLM deployment; each objective remains an explicit immutable job contract.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


FORMAT_VERSION = 1
MAX_TRAINING_SEQUENCE_LENGTH = 32768
CHECKPOINT_INTERVAL_STEPS = 20
DEFAULT_ENDPOINT = "http://127.0.0.1:8002/v1"
DEFAULT_LAN_ENDPOINT = "http://10.0.0.34:8002/v1"
SUPPORTED_PROFILES = {"micro_contract_validation"}
TERMINAL_STATES = {"complete", "failed"}
NON_TERMINAL_STATES = {
    "pending",
    "validating",
    "sft_running",
    "dpo_running",
    "rl_running",
    "awr_running",
    "deploying",
    "health_check",
}


class RunnerError(Exception):
    """Base runner exception with a failed stage."""

    failed_stage = "runner"


class ValidationError(RunnerError):
    failed_stage = "validating"


class StageError(RunnerError):
    def __init__(self, failed_stage: str, message: str):
        super().__init__(message)
        self.failed_stage = failed_stage


@dataclass(frozen=True)
class FixtureConfig:
    fail_stage: str = ""
    sleep_seconds: float = 0.0


@dataclass(frozen=True)
class RunnerConfig:
    jobs_root: Path
    workspace_root: Path
    mode: str = "real"
    once: bool = True
    poll_interval_seconds: float = 5.0
    fixture: FixtureConfig = FixtureConfig()


@dataclass
class ValidatedInput:
    path: Path
    rows: int
    sha256: str


@dataclass
class ValidatedJob:
    manifest: dict[str, Any]
    job_id: str
    base_checkpoint: str
    output_checkpoint: str
    max_sequence_length: int
    training_profile: str
    sft_enabled: bool
    dpo_enabled: bool
    sft_input: ValidatedInput | None
    dpo_input: ValidatedInput | None
    deployment_enabled: bool
    served_model_name: str
    assistant_reasoning: str
    thinking_max_chars: int
    max_gpu_memory_gib: float = 110.0
    dpo_execution_mode: str = "batched"
    rl_enabled: bool = False
    rl_input: ValidatedInput | None = None
    rl_clip_epsilon: float = 0.20
    rl_kl_beta: float = 0.01
    rl_epochs: int = 1
    awr_enabled: bool = False
    awr_input: ValidatedInput | None = None
    awr_epochs: int = 1


@dataclass
class StageOutput:
    session_dir: Path
    checkpoint: Path
    command: list[str]
    log_path: Path
    metrics: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_label(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return clean.strip("_") or "job"


def fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def append_event(job_dir: Path, event: str, state: str, stage: str | None, message: str = "") -> None:
    payload = {
        "timestamp": utc_now(),
        "event": event,
        "job_id": job_dir.name,
        "state": state,
        "stage": stage,
    }
    if message:
        payload["message"] = message[:500]
    path = job_dir / "trainer_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_status(
    job_dir: Path,
    state: str,
    message: str,
    stage: str | None = None,
    pid: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "job_id": job_dir.name,
        "state": state,
        "stage": stage,
        "updated_at": utc_now(),
        "message": message,
    }
    if pid is not None:
        payload["pid"] = pid
    atomic_write_json(job_dir / "status.json", payload)
    append_event(job_dir, "status", state, stage, message)


def write_success_result(
    job_dir: Path,
    job: ValidatedJob,
    sft: StageOutput | None,
    dpo: StageOutput | None,
    rl: StageOutput | None,
    awr: StageOutput | None,
    endpoint: str,
    health_check: dict[str, Any],
) -> None:
    if sum(item is not None for item in (dpo, rl, awr)) != 1:
        raise RunnerError("exactly one terminal training stage is required")
    payload = {
        "format_version": FORMAT_VERSION,
        "job_id": job.job_id,
        "status": "complete",
        "base_checkpoint": job.base_checkpoint,
        "sft_checkpoint": str(sft.checkpoint) if sft is not None else None,
        "final_checkpoint": str((awr or rl or dpo).checkpoint),
        "served_model_name": job.served_model_name,
        "endpoint": endpoint,
        "health_check": health_check,
        "metrics": {
            "sft": sft.metrics if sft is not None else {},
            "dpo": dpo.metrics if dpo is not None else {},
            "rl": rl.metrics if rl is not None else {},
            "awr": awr.metrics if awr is not None else {},
        },
        "completed_at": utc_now(),
    }
    if dpo is not None:
        effective_mode = resolve_dpo_execution_mode(
            job.dpo_execution_mode, job.max_sequence_length
        )
        max_prompt, max_completion, max_length = dpo_lengths(
            job.max_sequence_length, effective_mode
        )
        payload["dpo_execution"] = {
            "requested_mode": job.dpo_execution_mode,
            "effective_mode": effective_mode,
            "requested_max_sequence_length": job.max_sequence_length,
            "effective_max_sequence_length": max_length,
            "max_prompt_length": max_prompt,
            "max_completion_length": max_completion,
        }
    if rl is not None:
        payload["rl_execution"] = {
            "execution_mode": "serialized",
            "requested_max_sequence_length": job.max_sequence_length,
            "effective_max_sequence_length": job.max_sequence_length,
            "clip_epsilon": job.rl_clip_epsilon,
            "kl_beta": job.rl_kl_beta,
            "epochs": job.rl_epochs,
        }
    if awr is not None:
        payload["awr_execution"] = {
            "execution_mode": "serialized",
            "requested_max_sequence_length": job.max_sequence_length,
            "effective_max_sequence_length": job.max_sequence_length,
            "epochs": job.awr_epochs,
            "objective": "sample_weighted_completion_nll",
        }
    atomic_write_json(job_dir / "result.json", payload)


def write_failed_result(job_dir: Path, failed_stage: str, error: str) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "job_id": job_dir.name,
        "status": "failed",
        "failed_stage": failed_stage,
        "error": error[:1000],
        "completed_at": utc_now(),
    }
    stage_summary_path = job_dir / "stage_sessions.json"
    if stage_summary_path.is_file():
        with contextlib.suppress(Exception):
            stage_summary = read_json(stage_summary_path)
            sft = stage_summary.get("sft")
            if isinstance(sft, dict):
                checkpoint = sft.get("checkpoint")
                if isinstance(checkpoint, str) and checkpoint:
                    payload["sft_checkpoint"] = checkpoint
                    payload["recoverable_dpo_only"] = failed_stage == "dpo"
    atomic_write_json(job_dir / "result.json", payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"{path.name} must contain a JSON object")
    return data


def safe_relative_path(job_dir: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValidationError(f"{field} must be a non-empty relative path string")
    posix = PurePosixPath(raw)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValidationError(f"{field} is not a safe relative path: {raw!r}")
    resolved = (job_dir / Path(*posix.parts)).resolve()
    job_root = job_dir.resolve()
    if not resolved.is_relative_to(job_root):
        raise ValidationError(f"{field} escapes the job directory: {raw!r}")
    return resolved


def iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValidationError(f"{path.name}:{line_no}: blank JSONL lines are not allowed")
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path.name}:{line_no}: malformed JSON: {exc}") from exc


def validate_thinking(
    path: Path,
    line_no: int,
    label: str,
    message: dict[str, Any],
    thinking_max_chars: int,
) -> None:
    thinking = message.get("thinking")
    if not isinstance(thinking, str) or not thinking.strip():
        raise ValidationError(f"{path.name}:{line_no}: {label}.thinking is required")
    if len(thinking) > thinking_max_chars:
        raise ValidationError(
            f"{path.name}:{line_no}: {label}.thinking exceeds "
            f"{thinking_max_chars} characters"
        )


def validate_sft_jsonl(
    path: Path,
    assistant_reasoning: str = "disabled",
    thinking_max_chars: int = 1800,
) -> int:
    rows = 0
    for line_no, obj in iter_jsonl(path):
        rows += 1
        if not isinstance(obj, dict):
            raise ValidationError(f"{path.name}:{line_no}: SFT row must be an object")
        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValidationError(f"{path.name}:{line_no}: SFT row must contain a non-empty messages array")
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(f"{path.name}:{line_no}: messages[{idx}] must be an object")
            if not isinstance(message.get("role"), str) or not message["role"]:
                raise ValidationError(f"{path.name}:{line_no}: messages[{idx}].role must be a non-empty string")
            if "content" not in message:
                raise ValidationError(f"{path.name}:{line_no}: messages[{idx}].content is required")
            content = message["content"]
            if not isinstance(content, (str, list)):
                raise ValidationError(
                    f"{path.name}:{line_no}: messages[{idx}].content must be a string or content-part list"
                )
        if assistant_reasoning == "required":
            final = messages[-1]
            if final.get("role") != "assistant":
                raise ValidationError(
                    f"{path.name}:{line_no}: SFT messages must end with an assistant message"
                )
            validate_thinking(
                path,
                line_no,
                f"messages[{len(messages) - 1}]",
                final,
                thinking_max_chars,
            )
    return rows


def validate_dpo_value(
    path: Path,
    line_no: int,
    key: str,
    value: Any,
    assistant_reasoning: str = "disabled",
    thinking_max_chars: int = 1800,
) -> None:
    if isinstance(value, str) and value:
        if assistant_reasoning == "required" and key in {"chosen", "rejected"}:
            raise ValidationError(
                f"{path.name}:{line_no}: DPO field {key!r} must be a message array "
                "when assistant reasoning is required"
            )
        return
    if isinstance(value, list) and value:
        for idx, message in enumerate(value):
            if not isinstance(message, dict):
                raise ValidationError(f"{path.name}:{line_no}: DPO field {key}[{idx}] must be an object")
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValidationError(
                    f"{path.name}:{line_no}: DPO field {key}[{idx}].role must be a non-empty string"
                )
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise ValidationError(
                    f"{path.name}:{line_no}: DPO field {key}[{idx}].content must be a non-empty string"
                )
        if assistant_reasoning == "required" and key in {"chosen", "rejected"}:
            final = value[-1]
            if final.get("role") != "assistant":
                raise ValidationError(
                    f"{path.name}:{line_no}: DPO field {key!r} must end with an assistant message"
                )
            validate_thinking(
                path,
                line_no,
                f"DPO field {key}[{len(value) - 1}]",
                final,
                thinking_max_chars,
            )
        return
    raise ValidationError(
        f"{path.name}:{line_no}: DPO field {key!r} must be a non-empty string "
        "or a non-empty array of message objects"
    )


def validate_dpo_jsonl(
    path: Path,
    assistant_reasoning: str = "disabled",
    thinking_max_chars: int = 1800,
) -> int:
    rows = 0
    for line_no, obj in iter_jsonl(path):
        rows += 1
        if not isinstance(obj, dict):
            raise ValidationError(f"{path.name}:{line_no}: DPO row must be an object")
        for key in ("prompt", "chosen", "rejected"):
            validate_dpo_value(
                path,
                line_no,
                key,
                obj.get(key),
                assistant_reasoning,
                thinking_max_chars,
            )
        if "meta" in obj and not isinstance(obj["meta"], (dict, list, str, int, float, bool, type(None))):
            raise ValidationError(f"{path.name}:{line_no}: DPO meta must be JSON-serializable")
    return rows


def validate_rl_reasoning_structure_negative(
    path: Path,
    line_no: int,
    obj: dict[str, Any],
    policy_steps: list[Any],
    thinking_max_chars: int,
) -> bool:
    evidence = obj.get("reward_evidence")
    if not isinstance(evidence, dict) or evidence.get("kind") != (
        "reasoning_structure_negative"
    ):
        return False
    if float(obj.get("reward", 0.0)) != -1.0 or len(policy_steps) != 1:
        raise ValidationError(
            f"{path.name}:{line_no}: reasoning structural negative must "
            "isolate one -1 response"
        )
    step = policy_steps[0]
    if not isinstance(step, dict):
        raise ValidationError(
            f"{path.name}:{line_no}: reasoning structural step must be an object"
        )
    reasons = step.get("reasoning_structure_reasons")
    reason = str(evidence.get("reason", ""))
    if (
        step.get("reasoning_structure_valid") is not False
        or reasons != [reason]
        or reason not in {"missing_thinking", "thinking_too_long"}
    ):
        raise ValidationError(
            f"{path.name}:{line_no}: reasoning structural evidence is inconsistent"
        )
    completion = step.get("completion")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValidationError(
            f"{path.name}:{line_no}: reasoning structural completion must "
            "contain one response"
        )
    message = completion[0]
    if not isinstance(message, dict):
        raise ValidationError(
            f"{path.name}:{line_no}: reasoning structural completion is invalid"
        )
    role = message.get("role")
    content = message.get("content")
    if role != "assistant" or not isinstance(content, str) or not content:
        raise ValidationError(
            f"{path.name}:{line_no}: reasoning structural completion must be "
            "a non-empty assistant response"
        )
    thinking = message.get("thinking")
    if reason == "missing_thinking":
        valid_violation = not isinstance(thinking, str) or not thinking.strip()
        observed_chars = 0
    else:
        valid_violation = (
            isinstance(thinking, str) and len(thinking) > thinking_max_chars
        )
        observed_chars = len(thinking) if isinstance(thinking, str) else 0
    if not valid_violation:
        raise ValidationError(
            f"{path.name}:{line_no}: declared {reason} violation is absent"
        )
    if int(step.get("thinking_chars", -1)) != observed_chars:
        raise ValidationError(
            f"{path.name}:{line_no}: thinking length evidence is inconsistent"
        )
    if int(step.get("thinking_max_chars", 0)) != thinking_max_chars:
        raise ValidationError(
            f"{path.name}:{line_no}: thinking limit evidence is inconsistent"
        )
    return True


def validate_rl_jsonl(
    path: Path,
    assistant_reasoning: str = "required",
    thinking_max_chars: int = 1800,
) -> int:
    rows = 0
    group_advantages: dict[str, list[float]] = {}
    identities: set[tuple[str, str]] = set()
    for line_no, obj in iter_jsonl(path):
        rows += 1
        if not isinstance(obj, dict):
            raise ValidationError(f"{path.name}:{line_no}: RL row must be an object")
        group_id = obj.get("group_id")
        rollout_id = obj.get("rollout_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValidationError(f"{path.name}:{line_no}: group_id is required")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValidationError(f"{path.name}:{line_no}: rollout_id is required")
        identity = (group_id, rollout_id)
        if identity in identities:
            raise ValidationError(
                f"{path.name}:{line_no}: duplicate RL rollout {group_id}/{rollout_id}"
            )
        identities.add(identity)
        for field in ("reward", "advantage", "policy_step_weight"):
            value = obj.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationError(
                    f"{path.name}:{line_no}: {field} must be a finite number"
                )
            if not math.isfinite(float(value)):
                raise ValidationError(
                    f"{path.name}:{line_no}: {field} must be a finite number"
                )
        if float(obj["policy_step_weight"]) <= 0.0:
            raise ValidationError(
                f"{path.name}:{line_no}: policy_step_weight must be positive"
            )
        policy_steps = obj.get("policy_steps")
        if not isinstance(policy_steps, list) or not policy_steps:
            raise ValidationError(f"{path.name}:{line_no}: policy_steps must be non-empty")
        expected_weight = 1.0 / len(policy_steps)
        if abs(float(obj["policy_step_weight"]) - expected_weight) > 1e-6:
            raise ValidationError(
                f"{path.name}:{line_no}: policy_step_weight must equal 1 / policy_steps"
            )
        reasoning_structure_negative = validate_rl_reasoning_structure_negative(
            path,
            line_no,
            obj,
            policy_steps,
            thinking_max_chars,
        )
        for step_index, policy_step in enumerate(policy_steps):
            label = f"RL policy_steps[{step_index}]"
            if not isinstance(policy_step, dict):
                raise ValidationError(f"{path.name}:{line_no}: {label} must be an object")
            validate_dpo_value(
                path,
                line_no,
                f"{label}.prompt",
                policy_step.get("prompt"),
                "disabled",
                thinking_max_chars,
            )
            if not reasoning_structure_negative:
                validate_dpo_value(
                    path,
                    line_no,
                    f"{label}.completion",
                    policy_step.get("completion"),
                    assistant_reasoning,
                    thinking_max_chars,
                )
        group_advantages.setdefault(group_id, []).append(float(obj["advantage"]))

    for group_id, advantages in group_advantages.items():
        if len(advantages) < 2:
            raise ValidationError(
                f"{path.name}: RL group {group_id!r} must contain at least two rollouts"
            )
        if max(advantages) - min(advantages) <= 1e-9:
            raise ValidationError(
                f"{path.name}: RL group {group_id!r} has no advantage diversity"
            )
        if abs(sum(advantages)) > 1e-4 * max(1, len(advantages)):
            raise ValidationError(
                f"{path.name}: RL group {group_id!r} advantages are not centered"
            )
    return rows


def validate_awr_jsonl(
    path: Path,
    assistant_reasoning: str = "required",
    thinking_max_chars: int = 1800,
) -> int:
    rows = 0
    identities: set[tuple[str, str]] = set()
    for line_no, obj in iter_jsonl(path):
        rows += 1
        if not isinstance(obj, dict):
            raise ValidationError(f"{path.name}:{line_no}: AWR row must be an object")
        if obj.get("objective") != "repair_distance_awr":
            raise ValidationError(
                f"{path.name}:{line_no}: objective must be repair_distance_awr"
            )
        group_id = obj.get("group_id")
        sample_id = obj.get("sample_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValidationError(f"{path.name}:{line_no}: group_id is required")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValidationError(f"{path.name}:{line_no}: sample_id is required")
        identity = (group_id, sample_id)
        if identity in identities:
            raise ValidationError(
                f"{path.name}:{line_no}: duplicate AWR sample {group_id}/{sample_id}"
            )
        identities.add(identity)
        weight = obj.get("sample_weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise ValidationError(
                f"{path.name}:{line_no}: sample_weight must be finite and positive"
            )
        validate_dpo_value(
            path,
            line_no,
            "prompt",
            obj.get("prompt"),
            "disabled",
            thinking_max_chars,
        )
        validate_dpo_value(
            path,
            line_no,
            "completion",
            obj.get("completion"),
            assistant_reasoning,
            thinking_max_chars,
        )
        completion = obj.get("completion")
        if (
            not isinstance(completion, list)
            or len(completion) != 1
            or completion[0].get("role") != "assistant"
        ):
            raise ValidationError(
                f"{path.name}:{line_no}: AWR completion must contain exactly one assistant message"
            )
        if assistant_reasoning == "required":
            validate_thinking(
                path,
                line_no,
                "AWR completion[0]",
                completion[0],
                thinking_max_chars,
            )
    return rows


def validate_input(
    job_dir: Path,
    manifest: dict[str, Any],
    stage: str,
    assistant_reasoning: str = "disabled",
    thinking_max_chars: int = 1800,
) -> ValidatedInput:
    try:
        spec = manifest["inputs"][stage]
    except KeyError as exc:
        raise ValidationError(f"inputs.{stage} is required") from exc
    if not isinstance(spec, dict):
        raise ValidationError(f"inputs.{stage} must be an object")
    path = safe_relative_path(job_dir, spec.get("path"), f"inputs.{stage}.path")
    if not path.exists() or not path.is_file():
        raise ValidationError(f"inputs.{stage}.path does not exist: {path}")
    expected_sha = spec.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
        raise ValidationError(f"inputs.{stage}.sha256 must be a 64-character hex digest")
    actual_sha = sha256_file(path)
    if actual_sha.lower() != expected_sha.lower():
        raise ValidationError(
            f"inputs.{stage}.sha256 mismatch: expected {expected_sha.lower()}, got {actual_sha.lower()}"
        )
    expected_rows = spec.get("rows")
    if not isinstance(expected_rows, int) or expected_rows < 0:
        raise ValidationError(f"inputs.{stage}.rows must be a non-negative integer")
    if stage == "sft":
        actual_rows = validate_sft_jsonl(
            path, assistant_reasoning, thinking_max_chars
        )
    elif stage == "dpo":
        actual_rows = validate_dpo_jsonl(
            path, assistant_reasoning, thinking_max_chars
        )
    elif stage == "rl":
        actual_rows = validate_rl_jsonl(
            path, assistant_reasoning, thinking_max_chars
        )
    elif stage == "awr":
        actual_rows = validate_awr_jsonl(
            path, assistant_reasoning, thinking_max_chars
        )
    else:
        raise ValidationError(f"unsupported input stage: {stage!r}")
    if actual_rows != expected_rows:
        raise ValidationError(f"inputs.{stage}.rows mismatch: expected {expected_rows}, got {actual_rows}")
    return ValidatedInput(path=path, rows=actual_rows, sha256=actual_sha)


def validate_manifest(job_dir: Path) -> ValidatedJob:
    manifest_path = job_dir / "job.json"
    if not manifest_path.exists():
        raise ValidationError("job.json is required")
    manifest = read_json(manifest_path)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValidationError(f"unsupported format_version: {manifest.get('format_version')!r}")
    job_id = manifest.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValidationError("job_id is required")
    if job_id != job_dir.name:
        raise ValidationError(f"job_id {job_id!r} must match job directory name {job_dir.name!r}")
    base_checkpoint = manifest.get("base_checkpoint")
    if not isinstance(base_checkpoint, str) or not base_checkpoint:
        raise ValidationError("base_checkpoint is required")
    output_checkpoint = manifest.get("output_checkpoint")
    if not isinstance(output_checkpoint, str) or not output_checkpoint:
        raise ValidationError("output_checkpoint is required")
    max_sequence_length = manifest.get("max_sequence_length", MAX_TRAINING_SEQUENCE_LENGTH)
    if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
        raise ValidationError("max_sequence_length must be a positive integer")
    if max_sequence_length > MAX_TRAINING_SEQUENCE_LENGTH:
        raise ValidationError(
            f"max_sequence_length {max_sequence_length} exceeds supported maximum "
            f"{MAX_TRAINING_SEQUENCE_LENGTH}"
        )
    training_profile = manifest.get("training_profile")
    if training_profile not in SUPPORTED_PROFILES:
        raise ValidationError(
            f"unsupported training_profile {training_profile!r}; supported={sorted(SUPPORTED_PROFILES)}"
        )
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise ValidationError("stages object is required")
    sft_stage = stages.get("sft")
    dpo_stage = stages.get("dpo")
    rl_stage = stages.get("rl", {"enabled": False, "overrides": {}})
    awr_stage = stages.get("awr", {"enabled": False, "overrides": {}})
    if not isinstance(sft_stage, dict) or not isinstance(dpo_stage, dict):
        raise ValidationError("stages.sft and stages.dpo objects are required")
    if not isinstance(rl_stage, dict):
        raise ValidationError("stages.rl must be an object when provided")
    if not isinstance(awr_stage, dict):
        raise ValidationError("stages.awr must be an object when provided")
    if not all(
        isinstance(stage.get("enabled"), bool)
        for stage in (sft_stage, dpo_stage, rl_stage, awr_stage)
    ):
        raise ValidationError(
            "all training stage enabled values must be booleans"
        )
    sft_enabled = sft_stage["enabled"]
    dpo_enabled = dpo_stage["enabled"]
    rl_enabled = rl_stage["enabled"]
    awr_enabled = awr_stage["enabled"]
    if sum((dpo_enabled, rl_enabled, awr_enabled)) != 1:
        raise ValidationError("exactly one of DPO, RL, or AWR must be enabled")
    if (rl_enabled or awr_enabled) and sft_enabled:
        raise ValidationError("RL/AWR jobs cannot include an SFT stage")
    if sft_stage.get("overrides", {}) or dpo_stage.get("overrides", {}):
        raise ValidationError("SFT/DPO stage overrides are not supported in V1")

    rl_overrides = rl_stage.get("overrides", {})
    if not isinstance(rl_overrides, dict):
        raise ValidationError("stages.rl.overrides must be an object")
    unknown_rl_overrides = set(rl_overrides) - {
        "clip_epsilon",
        "kl_beta",
        "epochs",
    }
    if unknown_rl_overrides:
        raise ValidationError(
            "unsupported RL overrides: " + ", ".join(sorted(unknown_rl_overrides))
        )
    if rl_enabled and rl_stage.get("execution_mode", "serialized") != "serialized":
        raise ValidationError("stages.rl.execution_mode must be 'serialized'")
    awr_overrides = awr_stage.get("overrides", {})
    if not isinstance(awr_overrides, dict):
        raise ValidationError("stages.awr.overrides must be an object")
    if set(awr_overrides) - {"epochs"}:
        raise ValidationError("unsupported AWR overrides")
    if awr_enabled and awr_stage.get("execution_mode", "serialized") != "serialized":
        raise ValidationError("stages.awr.execution_mode must be 'serialized'")
    awr_epochs = awr_overrides.get("epochs", 1)
    if not isinstance(awr_epochs, int) or isinstance(awr_epochs, bool) or awr_epochs != 1:
        raise ValidationError("AWR currently requires exactly one epoch")
    rl_clip_epsilon = rl_overrides.get("clip_epsilon", 0.20)
    rl_kl_beta = rl_overrides.get("kl_beta", 0.01)
    rl_epochs = rl_overrides.get("epochs", 1)
    if (
        not isinstance(rl_clip_epsilon, (int, float))
        or isinstance(rl_clip_epsilon, bool)
        or not 0.0 < float(rl_clip_epsilon) < 1.0
    ):
        raise ValidationError("RL clip_epsilon must be in (0, 1)")
    if (
        not isinstance(rl_kl_beta, (int, float))
        or isinstance(rl_kl_beta, bool)
        or float(rl_kl_beta) < 0.0
    ):
        raise ValidationError("RL kl_beta must be non-negative")
    if not isinstance(rl_epochs, int) or isinstance(rl_epochs, bool) or rl_epochs != 1:
        raise ValidationError("RL currently requires exactly one epoch")

    reasoning = manifest.get("assistant_reasoning", {})
    if reasoning is None:
        reasoning = {}
    if not isinstance(reasoning, dict):
        raise ValidationError("assistant_reasoning must be an object")
    assistant_reasoning = reasoning.get("mode", "disabled")
    if assistant_reasoning not in {"required", "disabled"}:
        raise ValidationError(
            "assistant_reasoning.mode must be 'required' or 'disabled'"
        )
    thinking_max_chars = reasoning.get("thinking_max_chars", 1800)
    if not isinstance(thinking_max_chars, int) or thinking_max_chars <= 0:
        raise ValidationError(
            "assistant_reasoning.thinking_max_chars must be a positive integer"
        )
    if reasoning.get("field", "thinking") != "thinking":
        raise ValidationError("assistant_reasoning.field must be 'thinking'")
    semantic_judging = reasoning.get("semantic_judging", "final_content_only")
    if semantic_judging not in {"final_content_only", "not_used"}:
        raise ValidationError(
            "assistant_reasoning.semantic_judging is invalid"
        )
    if (rl_enabled or awr_enabled) and assistant_reasoning != "required":
        raise ValidationError("RL/AWR jobs require assistant reasoning")
    if awr_enabled and semantic_judging != "not_used":
        raise ValidationError("AWR jobs must declare semantic_judging=not_used")

    dpo_execution_mode = manifest.get("dpo_execution_mode", "batched")
    if dpo_execution_mode not in {"batched", "split_backward", "auto"}:
        raise ValidationError(
            "dpo_execution_mode must be 'batched', 'split_backward', or 'auto'"
        )
    max_gpu_memory_gib = manifest.get("max_gpu_memory_gib", 110.0)
    if (
        isinstance(max_gpu_memory_gib, bool)
        or not isinstance(max_gpu_memory_gib, (int, float))
        or not math.isfinite(float(max_gpu_memory_gib))
        or float(max_gpu_memory_gib) < 0.0
    ):
        raise ValidationError(
            "max_gpu_memory_gib must be a finite non-negative number"
        )

    sft_input = (
        validate_input(
            job_dir,
            manifest,
            "sft",
            assistant_reasoning,
            thinking_max_chars,
        )
        if sft_enabled
        else None
    )
    dpo_input = (
        validate_input(
            job_dir,
            manifest,
            "dpo",
            assistant_reasoning,
            thinking_max_chars,
        )
        if dpo_enabled
        else None
    )
    rl_input = (
        validate_input(
            job_dir,
            manifest,
            "rl",
            assistant_reasoning,
            thinking_max_chars,
        )
        if rl_enabled
        else None
    )
    awr_input = (
        validate_input(
            job_dir,
            manifest,
            "awr",
            assistant_reasoning,
            thinking_max_chars,
        )
        if awr_enabled
        else None
    )

    deployment = manifest.get("deployment", {})
    if deployment is None:
        deployment = {}
    if not isinstance(deployment, dict):
        raise ValidationError("deployment must be an object")
    deployment_enabled = deployment.get("enabled", True) is True
    served_model_name = deployment.get("served_model_name") or output_checkpoint
    if not isinstance(served_model_name, str) or not served_model_name:
        raise ValidationError("deployment.served_model_name must be a non-empty string when provided")

    return ValidatedJob(
        manifest=manifest,
        job_id=job_id,
        base_checkpoint=base_checkpoint,
        output_checkpoint=output_checkpoint,
        max_sequence_length=max_sequence_length,
        training_profile=training_profile,
        sft_enabled=sft_enabled,
        dpo_enabled=dpo_enabled,
        sft_input=sft_input,
        dpo_input=dpo_input,
        deployment_enabled=deployment_enabled,
        served_model_name=served_model_name,
        assistant_reasoning=assistant_reasoning,
        thinking_max_chars=thinking_max_chars,
        max_gpu_memory_gib=float(max_gpu_memory_gib),
        dpo_execution_mode=dpo_execution_mode,
        rl_enabled=rl_enabled,
        rl_input=rl_input,
        rl_clip_epsilon=float(rl_clip_epsilon),
        rl_kl_beta=float(rl_kl_beta),
        rl_epochs=rl_epochs,
        awr_enabled=awr_enabled,
        awr_input=awr_input,
        awr_epochs=awr_epochs,
    )


def ensure_jobs_root(jobs_root: Path) -> None:
    for name in ("incoming", "running", "completed", "failed"):
        (jobs_root / name).mkdir(parents=True, exist_ok=True)


def read_state(job_dir: Path) -> str | None:
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        status = read_json(status_path)
    except Exception:
        return None
    state = status.get("state")
    return state if isinstance(state, str) else None


def terminal_exists(jobs_root: Path, job_id: str) -> bool:
    return (jobs_root / "completed" / job_id).exists() or (jobs_root / "failed" / job_id).exists()


def discover_candidate(jobs_root: Path) -> tuple[Path, str] | None:
    running = sorted((jobs_root / "running").iterdir(), key=lambda p: (p.stat().st_mtime, p.name))
    for path in running:
        if not path.is_dir():
            continue
        state = read_state(path)
        if state not in TERMINAL_STATES:
            return path, "running"

    incoming = []
    for path in (jobs_root / "incoming").iterdir():
        if not path.is_dir() or not (path / "READY").exists():
            continue
        if terminal_exists(jobs_root, path.name):
            continue
        ready_mtime = (path / "READY").stat().st_mtime
        incoming.append((ready_mtime, path.name, path))
    if not incoming:
        return None
    incoming.sort()
    return incoming[0][2], "incoming"


def claim_job(jobs_root: Path, candidate: Path, origin: str) -> tuple[Path, bool]:
    if origin == "running":
        return candidate, True
    target = jobs_root / "running" / candidate.name
    if target.exists():
        raise RunnerError(f"running job already exists for {candidate.name}")
    os.replace(candidate, target)
    fsync_dir(jobs_root / "incoming")
    fsync_dir(jobs_root / "running")
    return target, False


@contextlib.contextmanager
def single_job_lock(jobs_root: Path):
    jobs_root.mkdir(parents=True, exist_ok=True)
    lock_path = jobs_root / "runner.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StageError("runner", f"another training runner already holds {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def python_executable(workspace_root: Path) -> str:
    venv_python = workspace_root / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def script_path(workspace_root: Path, script_name: str) -> Path:
    return workspace_root / "qwen35_9b_fullft" / "scripts" / script_name


def write_command_log_header(log_path: Path, command: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        header = {
            "timestamp": utc_now(),
            "command": command,
        }
        handle.write((json.dumps(header, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def run_streamed_command(
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> int:
    write_command_log_header(log_path, command)
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=env,
        )
        return process.wait()


def run_captured_command(command: list[str], log_path: Path) -> tuple[int, str]:
    write_command_log_header(log_path, command)
    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(process.stdout)
        handle.flush()
        os.fsync(handle.fileno())
    return process.returncode, process.stdout


def create_real_session(
    config: RunnerConfig,
    job_dir: Path,
    stage: str,
    input_name: str,
    label: str,
    log_path: Path,
) -> Path:
    command = [
        python_executable(config.workspace_root),
        str(script_path(config.workspace_root, "create_session.py")),
        "--workspace-root",
        str(config.workspace_root),
        "--dataset-root",
        str(job_dir),
        "--jsonl-pattern",
        input_name,
        "--label",
        label,
        "--notes",
        f"remote training runner {stage} stage for {job_dir.name}",
    ]
    rc, output = run_captured_command(command, log_path)
    if rc != 0:
        raise StageError(stage, f"create_session.py failed for {stage} with exit code {rc}")
    match = re.search(r"Session created:\s*(.+)", output)
    if not match:
        raise StageError(stage, f"create_session.py did not report a session path for {stage}")
    session_dir = Path(match.group(1).strip()).resolve()
    if not session_dir.exists():
        raise StageError(stage, f"created session path does not exist: {session_dir}")
    return session_dir


def create_fixture_session(config: RunnerConfig, job_dir: Path, stage: str, label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    session_dir = (
        config.workspace_root
        / "qwen35_9b_fullft"
        / "runs"
        / f"{timestamp}_{sanitize_label(label)}"
    )
    for subdir in ("metadata", "logs", "checkpoints", "artifacts/full_model"):
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        session_dir / "metadata" / "session.json",
        {
            "created_at_utc": utc_now(),
            "session_id": session_dir.name,
            "session_dir": str(session_dir),
            "notes": f"fixture {stage} session for {job_dir.name}",
            "status": "created",
        },
    )
    return session_dir


def create_stage_session(
    config: RunnerConfig,
    job_dir: Path,
    stage: str,
    input_path: Path,
    label: str,
    log_path: Path,
) -> Path:
    if config.mode == "fixture":
        return create_fixture_session(config, job_dir, stage, label)
    return create_real_session(config, job_dir, stage, input_path.name, label, log_path)


def build_sft_command(config: RunnerConfig, job: ValidatedJob, session_dir: Path) -> list[str]:
    if config.mode == "fixture":
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_fake_stage",
            "--stage",
            "sft",
            "--session-dir",
            str(session_dir),
        ]
        if config.fixture.fail_stage == "sft":
            command.append("--fail")
        if config.fixture.sleep_seconds:
            command += ["--sleep", str(config.fixture.sleep_seconds)]
        return command

    return [
        python_executable(config.workspace_root),
        str(script_path(config.workspace_root, "train_session.py")),
        "--session-dir",
        str(session_dir),
        "--model-name",
        job.base_checkpoint,
        "--max-seq-length",
        str(job.max_sequence_length),
        "--num-train-epochs",
        "1.0",
        "--truncation-side",
        "left",
        "--attn-implementation",
        "sdpa",
        "--device-map",
        "cuda:0",
        "--per-device-train-batch-size",
        "1",
        "--dataset-num-proc",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--gradient-checkpointing",
        "unsloth",
        "--precision",
        "auto",
        "--torch-dtype",
        "bfloat16",
        "--unsloth-mixed-precision",
        "auto",
        "--learning-rate",
        "1e-5",
        "--warmup-steps",
        "0",
        "--seed",
        "3413",
        "--logging-steps",
        "1",
        "--save-steps",
        str(CHECKPOINT_INTERVAL_STEPS),
        "--save-total-limit",
        "4",
        "--max-gpu-memory-gib",
        str(job.max_gpu_memory_gib),
        "--cuda-memory-fraction",
        "0.88",
        "--cuda-alloc-conf",
        "expandable_segments:True,max_split_size_mb:256",
        "--causal-loss-mode",
        "active_chunked_no_upcast",
        "--causal-loss-chunk-tokens",
        "2048",
        "--checkpoint-max-shard-size",
        "512MB",
        "--checkpoint-safe-serialization",
        "true",
        "--full-finetuning",
        "--no-load-in-4bit",
        "--disable-unsloth-compile",
        "--disable-moe-triton",
        "--disable-flex-attention",
        "--disable-cce",
        "--no-packing",
        "--assistant-only-loss",
        "--loss-target",
        "final_assistant",
        "--final-assistant-preview-rows",
        "8",
        "--final-assistant-preview-max-chars",
        "2400",
        "--group-by-length",
        "--skip-merged-export",
        "--skip-gguf-export",
        "--checkpoint-presave-gc",
        "--checkpoint-presave-empty-cache",
        "--checkpoint-presave-disable-cuda-history",
        "--resume-torch-load-mmap",
    ]


DPO_BATCHED_MAX_SEQUENCE_LENGTH = 16384


def resolve_dpo_execution_mode(requested_mode: str, max_sequence_length: int) -> str:
    if requested_mode == "auto":
        return (
            "split_backward"
            if max_sequence_length > DPO_BATCHED_MAX_SEQUENCE_LENGTH
            else "batched"
        )
    return requested_mode


def dpo_lengths(max_sequence_length: int, effective_mode: str) -> tuple[int, int, int]:
    max_length = (
        min(DPO_BATCHED_MAX_SEQUENCE_LENGTH, max_sequence_length)
        if effective_mode == "batched"
        else max_sequence_length
    )
    max_completion = min(1536, max(256, max_length // 4))
    max_prompt = max(256, max_length - max_completion)
    return max_prompt, max_completion, max_length


def build_dpo_command(
    config: RunnerConfig,
    job: ValidatedJob,
    session_dir: Path,
    model_checkpoint: Path | str,
) -> list[str]:
    if config.mode == "fixture":
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_fake_stage",
            "--stage",
            "dpo",
            "--session-dir",
            str(session_dir),
            "--model-checkpoint",
            str(model_checkpoint),
        ]
        if config.fixture.fail_stage == "dpo":
            command.append("--fail")
        if config.fixture.sleep_seconds:
            command += ["--sleep", str(config.fixture.sleep_seconds)]
        return command

    effective_mode = resolve_dpo_execution_mode(
        job.dpo_execution_mode, job.max_sequence_length
    )
    max_prompt, max_completion, max_length = dpo_lengths(
        job.max_sequence_length, effective_mode
    )
    command = [
        python_executable(config.workspace_root),
        str(script_path(config.workspace_root, "train_dpo_session.py")),
        "--session-dir",
        str(session_dir),
        "--model-name",
        str(model_checkpoint),
        "--attn-implementation",
        "sdpa",
        "--device-map",
        "cuda:0",
        "--max-prompt-length",
        str(max_prompt),
        "--max-completion-length",
        str(max_completion),
        "--max-length",
        str(max_length),
        "--truncation-mode",
        "keep_end",
        "--dpo-execution-mode",
        effective_mode,
        "--requested-dpo-execution-mode",
        job.dpo_execution_mode,
        "--num-train-epochs",
        "1.0",
        "--per-device-train-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--learning-rate",
        "1e-6",
        "--warmup-steps",
        "0",
        "--weight-decay",
        "0.01",
        "--logging-steps",
        "1",
        "--save-steps",
        str(CHECKPOINT_INTERVAL_STEPS),
        "--save-total-limit",
        "4",
        "--optim",
        "adamw_8bit",
        "--seed",
        "3413",
        "--beta",
        "0.05",
        "--loss-type",
        "sigmoid",
        "--precompute-ref-batch-size",
        "1",
        "--dataset-num-proc",
        "1",
        "--precision",
        "auto",
        "--torch-dtype",
        "bfloat16",
        "--max-gpu-memory-gib",
        str(job.max_gpu_memory_gib),
        "--cuda-memory-fraction",
        "0.88",
        "--cuda-alloc-conf",
        "expandable_segments:True,max_split_size_mb:256",
        "--checkpoint-max-shard-size",
        "512MB",
        "--checkpoint-safe-serialization",
        "true",
        "--precompute-ref-log-probs",
        "--use-logits-to-keep",
        "--resume-warm-marker-path",
        str(session_dir / "metadata" / "resume_warm_marker.json"),
        "--checkpoint-save-marker-path",
        str(session_dir / "metadata" / "checkpoint_save_marker.json"),
        "--checkpoint-presave-gc",
        "--checkpoint-presave-empty-cache",
    ]
    return command


def build_rl_command(
    config: RunnerConfig,
    job: ValidatedJob,
    session_dir: Path,
    model_checkpoint: Path | str,
) -> list[str]:
    if config.mode == "fixture":
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_fake_stage",
            "--stage",
            "rl",
            "--session-dir",
            str(session_dir),
            "--model-checkpoint",
            str(model_checkpoint),
        ]
        if config.fixture.fail_stage == "rl":
            command.append("--fail")
        if config.fixture.sleep_seconds:
            command += ["--sleep", str(config.fixture.sleep_seconds)]
        return command

    return [
        python_executable(config.workspace_root),
        str(script_path(config.workspace_root, "train_rl_session.py")),
        "--session-dir",
        str(session_dir),
        "--model-name",
        str(model_checkpoint),
        "--attn-implementation",
        "sdpa",
        "--device-map",
        "cuda:0",
        "--max-length",
        str(job.max_sequence_length),
        "--num-train-epochs",
        str(job.rl_epochs),
        "--learning-rate",
        "5e-7",
        "--warmup-steps",
        "0",
        "--weight-decay",
        "0.01",
        "--logging-steps",
        "1",
        "--save-steps",
        "10,20,40,60",
        "--save-total-limit",
        "4",
        "--optim",
        "adamw_8bit",
        "--seed",
        "3413",
        "--clip-epsilon",
        str(job.rl_clip_epsilon),
        "--kl-beta",
        str(job.rl_kl_beta),
        "--precision",
        "auto",
        "--torch-dtype",
        "bfloat16",
        "--max-gpu-memory-gib",
        str(job.max_gpu_memory_gib),
        "--cuda-memory-fraction",
        "0.88",
        "--cuda-alloc-conf",
        "expandable_segments:True,max_split_size_mb:256",
        "--checkpoint-max-shard-size",
        "512MB",
        "--checkpoint-safe-serialization",
        "true",
        "--frozen-eval-max-sequences",
        "16",
    ]


def build_awr_command(
    config: RunnerConfig,
    job: ValidatedJob,
    session_dir: Path,
    model_checkpoint: Path | str,
) -> list[str]:
    if config.mode == "fixture":
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_fake_stage",
            "--stage",
            "awr",
            "--session-dir",
            str(session_dir),
            "--model-checkpoint",
            str(model_checkpoint),
        ]
        if config.fixture.fail_stage == "awr":
            command.append("--fail")
        if config.fixture.sleep_seconds:
            command += ["--sleep", str(config.fixture.sleep_seconds)]
        return command

    return [
        python_executable(config.workspace_root),
        str(script_path(config.workspace_root, "train_awr_session.py")),
        "--session-dir",
        str(session_dir),
        "--model-name",
        str(model_checkpoint),
        "--attn-implementation",
        "sdpa",
        "--device-map",
        "cuda:0",
        "--max-length",
        str(job.max_sequence_length),
        "--num-train-epochs",
        str(job.awr_epochs),
        "--learning-rate",
        "5e-7",
        "--warmup-steps",
        "0",
        "--weight-decay",
        "0.01",
        "--logging-steps",
        "1",
        "--save-steps",
        "10,20,40,60",
        "--save-total-limit",
        "4",
        "--optim",
        "adamw_8bit",
        "--seed",
        "3413",
        "--precision",
        "auto",
        "--torch-dtype",
        "bfloat16",
        "--max-gpu-memory-gib",
        str(job.max_gpu_memory_gib),
        "--cuda-memory-fraction",
        "0.88",
        "--cuda-alloc-conf",
        "expandable_segments:True,max_split_size_mb:256",
        "--checkpoint-max-shard-size",
        "512MB",
        "--checkpoint-safe-serialization",
        "true",
        "--frozen-eval-max-sequences",
        "16",
    ]


def verify_checkpoint(checkpoint: Path, config: RunnerConfig, stage: str) -> None:
    if not checkpoint.exists() or not checkpoint.is_dir():
        raise StageError(stage, f"checkpoint directory is missing: {checkpoint}")
    if config.mode == "fixture":
        marker = checkpoint / "FAKE_CHECKPOINT"
        if not marker.exists():
            raise StageError(stage, f"fixture checkpoint marker is missing: {marker}")
        return

    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise StageError(stage, f"checkpoint config.json is missing: {checkpoint}")
    has_weights = any(checkpoint.glob("*.safetensors")) or any(checkpoint.glob("*.bin"))
    has_index = (checkpoint / "model.safetensors.index.json").exists() or (
        checkpoint / "pytorch_model.bin.index.json"
    ).exists()
    if not has_weights and not has_index:
        raise StageError(stage, f"checkpoint has no recognized weight files: {checkpoint}")

    command = [
        python_executable(config.workspace_root),
        "-c",
        (
            "from transformers import AutoConfig, AutoTokenizer; "
            "import sys; p=sys.argv[1]; "
            "AutoConfig.from_pretrained(p, local_files_only=True); "
            "AutoTokenizer.from_pretrained(p, local_files_only=True); "
            "print('checkpoint_loadable')"
        ),
        str(checkpoint),
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0:
        error = (process.stderr or process.stdout).strip().splitlines()[-1:]
        raise StageError(stage, f"checkpoint metadata load failed: {' '.join(error)}")


def load_stage_metrics(session_dir: Path, stage: str) -> dict[str, Any]:
    metrics_path = session_dir / "metadata" / "train_metrics.json"
    if not metrics_path.is_file():
        raise StageError(stage, f"training metrics are missing: {metrics_path}")
    try:
        metrics = read_json(metrics_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StageError(
            stage, f"unable to read training metrics {metrics_path}: {exc}"
        ) from exc
    if not metrics:
        raise StageError(stage, f"training metrics are empty: {metrics_path}")
    return metrics


def run_sft_stage(config: RunnerConfig, job_dir: Path, job: ValidatedJob) -> StageOutput:
    if job.sft_input is None:
        raise StageError("sft", "SFT input is unavailable for a DPO-only job")
    stage = "sft"
    logs_dir = job_dir / "logs"
    label = f"contract_{job.output_checkpoint}_sft"
    log_path = logs_dir / "sft.log"
    session_dir = create_stage_session(config, job_dir, stage, job.sft_input.path, label, log_path)
    command = build_sft_command(config, job, session_dir)
    write_status(job_dir, "sft_running", "SFT training started", stage=stage)
    rc = run_streamed_command(command, log_path)
    if rc != 0:
        raise StageError(stage, f"SFT command failed with exit code {rc}; see {log_path}")
    checkpoint = session_dir / "artifacts" / "full_model"
    verify_checkpoint(checkpoint, config, stage)
    metrics = load_stage_metrics(session_dir, stage)
    return StageOutput(
        session_dir=session_dir,
        checkpoint=checkpoint,
        command=command,
        log_path=log_path,
        metrics=metrics,
    )


def run_dpo_stage(
    config: RunnerConfig,
    job_dir: Path,
    job: ValidatedJob,
    model_checkpoint: Path | str,
) -> StageOutput:
    stage = "dpo"
    logs_dir = job_dir / "logs"
    label = f"contract_{job.output_checkpoint}_dpo"
    log_path = logs_dir / "dpo.log"
    session_dir = create_stage_session(config, job_dir, stage, job.dpo_input.path, label, log_path)
    command = build_dpo_command(config, job, session_dir, model_checkpoint)
    write_status(job_dir, "dpo_running", "DPO training started", stage=stage)
    rc = run_streamed_command(command, log_path)
    if rc != 0:
        raise StageError(stage, f"DPO command failed with exit code {rc}; see {log_path}")
    checkpoint = session_dir / "artifacts" / "full_model"
    verify_checkpoint(checkpoint, config, stage)
    metrics = load_stage_metrics(session_dir, stage)
    return StageOutput(
        session_dir=session_dir,
        checkpoint=checkpoint,
        command=command,
        log_path=log_path,
        metrics=metrics,
    )


def run_rl_stage(
    config: RunnerConfig,
    job_dir: Path,
    job: ValidatedJob,
    model_checkpoint: Path | str,
) -> StageOutput:
    if job.rl_input is None:
        raise StageError("rl", "RL input is unavailable for this job")
    stage = "rl"
    logs_dir = job_dir / "logs"
    label = f"contract_{job.output_checkpoint}_rl"
    log_path = logs_dir / "rl.log"
    session_dir = create_stage_session(
        config, job_dir, stage, job.rl_input.path, label, log_path
    )
    command = build_rl_command(config, job, session_dir, model_checkpoint)
    write_status(job_dir, "rl_running", "Serialized RL training started", stage=stage)
    rc = run_streamed_command(command, log_path)
    if rc != 0:
        raise StageError(stage, f"RL command failed with exit code {rc}; see {log_path}")
    checkpoint = session_dir / "artifacts" / "full_model"
    verify_checkpoint(checkpoint, config, stage)
    metrics = load_stage_metrics(session_dir, stage)
    return StageOutput(
        session_dir=session_dir,
        checkpoint=checkpoint,
        command=command,
        log_path=log_path,
        metrics=metrics,
    )


def run_awr_stage(
    config: RunnerConfig,
    job_dir: Path,
    job: ValidatedJob,
    model_checkpoint: Path | str,
) -> StageOutput:
    if job.awr_input is None:
        raise StageError("awr", "AWR input is unavailable for this job")
    stage = "awr"
    logs_dir = job_dir / "logs"
    label = f"contract_{job.output_checkpoint}_awr"
    log_path = logs_dir / "awr.log"
    session_dir = create_stage_session(
        config, job_dir, stage, job.awr_input.path, label, log_path
    )
    command = build_awr_command(config, job, session_dir, model_checkpoint)
    write_status(
        job_dir,
        "awr_running",
        "Serialized repair-distance AWR training started",
        stage=stage,
    )
    rc = run_streamed_command(command, log_path)
    if rc != 0:
        raise StageError(stage, f"AWR command failed with exit code {rc}; see {log_path}")
    checkpoint = session_dir / "artifacts" / "full_model"
    verify_checkpoint(checkpoint, config, stage)
    metrics = load_stage_metrics(session_dir, stage)
    return StageOutput(
        session_dir=session_dir,
        checkpoint=checkpoint,
        command=command,
        log_path=log_path,
        metrics=metrics,
    )


def prepare_training_environment(config: RunnerConfig, job_dir: Path) -> None:
    """Release resident inference GPU memory before loading a training model."""
    stage = "prepare_training"
    write_status(
        job_dir,
        "preparing_training",
        "Stopping resident vLLM before training",
        stage=stage,
    )
    if config.mode == "fixture":
        if config.fixture.fail_stage == stage:
            raise StageError(stage, "fixture training preparation failure requested")
        atomic_write_json(
            job_dir / "training_environment.json",
            {"mode": "fixture", "serving_stopped": False},
        )
        return

    log_path = job_dir / "logs" / "prepare_training.log"
    stop_script = script_path(config.workspace_root, "stop_vllm_fullft_bf16_openai.sh")
    rc = run_streamed_command([str(stop_script)], log_path)
    if rc != 0:
        raise StageError(
            stage,
            f"vLLM stop command failed with exit code {rc}; see {log_path}",
        )
    atomic_write_json(
        job_dir / "training_environment.json",
        {"mode": "real", "serving_stopped": True, "log_path": str(log_path)},
    )


def deploy_fixture(job_dir: Path, job: ValidatedJob, final_checkpoint: Path, config: RunnerConfig) -> str:
    if config.fixture.fail_stage == "deploy":
        raise StageError("deploying", "fixture deployment failure requested")
    payload = {
        "served_model_name": job.served_model_name,
        "checkpoint": str(final_checkpoint),
        "endpoint": DEFAULT_ENDPOINT,
        "mode": "fixture",
        "assistant_reasoning": job.assistant_reasoning,
    }
    atomic_write_json(job_dir / "deployment.json", payload)
    return DEFAULT_ENDPOINT


def deploy_real(config: RunnerConfig, job_dir: Path, job: ValidatedJob, final_checkpoint: Path) -> str:
    log_path = job_dir / "logs" / "deploy.log"
    stop_script = script_path(config.workspace_root, "stop_vllm_fullft_bf16_openai.sh")
    start_script = script_path(config.workspace_root, "start_vllm_fullft_bf16_openai.sh")
    stop_command = [str(stop_script)]
    rc = run_streamed_command(stop_command, log_path)
    if rc != 0:
        raise StageError("deploying", f"vLLM stop command failed with exit code {rc}; see {log_path}")

    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": str(final_checkpoint),
            "SERVED_MODEL_NAME": job.served_model_name,
            "MAX_MODEL_LEN": "65536",
            "GPU_MEMORY_UTILIZATION": "0.70",
            "MAX_NUM_SEQS": "1",
            "MAX_NUM_BATCHED_TOKENS": "32768",
            "PORT": "8002",
            "READY_WAIT_SEC": "900",
            "ENABLE_THINKING": (
                "true" if job.assistant_reasoning == "required" else "false"
            ),
        }
    )
    start_command = [str(start_script)]
    rc = run_streamed_command(start_command, log_path, env=env)
    if rc != 0:
        raise StageError("deploying", f"vLLM start command failed with exit code {rc}; see {log_path}")
    return DEFAULT_ENDPOINT


def deploy(config: RunnerConfig, job_dir: Path, job: ValidatedJob, final_checkpoint: Path) -> str:
    write_status(job_dir, "deploying", "Deploying final checkpoint with vLLM", stage="deploying")
    if not job.deployment_enabled:
        raise StageError("deploying", "deployment.enabled=false is not supported by V1 contract validation")
    if config.mode == "fixture":
        return deploy_fixture(job_dir, job, final_checkpoint, config)
    return deploy_real(config, job_dir, job, final_checkpoint)


def health_check_fixture(config: RunnerConfig, job: ValidatedJob) -> dict[str, Any]:
    if config.fixture.fail_stage == "health_check":
        raise StageError("health_check", "fixture health-check failure requested")
    return {
        "passed": True,
        "latency_ms": 0,
        "assistant_reasoning": job.assistant_reasoning,
    }


def health_check_real(
    endpoint: str,
    model: str,
    assistant_reasoning: str = "disabled",
) -> dict[str, Any]:
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Think briefly, then return exactly: ok",
            }
        ],
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {
            "enable_thinking": assistant_reasoning == "required"
        },
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise StageError("health_check", f"health-check request failed: {exc}") from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    choices = data.get("choices") if isinstance(data, dict) else None
    content = ""
    reasoning = ""
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content") or ""
            reasoning = (
                message.get("reasoning_content")
                or message.get("reasoning")
                or message.get("thinking")
                or ""
            )
    if not isinstance(content, str) or not content.strip():
        raise StageError("health_check", "health-check response was empty")
    if assistant_reasoning == "required" and (
        not isinstance(reasoning, str) or not reasoning.strip()
    ):
        raise StageError(
            "health_check",
            "health-check response did not contain separate assistant reasoning",
        )
    return {
        "passed": True,
        "latency_ms": latency_ms,
        "assistant_reasoning": assistant_reasoning,
        "reasoning_chars": len(reasoning),
    }


def health_check(config: RunnerConfig, job_dir: Path, job: ValidatedJob, endpoint: str) -> dict[str, Any]:
    write_status(job_dir, "health_check", "Running deployment health check", stage="health_check")
    if config.mode == "fixture":
        return health_check_fixture(config, job)
    return health_check_real(
        endpoint,
        job.served_model_name,
        job.assistant_reasoning,
    )


def write_stage_summary(
    job_dir: Path,
    sft: StageOutput | None,
    dpo: StageOutput | None,
    rl: StageOutput | None = None,
    awr: StageOutput | None = None,
) -> None:
    payload: dict[str, Any] = {
        "sft": None,
        "dpo": None,
        "rl": None,
        "awr": None,
    }
    if sft is not None:
        payload["sft"] = {
            "session_dir": str(sft.session_dir),
            "checkpoint": str(sft.checkpoint),
            "log_path": str(sft.log_path),
            "command": sft.command,
            "metrics": sft.metrics,
        }
    if dpo is not None:
        payload["dpo"] = {
            "session_dir": str(dpo.session_dir),
            "checkpoint": str(dpo.checkpoint),
            "log_path": str(dpo.log_path),
            "command": dpo.command,
            "metrics": dpo.metrics,
        }
    if rl is not None:
        payload["rl"] = {
            "session_dir": str(rl.session_dir),
            "checkpoint": str(rl.checkpoint),
            "log_path": str(rl.log_path),
            "command": rl.command,
            "metrics": rl.metrics,
        }
    if awr is not None:
        payload["awr"] = {
            "session_dir": str(awr.session_dir),
            "checkpoint": str(awr.checkpoint),
            "log_path": str(awr.log_path),
            "command": awr.command,
            "metrics": awr.metrics,
        }
    atomic_write_json(job_dir / "stage_sessions.json", payload)


def move_terminal(jobs_root: Path, job_dir: Path, terminal: str) -> Path:
    target_root = jobs_root / ("completed" if terminal == "complete" else "failed")
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / job_dir.name
    if target.exists():
        raise RunnerError(f"terminal job directory already exists: {target}")
    os.replace(job_dir, target)
    fsync_dir(jobs_root / "running")
    fsync_dir(target_root)
    return target


def process_job(config: RunnerConfig, job_dir: Path, resumed_running: bool = False) -> Path:
    previous_state = read_state(job_dir)
    if resumed_running and previous_state in {
        "preparing_training",
        "sft_running",
        "dpo_running",
        "rl_running",
        "awr_running",
        "deploying",
        "health_check",
    }:
        raise StageError(
            previous_state,
            f"found abandoned running job in state {previous_state}; automatic ambiguous resume is disabled",
        )

    write_status(job_dir, "pending", "Job claimed by remote training runner", stage=None, pid=os.getpid())
    write_status(job_dir, "validating", "Validating job manifest and inputs", stage="validating", pid=os.getpid())
    job = validate_manifest(job_dir)

    prepare_training_environment(config, job_dir)
    sft: StageOutput | None = None
    dpo: StageOutput | None = None
    rl: StageOutput | None = None
    awr: StageOutput | None = None
    if job.awr_enabled:
        awr = run_awr_stage(config, job_dir, job, job.base_checkpoint)
    elif job.rl_enabled:
        rl = run_rl_stage(config, job_dir, job, job.base_checkpoint)
    else:
        sft = run_sft_stage(config, job_dir, job) if job.sft_enabled else None
        if sft is not None:
            # Persist the verified checkpoint before DPO so a failed bootstrap
            # can be resumed as an immutable DPO-only replacement job.
            write_stage_summary(job_dir, sft, None)
        dpo_model_checkpoint: Path | str = (
            sft.checkpoint if sft is not None else job.base_checkpoint
        )
        dpo = run_dpo_stage(config, job_dir, job, dpo_model_checkpoint)
    write_stage_summary(job_dir, sft, dpo, rl, awr)
    final_checkpoint = (awr or rl or dpo).checkpoint
    endpoint = deploy(config, job_dir, job, final_checkpoint)
    health = health_check(config, job_dir, job, endpoint)
    write_success_result(job_dir, job, sft, dpo, rl, awr, endpoint, health)
    write_status(job_dir, "complete", "Job complete", stage=None)
    return move_terminal(config.jobs_root, job_dir, "complete")


def fail_job(config: RunnerConfig, job_dir: Path, failed_stage: str, error: str) -> Path:
    with contextlib.suppress(Exception):
        write_failed_result(job_dir, failed_stage, error)
    with contextlib.suppress(Exception):
        write_status(job_dir, "failed", error, stage=failed_stage)
    if job_dir.parent.name == "running":
        return move_terminal(config.jobs_root, job_dir, "failed")
    return job_dir


def run_once(config: RunnerConfig) -> int:
    ensure_jobs_root(config.jobs_root)
    with single_job_lock(config.jobs_root):
        candidate = discover_candidate(config.jobs_root)
        if candidate is None:
            return 0
        try:
            job_dir, resumed = claim_job(config.jobs_root, candidate[0], candidate[1])
            try:
                process_job(config, job_dir, resumed_running=resumed)
                return 0
            except RunnerError as exc:
                failed_stage = getattr(exc, "failed_stage", "runner")
                fail_job(config, job_dir, failed_stage, str(exc))
                return 1
            except Exception as exc:  # Keep failed jobs explicit and terminal.
                fail_job(config, job_dir, "runner", f"unexpected runner error: {exc}")
                return 1
        except StageError:
            raise


def run_loop(config: RunnerConfig) -> int:
    while True:
        rc = run_once(config)
        if config.once:
            return rc
        time.sleep(config.poll_interval_seconds)


def fake_stage_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["sft", "dpo", "rl", "awr"])
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--model-checkpoint", default="")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.sleep:
        time.sleep(args.sleep)
    if args.fail:
        print(f"fixture {args.stage} failure requested", file=sys.stderr)
        return 17
    session_dir = Path(args.session_dir)
    checkpoint = session_dir / "artifacts" / "full_model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "FAKE_CHECKPOINT").write_text(f"{args.stage}\n", encoding="utf-8")
    atomic_write_json(checkpoint / "config.json", {"fixture": True, "stage": args.stage})
    atomic_write_json(session_dir / "metadata" / "train_metrics.json", {"stage": args.stage})
    print(f"fixture {args.stage} complete: {checkpoint}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote Hayabusa training jobs.")
    parser.add_argument("--jobs-root", required=True, help="Root containing incoming/running/completed/failed.")
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Training repository root.",
    )
    parser.add_argument("--once", action="store_true", help="Process at most one job and exit.")
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--mode", choices=["real", "fixture"], default="real")
    parser.add_argument("--fixture-mode", action="store_true", help="Alias for --mode fixture.")
    parser.add_argument(
        "--fixture-fail-stage",
        choices=["", "prepare_training", "sft", "dpo", "rl", "awr", "deploy", "health_check"],
        default="",
    )
    parser.add_argument("--fixture-sleep-seconds", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_fake_stage":
        return fake_stage_main(argv[1:])
    args = parse_args(argv)
    mode = "fixture" if args.fixture_mode else args.mode
    config = RunnerConfig(
        jobs_root=Path(args.jobs_root).expanduser().resolve(),
        workspace_root=Path(args.workspace_root).expanduser().resolve(),
        mode=mode,
        once=args.once,
        poll_interval_seconds=args.poll_interval_seconds,
        fixture=FixtureConfig(
            fail_stage=args.fixture_fail_stage,
            sleep_seconds=args.fixture_sleep_seconds,
        ),
    )
    try:
        return run_loop(config)
    except StageError as exc:
        print(str(exc), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
