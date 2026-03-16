#!/usr/bin/env python3
"""Run short full-FT quality probes and rank candidates by holdout eval loss."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run(cmd: list[str], cwd: Path) -> tuple[int, float]:
    started = time.time()
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return proc.returncode, time.time() - started


def sanitize_label_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def parse_lrs(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise ValueError("At least one learning rate is required.")
    return values


def detect_new_session(runs_root: Path, before: set[str], label: str) -> Path:
    after = {p.name for p in runs_root.iterdir() if p.is_dir()}
    created = sorted(after - before)
    if created:
        return runs_root / created[-1]
    candidates = sorted(runs_root.glob(f"*{label}"))
    if not candidates:
        raise RuntimeError(f"Failed to detect session for label={label}")
    return candidates[-1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_session(session_dir: Path, lr: float, rc: int, runtime_s: float) -> dict[str, Any]:
    metadata_dir = session_dir / "metadata"
    train_metrics = load_json(metadata_dir / "train_metrics.json") if (metadata_dir / "train_metrics.json").exists() else {}
    log_history = []
    history_path = metadata_dir / "train_log_history.json"
    if history_path.exists():
        payload = load_json(history_path)
        if isinstance(payload, dict) and isinstance(payload.get("log_history"), list):
            log_history = payload["log_history"]

    eval_losses: list[float] = []
    train_losses: list[float] = []
    grad_norms: list[float] = []
    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        eval_loss = entry.get("eval_loss")
        if isinstance(eval_loss, (float, int)):
            eval_losses.append(float(eval_loss))
        train_loss = entry.get("loss")
        if isinstance(train_loss, (float, int)):
            train_losses.append(float(train_loss))
        grad_norm = entry.get("grad_norm")
        if isinstance(grad_norm, (float, int)):
            grad_norms.append(float(grad_norm))

    return {
        "session_dir": str(session_dir),
        "learning_rate": lr,
        "return_code": rc,
        "runtime_s": round(runtime_s, 3),
        "train_runtime_s": train_metrics.get("train_runtime"),
        "train_loss_final": train_metrics.get("train_loss"),
        "eval_loss_first": eval_losses[0] if eval_losses else None,
        "eval_loss_best": min(eval_losses) if eval_losses else None,
        "eval_loss_last": eval_losses[-1] if eval_losses else None,
        "train_loss_first_log": train_losses[0] if train_losses else None,
        "train_loss_last_log": train_losses[-1] if train_losses else None,
        "grad_norm_last_log": grad_norms[-1] if grad_norms else None,
        "num_eval_points": len(eval_losses),
        "num_train_log_points": len(train_losses),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("results", [])
    ranked_eval = [row for row in rows if row.get("eval_loss_best") is not None and row.get("return_code") == 0]
    ranked_eval.sort(key=lambda x: x["eval_loss_best"])
    ranked_train = [row for row in rows if row.get("return_code") == 0 and row.get("train_loss_last_log") is not None]
    ranked_train.sort(key=lambda x: x["train_loss_last_log"])

    lines = [
        "# Qwen3.5 9B Quality Probe",
        "",
        f"- Generated at: `{payload['created_at_utc']}`",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Model: `{payload['model_name']}`",
        f"- Max seq length: `{payload['max_seq_length']}`",
        f"- Max steps: `{payload['max_steps']}`",
        f"- Holdout ratio: `{payload['eval_holdout_ratio']}`",
        "",
        "## Ranked by Best Eval Loss",
        "",
    ]
    if ranked_eval:
        for idx, row in enumerate(ranked_eval, start=1):
            lines.append(
                f"{idx}. lr=`{row['learning_rate']}` "
                f"best_eval_loss=`{row['eval_loss_best']}` "
                f"last_eval_loss=`{row['eval_loss_last']}` "
                f"session=`{row['session_dir']}`"
            )
    else:
        lines.append("No successful probe run with eval loss found.")

    lines.extend(["", "## Ranked by Last Train Loss", ""])
    if ranked_train:
        for idx, row in enumerate(ranked_train, start=1):
            lines.append(
                f"{idx}. lr=`{row['learning_rate']}` "
                f"train_loss_first=`{row['train_loss_first_log']}` "
                f"train_loss_last=`{row['train_loss_last_log']}` "
                f"session=`{row['session_dir']}`"
            )
    else:
        lines.append("No successful probe run with train-loss logs found.")

    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| lr | rc | best eval | last eval | final train loss | last train loss | eval points | session |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('learning_rate')} | "
            f"{row.get('return_code')} | "
            f"{row.get('eval_loss_best')} | "
            f"{row.get('eval_loss_last')} | "
            f"{row.get('train_loss_final')} | "
            f"{row.get('train_loss_last_log')} | "
            f"{row.get('num_eval_points')} | "
            f"{row.get('session_dir')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short holdout-eval quality probes for Qwen3.5 9B full FT.")
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--dataset-root",
        default="qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--label-prefix", default="quality_probe32k_qwen35_9b_instruct")
    parser.add_argument("--learning-rates", default="5e-6,1e-5,2e-5")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-holdout-ratio", type=float, default=0.1)
    parser.add_argument("--eval-max-samples", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3413)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lrs = parse_lrs(args.learning_rates)

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    runs_root = workspace_root / "qwen35_9b_fullft" / "runs"
    scripts_dir = workspace_root / "qwen35_9b_fullft" / "scripts"
    docs_dir = workspace_root / "qwen35_9b_fullft" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (workspace_root / dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    stamp = utc_now_compact()
    result_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else docs_dir / f"quality_probe_{stamp}.json"
    )
    result_md = (
        Path(args.output_md).expanduser().resolve()
        if args.output_md
        else docs_dir / f"quality_probe_{stamp}.md"
    )

    results: list[dict[str, Any]] = []
    for lr in lrs:
        lr_tag = sanitize_label_part(f"{lr:.2e}".replace("+", ""))
        label = sanitize_label_part(f"{args.label_prefix}_lr_{lr_tag}")

        before = {p.name for p in runs_root.iterdir() if p.is_dir()} if runs_root.exists() else set()
        create_cmd = [
            sys.executable,
            str(scripts_dir / "create_session.py"),
            "--workspace-root",
            str(workspace_root),
            "--dataset-root",
            str(dataset_root),
            "--label",
            label,
            "--notes",
            (
                "Short quality probe run "
                f"for Qwen3.5-9B-Base full FT, lr={lr}"
            ),
        ]
        rc_create, _ = run(create_cmd, workspace_root)
        if rc_create != 0:
            result = {
                "session_dir": "",
                "learning_rate": lr,
                "return_code": rc_create,
                "runtime_s": None,
                "error": "create_session_failed",
            }
            results.append(result)
            if not args.continue_on_error:
                break
            continue

        session_dir = detect_new_session(runs_root, before, label)
        train_cmd = [
            sys.executable,
            str(scripts_dir / "train_session.py"),
            "--session-dir",
            str(session_dir),
            "--model-name",
            args.model_name,
            "--max-seq-length",
            str(args.max_seq_length),
            "--truncation-side",
            "left",
            "--attn-implementation",
            args.attn_implementation,
            "--device-map",
            args.device_map,
            "--max-steps",
            str(args.max_steps),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--per-device-eval-batch-size",
            str(args.per_device_eval_batch_size),
            "--dataset-num-proc",
            str(args.dataset_num_proc),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--gradient-checkpointing",
            "true",
            "--precision",
            "auto",
            "--torch-dtype",
            "bfloat16",
            "--learning-rate",
            str(lr),
            "--warmup-steps",
            str(args.warmup_steps),
            "--weight-decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
            "--logging-steps",
            "1",
            "--save-strategy",
            "no",
            "--save-total-limit",
            "1",
            "--max-gpu-memory-gib",
            "112",
            "--full-finetuning",
            "--no-load-in-4bit",
            "--disable-unsloth-compile",
            "--disable-moe-triton",
            "--disable-flex-attention",
            "--disable-cce",
            "--no-packing",
            "--assistant-only-loss",
            "--group-by-length",
            "--skip-final-save",
            "--skip-merged-export",
            "--skip-gguf-export",
        ]
        if args.eval_holdout_ratio > 0:
            train_cmd.extend(
                [
                    "--eval-holdout-ratio",
                    str(args.eval_holdout_ratio),
                    "--eval-strategy",
                    "steps",
                    "--eval-steps",
                    str(args.eval_steps),
                    "--eval-max-samples",
                    str(args.eval_max_samples),
                    "--metric-for-best-model",
                    "eval_loss",
                ]
            )
        if args.max_samples > 0:
            train_cmd.extend(["--max-samples", str(args.max_samples)])

        rc_train, runtime_s = run(train_cmd, workspace_root)
        result = summarize_session(session_dir, lr, rc_train, runtime_s)
        results.append(result)
        if rc_train != 0 and not args.continue_on_error:
            break

    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workspace_root": str(workspace_root),
        "dataset_root": str(dataset_root),
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "eval_holdout_ratio": args.eval_holdout_ratio,
        "eval_max_samples": args.eval_max_samples,
        "eval_steps": args.eval_steps,
        "learning_rates": lrs,
        "results": results,
    }
    result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(result_md, payload)

    successful_eval = [row for row in results if row.get("return_code") == 0 and row.get("eval_loss_best") is not None]
    successful_train = [row for row in results if row.get("return_code") == 0 and row.get("train_loss_last_log") is not None]
    if successful_eval:
        best_eval = sorted(successful_eval, key=lambda x: x["eval_loss_best"])[0]
        print(
            "Best candidate by eval:",
            f"lr={best_eval['learning_rate']},",
            f"best_eval_loss={best_eval['eval_loss_best']},",
            f"session={best_eval['session_dir']}",
        )
    if successful_train:
        best_train = sorted(successful_train, key=lambda x: x["train_loss_last_log"])[0]
        print(
            "Best candidate by train loss:",
            f"lr={best_train['learning_rate']},",
            f"last_train_loss={best_train['train_loss_last_log']},",
            f"session={best_train['session_dir']}",
        )
    if not successful_eval and not successful_train:
        print("No successful run with eval or train-loss metrics was produced.")
    print(f"Wrote summary JSON: {result_json}")
    print(f"Wrote summary MD:   {result_md}")


if __name__ == "__main__":
    main()
