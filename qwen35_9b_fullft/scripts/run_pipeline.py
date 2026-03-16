#!/usr/bin/env python3
"""One-command workflow: create session -> train -> optional package for Ollama."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full fine-tuning pipeline for one datasets_* folder."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset folder or JSONL file path (relative or absolute).",
    )
    parser.add_argument("--label", default="run")
    parser.add_argument("--notes", default="")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument(
        "--truncation-side",
        default="left",
        choices=["right", "left"],
        help="How to truncate overlength samples. 'right' keeps the start.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--hf-cache-dir",
        default="",
        help="Shared Hugging Face cache path. Default is workspace-local cache.",
    )
    parser.add_argument(
        "--python-headers-root",
        default="",
        help="Optional local python header include root (e.g. .local_py312dev/usr/include).",
    )
    parser.add_argument(
        "--triton-ptxas-path",
        default="auto",
        help="Path to ptxas used by Triton JIT. 'auto' prefers /usr/local/cuda/bin/ptxas.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        default=-1.0,
        help=(
            "Hard-stop GPU memory guard passed to train_session.py. "
            "-1=auto (110 GiB for Qwen3.5-9B), 0=disabled."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=4,
        help="Max parallel dataset tokenization workers used by SFTTrainer.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
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
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype passed to FastLanguageModel.from_pretrained.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Optional compute dtype override for 4-bit quantization kernels.",
    )
    parser.add_argument(
        "--unsloth-mixed-precision",
        default="auto",
        choices=["auto", "float32", "bfloat16"],
        help="Optional UNSLOTH_MIXED_PRECISION override.",
    )
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=False)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--full-finetuning", dest="full_finetuning", action="store_true", default=True)
    parser.add_argument("--no-full-finetuning", dest="full_finetuning", action="store_false")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--disable-unsloth-compile", action="store_true", default=True)
    parser.add_argument("--enable-unsloth-compile", dest="disable_unsloth_compile", action="store_false")
    parser.add_argument("--disable-moe-triton", action="store_true", default=True)
    parser.add_argument("--enable-moe-triton", dest="disable_moe_triton", action="store_false")
    parser.add_argument(
        "--disable-flex-attention",
        action="store_true",
        default=True,
        help="Set UNSLOTH_ENABLE_FLEX_ATTENTION=0 during train_session.",
    )
    parser.add_argument(
        "--enable-flex-attention",
        dest="disable_flex_attention",
        action="store_false",
    )
    parser.add_argument(
        "--disable-cce",
        action="store_true",
        default=True,
        help="Set UNSLOTH_ENABLE_CCE=0 during train_session.",
    )
    parser.add_argument(
        "--enable-cce",
        dest="disable_cce",
        action="store_false",
    )
    parser.add_argument("--packing", action="store_true", default=False)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument("--assistant-only-loss", action="store_true", default=True)
    parser.add_argument("--no-assistant-only-loss", dest="assistant_only_loss", action="store_false")
    parser.add_argument("--group-by-length", action="store_true", default=True)
    parser.add_argument("--no-group-by-length", dest="group_by_length", action="store_false")
    parser.add_argument("--skip-merged-export", action="store_true")
    parser.add_argument("--skip-gguf-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--ollama-create", action="store_true")
    parser.add_argument(
        "--ollama-model-prefix",
        default="qwen35-9b-ft",
        help="Final Ollama tag will be <prefix>:<session_id>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    scripts_dir = workspace_root / "qwen35_9b_fullft" / "scripts"
    runs_root = workspace_root / "qwen35_9b_fullft" / "runs"

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = (workspace_root / dataset_root).resolve()

    before = {p.name for p in runs_root.iterdir() if p.is_dir()} if runs_root.exists() else set()

    create_cmd = [
        sys.executable,
        str(scripts_dir / "create_session.py"),
        "--workspace-root",
        str(workspace_root),
        "--dataset-root",
        str(dataset_root),
        "--label",
        args.label,
        "--notes",
        args.notes,
    ]
    run(create_cmd, workspace_root)

    after = {p.name for p in runs_root.iterdir() if p.is_dir()}
    new_sessions = sorted(after - before)
    if not new_sessions:
        raise RuntimeError("Failed to detect new session directory.")
    session_id = new_sessions[-1]
    session_dir = runs_root / session_id

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
        args.truncation_side,
        "--attn-implementation",
        args.attn_implementation,
        "--device-map",
        args.device_map,
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--max-steps",
        str(args.max_steps),
        "--max-gpu-memory-gib",
        str(args.max_gpu_memory_gib),
        "--max-samples",
        str(args.max_samples),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--dataset-num-proc",
        str(args.dataset_num_proc),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--gradient-checkpointing",
        args.gradient_checkpointing,
        "--precision",
        args.precision,
        "--torch-dtype",
        args.torch_dtype,
        "--bnb-4bit-compute-dtype",
        args.bnb_4bit_compute_dtype,
        "--unsloth-mixed-precision",
        args.unsloth_mixed_precision,
        "--learning-rate",
        str(args.learning_rate),
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    if args.load_in_4bit:
        train_cmd.append("--load-in-4bit")
    else:
        train_cmd.append("--no-load-in-4bit")
    if args.full_finetuning:
        train_cmd.append("--full-finetuning")
    else:
        train_cmd.append("--no-full-finetuning")
    if args.hf_cache_dir:
        train_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.python_headers_root:
        train_cmd.extend(["--python-headers-root", args.python_headers_root])
    if args.triton_ptxas_path:
        train_cmd.extend(["--triton-ptxas-path", args.triton_ptxas_path])
    if args.disable_unsloth_compile:
        train_cmd.append("--disable-unsloth-compile")
    else:
        train_cmd.append("--enable-unsloth-compile")
    if args.disable_moe_triton:
        train_cmd.append("--disable-moe-triton")
    else:
        train_cmd.append("--enable-moe-triton")
    if args.disable_flex_attention:
        train_cmd.append("--disable-flex-attention")
    else:
        train_cmd.append("--enable-flex-attention")
    if args.disable_cce:
        train_cmd.append("--disable-cce")
    else:
        train_cmd.append("--enable-cce")
    if args.packing:
        train_cmd.append("--packing")
    else:
        train_cmd.append("--no-packing")
    if args.assistant_only_loss:
        train_cmd.append("--assistant-only-loss")
    else:
        train_cmd.append("--no-assistant-only-loss")
    if args.group_by_length:
        train_cmd.append("--group-by-length")
    else:
        train_cmd.append("--no-group-by-length")
    if args.skip_merged_export:
        train_cmd.append("--skip-merged-export")
    if args.skip_gguf_export:
        train_cmd.append("--skip-gguf-export")
    if args.dry_run:
        train_cmd.append("--dry-run")
    run(train_cmd, workspace_root)

    if args.skip_ollama or args.dry_run or args.skip_gguf_export:
        print(f"Pipeline finished for session: {session_id}")
        print(f"Session dir: {session_dir}")
        return

    ollama_tag = f"{args.ollama_model_prefix}:{session_id}"
    modelfile_cmd = [
        sys.executable,
        str(scripts_dir / "make_ollama_modelfile.py"),
        "--session-dir",
        str(session_dir),
        "--model-name",
        ollama_tag,
    ]
    if args.ollama_create:
        modelfile_cmd.append("--create")
    run(modelfile_cmd, workspace_root)

    print(f"Pipeline finished for session: {session_id}")
    print(f"Session dir: {session_dir}")
    print(f"Ollama tag: {ollama_tag}")


if __name__ == "__main__":
    main()
