#!/usr/bin/env python3
"""Build canonical round-2 continuation datasets from trajectory bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class BucketStats:
    files: int = 0
    rows: int = 0
    thinking_fields_removed: int = 0
    token_lengths: list[int] = field(default_factory=list)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge round-2 SFT and DPO datasets into canonical local JSONL files."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing the unpacked trajectory bundles.",
    )
    parser.add_argument(
        "--sft-output",
        required=True,
        help="Output JSONL path for merged SFT rows.",
    )
    parser.add_argument(
        "--dpo-output",
        required=True,
        help="Output JSONL path for merged DPO rows.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3.5-9B",
        help="Tokenizer model used for token-length analysis.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32768,
        help="Reference truncation threshold for reporting.",
    )
    return parser.parse_args()


def is_ignored_path(path: Path) -> bool:
    ignored_parts = {"__MACOSX"}
    return any(part in ignored_parts for part in path.parts)


def list_target_files(root: Path, filename: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(filename)
        if path.is_file() and not is_ignored_path(path)
    )


def strip_assistant_thinking(messages: list[dict[str, Any]]) -> int:
    removed = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "thinking" in message:
            del message["thinking"]
            removed += 1
    return removed


def render_messages(messages: list[dict[str, Any]], tokenizer: Any) -> int:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * q)
    return sorted_values[index]


def summarize_lengths(lengths: list[int], max_seq_length: int) -> dict[str, Any]:
    ordered = sorted(lengths)
    if not ordered:
        return {
            "rows": 0,
            "min": 0,
            "mean": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            f"rows_gt_{max_seq_length}": 0,
        }
    return {
        "rows": len(ordered),
        "min": ordered[0],
        "mean": int(mean(ordered)),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        f"rows_gt_{max_seq_length}": sum(value > max_seq_length for value in ordered),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_reports_markdown(report: dict[str, Any]) -> str:
    max_seq_length = report["max_seq_length"]
    sft_over_key = f"rows_gt_{max_seq_length}"
    dpo_over_key = f"pairs_with_side_gt_{max_seq_length}"
    lines = [
        "# Round 2 Dataset Prep Report",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- input_root: `{report['input_root']}`",
        f"- tokenizer_model: `{report['tokenizer_model']}`",
        f"- max_seq_length_reference: `{max_seq_length}`",
        "",
        "## Outputs",
        f"- SFT: `{report['outputs']['sft_path']}`",
        f"- SFT rows: `{report['outputs']['sft_rows']}`",
        f"- SFT sha256: `{report['outputs']['sft_sha256']}`",
        f"- DPO: `{report['outputs']['dpo_path']}`",
        f"- DPO rows: `{report['outputs']['dpo_rows']}`",
        f"- DPO sha256: `{report['outputs']['dpo_sha256']}`",
        "",
        "## SFT Inputs",
        f"- `train_run_sft.jsonl` files: `{report['inputs']['train_run_sft']['files']}`",
        f"- `train_run_sft.jsonl` rows: `{report['inputs']['train_run_sft']['rows']}`",
        f"- `train_dbg_sft.jsonl` files: `{report['inputs']['train_dbg_sft']['files']}`",
        f"- `train_dbg_sft.jsonl` rows: `{report['inputs']['train_dbg_sft']['rows']}`",
        f"- combined SFT rows: `{report['inputs']['combined_sft']['rows']}`",
        "",
        "## SFT Truncation Stats",
        f"- `train_run_sft` rows over `{max_seq_length}`: `{report['token_stats']['train_run_sft'][sft_over_key]}`",
        f"- `train_dbg_sft` rows over `{max_seq_length}`: `{report['token_stats']['train_dbg_sft'][sft_over_key]}`",
        f"- combined SFT rows over `{max_seq_length}`: `{report['token_stats']['combined_sft'][sft_over_key]}`",
        f"- combined SFT `p95`: `{report['token_stats']['combined_sft']['p95']}`",
        f"- combined SFT `max`: `{report['token_stats']['combined_sft']['max']}`",
        "",
        "## DPO Inputs",
        f"- `train_dbg_dpo.jsonl` files: `{report['inputs']['train_dbg_dpo']['files']}`",
        f"- `train_dbg_dpo.jsonl` rows: `{report['inputs']['train_dbg_dpo']['rows']}`",
        "",
        "## DPO Length Stats",
        f"- DPO pairs with chosen/rejected side over `{max_seq_length}`: `{report['token_stats']['train_dbg_dpo'][dpo_over_key]}`",
        f"- DPO `p95` max-side length: `{report['token_stats']['train_dbg_dpo']['p95']}`",
        f"- DPO max-side length: `{report['token_stats']['train_dbg_dpo']['max']}`",
        "",
        "## Decision",
        "- Keep overlength SFT rows in the merged continuation dataset.",
        "- Use left truncation (`truncation_side=left`) in training so the newest context and final assistant target are preserved.",
        "- Keep DPO as a separate second-stage dataset; do not fold it into the first continuation SFT run.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    sft_output = Path(args.sft_output).expanduser().resolve()
    dpo_output = Path(args.dpo_output).expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    run_sft_files = list_target_files(input_root, "train_run_sft.jsonl")
    dbg_sft_files = list_target_files(input_root, "train_dbg_sft.jsonl")
    dbg_dpo_files = list_target_files(input_root, "train_dbg_dpo.jsonl")

    stats = {
        "train_run_sft": BucketStats(files=len(run_sft_files)),
        "train_dbg_sft": BucketStats(files=len(dbg_sft_files)),
        "train_dbg_dpo": BucketStats(files=len(dbg_dpo_files)),
        "combined_sft": BucketStats(),
    }

    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []

    for bucket_name, files in (
        ("train_run_sft", run_sft_files),
        ("train_dbg_sft", dbg_sft_files),
    ):
        bucket = stats[bucket_name]
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    row = json.loads(raw)
                    messages = row["messages"]
                    bucket.rows += 1
                    removed = strip_assistant_thinking(messages)
                    bucket.thinking_fields_removed += removed
                    stats["combined_sft"].thinking_fields_removed += removed
                    token_length = render_messages(messages, tokenizer)
                    bucket.token_lengths.append(token_length)
                    stats["combined_sft"].token_lengths.append(token_length)
                    stats["combined_sft"].rows += 1
                    sft_rows.append({"messages": messages})

    for path in dbg_dpo_files:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                row = json.loads(raw)
                prompt = row["prompt"]
                chosen = row["chosen"]
                rejected = row["rejected"]
                removed = 0
                removed += strip_assistant_thinking(prompt)
                removed += strip_assistant_thinking(chosen)
                removed += strip_assistant_thinking(rejected)
                stats["train_dbg_dpo"].thinking_fields_removed += removed
                max_side_tokens = max(
                    render_messages(prompt + chosen, tokenizer),
                    render_messages(prompt + rejected, tokenizer),
                )
                stats["train_dbg_dpo"].rows += 1
                stats["train_dbg_dpo"].token_lengths.append(max_side_tokens)
                dpo_rows.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "meta": row.get("meta", {}),
                    }
                )

    write_jsonl(sft_output, sft_rows)
    write_jsonl(dpo_output, dpo_rows)

    report = {
        "created_at_utc": utc_now(),
        "input_root": str(input_root),
        "tokenizer_model": args.tokenizer_model,
        "max_seq_length": args.max_seq_length,
        "inputs": {
            "train_run_sft": {
                "files": stats["train_run_sft"].files,
                "rows": stats["train_run_sft"].rows,
                "thinking_fields_removed": stats["train_run_sft"].thinking_fields_removed,
            },
            "train_dbg_sft": {
                "files": stats["train_dbg_sft"].files,
                "rows": stats["train_dbg_sft"].rows,
                "thinking_fields_removed": stats["train_dbg_sft"].thinking_fields_removed,
            },
            "combined_sft": {
                "files": stats["train_run_sft"].files + stats["train_dbg_sft"].files,
                "rows": stats["combined_sft"].rows,
                "thinking_fields_removed": stats["combined_sft"].thinking_fields_removed,
            },
            "train_dbg_dpo": {
                "files": stats["train_dbg_dpo"].files,
                "rows": stats["train_dbg_dpo"].rows,
                "thinking_fields_removed": stats["train_dbg_dpo"].thinking_fields_removed,
            },
        },
        "token_stats": {
            "train_run_sft": summarize_lengths(
                stats["train_run_sft"].token_lengths, args.max_seq_length
            ),
            "train_dbg_sft": summarize_lengths(
                stats["train_dbg_sft"].token_lengths, args.max_seq_length
            ),
            "combined_sft": summarize_lengths(
                stats["combined_sft"].token_lengths, args.max_seq_length
            ),
            "train_dbg_dpo": {
                **summarize_lengths(
                    stats["train_dbg_dpo"].token_lengths, args.max_seq_length
                ),
                f"pairs_with_side_gt_{args.max_seq_length}": sum(
                    value > args.max_seq_length
                    for value in stats["train_dbg_dpo"].token_lengths
                ),
            },
        },
        "outputs": {
            "sft_path": str(sft_output),
            "sft_rows": len(sft_rows),
            "sft_sha256": file_sha256(sft_output),
            "dpo_path": str(dpo_output),
            "dpo_rows": len(dpo_rows),
            "dpo_sha256": file_sha256(dpo_output),
        },
    }

    report_json_path = sft_output.with_suffix(".meta.json")
    report_md_path = sft_output.with_suffix(".meta.md")
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md_path.write_text(build_reports_markdown(report), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote SFT dataset: {sft_output}")
    print(f"Wrote DPO dataset: {dpo_output}")
    print(f"Wrote report: {report_json_path}")


if __name__ == "__main__":
    main()
