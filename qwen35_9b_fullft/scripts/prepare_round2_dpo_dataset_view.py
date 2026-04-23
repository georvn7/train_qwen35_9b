#!/usr/bin/env python3
"""Build a chat-templated round-2 DPO dataset view with explicit token stats."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def summarize(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": int(mean(ordered)),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a chat-templated DPO JSONL view for round-2 Qwen3.5 9B training."
    )
    parser.add_argument("--input-jsonl", required=True, help="Raw conversational DPO JSONL path.")
    parser.add_argument("--output-jsonl", required=True, help="Prepared text DPO JSONL output path.")
    parser.add_argument(
        "--tokenizer-model",
        required=True,
        help="Tokenizer/model path used to render the chat template.",
    )
    parser.add_argument("--max-prompt-length", type=int, default=14848)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--max-length", type=int, default=16384)
    return parser.parse_args()


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Round 2 DPO Prepared Dataset Report",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- input_jsonl: `{report['input_jsonl']}`",
        f"- output_jsonl: `{report['output_jsonl']}`",
        f"- tokenizer_model: `{report['tokenizer_model']}`",
        f"- rows: `{report['rows']}`",
        f"- output_sha256: `{report['output_sha256']}`",
        "",
        "## Recipe Budget",
        f"- max_prompt_length: `{report['recipe']['max_prompt_length']}`",
        f"- max_completion_length: `{report['recipe']['max_completion_length']}`",
        f"- max_length: `{report['recipe']['max_length']}`",
        f"- truncation_mode: `{report['recipe']['truncation_mode']}`",
        "",
        "## Prompt Lengths",
        f"- summary: `{report['prompt_tokens']}`",
        f"- rows_over_prompt_budget: `{report['budget_counts']['prompt_over_budget']}`",
        "",
        "## Chosen Lengths",
        f"- summary: `{report['chosen_tokens']}`",
        f"- rows_over_completion_budget: `{report['budget_counts']['chosen_over_budget']}`",
        "",
        "## Rejected Lengths",
        f"- summary: `{report['rejected_tokens']}`",
        f"- rows_over_completion_budget: `{report['budget_counts']['rejected_over_budget']}`",
        "",
        "## Side Lengths",
        f"- chosen_side_summary: `{report['chosen_side_tokens']}`",
        f"- rejected_side_summary: `{report['rejected_side_tokens']}`",
        f"- rows_over_side_budget: `{report['budget_counts']['pair_side_over_budget']}`",
        "",
        "## Decision",
        "- Use this prepared chat-templated view as the canonical DPO dataset input.",
        "- Keep prompt budget at 14848 and completion budget at 1536.",
        "- Start DPO from the finished round-2 full-FT model, not from scratch.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    meta_json = output_jsonl.with_suffix(output_jsonl.suffix + ".meta.json")
    meta_md = output_jsonl.with_suffix(output_jsonl.suffix + ".meta.md")

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_jsonl}")

    from transformers import AutoTokenizer
    from trl.data_utils import maybe_apply_chat_template

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.tokenizer_model).expanduser()))

    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    chosen_side_lengths: list[int] = []
    rejected_side_lengths: list[int] = []
    prompt_over_budget = 0
    chosen_over_budget = 0
    rejected_over_budget = 0
    pair_side_over_budget = 0

    prepared_rows: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for row_index, raw in enumerate(handle, start=1):
            row = json.loads(raw)
            rendered = maybe_apply_chat_template(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                },
                tokenizer=tokenizer,
            )

            prompt = rendered["prompt"]
            chosen = rendered["chosen"]
            rejected = rendered["rejected"]

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
            rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

            eos_len = 1 if tokenizer.eos_token_id is not None else 0
            prompt_len = len(prompt_ids)
            chosen_len = len(chosen_ids) + eos_len
            rejected_len = len(rejected_ids) + eos_len
            chosen_side_len = prompt_len + chosen_len
            rejected_side_len = prompt_len + rejected_len

            prompt_lengths.append(prompt_len)
            chosen_lengths.append(chosen_len)
            rejected_lengths.append(rejected_len)
            chosen_side_lengths.append(chosen_side_len)
            rejected_side_lengths.append(rejected_side_len)

            if prompt_len > args.max_prompt_length:
                prompt_over_budget += 1
            if chosen_len > args.max_completion_length:
                chosen_over_budget += 1
            if rejected_len > args.max_completion_length:
                rejected_over_budget += 1
            if max(chosen_side_len, rejected_side_len) > args.max_length:
                pair_side_over_budget += 1

            prepared_row: dict[str, Any] = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "row_index": row_index,
            }
            if "meta" in row:
                prepared_row["meta"] = row["meta"]
            prepared_rows.append(prepared_row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in prepared_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "created_at_utc": utc_now(),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "tokenizer_model": str(Path(args.tokenizer_model).expanduser().resolve()),
        "rows": len(prepared_rows),
        "recipe": {
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "max_length": args.max_length,
            "truncation_mode": "keep_end",
        },
        "prompt_tokens": summarize(prompt_lengths),
        "chosen_tokens": summarize(chosen_lengths),
        "rejected_tokens": summarize(rejected_lengths),
        "chosen_side_tokens": summarize(chosen_side_lengths),
        "rejected_side_tokens": summarize(rejected_side_lengths),
        "budget_counts": {
            "prompt_over_budget": prompt_over_budget,
            "chosen_over_budget": chosen_over_budget,
            "rejected_over_budget": rejected_over_budget,
            "pair_side_over_budget": pair_side_over_budget,
        },
        "output_sha256": file_sha256(output_jsonl),
    }

    meta_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    meta_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"Prepared DPO dataset: {output_jsonl}")
    print(f"Rows: {len(prepared_rows)}")
    print(f"Prompt over budget: {prompt_over_budget}")
    print(f"Chosen over budget: {chosen_over_budget}")
    print(f"Rejected over budget: {rejected_over_budget}")
    print(f"Side over budget: {pair_side_over_budget}")


if __name__ == "__main__":
    main()
