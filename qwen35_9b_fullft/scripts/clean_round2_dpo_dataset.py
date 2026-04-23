#!/usr/bin/env python3
"""Clean round-2 DPO dataset by dropping invalid preference rows and normalizing schema fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "action_subject",
    "action_type",
    "breakpoints",
    "invocation",
    "line_number",
    "motivation",
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the round-2 conversational DPO dataset into a canonical training file."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def normalize_response(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = json.loads(raw_text)
    except Exception as exc:
        return None, f"json_parse_error:{type(exc).__name__}"

    if not isinstance(obj, dict):
        return None, "json_not_object"

    if "breakpoints" not in obj:
        obj["breakpoints"] = []

    for field in REQUIRED_FIELDS:
        if field not in obj:
            return None, f"missing_required_field:{field}"

    action_type = str(obj.get("action_type", "")).strip()
    action_subject = str(obj.get("action_subject", "")).strip()
    if not action_type:
        return None, "empty_action_type"
    if not action_subject:
        return None, "empty_action_subject"
    if not isinstance(obj.get("breakpoints"), list):
        return None, "breakpoints_not_list"
    return obj, None


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Round 2 DPO Cleaned Dataset Report",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- input_jsonl: `{report['input_jsonl']}`",
        f"- output_jsonl: `{report['output_jsonl']}`",
        f"- input_rows: `{report['input_rows']}`",
        f"- output_rows: `{report['output_rows']}`",
        f"- dropped_rows: `{report['dropped_rows']}`",
        f"- normalized_breakpoints_rows: `{report['normalized_breakpoints_rows']}`",
        f"- output_sha256: `{report['output_sha256']}`",
        "",
        "## Drop Reasons",
    ]
    for reason, count in sorted(report["drop_reasons"].items()):
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "## Decision",
            "- Keep all structurally valid rows.",
            "- Drop rows whose chosen or rejected action JSON is unusable for DPO training.",
            "- Normalize missing `breakpoints` to `[]` so the schema stays stable.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    meta_json = output_jsonl.with_suffix(output_jsonl.suffix + ".meta.json")
    meta_md = output_jsonl.with_suffix(output_jsonl.suffix + ".meta.md")

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_jsonl}")

    rows_in = 0
    rows_out = 0
    normalized_breakpoints_rows = 0
    drop_reasons: dict[str, int] = {}
    cleaned_rows: list[dict[str, Any]] = []

    with input_jsonl.open("r", encoding="utf-8") as handle:
        for row_index, raw in enumerate(handle, start=1):
            row = json.loads(raw)
            rows_in += 1

            chosen_message = row["chosen"][0]
            rejected_message = row["rejected"][0]
            chosen_obj, chosen_error = normalize_response(chosen_message["content"])
            rejected_obj, rejected_error = normalize_response(rejected_message["content"])

            if chosen_error is not None:
                drop_reasons[f"chosen:{chosen_error}"] = drop_reasons.get(f"chosen:{chosen_error}", 0) + 1
                continue
            if rejected_error is not None:
                drop_reasons[f"rejected:{rejected_error}"] = drop_reasons.get(f"rejected:{rejected_error}", 0) + 1
                continue

            if "breakpoints" not in json.loads(chosen_message["content"]):
                normalized_breakpoints_rows += 1
            if "breakpoints" not in json.loads(rejected_message["content"]):
                normalized_breakpoints_rows += 1

            chosen_message = dict(chosen_message)
            rejected_message = dict(rejected_message)
            chosen_message["content"] = json.dumps(chosen_obj, ensure_ascii=False)
            rejected_message["content"] = json.dumps(rejected_obj, ensure_ascii=False)

            cleaned_row = dict(row)
            cleaned_row["chosen"] = [chosen_message]
            cleaned_row["rejected"] = [rejected_message]
            cleaned_rows.append(cleaned_row)
            rows_out += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in cleaned_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "created_at_utc": utc_now(),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_rows": rows_in,
        "output_rows": rows_out,
        "dropped_rows": rows_in - rows_out,
        "normalized_breakpoints_rows": normalized_breakpoints_rows,
        "drop_reasons": drop_reasons,
        "output_sha256": file_sha256(output_jsonl),
    }
    meta_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    meta_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"Cleaned DPO dataset: {output_jsonl}")
    print(f"input_rows={rows_in}")
    print(f"output_rows={rows_out}")
    print(f"dropped_rows={rows_in - rows_out}")
    print(f"normalized_breakpoints_rows={normalized_breakpoints_rows}")


if __name__ == "__main__":
    main()
