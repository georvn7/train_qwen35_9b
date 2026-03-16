#!/usr/bin/env python3
"""Sanitize debugger chat datasets for stable structured-output fine-tuning.

This script rewrites input JSONL chat rows into a stricter training format:
- Keep first system message (optional), final user message, final assistant target.
- Final assistant target is normalized to one strict JSON action object.
- Optional thinking handling:
  - off: drop thinking
  - compact: keep a short cleaned thinking summary
  - keep: keep cleaned thinking text
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_ACTION_KEYS = [
    "action_subject",
    "action_type",
    "breakpoints",
    "invocation",
    "line_number",
    "motivation",
]
OUTPUT_LEAK_MARKERS = ["<|", "\"cmd\":", "bash -lc", "```"]


@dataclass
class Stats:
    files_scanned: int = 0
    rows_scanned: int = 0
    rows_written: int = 0
    rows_with_multi_assistant: int = 0
    rows_with_thinking_input: int = 0
    rows_with_thinking_output: int = 0
    content_leak_markers_output: int = 0
    thinking_leak_markers_output: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanitize debugger JSONL datasets for parser-stable training."
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="Input dataset root containing *.jsonl files (repeatable).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root directory where sanitized files are written.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["off", "compact", "keep"],
        default="off",
        help="How to handle assistant thinking field.",
    )
    parser.add_argument(
        "--compact-max-chars",
        type=int,
        default=220,
        help="Max chars for compact thinking mode.",
    )
    parser.add_argument(
        "--compact-max-words",
        type=int,
        default=36,
        help="Max words for compact thinking mode.",
    )
    parser.add_argument(
        "--drop-system",
        action="store_true",
        help="Drop system message from output rows.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional report json path. Defaults to <output-root>/sanitization_report.json.",
    )
    parser.add_argument(
        "--report-md",
        default="",
        help="Optional report markdown path. Defaults to <output-root>/sanitization_report.md.",
    )
    return parser.parse_args()


def list_jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def find_first_system(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def find_final_user_and_assistant(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    final_assistant_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                final_assistant_idx = idx
                break
    if final_assistant_idx < 0:
        return None, None

    final_user_idx = -1
    for idx in range(final_assistant_idx - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                final_user_idx = idx
                break
    if final_user_idx < 0:
        return None, None

    return messages[final_user_idx], messages[final_assistant_idx]


def _as_int_or_default(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def normalize_action_object(obj: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    action_type = obj.get("action_type")
    action_subject = obj.get("action_subject")
    if not isinstance(action_type, str) or not action_type.strip():
        return None, "missing_action_type"
    if not isinstance(action_subject, str) or not action_subject.strip():
        action_subject = "none"

    breakpoints = obj.get("breakpoints", [])
    if not isinstance(breakpoints, list):
        breakpoints = []

    motivation = obj.get("motivation", "")
    if not isinstance(motivation, str):
        motivation = str(motivation)

    normalized = {
        "action_subject": action_subject,
        "action_type": action_type,
        "breakpoints": breakpoints,
        "invocation": _as_int_or_default(obj.get("invocation"), 1),
        "line_number": _as_int_or_default(obj.get("line_number"), 0),
        "motivation": motivation,
    }

    return normalized, ""


def has_leak_markers(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in OUTPUT_LEAK_MARKERS)


def clean_thinking_text(text: str) -> str:
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = text.replace("```", " ")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if (
            "analysis" in lowered
            or "commentary" in lowered
            or "\"cmd\":" in lowered
            or "bash -lc" in lowered
            or "<|" in lowered
        ):
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def compact_thinking(text: str, max_words: int, max_chars: int) -> str:
    cleaned = clean_thinking_text(text)
    if not cleaned:
        cleaned = (
            "Choose the next debugging action from the evidence and emit one schema-compliant JSON object."
        )

    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    if not sentence:
        sentence = cleaned
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words])
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 1].rstrip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def sanitize_row(
    row: dict[str, Any],
    thinking_mode: str,
    compact_max_words: int,
    compact_max_chars: int,
    drop_system: bool,
) -> tuple[dict[str, Any] | None, str, bool, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "invalid_messages", False, False

    assistant_count = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
    user_msg, assistant_msg = find_final_user_and_assistant(messages)
    if user_msg is None or assistant_msg is None:
        return None, "missing_final_user_or_assistant", assistant_count > 1, False

    content = assistant_msg.get("content")
    if not isinstance(content, str):
        return None, "assistant_content_not_string", assistant_count > 1, False
    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        return None, "assistant_content_not_json", assistant_count > 1, False
    if not isinstance(parsed, dict):
        return None, "assistant_content_not_json_object", assistant_count > 1, False

    normalized_action, reason = normalize_action_object(parsed)
    if normalized_action is None:
        return None, reason, assistant_count > 1, False

    out_messages: list[dict[str, str]] = []
    system_text = find_first_system(messages)
    if system_text and not drop_system:
        out_messages.append({"role": "system", "content": system_text})
    out_messages.append({"role": "user", "content": str(user_msg["content"])})

    out_assistant: dict[str, str] = {
        "role": "assistant",
        "content": json.dumps(normalized_action, ensure_ascii=False, separators=(",", ":")),
    }

    input_thinking = assistant_msg.get("thinking")
    has_input_thinking = isinstance(input_thinking, str) and bool(input_thinking.strip())
    if thinking_mode == "keep":
        if has_input_thinking:
            cleaned = clean_thinking_text(input_thinking)
            if cleaned:
                out_assistant["thinking"] = cleaned
    elif thinking_mode == "compact":
        compact = compact_thinking(
            input_thinking if isinstance(input_thinking, str) else "",
            max_words=compact_max_words,
            max_chars=compact_max_chars,
        )
        if compact:
            out_assistant["thinking"] = compact
    out_messages.append(out_assistant)

    out_row: dict[str, Any] = {"messages": out_messages}
    if isinstance(row.get("name"), str):
        out_row["name"] = row["name"]

    has_output_thinking = isinstance(out_assistant.get("thinking"), str)
    return out_row, "", assistant_count > 1, has_input_thinking and has_output_thinking


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Dataset Sanitization Report",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- thinking_mode: `{report['thinking_mode']}`",
        f"- input_roots: `{', '.join(report['input_roots'])}`",
        f"- output_root: `{report['output_root']}`",
        "",
        "## Totals",
        f"- files_scanned: `{report['stats']['files_scanned']}`",
        f"- rows_scanned: `{report['stats']['rows_scanned']}`",
        f"- rows_written: `{report['stats']['rows_written']}`",
        f"- rows_dropped: `{report['rows_dropped']}`",
        f"- rows_with_multi_assistant: `{report['stats']['rows_with_multi_assistant']}`",
        f"- rows_with_thinking_input: `{report['stats']['rows_with_thinking_input']}`",
        f"- rows_with_thinking_output: `{report['stats']['rows_with_thinking_output']}`",
        "",
        "## Output Leak Checks",
        f"- content_leak_markers_output: `{report['stats']['content_leak_markers_output']}`",
        f"- thinking_leak_markers_output: `{report['stats']['thinking_leak_markers_output']}`",
        "",
        "## Rejection Reasons",
    ]
    for reason, count in sorted(report["reject_reasons"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {reason}: `{count}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = [Path(item).expanduser().resolve() for item in args.dataset_root]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"dataset root not found: {root}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else output_root / "sanitization_report.json"
    )
    report_md = (
        Path(args.report_md).expanduser().resolve()
        if args.report_md
        else output_root / "sanitization_report.md"
    )

    stats = Stats()
    reject_reasons: Counter[str] = Counter()
    output_files: list[str] = []

    for root in roots:
        files = list_jsonl_files(root)
        stats.files_scanned += len(files)
        root_label = root.name
        for in_file in files:
            rel_path = in_file.relative_to(root)
            out_file = output_root / root_label / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with in_file.open("r", encoding="utf-8") as src, out_file.open("w", encoding="utf-8") as dst:
                for line in src:
                    raw = line.strip()
                    if not raw:
                        continue
                    stats.rows_scanned += 1
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        reject_reasons["invalid_jsonl_row"] += 1
                        continue

                    out_row, reason, had_multi_assistant, had_thinking_in_and_out = sanitize_row(
                        row=row,
                        thinking_mode=args.thinking_mode,
                        compact_max_words=args.compact_max_words,
                        compact_max_chars=args.compact_max_chars,
                        drop_system=args.drop_system,
                    )
                    if had_multi_assistant:
                        stats.rows_with_multi_assistant += 1
                    last_assistant = None
                    if isinstance(row.get("messages"), list):
                        for msg in reversed(row["messages"]):
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                last_assistant = msg
                                break
                    if isinstance(last_assistant, dict):
                        th = last_assistant.get("thinking")
                        if isinstance(th, str) and th.strip():
                            stats.rows_with_thinking_input += 1

                    if out_row is None:
                        reject_reasons[reason] += 1
                        continue

                    final_assistant = out_row["messages"][-1]
                    if has_leak_markers(str(final_assistant.get("content", ""))):
                        stats.content_leak_markers_output += 1
                    out_thinking = final_assistant.get("thinking", "")
                    if isinstance(out_thinking, str) and out_thinking.strip():
                        stats.rows_with_thinking_output += 1
                        if has_leak_markers(out_thinking):
                            stats.thinking_leak_markers_output += 1
                    elif had_thinking_in_and_out:
                        stats.rows_with_thinking_output += 1

                    dst.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    stats.rows_written += 1
                    written += 1
            if written > 0:
                output_files.append(str(out_file))

    report = {
        "created_at_utc": utc_now(),
        "thinking_mode": args.thinking_mode,
        "input_roots": [str(path) for path in roots],
        "output_root": str(output_root),
        "stats": {
            "files_scanned": stats.files_scanned,
            "rows_scanned": stats.rows_scanned,
            "rows_written": stats.rows_written,
            "rows_with_multi_assistant": stats.rows_with_multi_assistant,
            "rows_with_thinking_input": stats.rows_with_thinking_input,
            "rows_with_thinking_output": stats.rows_with_thinking_output,
            "content_leak_markers_output": stats.content_leak_markers_output,
            "thinking_leak_markers_output": stats.thinking_leak_markers_output,
        },
        "rows_dropped": stats.rows_scanned - stats.rows_written,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "output_files_count": len(output_files),
        "output_files_sample": output_files[:20],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report_md(report_md, report)

    print(f"Sanitized rows: {stats.rows_written}/{stats.rows_scanned}")
    print(f"Output root: {output_root}")
    print(f"Report JSON: {report_json}")
    print(f"Report MD: {report_md}")


if __name__ == "__main__":
    main()
