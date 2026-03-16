#!/usr/bin/env python3
"""Build a dataset variant that preserves all messages/rows and removes assistant thinking only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Stats:
    files_scanned: int = 0
    rows_scanned: int = 0
    rows_written: int = 0
    rows_invalid_json: int = 0
    rows_invalid_messages: int = 0
    assistant_messages_scanned: int = 0
    assistant_messages_with_thinking: int = 0
    thinking_fields_removed: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preserve all dataset rows/messages but remove assistant.thinking fields."
    )
    p.add_argument("--input-root", required=True, help="Input dataset root (recursive *.jsonl).")
    p.add_argument("--output-root", required=True, help="Output dataset root.")
    return p.parse_args()


def list_jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def write_report(output_root: Path, stats: Stats, input_root: Path) -> None:
    report = {
        "created_at_utc": utc_now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "stats": asdict(stats),
    }
    report_json = output_root / "strip_thinking_report.json"
    report_md = output_root / "strip_thinking_report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Strip Thinking Report",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- input_root: `{report['input_root']}`",
        f"- output_root: `{report['output_root']}`",
        "",
        "## Totals",
        f"- files_scanned: `{stats.files_scanned}`",
        f"- rows_scanned: `{stats.rows_scanned}`",
        f"- rows_written: `{stats.rows_written}`",
        f"- rows_invalid_json: `{stats.rows_invalid_json}`",
        f"- rows_invalid_messages: `{stats.rows_invalid_messages}`",
        f"- assistant_messages_scanned: `{stats.assistant_messages_scanned}`",
        f"- assistant_messages_with_thinking: `{stats.assistant_messages_with_thinking}`",
        f"- thinking_fields_removed: `{stats.thinking_fields_removed}`",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    files = list_jsonl_files(input_root)
    for src in files:
        stats.files_scanned += 1
        rel = src.relative_to(input_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_lines: list[str] = []
        with src.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                stats.rows_scanned += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    stats.rows_invalid_json += 1
                    continue

                messages = row.get("messages")
                if not isinstance(messages, list):
                    stats.rows_invalid_messages += 1
                    continue

                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "assistant":
                        continue
                    stats.assistant_messages_scanned += 1
                    thinking = msg.get("thinking")
                    if isinstance(thinking, str) and thinking.strip():
                        stats.assistant_messages_with_thinking += 1
                    if "thinking" in msg:
                        del msg["thinking"]
                        stats.thinking_fields_removed += 1

                out_lines.append(json.dumps(row, ensure_ascii=False))
                stats.rows_written += 1

        dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    write_report(output_root=output_root, stats=stats, input_root=input_root)

    print(f"Stripped thinking rows: {stats.rows_written}/{stats.rows_scanned}")
    print(f"Output root: {output_root}")
    print(f"Report: {output_root / 'strip_thinking_report.md'}")


if __name__ == "__main__":
    main()
