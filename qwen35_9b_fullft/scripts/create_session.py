#!/usr/bin/env python3
"""Create a reproducible fine-tuning session directory with dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


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


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def collect_jsonl_files(roots: Iterable[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.match(pattern):
            files.append(root)
            continue
        for path in root.rglob(pattern):
            if path.is_file():
                files.append(path)
    # de-duplicate and sort for stable manifests
    return sorted(set(files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fine-tuning session and dataset manifest."
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository/workspace root path.",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Where session directories are stored. Defaults to <workspace>/qwen35_9b_fullft/runs.",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="Dataset root directory (repeatable). If omitted, uses all datasets_* dirs in workspace root.",
    )
    parser.add_argument(
        "--jsonl-pattern",
        default="*.jsonl",
        help="File name pattern for dataset files.",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Short label appended to the session id.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Freeform notes saved in metadata/session.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    runs_root = (
        Path(args.runs_root).expanduser().resolve()
        if args.runs_root
        else workspace_root / "qwen35_9b_fullft" / "runs"
    )

    if args.dataset_root:
        dataset_roots = []
        for root in args.dataset_root:
            path = Path(root).expanduser()
            if not path.is_absolute():
                path = workspace_root / path
            dataset_roots.append(path.resolve())
    else:
        dataset_roots = sorted(
            [path for path in workspace_root.glob("datasets_*") if path.is_dir()]
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in args.label)
    session_id = f"{timestamp}_{clean_label}"
    session_dir = runs_root / session_id

    metadata_dir = session_dir / "metadata"
    artifacts_dir = session_dir / "artifacts"
    checkpoints_dir = session_dir / "checkpoints"
    logs_dir = session_dir / "logs"

    for directory in (
        metadata_dir,
        artifacts_dir / "full_model",
        artifacts_dir / "adapter",
        artifacts_dir / "merged_16bit",
        artifacts_dir / "gguf",
        checkpoints_dir,
        logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    jsonl_files = collect_jsonl_files(dataset_roots, args.jsonl_pattern)
    manifest_entries = []
    total_lines = 0
    total_bytes = 0
    for path in jsonl_files:
        lines = line_count(path)
        size = path.stat().st_size
        total_lines += lines
        total_bytes += size
        entry = {
            "size_bytes": size,
            "num_lines": lines,
            "sha256": file_sha256(path),
        }
        if path.is_relative_to(workspace_root):
            entry["relative_path"] = str(path.relative_to(workspace_root))
        else:
            entry["absolute_path"] = str(path)
        manifest_entries.append(entry)

    manifest = {
        "created_at_utc": utc_now(),
        "workspace_root": str(workspace_root),
        "dataset_roots": [
            str(root.relative_to(workspace_root)) if root.is_relative_to(workspace_root) else str(root)
            for root in dataset_roots
        ],
        "jsonl_pattern": args.jsonl_pattern,
        "num_files": len(manifest_entries),
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "files": manifest_entries,
    }

    session_meta = {
        "created_at_utc": utc_now(),
        "session_id": session_id,
        "session_dir": str(session_dir),
        "notes": args.notes,
        "status": "created",
    }

    manifest_path = metadata_dir / "dataset_manifest.json"
    session_meta_path = metadata_dir / "session.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    session_meta_path.write_text(json.dumps(session_meta, indent=2), encoding="utf-8")

    print(f"Session created: {session_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Dataset files: {len(manifest_entries)}")
    print(f"Total rows: {total_lines}")
    print("")
    print("Next step:")
    print(
        "python qwen35_9b_fullft/scripts/train_session.py "
        f"--session-dir {session_dir}"
    )


if __name__ == "__main__":
    main()
