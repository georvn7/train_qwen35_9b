#!/usr/bin/env python3
"""Analyze sample context lengths and suggest sequence-length strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze context lengths from session manifest or dataset folder."
    )
    parser.add_argument("--session-dir", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.5-9B",
        help="Tokenizer model id for token-accurate estimates.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Optional cap on number of samples processed.",
    )
    parser.add_argument(
        "--bins",
        default="512,1024,2048,4096,8192,16384,32768",
        help="Comma-separated max-token bins.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional explicit output path. Default is <session>/metadata/context_stats.json when --session-dir is used.",
    )
    return parser.parse_args()


def collect_files(args: argparse.Namespace) -> list[Path]:
    if args.session_dir:
        session_dir = Path(args.session_dir).expanduser().resolve()
        manifest_path = session_dir / "metadata" / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        workspace_root = Path(manifest["workspace_root"])
        files: list[Path] = []
        for entry in manifest["files"]:
            if "relative_path" in entry:
                files.append((workspace_root / entry["relative_path"]).resolve())
                continue
            if "absolute_path" in entry:
                files.append(Path(entry["absolute_path"]).expanduser().resolve())
                continue
            raise ValueError(f"Manifest entry missing relative_path/absolute_path: {entry}")
        return files

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        return sorted(dataset_root.rglob("*.jsonl"))

    raise ValueError("Provide either --session-dir or --dataset-root")


def load_tokenizer(model_name: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None


def messages_to_text(messages: list[dict[str, Any]], tokenizer: Any | None) -> str:
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return "\n".join(f"{m.get('role','unknown')}: {m.get('content','')}" for m in messages)


def estimate_tokens(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # fallback heuristic
    return max(1, len(text) // 4)


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def recommend_max_seq_length(p95: int) -> int:
    candidates = [512, 1024, 2048, 4096, 8192, 12288, 16384, 32768]
    for c in candidates:
        if p95 <= c:
            return c
    return 32768


def main() -> None:
    args = parse_args()
    files = collect_files(args)
    tokenizer = load_tokenizer(args.model_name)
    bins = sorted(int(part.strip()) for part in args.bins.split(",") if part.strip())
    bucket_counts = {str(boundary): 0 for boundary in bins}
    overflow_key = f">{bins[-1]}"
    bucket_counts[overflow_key] = 0

    lengths: list[int] = []
    processed = 0
    format_counts = {"messages": 0, "text": 0, "other": 0}

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if args.sample_limit > 0 and processed >= args.sample_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                if isinstance(row, dict) and "messages" in row:
                    text = messages_to_text(row["messages"], tokenizer)
                    format_counts["messages"] += 1
                elif isinstance(row, dict) and "text" in row:
                    text = str(row["text"])
                    format_counts["text"] += 1
                else:
                    format_counts["other"] += 1
                    text = json.dumps(row)

                n_tokens = estimate_tokens(text, tokenizer)
                lengths.append(n_tokens)
                processed += 1

                assigned = False
                for boundary in bins:
                    if n_tokens <= boundary:
                        bucket_counts[str(boundary)] += 1
                        assigned = True
                        break
                if not assigned:
                    bucket_counts[overflow_key] += 1
            if args.sample_limit > 0 and processed >= args.sample_limit:
                break

    lengths.sort()
    p50 = percentile(lengths, 0.50)
    p90 = percentile(lengths, 0.90)
    p95 = percentile(lengths, 0.95)
    p99 = percentile(lengths, 0.99)
    p100 = lengths[-1] if lengths else 0

    report = {
        "files_scanned": len(files),
        "samples_scanned": processed,
        "tokenizer_model": args.model_name,
        "tokenizer_loaded": tokenizer is not None,
        "format_counts": format_counts,
        "token_length_stats": {
            "min": lengths[0] if lengths else 0,
            "mean": int(mean(lengths)) if lengths else 0,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "max": p100,
        },
        "bucket_counts": bucket_counts,
        "recommendation": {
            "max_seq_length": recommend_max_seq_length(p95),
            "group_by_length": True,
            "packing_for_short_data": p50 < 0.6 * recommend_max_seq_length(p95),
            "note": "Bucket by sample length, not file. Keep one manifest and let batching handle length groups.",
        },
    }

    output_path = None
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
    elif args.session_dir:
        output_path = (
            Path(args.session_dir).expanduser().resolve() / "metadata" / "context_stats.json"
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote: {output_path}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
