#!/usr/bin/env python3
"""Create an Ollama Modelfile from a trained session GGUF export."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def load_gpt_oss_template() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    template_path = repo_root / "templates" / "ollama_gpt_oss_template.tmpl"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing GPT-OSS template file: {template_path}")
    return template_path.read_text(encoding="utf-8").rstrip()


def find_gguf_file(gguf_dir: Path) -> Path:
    candidates = sorted(gguf_dir.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(f"No .gguf files found in {gguf_dir}")
    # Prefer quantized outputs for serving; avoid selecting full f16 when both exist.
    quantized = [p for p in candidates if "f16" not in p.name.lower()]
    if quantized:
        return min(quantized, key=lambda p: p.stat().st_size)
    return min(candidates, key=lambda p: p.stat().st_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Modelfile and optionally create Ollama model."
    )
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--model-name", required=True, help="Ollama model tag to create.")
    parser.add_argument(
        "--gguf-file",
        default="",
        help="Optional explicit GGUF path. Defaults to largest file in artifacts/gguf.",
    )
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant tuned for this environment.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Write raw prompt template only (not recommended for gpt-oss).",
    )
    parser.add_argument("--create", action="store_true", help="Run `ollama create` after writing the file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpt_oss_template = load_gpt_oss_template()
    session_dir = Path(args.session_dir).expanduser().resolve()
    gguf_dir = session_dir / "artifacts" / "gguf"
    ollama_dir = session_dir / "ollama"
    ollama_dir.mkdir(parents=True, exist_ok=True)

    if args.gguf_file:
        gguf_file = Path(args.gguf_file).expanduser().resolve()
    else:
        gguf_file = find_gguf_file(gguf_dir)

    if not gguf_file.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_file}")

    modelfile_path = ollama_dir / "Modelfile"
    lines = [
        f"FROM {gguf_file}",
        f"PARAMETER num_ctx {args.num_ctx}",
        f"PARAMETER temperature {args.temperature}",
        f'SYSTEM """{args.system_prompt}"""',
    ]
    if args.disable_chat_template:
        lines.append("TEMPLATE {{ .Prompt }}")
    else:
        lines.extend(
            [
                'TEMPLATE """',
                gpt_oss_template,
                '"""',
            ]
        )
    lines.append("")
    content = "\n".join(lines)
    modelfile_path.write_text(content, encoding="utf-8")

    print(f"Modelfile written: {modelfile_path}")
    print(f"Model tag: {args.model_name}")

    if args.create:
        cmd = ["ollama", "create", args.model_name, "-f", str(modelfile_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
