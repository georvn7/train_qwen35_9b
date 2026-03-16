#!/usr/bin/env python3
"""Create a vLLM-compatible Qwen3.5 full-FT model copy by rewriting config only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_config(full_cfg: dict, base_cfg: dict) -> dict:
    top_skip = {
        "architectures",
        "model_type",
        "transformers_version",
        "unsloth_version",
    }
    text_cfg = {k: v for k, v in full_cfg.items() if k not in top_skip}
    text_cfg["model_type"] = "qwen3_5_text"

    out = dict(base_cfg)
    out["model_type"] = "qwen3_5"
    out["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    out["tie_word_embeddings"] = full_cfg.get("tie_word_embeddings", False)
    out["text_config"] = text_cfg
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    full_dir = Path(args.full_model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_config_path = full_dir / "config.json"
    if not full_config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {full_config_path}")

    base_config_path = Path(hf_hub_download(args.base_model_id, "config.json"))
    full_cfg = json.loads(full_config_path.read_text())
    base_cfg = json.loads(base_config_path.read_text())
    out_cfg = build_config(full_cfg, base_cfg)

    for src in full_dir.iterdir():
        if src.name == "config.json":
            continue
        dst = out_dir / src.name
        if src.is_file():
            link_or_copy(src, dst)
        elif src.is_dir():
            if dst.exists():
                continue
            shutil.copytree(src, dst, copy_function=shutil.copy2)

    (out_dir / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")
    print(f"Wrote vLLM-compatible model dir: {out_dir}")


if __name__ == "__main__":
    main()
