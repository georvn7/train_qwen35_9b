#!/usr/bin/env python3
"""Run schema20 scoring directly with a local/HF causal LM (no Ollama required)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_python_headers() -> None:
    """Allow Triton JIT C launcher compile on systems without /usr/include/python3.12."""
    default_root = Path("/home/georvn/train_qwen35_9b/.local_py312dev/usr/include")
    py_root = default_root / "python3.12"
    if not py_root.exists():
        return
    current = os.environ.get("CPATH", "")
    add_parts = [str(py_root), str(default_root)]
    if current:
        add_parts.append(current)
    os.environ["CPATH"] = ":".join(add_parts)


ensure_python_headers()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


@dataclass
class CaseResult:
    index: int
    name: str
    structured_score: float
    latency_s: float
    action_type: str
    action_subject: str
    parsed_keys: list[str]
    preview: str
    output_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "structured_score": self.structured_score,
            "latency_s": round(self.latency_s, 3),
            "action_type": self.action_type,
            "action_subject": self.action_subject,
            "parsed_keys": self.parsed_keys,
            "preview": self.preview,
            "output_text": self.output_text,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run schema20 local HF benchmark scorer.")
    parser.add_argument(
        "--workspace-root",
        default="/home/georvn/train_qwen35_9b",
        help="Workspace root used for default eval/scorer paths.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id or local path (e.g. artifacts/full_model).",
    )
    parser.add_argument(
        "--eval-file",
        default="qwen35_9b_fullft/evals/agent_cases_20_schema_final_v1.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--start-case-index", type=int, default=1)
    parser.add_argument("--end-case-index", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model thinking in chat template (default: disabled).",
    )
    parser.add_argument(
        "--store-full-output",
        action="store_true",
        help="Include full generated text for each case in JSON output.",
    )
    return parser.parse_args()


def load_schema_scorer(workspace_root: Path):
    scorer_path = workspace_root / "qwen35_9b_fullft" / "scripts" / "benchmark_ollama_chat.py"
    spec = importlib.util.spec_from_file_location("schema_scorer", str(scorer_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def build_prompt(tokenizer, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    chunks = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    chunks.append("assistant:")
    return "\n".join(chunks)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("results", [])
    lines = [
        "# Schema20 Local HF Report",
        "",
        f"- model: `{payload.get('model')}`",
        f"- eval_file: `{payload.get('eval_file')}`",
        f"- cases: `{payload.get('num_cases')}`",
        f"- avg_structured_score: `{payload.get('avg_structured_score')}`",
        f"- median_structured_score: `{payload.get('median_structured_score')}`",
        f"- min_structured_score: `{payload.get('min_structured_score')}`",
        f"- max_structured_score: `{payload.get('max_structured_score')}`",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        "",
        "| idx | structured | latency_s | action_type | action_subject | name |",
        "|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('index')} | "
            f"{row.get('structured_score')} | "
            f"{row.get('latency_s')} | "
            f"{row.get('action_type')} | "
            f"{row.get('action_subject')} | "
            f"{row.get('name')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    eval_file = Path(args.eval_file).expanduser()
    if not eval_file.is_absolute():
        eval_file = workspace_root / eval_file
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    model_ref = str(Path(args.model).expanduser()) if Path(args.model).expanduser().exists() else args.model

    out_dir = workspace_root / "qwen35_9b_fullft" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    out_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else out_dir / f"schema20_local_hf_{stamp}.json"
    )
    out_md = (
        Path(args.output_md).expanduser().resolve()
        if args.output_md
        else out_dir / f"schema20_local_hf_{stamp}.md"
    )

    scorer = load_schema_scorer(workspace_root)
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    start_idx = max(1, args.start_case_index)
    end_idx = args.end_case_index if args.end_case_index > 0 else len(cases)
    selected = cases[start_idx - 1 : end_idx]
    if not selected:
        raise ValueError("No cases selected.")

    print(f"loading model={model_ref}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"model_loaded_sec={time.time() - t0:.2f}")

    rows: list[CaseResult] = []
    scores: list[float] = []
    for offset, case in enumerate(selected, start=start_idx):
        name = str(case.get("name", f"case_{offset}"))
        messages = case.get("messages", [])
        expected = case.get("expected", {})
        prompt = build_prompt(
            tokenizer,
            messages if isinstance(messages, list) else [],
            enable_thinking=args.enable_thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start = time.time()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - start
        gen_ids = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        score = scorer._score_structured_debug(
            text,
            expected if isinstance(expected, dict) else None,
        )
        structured = float(score.get("structured_score", 0.0)) if isinstance(score, dict) else 0.0
        action_type = str(score.get("action_type", "")) if isinstance(score, dict) else ""
        action_subject = str(score.get("action_subject", "")) if isinstance(score, dict) else ""
        parsed_keys: list[str] = []
        if isinstance(score, dict):
            preview = score.get("json_candidates_preview", [])
            if isinstance(preview, list) and preview:
                keys = preview[0].get("keys", [])
                if isinstance(keys, list):
                    parsed_keys = [str(k) for k in keys]
        row = CaseResult(
            index=offset,
            name=name,
            structured_score=structured,
            latency_s=latency,
            action_type=action_type,
            action_subject=action_subject,
            parsed_keys=parsed_keys,
            preview=text[:280],
            output_text=text if args.store_full_output else None,
        )
        rows.append(row)
        scores.append(structured)
        print(
            f"case={offset}/{start_idx - 1 + len(selected)} "
            f"structured={structured:.3f} latency_s={latency:.2f} name={name}"
        )

    payload = {
        "model": model_ref,
        "eval_file": str(eval_file),
        "num_cases": len(rows),
        "avg_structured_score": (sum(scores) / len(scores)) if scores else 0.0,
        "median_structured_score": (sorted(scores)[len(scores) // 2]) if scores else 0.0,
        "min_structured_score": min(scores) if scores else 0.0,
        "max_structured_score": max(scores) if scores else 0.0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": [row.to_dict() for row in rows],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)

    print(f"avg_structured_score={payload['avg_structured_score']:.4f}")
    print(f"report_json={out_json}")
    print(f"report_md={out_md}")


if __name__ == "__main__":
    main()
