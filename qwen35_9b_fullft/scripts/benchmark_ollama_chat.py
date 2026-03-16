#!/usr/bin/env python3
"""Benchmark Ollama chat endpoints on a case file."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an Ollama model through /v1/chat/completions or /api/chat."
    )
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. gpt-oss:20b")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama server base URL.",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=["v1", "api_chat"],
        default="v1",
        help="Use OpenAI-compatible v1 endpoint or native Ollama /api/chat endpoint.",
    )
    parser.add_argument(
        "--messages-file",
        required=True,
        help=(
            "JSON list with chat messages. "
            "Format: [{\"name\":\"...\",\"messages\":[{\"role\":\"user\",\"content\":\"...\"}],\"expected\":{...}}]"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["", "low", "medium", "high"],
        help="Reasoning-effort hint for models/endpoints that support it.",
    )
    parser.add_argument(
        "--verbosity",
        default="medium",
        choices=["", "low", "medium", "high"],
        help="Verbosity hint for models/endpoints that support it.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=True,
        help="Enable thinking for /api/chat (ignored by /v1).",
    )
    parser.add_argument(
        "--no-think",
        dest="think",
        action="store_false",
        help="Disable thinking for /api/chat.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--runs-per-case", type=int, default=1)
    parser.add_argument(
        "--start-case-index",
        type=int,
        default=1,
        help="1-based inclusive case index to start from.",
    )
    parser.add_argument(
        "--end-case-index",
        type=int,
        default=0,
        help="1-based inclusive case index to stop at. 0 means all remaining cases.",
    )
    parser.add_argument(
        "--force-tool-choice-none",
        action="store_true",
        default=True,
        help="Send tool_choice='none' to avoid unintended tool-call parsing behavior.",
    )
    parser.add_argument(
        "--allow-tool-choice",
        dest="force_tool_choice_none",
        action="store_false",
        help="Do not force tool_choice='none'.",
    )
    parser.add_argument(
        "--include-full-output",
        action="store_true",
        help="Store full assistant content and reasoning per run in JSON.",
    )
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--progress-log-file", default="")
    return parser.parse_args()


def _score_debug_style(text: str) -> dict[str, Any]:
    import re

    lowered = text.lower()
    required = {
        "root_cause": bool(re.search(r"\broot cause\b", lowered)),
        "minimal_fix": bool(re.search(r"\bminimal fix\b", lowered)),
        "verification": bool(re.search(r"\bverification\b", lowered)),
    }
    forbidden_tokens = ["log_summary", "transcript", "what do you do?"]
    forbidden_hits = [token for token in forbidden_tokens if token in lowered]
    leading = lowered[:120]
    analysis_leak = bool(
        re.search(
            r"^\s*(analysis|thinking\.\.\.|channel\s*[:=]\s*analysis|<\|channel\|>\s*analysis)",
            leading,
        )
    )
    raw_score = (sum(required.values()) / 3.0) - 0.25 * len(forbidden_hits) - (
        0.5 if analysis_leak else 0.0
    )
    return {
        "required": required,
        "forbidden_hits": forbidden_hits,
        "analysis_leak": analysis_leak,
        "style_score": max(0.0, min(1.0, raw_score)),
    }


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "\"":
                    in_string = False
                continue
            if ch == "\"":
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        span = (start, idx + 1)
                        if span not in seen_spans:
                            found.append(parsed)
                            seen_spans.add(span)
                    break
        start = text.find("{", start + 1)
    return found


def _to_words(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    return len(value.split())


def _score_structured_debug(text: str, expected: dict[str, Any] | None) -> dict[str, Any]:
    import re

    parsed_objects = _extract_json_objects(text)

    details: dict[str, Any] = {
        "has_json": bool(parsed_objects),
        "num_json_objects": len(parsed_objects),
        "json_candidates_preview": [
            {
                "keys": sorted(list(obj.keys()))[:10],
                "has_action_type": "action_type" in obj,
                "has_action_subject": "action_subject" in obj,
            }
            for obj in parsed_objects[:5]
        ],
    }

    def score_candidate(obj: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        action_type = str(obj.get("action_type", "")).strip()
        action_subject = str(obj.get("action_subject", "")).strip()
        motivation = obj.get("motivation")
        line_number = obj.get("line_number")
        breakpoints = obj.get("breakpoints")
        log_summary = obj.get("log_summary", "")
        debug_notes = obj.get("debug_notes", "")
        has_action_type = bool(action_type)
        has_action_subject = bool(action_subject)
        has_line_number = isinstance(line_number, (int, float))
        has_motivation = isinstance(motivation, str) and bool(motivation.strip())
        has_breakpoints = isinstance(breakpoints, list)

        if expected:
            exp_action_type = str(expected.get("action_type", "")).strip()
            exp_subject = str(expected.get("action_subject", "")).strip()
            exp_subject_regex = str(expected.get("action_subject_regex", "")).strip()
            exp_line_number = expected.get("line_number")
            exp_breakpoints_min = expected.get("breakpoints_min")
            exp_breakpoints_exact = expected.get("breakpoints_exact")
            exp_breakpoints_nonempty = expected.get("breakpoints_nonempty")
            max_words = expected.get("max_words")

            action_type_match = has_action_type and (
                (not exp_action_type) or action_type == exp_action_type
            )
            if exp_subject_regex:
                try:
                    subject_match = bool(re.search(exp_subject_regex, action_subject))
                except re.error:
                    subject_match = False
            else:
                subject_match = has_action_subject and (
                    (not exp_subject) or action_subject == exp_subject
                )
            if exp_line_number is None:
                line_match = has_line_number
            else:
                line_match = has_line_number and float(line_number) == float(exp_line_number)
            if not has_breakpoints:
                breakpoints_match = False
            else:
                bp_len = len(breakpoints)
                breakpoints_match = True
                if isinstance(exp_breakpoints_exact, (int, float)):
                    breakpoints_match = breakpoints_match and (bp_len == int(exp_breakpoints_exact))
                if isinstance(exp_breakpoints_min, (int, float)):
                    breakpoints_match = breakpoints_match and (bp_len >= int(exp_breakpoints_min))
                if isinstance(exp_breakpoints_nonempty, bool):
                    breakpoints_match = breakpoints_match and (
                        (bp_len > 0) == exp_breakpoints_nonempty
                    )
            words = len(text.split())
            verbosity_match = True
            if isinstance(max_words, (int, float)) and max_words > 0:
                verbosity_match = words <= int(max_words)
            scores = {
                "action_type_match": 1.0 if action_type_match else 0.0,
                "action_subject_match": 1.0 if subject_match else 0.0,
                "line_number_match": 1.0 if line_match else 0.0,
                "breakpoints_match": 1.0 if breakpoints_match else 0.0,
                "verbosity_match": 1.0 if verbosity_match else 0.0,
                "json_parse": 1.0,
            }
            composite = (
                0.30 * scores["action_type_match"]
                + 0.30 * scores["action_subject_match"]
                + 0.15 * scores["line_number_match"]
                + 0.15 * scores["breakpoints_match"]
                + 0.10 * scores["verbosity_match"]
            )
        else:
            scores = {
                "has_action_type": 1.0 if has_action_type else 0.0,
                "has_action_subject": 1.0 if has_action_subject else 0.0,
                "has_line_number": 1.0 if has_line_number else 0.0,
                "has_breakpoints": 1.0 if has_breakpoints else 0.0,
                "has_motivation": 1.0 if has_motivation else 0.0,
                "json_parse": 1.0,
            }
            composite = (
                0.25 * scores["has_action_type"]
                + 0.25 * scores["has_action_subject"]
                + 0.15 * scores["has_line_number"]
                + 0.15 * scores["has_breakpoints"]
                + 0.20 * scores["json_parse"]
            )

        candidate_details = {
            "action_type": action_type,
            "action_subject": action_subject,
            "line_number": line_number,
            "breakpoints_count": len(breakpoints) if isinstance(breakpoints, list) else None,
            "motivation_words": _to_words(motivation),
            "log_summary_words": _to_words(log_summary),
            "debug_notes_words": _to_words(debug_notes),
            "scores": scores,
        }
        return max(0.0, min(1.0, composite)), candidate_details

    if not parsed_objects:
        details["structured_score"] = 0.0
        details["scores"] = {"json_parse": 0.0}
        if expected:
            details["expected"] = expected
        return details

    scored = [score_candidate(obj) for obj in parsed_objects]
    best_score, best_details = max(scored, key=lambda pair: pair[0])
    details.update(best_details)
    details["structured_score"] = best_score
    if expected:
        details["expected"] = expected
    return details


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("messages-file must be a JSON list")
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"case {idx} is not an object")
        name = str(item.get("name") or f"case_{idx:02d}")
        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"case {idx} missing non-empty messages")
        expected = item.get("expected") if isinstance(item.get("expected"), dict) else None
        cases.append({"name": name, "messages": messages, "expected": expected})
    return cases


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _extract_v1_response(resp: dict[str, Any]) -> dict[str, Any]:
    choices = resp.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    usage = resp.get("usage") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    return {
        "content": content,
        "reasoning": reasoning,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_api_chat_response(resp: dict[str, Any]) -> dict[str, Any]:
    message = resp.get("message") or {}
    content = message.get("content") or ""
    reasoning = (
        message.get("reasoning")
        or message.get("thinking")
        or resp.get("reasoning")
        or resp.get("thinking")
        or ""
    )
    prompt_tokens = int(resp.get("prompt_eval_count") or 0)
    completion_tokens = int(resp.get("eval_count") or 0)
    total_tokens = prompt_tokens + completion_tokens
    return {
        "content": content,
        "reasoning": reasoning,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    base_url = args.base_url.rstrip("/")
    if args.endpoint_mode == "v1":
        endpoint = f"{base_url}/v1/chat/completions"
    else:
        endpoint = f"{base_url}/api/chat"
    messages_path = Path(args.messages_file).expanduser().resolve()
    out_path = Path(args.save_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    progress_fh = None
    if args.progress_log_file:
        progress_path = Path(args.progress_log_file).expanduser().resolve()
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_fh = progress_path.open("a", encoding="utf-8")
    else:
        progress_path = None

    def log(line: str = "") -> None:
        print(line, flush=True)
        if progress_fh is not None:
            progress_fh.write(line + "\n")
            progress_fh.flush()

    log(f"endpoint: {endpoint}")
    log(f"model: {args.model}")
    log(f"messages_file: {messages_path}")
    if progress_path is not None:
        log(f"progress_log: {progress_path}")

    cases = _load_cases(messages_path)
    total_cases = len(cases)
    log(f"cases: {total_cases}")
    log("")

    bench_rows: list[dict[str, Any]] = []

    start_idx = max(1, args.start_case_index)
    end_idx = args.end_case_index if args.end_case_index > 0 else total_cases
    if end_idx < start_idx:
        raise ValueError(
            f"Invalid range: start-case-index={start_idx} end-case-index={end_idx}"
        )
    selected = [
        (idx, case) for idx, case in enumerate(cases, start=1) if start_idx <= idx <= end_idx
    ]
    total_selected = len(selected)
    if total_selected == 0:
        raise ValueError("No cases selected by start/end case range.")

    for run_pos, (case_index, case) in enumerate(selected, start=1):
        name = case["name"]
        expected = case.get("expected")
        run_rows: list[dict[str, Any]] = []
        error_text: str | None = None

        for run_index in range(1, args.runs_per_case + 1):
            try:
                if args.endpoint_mode == "v1":
                    payload = {
                        "model": args.model,
                        "messages": case["messages"],
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "stream": False,
                    }
                    if args.force_tool_choice_none:
                        payload["tool_choice"] = "none"
                    if args.reasoning_effort:
                        payload["reasoning_effort"] = args.reasoning_effort
                    if args.verbosity:
                        payload["verbosity"] = args.verbosity
                    options: dict[str, Any] = {}
                    if args.num_ctx > 0:
                        options["num_ctx"] = args.num_ctx
                    if args.top_p != 1.0:
                        options["top_p"] = args.top_p
                    if options:
                        payload["options"] = options
                else:
                    payload = {
                        "model": args.model,
                        "messages": case["messages"],
                        "stream": False,
                        "think": args.think,
                        "options": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "num_predict": args.max_tokens,
                        },
                    }
                    if args.num_ctx > 0:
                        payload["options"]["num_ctx"] = args.num_ctx
                    if args.reasoning_effort:
                        payload["options"]["reasoning_effort"] = args.reasoning_effort
                    if args.verbosity:
                        payload["options"]["verbosity"] = args.verbosity

                started = time.perf_counter()
                resp = _post_json(endpoint, payload, args.timeout_seconds)
                elapsed = time.perf_counter() - started
            except urllib.error.HTTPError as exc:
                # Some Ollama /v1 500 responses can carry very large or malformed bodies.
                # Avoid blocking on reading the entire body; capture only lightweight metadata.
                reason = str(getattr(exc, "reason", "") or "").strip()
                body_preview = ""
                fp = getattr(exc, "fp", None)
                if fp is not None:
                    try:
                        raw = fp.read(4096)
                        if isinstance(raw, bytes) and raw:
                            body_preview = raw.decode("utf-8", errors="replace")
                    except Exception:
                        body_preview = ""
                if body_preview:
                    error_text = f"HTTP {exc.code}: {reason} {body_preview}"
                elif reason:
                    error_text = f"HTTP {exc.code}: {reason}"
                else:
                    error_text = f"HTTP {exc.code}"
                break
            except Exception as exc:
                error_text = f"request failed: {exc!r}"
                break

            parsed = (
                _extract_v1_response(resp)
                if args.endpoint_mode == "v1"
                else _extract_api_chat_response(resp)
            )
            content = parsed["content"]
            reasoning = parsed["reasoning"]
            prompt_tokens = parsed["prompt_tokens"]
            completion_tokens = parsed["completion_tokens"]
            total_tokens = parsed["total_tokens"]

            structured_content = _score_structured_debug(content, expected)
            style_content = _score_debug_style(content)
            combined = (reasoning + "\n" + content).strip() if reasoning else content
            structured_combined = _score_structured_debug(combined, expected)
            style_combined = _score_debug_style(combined)

            run_row = {
                "run_index": run_index,
                "latency_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "end_to_end_tokens_per_second": (
                    completion_tokens / max(elapsed, 1e-9) if completion_tokens > 0 else 0.0
                ),
                "content_preview": content[:400],
                "reasoning_preview": reasoning[:400],
                "structured_content": structured_content,
                "debug_style_content": style_content,
                "structured_combined": structured_combined,
                "debug_style_combined": style_combined,
                **(
                    {
                        "content_text": content,
                        "reasoning_text": reasoning,
                    }
                    if args.include_full_output
                    else {}
                ),
            }
            run_rows.append(run_row)

        if error_text is not None:
            log(f"[{run_pos}/{total_selected}] {name}: FAILED ({error_text})")
            bench_rows.append(
                {
                    "case_index": case_index,
                    "case_name": name,
                    "expected": expected,
                    "error": error_text,
                    "runs": run_rows,
                }
            )
            partial = {
                "model": args.model,
                "base_url": args.base_url,
                "endpoint_mode": args.endpoint_mode,
                "endpoint": endpoint,
                "messages_file": str(messages_path),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_ctx": args.num_ctx,
                "reasoning_effort": args.reasoning_effort,
                "verbosity": args.verbosity,
                "think": args.think,
                "runs_per_case": args.runs_per_case,
                "start_case_index": start_idx,
                "end_case_index": end_idx,
                "completed_cases": len(bench_rows),
                "aggregate": {
                    "num_cases": len(bench_rows),
                    "num_successful_cases": len([r for r in bench_rows if "error" not in r]),
                    "num_failed_cases": len([r for r in bench_rows if "error" in r]),
                },
                "results": bench_rows,
            }
            out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            continue

        avg_tps = statistics.mean(r["end_to_end_tokens_per_second"] for r in run_rows)
        avg_latency = statistics.mean(r["latency_seconds"] for r in run_rows)
        avg_structured_content = statistics.mean(
            r["structured_content"]["structured_score"] for r in run_rows
        )
        avg_structured_combined = statistics.mean(
            r["structured_combined"]["structured_score"] for r in run_rows
        )
        avg_style_content = statistics.mean(r["debug_style_content"]["style_score"] for r in run_rows)
        avg_style_combined = statistics.mean(
            r["debug_style_combined"]["style_score"] for r in run_rows
        )
        last = run_rows[-1]

        log(
            f"[{run_pos}/{total_selected}] {name}: "
            f"pt={last['prompt_tokens']} ct={last['completion_tokens']} "
            f"lat={avg_latency:.2f}s e2e_tps={avg_tps:.2f} "
            f"structured(content)={avg_structured_content:.2f} "
            f"structured(reasoning+content)={avg_structured_combined:.2f}"
        )
        log(
            f"  style(content)={avg_style_content:.2f} style(reasoning+content)={avg_style_combined:.2f}"
        )
        log(f"  content preview: {last['content_preview'][:180].replace(chr(10), ' ')}")

        bench_rows.append(
            {
                "case_name": name,
                "case_index": case_index,
                "expected": expected,
                "runs": run_rows,
                "avg_latency_seconds": avg_latency,
                "avg_end_to_end_tokens_per_second": avg_tps,
                "avg_structured_content_score": avg_structured_content,
                "avg_structured_combined_score": avg_structured_combined,
                "avg_debug_style_content_score": avg_style_content,
                "avg_debug_style_combined_score": avg_style_combined,
            }
        )

        # Write partial progress after each case.
        good_rows = [row for row in bench_rows if "error" not in row]
        partial = {
            "model": args.model,
            "base_url": args.base_url,
            "endpoint_mode": args.endpoint_mode,
            "endpoint": endpoint,
            "messages_file": str(messages_path),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_ctx": args.num_ctx,
            "reasoning_effort": args.reasoning_effort,
            "verbosity": args.verbosity,
            "think": args.think,
            "runs_per_case": args.runs_per_case,
            "start_case_index": start_idx,
            "end_case_index": end_idx,
            "completed_cases": len(bench_rows),
            "aggregate": {
                "num_cases": len(bench_rows),
                "num_successful_cases": len(good_rows),
                "num_failed_cases": len([row for row in bench_rows if "error" in row]),
                "avg_end_to_end_tokens_per_second": (
                    statistics.mean(row["avg_end_to_end_tokens_per_second"] for row in good_rows)
                    if good_rows
                    else None
                ),
                "avg_structured_content_score": (
                    statistics.mean(row["avg_structured_content_score"] for row in good_rows)
                    if good_rows
                    else None
                ),
                "avg_structured_combined_score": (
                    statistics.mean(row["avg_structured_combined_score"] for row in good_rows)
                    if good_rows
                    else None
                ),
                "avg_debug_style_content_score": (
                    statistics.mean(row["avg_debug_style_content_score"] for row in good_rows)
                    if good_rows
                    else None
                ),
                "avg_debug_style_combined_score": (
                    statistics.mean(row["avg_debug_style_combined_score"] for row in good_rows)
                    if good_rows
                    else None
                ),
            },
            "results": bench_rows,
        }
        out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    log("")
    log(f"saved: {out_path}")
    if progress_fh is not None:
        progress_fh.close()


if __name__ == "__main__":
    main()
