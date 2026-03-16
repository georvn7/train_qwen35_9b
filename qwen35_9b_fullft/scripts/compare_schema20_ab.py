#!/usr/bin/env python3
"""Compare two schema20 benchmark result JSON files and emit strict-gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate schema20 benchmark outputs.")
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--compare-json", required=True)
    parser.add_argument("--summary-md", required=True)
    return parser.parse_args()


def avg(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def by_index(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("results", []):
        idx = row.get("case_index")
        if isinstance(idx, int):
            out[idx] = row
    return out


def build_comparison(
    base: dict[str, Any],
    cand: dict[str, Any],
    baseline_model: str,
    candidate_model: str,
) -> dict[str, Any]:
    b = by_index(base)
    c = by_index(cand)
    indices = sorted(set(b) | set(c))

    rows: list[dict[str, Any]] = []
    for idx in indices:
        br = b.get(idx)
        cr = c.get(idx)
        row: dict[str, Any] = {
            "case_index": idx,
            "case_name": (br or cr or {}).get("case_name"),
            "baseline_error": br is None or ("error" in br),
            "candidate_error": cr is None or ("error" in cr),
        }
        if br is not None and "error" not in br:
            row["baseline"] = {
                "structured_content": br.get("avg_structured_content_score"),
                "structured_combined": br.get("avg_structured_combined_score"),
                "e2e_tps": br.get("avg_end_to_end_tokens_per_second"),
                "completion_tokens": br.get("completion_tokens"),
            }
        if cr is not None and "error" not in cr:
            row["candidate"] = {
                "structured_content": cr.get("avg_structured_content_score"),
                "structured_combined": cr.get("avg_structured_combined_score"),
                "e2e_tps": cr.get("avg_end_to_end_tokens_per_second"),
                "completion_tokens": cr.get("completion_tokens"),
            }
        if "baseline" in row and "candidate" in row:
            row["delta"] = {
                "structured_content": row["candidate"]["structured_content"] - row["baseline"]["structured_content"],
                "structured_combined": row["candidate"]["structured_combined"] - row["baseline"]["structured_combined"],
                "e2e_tps": row["candidate"]["e2e_tps"] - row["baseline"]["e2e_tps"],
            }
        rows.append(row)

    ok_rows = [
        r for r in rows if (not r["baseline_error"]) and (not r["candidate_error"]) and ("delta" in r)
    ]
    new_failures = [r for r in rows if (not r["baseline_error"]) and r["candidate_error"]]

    base_agg = base.get("aggregate", {})
    cand_agg = cand.get("aggregate", {})
    no_regression = (
        (cand_agg.get("num_failed_cases", 999) <= base_agg.get("num_failed_cases", 999))
        and ((cand_agg.get("avg_structured_content_score") or -1.0) >= (base_agg.get("avg_structured_content_score") or -1.0))
        and ((cand_agg.get("avg_structured_combined_score") or -1.0) >= (base_agg.get("avg_structured_combined_score") or -1.0))
    )
    candidate_clean = (cand_agg.get("num_failed_cases", 999) == 0) and (cand_agg.get("num_successful_cases", 0) > 0)
    strict_gate = no_regression and candidate_clean and (len(new_failures) == 0)

    return {
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "num_cases": len(indices),
        "num_pairwise_success": len(ok_rows),
        "num_new_failures": len(new_failures),
        "aggregate": {
            "baseline": base_agg,
            "candidate": cand_agg,
            "delta": {
                "avg_structured_content_score": (cand_agg.get("avg_structured_content_score") or 0.0)
                - (base_agg.get("avg_structured_content_score") or 0.0),
                "avg_structured_combined_score": (cand_agg.get("avg_structured_combined_score") or 0.0)
                - (base_agg.get("avg_structured_combined_score") or 0.0),
                "avg_end_to_end_tokens_per_second": (cand_agg.get("avg_end_to_end_tokens_per_second") or 0.0)
                - (base_agg.get("avg_end_to_end_tokens_per_second") or 0.0),
            },
        },
        "pairwise_avg_delta": {
            "structured_content": avg([r["delta"]["structured_content"] for r in ok_rows]),
            "structured_combined": avg([r["delta"]["structured_combined"] for r in ok_rows]),
            "e2e_tps": avg([r["delta"]["e2e_tps"] for r in ok_rows]),
        },
        "strict_gate_promote": strict_gate,
        "new_failures": [
            {"case_index": r["case_index"], "case_name": r["case_name"]} for r in new_failures
        ],
        "rows": rows,
    }


def write_summary(path: Path, cmp: dict[str, Any]) -> None:
    base_agg = cmp["aggregate"]["baseline"]
    cand_agg = cmp["aggregate"]["candidate"]
    delta_agg = cmp["aggregate"]["delta"]

    lines = [
        "# A/B Summary (schema20 strict gate)",
        "",
        "## Aggregate",
        f"- Baseline model: `{cmp['baseline_model']}`",
        f"- Candidate model: `{cmp['candidate_model']}`",
        f"- Cases: {cmp['num_cases']}",
        f"- Baseline success/fail: {base_agg.get('num_successful_cases')}/{base_agg.get('num_failed_cases')}",
        f"- Candidate success/fail: {cand_agg.get('num_successful_cases')}/{cand_agg.get('num_failed_cases')}",
        f"- Baseline avg structured(content): {base_agg.get('avg_structured_content_score')}",
        f"- Candidate avg structured(content): {cand_agg.get('avg_structured_content_score')}",
        f"- Delta structured(content): {delta_agg.get('avg_structured_content_score')}",
        f"- Baseline avg structured(reasoning+content): {base_agg.get('avg_structured_combined_score')}",
        f"- Candidate avg structured(reasoning+content): {cand_agg.get('avg_structured_combined_score')}",
        f"- Delta structured(reasoning+content): {delta_agg.get('avg_structured_combined_score')}",
        f"- Baseline avg e2e t/s: {base_agg.get('avg_end_to_end_tokens_per_second')}",
        f"- Candidate avg e2e t/s: {cand_agg.get('avg_end_to_end_tokens_per_second')}",
        f"- Delta e2e t/s: {delta_agg.get('avg_end_to_end_tokens_per_second')}",
        "",
        "## Pairwise (success-success only)",
        f"- Compared cases: {cmp['num_pairwise_success']}",
        f"- Avg delta structured(content): {cmp['pairwise_avg_delta']['structured_content']}",
        f"- Avg delta structured(reasoning+content): {cmp['pairwise_avg_delta']['structured_combined']}",
        f"- Avg delta e2e t/s: {cmp['pairwise_avg_delta']['e2e_tps']}",
        "",
        "## New Failures (baseline success -> candidate fail)",
    ]
    if cmp["new_failures"]:
        for row in cmp["new_failures"]:
            lines.append(f"- case {row['case_index']:02d}: {row['case_name']}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Strict Gate",
        f"- Promote candidate: {'YES' if cmp['strict_gate_promote'] else 'NO'}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))

    comparison = build_comparison(base, cand, args.baseline_model, args.candidate_model)

    compare_path = Path(args.compare_json)
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(summary_path, comparison)

    print(f"compare_json_written {compare_path}")
    print(f"summary_written {summary_path}")
    print(f"strict_gate_promote {comparison['strict_gate_promote']}")


if __name__ == "__main__":
    main()
