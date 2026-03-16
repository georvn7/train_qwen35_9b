# Schema20 Eval Gate (Reused from 120B Workflow)

Date: `2026-03-05`

## Fixed Eval Set

- Cases file:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/evals/agent_cases_20_schema_final_v1.json`
- Case count: `20`

## Benchmark Runner

- Script:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/benchmark_ollama_chat.py`
- Endpoint mode used for gating:
  - Ollama OpenAI-compatible `/v1/chat/completions`
- Decode controls:
  - `temperature=0.0`
  - `top_p=1.0`
  - `num_ctx=32768`
  - `max_tokens=4096`
  - `reasoning_effort=medium`
  - `verbosity=medium`

## Structured Scoring (Exact Reuse)

For each case response, structured score uses these weights:

- `action_type` match: `0.30`
- `action_subject` match (exact/regex): `0.30`
- `line_number` match: `0.15`
- `breakpoints` match: `0.15`
- verbosity bound match: `0.10`

JSON parse failure yields `0` structured score.

## A/B Strict Gate

Comparator script:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/compare_schema20_ab.py`

Promotion requires all of:

- candidate has `0` failed cases,
- no new failures vs baseline,
- no regression in aggregate structured(content),
- no regression in aggregate structured(reasoning+content),
- candidate failed-cases not greater than baseline.

## One-Command Runner

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/run_ab_schema20_ollama.sh`
