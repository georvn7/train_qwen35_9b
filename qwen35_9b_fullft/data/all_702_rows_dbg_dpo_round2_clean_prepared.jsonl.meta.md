# Round 2 DPO Prepared Dataset Report

- created_at_utc: `2026-04-21T03:22:11Z`
- input_jsonl: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean.jsonl`
- output_jsonl: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean_prepared.jsonl`
- tokenizer_model: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model`
- rows: `702`
- output_sha256: `a1cffa42f265bec3ed3ab10c3d4b7f667c30cf993c88ade5e362a2b55503dc3c`

## Recipe Budget
- max_prompt_length: `14848`
- max_completion_length: `1536`
- max_length: `16384`
- truncation_mode: `keep_end`

## Prompt Lengths
- summary: `{'min': 1251, 'mean': 5527, 'p50': 3586, 'p95': 14851, 'p99': 22854, 'max': 27049}`
- rows_over_prompt_budget: `36`

## Chosen Lengths
- summary: `{'min': 142, 'mean': 346, 'p50': 329, 'p95': 491, 'p99': 551, 'max': 1085}`
- rows_over_completion_budget: `0`

## Rejected Lengths
- summary: `{'min': 65, 'mean': 118, 'p50': 114, 'p95': 181, 'p99': 225, 'max': 402}`
- rows_over_completion_budget: `0`

## Side Lengths
- chosen_side_summary: `{'min': 1486, 'mean': 5874, 'p50': 3936, 'p95': 15149, 'p99': 23193, 'max': 27402}`
- rejected_side_summary: `{'min': 1347, 'mean': 5646, 'p50': 3711, 'p95': 14979, 'p99': 22953, 'max': 27127}`
- rows_over_side_budget: `24`

## Decision
- Use this prepared chat-templated view as the canonical DPO dataset input.
- Keep prompt budget at 14848 and completion budget at 1536.
- Start DPO from the finished round-2 full-FT model, not from scratch.
