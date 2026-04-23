# Round 2 DPO Prepared Dataset Report

- created_at_utc: `2026-04-21T01:44:15Z`
- input_jsonl: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_706_rows_dbg_dpo_round2.jsonl`
- output_jsonl: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_706_rows_dbg_dpo_round2_prepared.jsonl`
- tokenizer_model: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model`
- rows: `706`
- output_sha256: `7214fb75ae90b294139874cf31613046d8dfa6490734ee5e4dfd6de1828cd87c`

## Recipe Budget
- max_prompt_length: `14848`
- max_completion_length: `1536`
- max_length: `16384`
- truncation_mode: `keep_end`

## Prompt Lengths
- summary: `{'min': 1005, 'mean': 5502, 'p50': 3573, 'p95': 14851, 'p99': 22854, 'max': 27049}`
- rows_over_prompt_budget: `36`

## Chosen Lengths
- summary: `{'min': 31, 'mean': 334, 'p50': 319, 'p95': 481, 'p99': 541, 'max': 1061}`
- rows_over_completion_budget: `0`

## Rejected Lengths
- summary: `{'min': 51, 'mean': 107, 'p50': 103, 'p95': 171, 'p99': 215, 'max': 378}`
- rows_over_completion_budget: `0`

## Side Lengths
- chosen_side_summary: `{'min': 1036, 'mean': 5836, 'p50': 3868, 'p95': 15139, 'p99': 23183, 'max': 27392}`
- rejected_side_summary: `{'min': 1063, 'mean': 5610, 'p50': 3680, 'p95': 14969, 'p99': 22943, 'max': 27117}`
- rows_over_side_budget: `24`

## Decision
- Use this prepared chat-templated view as the canonical DPO dataset input.
- Keep prompt budget at 14848 and completion budget at 1536.
- Start DPO from the finished round-2 full-FT model, not from scratch.
