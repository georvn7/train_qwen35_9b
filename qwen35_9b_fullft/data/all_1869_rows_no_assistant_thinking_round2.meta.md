# Round 2 Dataset Prep Report

- created_at_utc: `2026-04-18T21:49:14Z`
- input_root: `/home/georvn/new_datasets`
- tokenizer_model: `Qwen/Qwen3.5-9B`
- max_seq_length_reference: `32768`

## Outputs
- SFT: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1869_rows_no_assistant_thinking_round2.jsonl`
- SFT rows: `1869`
- SFT sha256: `b0d3c21f330d8a73f3cf96487508efe4f2710360acf397bc085eb3e70b3ca018`
- DPO: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_706_rows_dbg_dpo_round2.jsonl`
- DPO rows: `706`
- DPO sha256: `82b945af6c86a51bc56eccf1a7035d2cd7d53ea4dd097f8525801f8a35039b4e`

## SFT Inputs
- `train_run_sft.jsonl` files: `138`
- `train_run_sft.jsonl` rows: `469`
- `train_dbg_sft.jsonl` files: `144`
- `train_dbg_sft.jsonl` rows: `1400`
- combined SFT rows: `1869`

## SFT Truncation Stats
- `train_run_sft` rows over `32768`: `124`
- `train_dbg_sft` rows over `32768`: `20`
- combined SFT rows over `32768`: `144`
- combined SFT `p95`: `38386`
- combined SFT `max`: `90235`

## DPO Inputs
- `train_dbg_dpo.jsonl` files: `144`
- `train_dbg_dpo.jsonl` rows: `706`

## DPO Length Stats
- DPO pairs with chosen/rejected side over `32768`: `0`
- DPO `p95` max-side length: `15137`
- DPO max-side length: `27390`

## Decision
- Keep overlength SFT rows in the merged continuation dataset.
- Use left truncation (`truncation_side=left`) in training so the newest context and final assistant target are preserved.
- Keep DPO as a separate second-stage dataset; do not fold it into the first continuation SFT run.
