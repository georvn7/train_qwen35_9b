# Round 2 Continuation

Date: `2026-04-18`

## Final Outcome

- Run completed successfully on `2026-04-20 07:35:58 PDT`.
- Session dir:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1`
- Final checkpoint:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/checkpoints/checkpoint-1869`
- Final full model:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model`
- Final trainer metrics:
  - `train_runtime=62581.8438` seconds (`~17h 23m 02s`)
  - `train_loss=0.23181603004614768`
  - `train_steps_per_second=0.03`
  - `epoch=1.0`

## Completion Notes

- Total train steps completed: `1869 / 1869`
- Final truncation stats:
  - `rows_truncated=144 / 1869`
  - `pct_rows_truncated=7.7047`
  - `max_original_tokens=90235`
  - `max_final_tokens=32768`
- Final full-model export completed cleanly after training.
- Export report:
  - merged `16-bit` export: `skipped`
  - `gguf` export: `skipped`

## Resume / Stability Notes

- The run did complete with the March `32K` recipe.
- During the long run, the external host-RAM guard tripped several times during resume/load phases, not during steady-state training.
- Recovery worked as designed:
  - the launcher retried automatically;
  - durable progress was preserved through checkpoints;
  - the run ultimately finished from the same session dir with final checkpoint `1869`.
- This means the recipe is valid for this continuation workload on Spark, but host-RAM headroom during resume remains a known operational sensitivity.

## Decision

- Continue from the previous full-FT model:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1/artifacts/full_model`
- Keep the training recipe as close as possible to the validated March 7 run:
  - `max_seq_length=32768`
  - `truncation_side=left`
  - `attn_implementation=sdpa`
  - `bf16`
  - `adamw_8bit`
  - `gradient_accumulation_steps=1`
  - `dataset_num_proc=1`
  - `cuda_memory_fraction=0.88`
  - `causal_loss_mode=active_chunked_no_upcast`
  - checkpoint shard / pre-save cleanup settings unchanged

## New Dataset Inputs

Round-2 dataset bundles:

- `/home/georvn/new_datasets/dataset2_dpo_sonnet46_oss120_oss20/dataset2_dpo_sonnet46_oss120_oss20`
- `/home/georvn/new_datasets/dataset_full_dpo_glm47_gpt120_gpt20_round3/dataset_full_dpo_glm47_gpt120_gpt20_round3`
- `/home/georvn/new_datasets/dataset_full_dpo_sonnet46_gpt5-oss120-oss2_round5/dataset_full_dpo_sonnet46_gpt5-oss120-oss2_round5`

Contained training files:

- `train_run_sft.jsonl`
- `train_dbg_sft.jsonl`
- `train_dbg_dpo.jsonl`

## Merged Local Outputs

Canonical round-2 merged datasets:

- SFT:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1869_rows_no_assistant_thinking_round2.jsonl`
- DPO:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_706_rows_dbg_dpo_round2.jsonl`

Prep report:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1869_rows_no_assistant_thinking_round2.meta.json`
- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1869_rows_no_assistant_thinking_round2.meta.md`

## Why Keep Truncated Rows

- We truncate from the start (`left`), so rows over `32768` keep the newest context and the final assistant target.
- This dataset family is debug-trace heavy, so preserving the tail is the correct bias.
- The new SFT union is heavier than the original March set, but dropping every over-`32K` row would throw away too much useful signal from the new round.

Measured pre-truncation token stats:

- Combined SFT rows: `1869`
- Combined SFT rows over `32768`: `144`
- `train_run_sft` rows over `32768`: `124 / 469`
- `train_dbg_sft` rows over `32768`: `20 / 1400`

Interpretation:

- `train_run_sft` is the heavy tail.
- `train_dbg_sft` remains mostly within the previous envelope.
- For the continuation SFT run, keep all rows and rely on the same left-truncation rule used in the validated recipe.

## DPO Plan

- Do not blend DPO into the first continuation SFT run.
- Keep real DPO as stage 2.
- Reason:
  - the SFT continuation can reuse the exact stable Spark recipe;
  - the DPO files are already in `prompt/chosen/rejected` format, but DPO needs its own trainer path and memory validation;
  - Spark stability is more important than maximizing change per run.

Observed DPO size:

- DPO rows: `706`
- Max chosen/rejected side over `32768`: `0`
- DPO `p95` max-side length: about `15137`

This makes DPO promising for a later stage, but still separate from the first continuation run.

## Canonical Commands

Build merged datasets:

```bash
cd /home/georvn/train_qwen35_9b
./.venv/bin/python qwen35_9b_fullft/scripts/prepare_round2_continuation_datasets.py \
  --input-root /home/georvn/new_datasets \
  --sft-output /home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1869_rows_no_assistant_thinking_round2.jsonl \
  --dpo-output /home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_706_rows_dbg_dpo_round2.jsonl
```

Launch continuation SFT run:

```bash
cd /home/georvn/train_qwen35_9b
qwen35_9b_fullft/scripts/run_train_qwen35_9b_round2_from_last_fullft_safe.sh
```

## Smoke Validation

- A 1-step continuation smoke from the previous full-FT model completed successfully on `2026-04-18`.
- Smoke session:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260418_215311_qwen35_9b_round2_cont_sft_1869_smoke1_fix1`
- What it proved:
  - previous full-FT model loads correctly as the new training starting point;
  - merged round-2 SFT dataset renders, left-truncates, and tokenizes correctly;
  - one full `32768`-context full-weight step completes;
  - checkpoint save completes with the same shard + pre-save cleanup policy as the March run.

Observed smoke details:

- dataset rows used: `1869`
- rows truncated by the pre-tokenization left-truncator: `144 / 1869`
- max original token length observed: `90235`
- max final token length after truncation: `32768`
- trainable parameters: `8,953,803,264 / 8,953,803,264`
- step-1 train loss: `1.633`
- step-1 runtime: about `59s`
- checkpoint pre-save peak reserved VRAM: about `89.9 GiB`
- checkpoint pre-save reported device usage before cleanup: about `90.7 GiB`
- checkpoint pre-save reported device usage after cleanup: about `41.7 GiB`

## Environment Compatibility Note

- The previous recipe itself did not need to change.
- A local compatibility patch was applied in the venv at:
  - `/home/georvn/train_qwen35_9b/.venv/lib/python3.12/site-packages/unsloth/models/_utils.py`
- Patch purpose:
  - import `auto_docstring` so current `Unsloth 2026.3.3` can re-exec modern `transformers` config classes that use `@auto_docstring`.
- This is an environment fix only. It is not a training-recipe change and it is not committed as repo code.
