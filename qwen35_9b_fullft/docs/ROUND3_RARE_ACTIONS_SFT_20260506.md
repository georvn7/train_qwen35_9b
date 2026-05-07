# Round 3 Rare-Actions SFT

Date: `2026-05-06`

## Purpose

Continue SFT from the finished round-2 DPO model using the rare-actions no-thinking dataset.

This is not a from-scratch run. The starting model is the last DPO-trained full model:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1/artifacts/full_model`

Training dataset:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/super-debug-v2-rare-actions-no-assistant-thinking.jsonl`

## Decision

Use the same Spark-safe 32K SFT recipe that completed the previous SFT runs, including the same checkpoint-based recovery strategy.

Key reason: this dataset is another no-thinking SFT continuation workload, so changing the recipe would add unnecessary risk. The DPO phase is complete; this run returns to supervised fine-tuning.

## Dataset Stats

Measured with the `Qwen/Qwen3.5-9B` tokenizer before training:

- rows: `841`
- bad JSON rows: `0`
- assistant thinking fields: `0`
- system messages: `841`
- user messages: `4169`
- assistant messages: `4169`
- min tokens: `1650`
- mean tokens: `13147`
- p50 tokens: `7295`
- p90 tokens: `27748`
- p95 tokens: `54249`
- p99 tokens: `87314`
- max tokens: `115527`
- rows over `32768`: `76`

Training uses left truncation, so over-budget rows keep the newest context and final assistant target.

Preprocessing in the production run confirmed:

- `rows_truncated=76/841`
- `max_original_tokens=115527`
- `max_final_tokens=32768`

## Recipe

- model start: finished round-2 DPO `artifacts/full_model`
- training mode: full fine-tune, no LoRA, no QLoRA
- max sequence length: `32768`
- truncation side: `left`
- precision: `bf16`
- optimizer path: `adamw_8bit`
- batch size per device: `1`
- gradient accumulation steps: `1`
- gradient checkpointing: `unsloth`
- attention implementation: `sdpa`
- learning rate: `1e-5`
- warmup steps: `50`
- epochs: `1`
- dataset workers: `1`
- assistant-only loss: enabled
- packing: disabled
- causal loss mode: `active_chunked_no_upcast`
- causal loss chunk tokens: `2048`
- CUDA allocator config: `expandable_segments:True,max_split_size_mb:256`
- CUDA memory fraction: `0.88`
- internal GPU guard: `110 GiB`

## Checkpoint Strategy

Same checkpoint-based strategy as the previous successful SFT run:

- `save_steps=50`
- `save_total_limit=4`
- checkpoint shard size: `512MB`
- safe serialization: enabled
- checkpoint pre-save `gc.collect()`: enabled
- checkpoint pre-save `torch.cuda.empty_cache()`: enabled
- checkpoint pre-save CUDA history disable: enabled
- resume load mmap: enabled

Operational rule:

- Resume from the newest valid checkpoint in the same session lineage.
- Do not restart from the DPO model if a valid run checkpoint exists.
- Keep checkpoint cadence dense enough to avoid losing many hours on Spark restarts.

## Launcher

Dedicated launcher:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/run_train_qwen35_9b_rare_actions_from_round2_dpo_safe.sh`

It delegates to the existing resume-safe SFT launcher:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/run_train_qwen35_9b_full1109_resume_safe.sh`

Production label:

- `qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1`

Production session:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1`

Session pointer:

- `/home/georvn/train_qwen35_9b/.state/session_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1.txt`

Log:

- `/home/georvn/train_qwen35_9b/logs/train_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1.log`

## Smoke Validation

One-step smoke session:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_044331_qwen35_9b_rare_actions_sft_from_round2_dpo_841_smoke1_v1`

Smoke result:

- loaded the DPO model successfully
- rendered, left-truncated, and tokenized the rare-actions dataset
- completed one full-weight backward step
- saved a checkpoint
- exported a full model artifact
- `train_loss=0.5703`

Smoke memory:

- pre-save RSS: `1457.4 MiB`
- pre-save MemAvailable: `57810.0 MiB`
- peak reserved torch memory: `109610.0 MiB`
- pre-save NVIDIA process memory: `60139.0 MiB`
- post-cleanup MemAvailable: `75818.7 MiB`
- post-cleanup NVIDIA process memory: `42471.0 MiB`

## Final Outcome

The production run completed successfully on `2026-05-06 17:22:01 PDT`.

Final session:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1`

Final checkpoint:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1/checkpoints/checkpoint-841`
- size: `40G`

Final full model:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1/artifacts/full_model`
- size: `17G`

Retained checkpoints after `save_total_limit=4`:

- `checkpoint-700`
- `checkpoint-750`
- `checkpoint-800`
- `checkpoint-841`

Final trainer metrics:

- steps: `841 / 841`
- epoch: `1`
- `train_runtime=70080` seconds (`~19h 28m`)
- `train_samples_per_second=0.012`
- `train_steps_per_second=0.012`
- `train_loss=0.3686`

Final logged loss block:

- step `805`: `0.3322`
- step `810`: `0.3959`
- step `815`: `0.3434`
- step `820`: `0.2368`
- step `825`: `0.2962`
- step `830`: `0.3028`
- step `835`: `0.3509`
- step `840`: `0.2457`

Final checkpoint/export memory:

- final checkpoint pre-save memory:
  - RSS: `2656.2 MiB`
  - MemAvailable: `48793.0 MiB`
  - torch peak reserved: `109668.0 MiB`
  - NVIDIA process memory: `67203.0 MiB`
- final checkpoint after cleanup:
  - MemAvailable: `74556.7 MiB`
  - NVIDIA process memory: `41953.0 MiB`
- post-save memory:
  - RSS: `2672.2 MiB`
  - MemAvailable: `73115.2 MiB`
  - torch peak reserved: `109668.0 MiB`
  - NVIDIA process memory: `41953.0 MiB`

## Completion Notes

- Final full-model export completed cleanly.
- No OOM, crash, or manual resume was needed after production launch.
- Checkpoint cleanup and retention behaved as intended.
- Next useful step is schema20/local quality probing against the previous DPO model and this rare-actions SFT continuation model.
