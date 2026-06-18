# Hayabusa 9B V3 Rare-Actions SFT, Final-Assistant-Only

Date: 2026-06-17

## Objective

Continue training from the completed v3 main SFT model, but change SFT loss semantics for v3 rare-actions onward:

- Keep the full trace as context.
- Compute loss only on the final assistant message in each trace.
- Preserve any final assistant thinking/reasoning when it is present in the source row.
- For this run, use the no-assistant-thinking rare-actions dataset.

This avoids training again on historical assistant turns inside a trace. Those historical turns are context only; each step should be supervised in the row where it is the final/current assistant turn.

## Starting Model

The run starts from the completed v3 main SFT full model:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260615_164632_hayabusa-qwen-9b_v3_main_sft_2611_from_rare_32k_v1/artifacts/full_model
```

The human-facing model name for this continuation is:

```text
hayabusa-9b
```

## Dataset

Training dataset:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/super-debug-v3-rare-actions-no-assistant-thinking.jsonl
```

Source dataset file:

```text
/home/georvn/train_qwen35_9b/super-debug-v3/super-debug-v3-rare-actions-no-assistant-thinking.jsonl
```

Reference-only with-thinking variant, not used for this run:

```text
/home/georvn/train_qwen35_9b/super-debug-v3/super-debug-v3-rare-actions-with-assistant-thinking.jsonl
```

## Label Semantics

The new `train_session.py` option is:

```text
--loss-target final_assistant
```

Implementation details:

- The dataset is pre-tokenized before trainer dataset preparation.
- Each row produces `input_ids` and `labels` directly.
- Labels are `-100` for all tokens before the final assistant target.
- Labels are real token IDs only for the final assistant message target.
- TRL `assistant_only_loss` is disabled effectively for this mode, because the explicit labels already define the loss mask.
- The full context remains in `input_ids`, so the model sees prior system/user/assistant turns.
- If a row has assistant `thinking` or `reasoning_content`, it is normalized into the Qwen chat template's `reasoning_content` field.
- If a row has top-level `motivation`, it is preserved unless already embedded in the assistant `content` JSON.

The Qwen chat template tokenizes the generation boundary slightly differently for full rows vs prompt-only rows around empty `<think>` blocks. The implementation uses the longest common token prefix when the boundary adjustment is tiny. Validation showed:

```text
boundary_adjusted_rows = 562
max_boundary_adjustment_tokens = 1
```

This is expected for no-thinking rows because the full target begins with the empty-think close sequence.

## Tokenization Validation

Tokenizer-only validation was run before GPU training.

Validation output directory:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/metadata/final_assistant_validation_hayabusa_9b_v3_rare
```

Key stats:

```text
rows = 562
max_seq_length = 32768
truncation_side = left
rows_truncated = 13
pct_rows_truncated = 2.3132
target_partially_truncated = 0
max_original_tokens = 40178
max_final_tokens = 32768
min_supervised_tokens_after_truncation = 95
max_supervised_tokens_after_truncation = 634
avg_supervised_tokens_after_truncation = 200.6708
```

Decoded preview file:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/metadata/final_assistant_validation_hayabusa_9b_v3_rare/final_assistant_supervised_span_previews.jsonl
```

Manual inspection confirmed the decoded supervised spans contain only the final assistant output, for example:

```text
</think>

{"action_subject":..., "action_type":..., "motivation":...}<|im_end|>
```

No previous assistant turn is labeled.

## Smoke Test

A 1-step GPU smoke test was completed before the full run.

Smoke session:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260617_163019_hayabusa-9b_v3_rare_actions_finalonly_smoke1_20260617_093019
```

Smoke result:

```text
steps = 1 / 1
train_loss = 1.164
grad_norm = 44.5
final_save = skipped
```

The smoke confirmed:

- Full fine-tuning, not LoRA/QLoRA.
- `8,953,803,264 / 8,953,803,264` parameters trainable.
- Final-assistant-only labels work through the trainer and loss path.
- No final model artifact was written for the smoke run.

## Full Run

Launcher:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/run_train_hayabusa_9b_v3_rare_actions_from_v3_main_final_only_safe.sh
```

Run session:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260617_163331_hayabusa-9b_v3_rare_actions_sft_562_from_v3_main_32k_finalonly_v1
```

Launch log:

```text
/home/georvn/train_qwen35_9b/logs/launch_hayabusa-9b_v3_rare_actions_sft_562_from_v3_main_32k_finalonly_v1.log
```

Recipe:

```text
max_seq_length = 32768
truncation_side = left
precision = bf16
optimizer = adamw_8bit
batch_size = 1
gradient_accumulation_steps = 1
gradient_checkpointing = unsloth
attention = sdpa
learning_rate = 1e-5
warmup_steps = 50
num_train_epochs = 1
save_steps = 50
save_total_limit = 4
checkpoint_max_shard_size = 512MB
checkpoint_safe_serialization = true
causal_loss_mode = active_chunked_no_upcast
causal_loss_chunk_tokens = 2048
cuda_memory_fraction = 0.88
max_gpu_memory_gib = 110
min_mem_avail_mib = 1536
```

Full-run startup confirmation:

```text
num_examples = 562
total_steps = 562
trainable_parameters = 8,953,803,264 / 8,953,803,264
loss_target = final_assistant
rows_truncated = 13 / 562
target_partially_truncated = 0
```
