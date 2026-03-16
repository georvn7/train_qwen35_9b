# Quality Recipe (DGX Spark, Qwen3.5-9B Full FT)

Date: `2026-03-05`

## Selected Recipe

- Model: `Qwen/Qwen3.5-9B` (post-trained instruct model)
- Training mode: full finetuning (`--full-finetuning`, `--no-load-in-4bit`)
- Context: `max_seq_length=12288`, `truncation_side=left`
- Precision/optimizer: `bf16` + `adamw_8bit`
- LR schedule: `learning_rate=1e-5`, `warmup_steps=50`, `lr_scheduler_type=linear`
- Batch/update shape: `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`
- Dataset preprocessing workers: `dataset_num_proc=1`
- Stability flags:
  - `--disable-unsloth-compile`
  - `--disable-moe-triton`
  - `--disable-flex-attention`
  - `--disable-cce`
- Memory guard:
  - `max_gpu_memory_gib=110`
- Checkpoint policy (full-FT-safe):
  - `save_steps=50`
  - `save_total_limit=4`

## Why This Recipe

- User-directed model policy is to keep post-training behavior and avoid base-only training.
- Initial instruct probe at `2e-5` had high step-1 loss/grad norm, so full-run LR is reduced to `1e-5` with longer warmup (`50`) for safer updates.
- Repeated 32K, 24K, and 16K full-FT probes were OOM-killed (or guard-tripped at the same boundary). On DGX Spark, `12288` is the first verified stable context for this full-weight recipe.
- Same stability flags and checkpoint cadence are kept from the validated Spark recipe.

## Key Evidence

- LR sweep summary:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/docs/quality_probe_20260305_090939.json`
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/docs/quality_probe_20260305_090939.md`
- Longer-subset confirmation:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_093103_confirm_long320_lr2e5_2step`
- Instruct-model comparison probe:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_093920_probe_instruct1step_64rows_lr2e5`
- Stability proof run for memory envelope:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_182218_qwen35_9b_instruct_memprobe_12k_guard110`
  - Completed `12/12` steps, wrote `checkpoint-12`, and finalized full-model artifacts.

## Launch Command

```bash
WORK=/home/georvn/train_qwen35_9b \
MODEL_NAME=Qwen/Qwen3.5-9B \
MAX_SEQ_LENGTH=12288 \
MAX_GPU_MEMORY_GIB=110 \
DATASET_NUM_PROC=1 \
GRADIENT_CHECKPOINTING=unsloth \
LEARNING_RATE=1e-5 \
WARMUP_STEPS=50 \
GRADIENT_ACCUMULATION_STEPS=1 \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=4 \
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/run_train_qwen35_9b_full1109_resume_safe.sh
```
