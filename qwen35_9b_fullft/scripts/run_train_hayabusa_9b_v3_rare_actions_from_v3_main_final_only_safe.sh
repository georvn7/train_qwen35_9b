#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-/home/georvn/train_qwen35_9b}"
START_FULL_MODEL="${START_FULL_MODEL:-$WORK/qwen35_9b_fullft/runs/20260615_164632_hayabusa-qwen-9b_v3_main_sft_2611_from_rare_32k_v1/artifacts/full_model}"
DATASET_ROOT="${DATASET_ROOT:-$WORK/qwen35_9b_fullft/data/super-debug-v3-rare-actions-no-assistant-thinking.jsonl}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-hayabusa-9b}"
LABEL="${LABEL:-hayabusa-9b_v3_rare_actions_sft_562_from_v3_main_32k_finalonly_v1}"
SESSION_PTR="${SESSION_PTR:-$WORK/.state/session_${LABEL}.txt}"
RUN_LOG="${RUN_LOG:-$WORK/logs/train_${LABEL}.log}"
FAILED_STEPS_TODO="${FAILED_STEPS_TODO:-$WORK/.state/train_${LABEL}_failed_steps_todo.txt}"
SESSION_NOTES="${SESSION_NOTES:-hayabusa-9b: Qwen3.5-9B continuation SFT on super-debug-v3 rare-actions no-assistant-thinking dataset, starting from completed hayabusa-qwen-9b v3 main model, with explicit final-assistant-only labels.}"

if [[ ! -d "$START_FULL_MODEL" ]]; then
  echo "ERROR: start full model not found: $START_FULL_MODEL" >&2
  exit 1
fi

if [[ ! -f "$DATASET_ROOT" ]]; then
  echo "ERROR: super-debug-v3 rare-actions dataset not found: $DATASET_ROOT" >&2
  exit 1
fi

export WORK
export MODEL_NAME="${MODEL_NAME:-$START_FULL_MODEL}"
export MODEL_DISPLAY_NAME
export DATASET_ROOT
export LABEL
export SESSION_PTR
export RUN_LOG
export FAILED_STEPS_TODO
export SESSION_NOTES

export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
export TRUNCATION_SIDE="${TRUNCATION_SIDE:-left}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-unsloth}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-50}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
export MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-110}"
export CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.88}"
export CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export CAUSAL_LOSS_MODE="${CAUSAL_LOSS_MODE:-active_chunked_no_upcast}"
export CAUSAL_LOSS_CHUNK_TOKENS="${CAUSAL_LOSS_CHUNK_TOKENS:-2048}"
export CHECKPOINT_MAX_SHARD_SIZE="${CHECKPOINT_MAX_SHARD_SIZE:-512MB}"
export CHECKPOINT_SAFE_SERIALIZATION="${CHECKPOINT_SAFE_SERIALIZATION:-true}"
export CHECKPOINT_PRESAVE_GC="${CHECKPOINT_PRESAVE_GC:-1}"
export CHECKPOINT_PRESAVE_EMPTY_CACHE="${CHECKPOINT_PRESAVE_EMPTY_CACHE:-1}"
export CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY="${CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY:-1}"
export RESUME_TORCH_LOAD_MMAP="${RESUME_TORCH_LOAD_MMAP:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
export SEED="${SEED:-3413}"

# New v3+ SFT semantics: full context in the prompt, loss only on the final assistant turn.
export LOSS_TARGET="${LOSS_TARGET:-final_assistant}"
export FINAL_ASSISTANT_PREVIEW_ROWS="${FINAL_ASSISTANT_PREVIEW_ROWS:-8}"
export FINAL_ASSISTANT_PREVIEW_MAX_CHARS="${FINAL_ASSISTANT_PREVIEW_MAX_CHARS:-2400}"

exec "$WORK/qwen35_9b_fullft/scripts/run_train_qwen35_9b_full1109_resume_safe.sh"
