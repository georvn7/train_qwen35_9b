#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Canonical single-user resident profile (stable API attrs for agents).
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8002}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b-fullft-bf16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

exec "${SCRIPT_DIR}/start_vllm_fullft_bf16_openai.sh"
