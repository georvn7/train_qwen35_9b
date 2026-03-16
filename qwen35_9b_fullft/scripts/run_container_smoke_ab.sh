#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Usage:
  run_container_smoke_ab.sh <label> <optimizer> [max_steps] [save_strategy] [save_steps] [cuda_memory_fraction] [attn_implementation] [cce_mode] [gpu_guard_gib] [min_mem_avail_mib] [internal_gpu_guard_gib] [cuda_alloc_conf] [causal_loss_mode] [causal_loss_chunk_tokens] [save_only_model]

Example:
  run_container_smoke_ab.sh smoke32k_paged paged_adamw_8bit 1 no 0 0 sdpa enable 100 8192 100 expandable_segments:True,max_split_size_mb:256 chunked_fp32 2048
USAGE
  exit 1
fi

LABEL="$1"
OPTIMIZER="$2"
MAX_STEPS="${3:-1}"
SAVE_STRATEGY="${4:-no}"
SAVE_STEPS="${5:-0}"
CUDA_MEMORY_FRACTION="${6:-0}"
ATTN_IMPLEMENTATION="${7:-sdpa}"
CCE_MODE="${8:-enable}"
GPU_GUARD_GIB="${9:-100}"
MIN_MEM_AVAIL_MIB="${10:-8192}"
INTERNAL_GPU_GUARD_GIB="${11:-100}"
CUDA_ALLOC_CONF="${12:-expandable_segments:True,max_split_size_mb:256}"
CAUSAL_LOSS_MODE="${13:-default}"
CAUSAL_LOSS_CHUNK_TOKENS="${14:-2048}"
SAVE_ONLY_MODEL="${15:-false}"
CHECKPOINT_MAX_SHARD_SIZE="${CHECKPOINT_MAX_SHARD_SIZE:-512MB}"
CHECKPOINT_SAFE_SERIALIZATION="${CHECKPOINT_SAFE_SERIALIZATION:-true}"
CHECKPOINT_PRESAVE_GC="${CHECKPOINT_PRESAVE_GC:-1}"
CHECKPOINT_PRESAVE_EMPTY_CACHE="${CHECKPOINT_PRESAVE_EMPTY_CACHE:-1}"
CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY="${CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY:-1}"
ENABLE_CUDA_DEBUG_HISTORY="${ENABLE_CUDA_DEBUG_HISTORY:-0}"
DEBUG_CUDA_HISTORY_MAX_ENTRIES="${DEBUG_CUDA_HISTORY_MAX_ENTRIES:-50000}"
DEBUG_CUDA_SNAPSHOT_ON_ERROR="${DEBUG_CUDA_SNAPSHOT_ON_ERROR:-0}"

CCE_FLAG="--disable-cce"
if [[ "$CCE_MODE" == "enable" ]]; then
  CCE_FLAG="--enable-cce"
fi
SAVE_ONLY_MODEL_FLAG="--no-save-only-model"
if [[ "$SAVE_ONLY_MODEL" == "true" ]]; then
  SAVE_ONLY_MODEL_FLAG="--save-only-model"
fi
CHECKPOINT_PRESAVE_GC_FLAG="--checkpoint-presave-gc"
if [[ "$CHECKPOINT_PRESAVE_GC" == "0" || "$CHECKPOINT_PRESAVE_GC" == "false" ]]; then
  CHECKPOINT_PRESAVE_GC_FLAG="--no-checkpoint-presave-gc"
fi
CHECKPOINT_PRESAVE_EMPTY_CACHE_FLAG="--checkpoint-presave-empty-cache"
if [[ "$CHECKPOINT_PRESAVE_EMPTY_CACHE" == "0" || "$CHECKPOINT_PRESAVE_EMPTY_CACHE" == "false" ]]; then
  CHECKPOINT_PRESAVE_EMPTY_CACHE_FLAG="--no-checkpoint-presave-empty-cache"
fi
CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY_FLAG="--checkpoint-presave-disable-cuda-history"
if [[ "$CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY" == "0" || "$CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY" == "false" ]]; then
  CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY_FLAG="--no-checkpoint-presave-disable-cuda-history"
fi
DEBUG_HISTORY_FLAG=""
DEBUG_HISTORY_MAX_ENTRIES_ARG=""
if [[ "$ENABLE_CUDA_DEBUG_HISTORY" == "1" || "$ENABLE_CUDA_DEBUG_HISTORY" == "true" ]]; then
  DEBUG_HISTORY_FLAG="--debug-cuda-memory-history"
  DEBUG_HISTORY_MAX_ENTRIES_ARG="--debug-cuda-memory-history-max-entries $DEBUG_CUDA_HISTORY_MAX_ENTRIES"
fi
DEBUG_SNAPSHOT_FLAG=""
if [[ "$DEBUG_CUDA_SNAPSHOT_ON_ERROR" == "1" || "$DEBUG_CUDA_SNAPSHOT_ON_ERROR" == "true" ]]; then
  DEBUG_SNAPSHOT_FLAG="--debug-cuda-snapshot-on-error"
fi
GPU_GUARD_MIB=$((GPU_GUARD_GIB * 1024))

WORKSPACE="/home/georvn/train_qwen35_9b"
RUNS_ROOT="$WORKSPACE/qwen35_9b_fullft/runs"
DATA_ROOT="$WORKSPACE/qwen35_9b_fullft/data"
SCRIPT_CREATE="$WORKSPACE/qwen35_9b_fullft/scripts/create_session.py"
SCRIPT_TRAIN="$WORKSPACE/qwen35_9b_fullft/scripts/train_session.py"
LOG_ROOT="$WORKSPACE/logs"

mkdir -p "$LOG_ROOT"

./.venv/bin/python "$SCRIPT_CREATE" \
  --workspace-root "$WORKSPACE" \
  --dataset-root "$DATA_ROOT" \
  --jsonl-pattern stress_top6_over32k_rows.jsonl \
  --label "$LABEL" \
  --notes "container smoke 32k; optimizer=$OPTIMIZER; max_steps=$MAX_STEPS; causal_loss_mode=$CAUSAL_LOSS_MODE; causal_loss_chunk_tokens=$CAUSAL_LOSS_CHUNK_TOKENS"

SESSION_DIR="$(ls -1dt "$RUNS_ROOT"/*"${LABEL}" | head -n 1)"
if [[ -z "${SESSION_DIR:-}" || ! -d "$SESSION_DIR" ]]; then
  echo "failed: session not created for label=$LABEL" >&2
  exit 2
fi

TRAIN_LOG="$SESSION_DIR/logs/train.stdout.log"
SUMMARY_LOG="$LOG_ROOT/smoke_summary_${LABEL}.txt"
MEM_LOG="$LOG_ROOT/mem_sample_${LABEL}.tsv"
mkdir -p "$SESSION_DIR/logs"

echo -e "timestamp\tgpu_used_mib\tgpu_peak_mib\tmem_avail_mib\tswap_used_mib" > "$MEM_LOG"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v /home/georvn/train_qwen35_9b:/home/georvn/train_qwen35_9b \
  -w /home/georvn/train_qwen35_9b \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  bash -lc "
set -euo pipefail
./.venv/bin/python \"$SCRIPT_TRAIN\" \
  --session-dir \"$SESSION_DIR\" \
  --model-name Qwen/Qwen3.5-9B \
  --max-seq-length 32768 \
  --truncation-side left \
  --attn-implementation \"$ATTN_IMPLEMENTATION\" \
  --device-map cuda:0 \
  --max-steps \"$MAX_STEPS\" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-5 \
  --warmup-steps 0 \
  --weight-decay 0.01 \
  --logging-steps 1 \
  --save-strategy \"$SAVE_STRATEGY\" \
  --save-steps \"$SAVE_STEPS\" \
  --save-total-limit 2 \
  $SAVE_ONLY_MODEL_FLAG \
  --checkpoint-max-shard-size \"$CHECKPOINT_MAX_SHARD_SIZE\" \
  --checkpoint-safe-serialization \"$CHECKPOINT_SAFE_SERIALIZATION\" \
  $CHECKPOINT_PRESAVE_GC_FLAG \
  $CHECKPOINT_PRESAVE_EMPTY_CACHE_FLAG \
  $CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY_FLAG \
  --optim \"$OPTIMIZER\" \
  --gradient-checkpointing true \
  --precision auto \
  --torch-dtype bfloat16 \
  --dataset-num-proc 1 \
  --no-packing \
  --assistant-only-loss \
  --group-by-length \
  --no-load-in-4bit \
  --full-finetuning \
  --force-causal-lm-loader \
  --disable-unsloth-compile \
  --disable-moe-triton \
  --disable-flex-attention \
  $CCE_FLAG \
  --skip-merged-export \
  --skip-gguf-export \
  --skip-final-save \
  --max-gpu-memory-gib \"$INTERNAL_GPU_GUARD_GIB\" \
  --cuda-memory-fraction \"$CUDA_MEMORY_FRACTION\" \
  --cuda-alloc-conf \"$CUDA_ALLOC_CONF\" \
  --causal-loss-mode \"$CAUSAL_LOSS_MODE\" \
  --causal-loss-chunk-tokens \"$CAUSAL_LOSS_CHUNK_TOKENS\" \
  --freeze-visual-modules \
  $DEBUG_HISTORY_FLAG \
  $DEBUG_HISTORY_MAX_ENTRIES_ARG \
  $DEBUG_SNAPSHOT_FLAG \
  > \"$TRAIN_LOG\" 2>&1
" &

DOCKER_PID=$!
GPU_PEAK=0
GUARD_TRIGGERED="no"
GUARD_REASON=""

while kill -0 "$DOCKER_PID" 2>/dev/null; do
  TS="$(date -Iseconds)"
  GPU_USED="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
    | awk '{gsub(/ /, "", $1); if ($1+0 > m) m=$1+0} END {print m+0}')"
  if [[ -z "${GPU_USED:-}" ]]; then GPU_USED=0; fi
  if (( GPU_USED > GPU_PEAK )); then GPU_PEAK="$GPU_USED"; fi
  MEM_AVAIL="$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)"
  SWAP_USED="$(awk '/SwapTotal/ {tot=$2} /SwapFree/ {free=$2} END {printf "%d", (tot-free)/1024}' /proc/meminfo)"
  echo -e "${TS}\t${GPU_USED}\t${GPU_PEAK}\t${MEM_AVAIL}\t${SWAP_USED}" >> "$MEM_LOG"
  if (( GPU_GUARD_MIB > 0 && GPU_USED >= GPU_GUARD_MIB )); then
    GUARD_TRIGGERED="yes"
    GUARD_REASON="gpu_used_mib=${GPU_USED} >= gpu_guard_mib=${GPU_GUARD_MIB}"
  elif (( MIN_MEM_AVAIL_MIB > 0 && MEM_AVAIL <= MIN_MEM_AVAIL_MIB )); then
    GUARD_TRIGGERED="yes"
    GUARD_REASON="mem_avail_mib=${MEM_AVAIL} <= min_mem_avail_mib=${MIN_MEM_AVAIL_MIB}"
  fi
  if [[ "$GUARD_TRIGGERED" == "yes" ]]; then
    echo "guard_triggered ts=${TS} reason=${GUARD_REASON}" >> "$MEM_LOG"
    kill -TERM "$DOCKER_PID" 2>/dev/null || true
    sleep 2
    pkill -f "train_session.py --session-dir $SESSION_DIR" 2>/dev/null || true
    sleep 1
    kill -KILL "$DOCKER_PID" 2>/dev/null || true
    pkill -9 -f "train_session.py --session-dir $SESSION_DIR" 2>/dev/null || true
    break
  fi
  sleep 1
done

set +e
wait "$DOCKER_PID"
RC=$?
set -e

SESSION_STATUS="$(./.venv/bin/python - <<PY
import json, pathlib
path = pathlib.Path("$SESSION_DIR") / "metadata" / "session.json"
if path.exists():
    data = json.loads(path.read_text())
    print(data.get("status", "unknown"))
else:
    print("missing_session_json")
PY
)"

TRAIN_ERROR_PRESENT="no"
if [[ -f "$SESSION_DIR/metadata/train_error.json" ]]; then
  TRAIN_ERROR_PRESENT="yes"
fi

{
  echo "label=$LABEL"
  echo "optimizer=$OPTIMIZER"
  echo "session_dir=$SESSION_DIR"
  echo "train_log=$TRAIN_LOG"
  echo "mem_log=$MEM_LOG"
  echo "rc=$RC"
  echo "session_status=$SESSION_STATUS"
  echo "train_error_present=$TRAIN_ERROR_PRESENT"
  echo "gpu_peak_mib=$GPU_PEAK"
  echo "gpu_guard_gib=$GPU_GUARD_GIB"
  echo "min_mem_avail_mib=$MIN_MEM_AVAIL_MIB"
  echo "internal_gpu_guard_gib=$INTERNAL_GPU_GUARD_GIB"
  echo "cuda_alloc_conf=$CUDA_ALLOC_CONF"
  echo "causal_loss_mode=$CAUSAL_LOSS_MODE"
  echo "causal_loss_chunk_tokens=$CAUSAL_LOSS_CHUNK_TOKENS"
  echo "save_only_model=$SAVE_ONLY_MODEL"
  echo "checkpoint_max_shard_size=$CHECKPOINT_MAX_SHARD_SIZE"
  echo "checkpoint_safe_serialization=$CHECKPOINT_SAFE_SERIALIZATION"
  echo "checkpoint_presave_gc=$CHECKPOINT_PRESAVE_GC"
  echo "checkpoint_presave_empty_cache=$CHECKPOINT_PRESAVE_EMPTY_CACHE"
  echo "checkpoint_presave_disable_cuda_history=$CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY"
  echo "enable_cuda_debug_history=$ENABLE_CUDA_DEBUG_HISTORY"
  echo "debug_cuda_history_max_entries=$DEBUG_CUDA_HISTORY_MAX_ENTRIES"
  echo "debug_cuda_snapshot_on_error=$DEBUG_CUDA_SNAPSHOT_ON_ERROR"
  echo "guard_triggered=$GUARD_TRIGGERED"
  echo "guard_reason=$GUARD_REASON"
} | tee "$SUMMARY_LOG"

if [[ "$RC" -ne 0 || "$SESSION_STATUS" != "trained" ]]; then
  exit 10
fi
